#!/usr/bin/env python3
"""Observe HTTP model responses in one existing trial network; no API calls.

Run with nsenter -t <trial-harbor-pid> -n. Only incoming responses from the
specified model address/port are captured. Persist metadata and short excerpts,
not requests, credentials, full reasoning, or packet payloads.
"""
import argparse
import collections
import gzip
import hashlib
import ipaddress
import json
from pathlib import Path
import socket
import struct
import time


def read_chunked(data):
    out = bytearray()
    offset = 0
    while True:
        end = data.find(b'\r\n', offset)
        if end < 0:
            return None
        size = int(data[offset:end].split(b';')[0], 16)
        offset = end + 2
        if not size:
            return bytes(out)
        if len(data) < offset + size + 2:
            return None
        out.extend(data[offset:offset + size])
        offset += size + 2


def response_metadata(data, requests=False, save_request=None):
    found = []
    offset = 0
    while True:
        start = data.find(b'POST /', offset) if requests else data.find(b'HTTP/1.1 ', offset)
        if start < 0:
            return found
        end = data.find(b'\r\n\r\n', start)
        if end < 0:
            return found
        head = data[start:end].decode('latin1').split('\r\n')
        headers = dict(line.lower().split(': ', 1) for line in head[1:] if ': ' in line)
        offset = end + 4
        body = data[offset:]
        if 'content-length' in headers:
            size = int(headers['content-length'])
            if len(body) < size:
                return found
            body = body[:size]
            offset += size
        elif headers.get('transfer-encoding') == 'chunked':
            body = read_chunked(body)
            if body is None:
                return found
        else:
            continue
        if headers.get('content-encoding') == 'gzip':
            body = gzip.decompress(body)
        try:
            response = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            continue
        if requests:
            if not isinstance(response.get('messages'), list):
                continue
            if save_request and not save_request.exists():
                save_request.write_text(json.dumps(response, ensure_ascii=False) + '\n')
                save_request.chmod(0o600)
            messages = response['messages']
            found.append({'request_parameters': {k: v for k, v in response.items()
                          if k != 'messages' and k not in {'tools', 'api_key'}},
                'body_sha256': hashlib.sha256(body).hexdigest(),
                'message_count': len(messages),
                'role_counts': dict(collections.Counter(m.get('role') for m in messages)),
                'empty_assistant_messages': sum(m.get('role') == 'assistant' and not m.get('content') for m in messages),
                'history_reasoning_characters': sum(len(m.get('reasoning_content') or '') for m in messages),
                'last_user_excerpt': next((str(m.get('content'))[-500:] for m in reversed(messages) if m.get('role') == 'user'), '')})
            continue
        if not response.get('choices'):
            continue
        choice = response['choices'][0]
        message = choice.get('message') or {}
        content = message.get('content') or ''
        reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
        found.append({
            'id': response.get('id'), 'created': response.get('created'),
            'model': response.get('model'), 'http_status': head[0],
            'finish_reason': choice.get('finish_reason'),
            'stop_reason': choice.get('stop_reason'), 'usage': response.get('usage'),
            'content_characters': len(content), 'reasoning_characters': len(reasoning),
            'content_excerpt': content[:600], 'reasoning_tail_excerpt': reasoning[-600:],
            'response_sha256': hashlib.sha256(body).hexdigest(),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-ip', default='10.119.27.14')
    parser.add_argument('--model-port', type=int, default=8200)
    parser.add_argument('--interface', default='tap0')
    parser.add_argument('--seconds', type=float, default=45)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-request-metadata', action='store_true')
    parser.add_argument('--save-first-request', type=Path,
                        help='Explicitly retain one request body locally for exact-context diagnostic replay')
    args = parser.parse_args()
    target = ipaddress.IPv4Address(args.model_ip).packed
    streams = collections.defaultdict(dict)
    count = size = 0
    started = time.time()
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3))
    sock.bind((args.interface, 0))
    sock.settimeout(1)
    try:
        while time.time() - started < args.seconds and size < 100_000_000:
            try:
                packet = sock.recv(65535)
            except socket.timeout:
                continue
            if len(packet) < 54 or packet[12:14] != b'\x08\x00':
                continue
            ip = packet[14:]
            if ip[9] != 6:
                continue
            tcp = ip[(ip[0] & 15) * 4:]
            source, dest, seq = struct.unpack('!HHI', tcp[:8])
            incoming = ip[12:16] == target and source == args.model_port
            outgoing = ip[16:20] == target and dest == args.model_port
            if not incoming and not (outgoing and args.include_request_metadata):
                continue
            payload = tcp[(tcp[12] >> 4) * 4:]
            if payload:
                streams[(source, dest, outgoing)][seq] = payload
                count += 1
                size += len(payload)
    finally:
        sock.close()
    responses = []
    requests = []
    for key, segments in streams.items():
        contiguous = bytearray()
        expected = None
        for seq, payload in sorted(segments.items()):
            if expected is not None and seq > expected:
                (requests if key[2] else responses).extend(response_metadata(bytes(contiguous), key[2], args.save_first_request))
                contiguous.clear()
            overlap = max(0, (expected or seq) - seq)
            contiguous.extend(payload[overlap:])
            expected = max(expected or 0, seq + len(payload))
        (requests if key[2] else responses).extend(response_metadata(bytes(contiguous), key[2], args.save_first_request))
    receipt = {'model_calls_added': 0, 'started_unix': started,
        'elapsed_seconds': time.time() - started, 'response_packets': count,
        'captured_payload_bytes': size, 'complete_responses': responses,
        'request_metadata': requests,
        'scope': f'{args.interface}: {args.model_ip}:{args.model_port} only',
        'saved_request': str(args.save_first_request) if args.save_first_request else None,
        'note': 'Passive sample of an existing trial; incomplete TCP/HTTP messages excluded. Full request retained only with explicit --save-first-request.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
