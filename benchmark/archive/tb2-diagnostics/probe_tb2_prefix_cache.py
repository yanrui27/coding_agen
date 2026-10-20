#!/usr/bin/env python3
"""Replay a saved diagnostic context with warm vs isolated prefix-cache keys.

No terminal actions or grading; results are excluded from benchmark scores.
Run only after reserving one model-call slot. All calls execute sequentially.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://10.119.27.14:8200/v1')
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--timeout', type=int, default=240)
    args = parser.parse_args()
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        os.environ.pop(key, None)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    saved = json.loads(args.request.read_text())
    first_empty = next(i for i, m in enumerate(saved['messages'])
                       if m['role'] == 'assistant' and not m.get('content'))
    contexts = {'first_failure': saved['messages'][:first_empty],
                'current_loop': saved['messages']}
    args.output.mkdir(parents=True, exist_ok=True)
    state = {'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'request_sha256': hashlib.sha256(args.request.read_bytes()).hexdigest(),
        'base_url': args.base_url, 'benchmark_scoring': False,
        'protocol': 'Same saved context and seed; only per-request cache_salt differs within each pair. Both arms cap output at 2048 diagnostic tokens.',
        'results': [], 'complete': False}
    def save():
        (args.output/'receipt.json').write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
    save()
    for context, messages in contexts.items():
        for isolated in (False, True):
            name = context + ('-fresh-cache' if isolated else '-shared-cache')
            payload = copy.deepcopy(saved)
            payload['messages'] = messages
            payload['max_tokens'] = args.max_tokens
            payload.pop('cache_salt', None)
            if isolated:
                payload['cache_salt'] = secrets.token_hex(32)
            request = urllib.request.Request(args.base_url.rstrip('/')+'/chat/completions',
                data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'}, method='POST')
            started = time.monotonic()
            row = {'name':name,'context_messages':len(messages),'seed':payload.get('seed'),
                   'cache_isolated':isolated,'max_tokens':args.max_tokens}
            try:
                with opener.open(request, timeout=args.timeout) as response:
                    result = json.load(response)
                path = args.output/(name+'.json')
                path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
                choice = result['choices'][0]
                content = choice['message'].get('content') or ''
                reasoning = choice['message'].get('reasoning_content') or choice['message'].get('reasoning') or ''
                row.update(finish_reason=choice.get('finish_reason'),usage=result.get('usage'),
                           content_characters=len(content),reasoning_characters=len(reasoning),
                           content_excerpt=content[:1200],reasoning_excerpt=reasoning[:1200],
                           response_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            except Exception as exc:
                row.update(error_type=type(exc).__name__,error=str(exc))
            row['elapsed_seconds'] = time.monotonic()-started
            state['results'].append(row)
            save()
            print(json.dumps(row, ensure_ascii=False), flush=True)
    state.update(complete=True, finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save()


if __name__ == '__main__':
    main()
