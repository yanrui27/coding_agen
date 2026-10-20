#!/usr/bin/env python3
"""Export the pinned HLE Parquet to full and question-image-free JSONL."""
import base64
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'datasets'
HLE = ROOT / 'hle'
MANIFESTS = PROJECT / 'archive/manifests'
SOURCE = HLE / 'data/test-00000-of-00001.parquet'
MEDIA_COLUMNS = ('image_preview', 'rationale_image')


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def image_extension(data):
    if data.startswith(b'\xff\xd8\xff'):
        return '.jpg'
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return '.png'
    if data.startswith((b'GIF87a', b'GIF89a')):
        return '.gif'
    if data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return '.webp'
    if data.startswith(b'BM'):
        return '.bmp'
    if data.startswith((b'II*\x00', b'MM\x00*')):
        return '.tiff'
    raise ValueError('Unrecognized image signature')


def decode_image(value):
    if not value:
        return None
    if isinstance(value, str):
        header, payload = value.split(',', 1)
        if not header.startswith('data:image/') or not header.endswith(';base64'):
            raise ValueError('Expected an embedded image data URI')
        return base64.b64decode(payload, validate=True)
    if value.get('bytes') is None:
        raise ValueError('Expected embedded image bytes in Parquet')
    return value['bytes']


def media_path(data):
    return 'images/' + hashlib.sha256(data).hexdigest() + image_extension(data)


def full_record(row, save_media=False):
    result = dict(row)
    for field in ('image', *MEDIA_COLUMNS):
        raw = decode_image(row[field])
        if raw is None:
            continue
        relative = media_path(raw)
        if save_media:
            destination = HLE / relative
            if destination.exists():
                if destination.read_bytes() != raw:
                    raise ValueError(f'Existing image content differs: {destination}')
            else:
                destination.write_bytes(raw)
        result[field] = relative if field == 'image' else {'bytes': None, 'path': relative}
    return result


def text_record(row):
    return {key: value for key, value in row.items() if key not in MEDIA_COLUMNS}


def verify_export():
    """Compare every exported record and media byte with the pinned source."""
    full_path = HLE / 'test.jsonl'
    text_path = HLE / 'text_only.jsonl'
    full_ids, text_ids, images = [], [], set()
    source_counts = {'image': 0, 'image_preview': 0, 'rationale_image': 0}
    text_with_rationale_images = 0
    with full_path.open(encoding='utf-8') as full, text_path.open(encoding='utf-8') as text:
        for batch in pq.ParquetFile(SOURCE).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                actual = json.loads(next(full))
                assert actual == full_record(row), row['id']
                full_ids.append(row['id'])
                for field in source_counts:
                    raw = decode_image(row[field])
                    if raw is not None:
                        source_counts[field] += 1
                        relative = media_path(raw)
                        assert (HLE / relative).read_bytes() == raw, relative
                        images.add(relative)
                if not row['image']:
                    actual_text = json.loads(next(text))
                    assert actual_text == text_record(row), row['id']
                    assert not actual_text['image']
                    assert all(field not in actual_text for field in MEDIA_COLUMNS)
                    text_ids.append(row['id'])
                    text_with_rationale_images += bool(row['rationale_image'])
        assert not full.read() and not text.read(), 'Unexpected extra JSONL rows'
    assert len(full_ids) == len(set(full_ids)) == 2500
    assert len(text_ids) == len(set(text_ids)) == 2158
    assert source_counts['image'] == 342
    assert images == {str(path.relative_to(HLE)) for path in (HLE / 'images').iterdir() if path.is_file()}
    artifacts = [full_path, text_path, *(HLE / name for name in sorted(images))]
    return {
        'verified': True,
        'source': str(SOURCE.relative_to(PROJECT)),
        'source_sha256': sha256(SOURCE),
        'full_jsonl': str(full_path.relative_to(PROJECT)),
        'text_only_jsonl': str(text_path.relative_to(PROJECT)),
        'full_rows': len(full_ids),
        'text_only_rows': len(text_ids),
        'question_image_rows': source_counts['image'],
        'text_only_filter': 'not bool(original image field); preserve original row order',
        'text_only_removed_columns': list(MEDIA_COLUMNS),
        'text_only_rows_with_reference_rationale_images_removed': text_with_rationale_images,
        'preserved': 'All non-media fields, including id, question, answer and rationale, are unchanged.',
        'full_media_encoding': 'image: relative path; image_preview/rationale_image: {bytes: null, path: relative path}; absent values unchanged',
        'media_directory': str((HLE / 'images').relative_to(PROJECT)),
        'media_references_by_column': source_counts,
        'unique_media_files': len(images),
        'media_bytes_unchanged': True,
        'artifacts': [{'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                       'sha256': sha256(path)} for path in artifacts],
    }, text_ids


def main():
    lock = json.loads((PROJECT / 'archive/benchmarks.lock.json').read_text())
    expected = lock['datasets']['hle']['source_verification']['sha256']
    assert sha256(SOURCE) == expected, 'Source differs from the verified official snapshot'
    (HLE / 'images').mkdir(exist_ok=True)
    full_temporary = HLE / 'test.jsonl.partial'
    text_temporary = HLE / 'text_only.jsonl.partial'
    with full_temporary.open('w', encoding='utf-8') as full, text_temporary.open('w', encoding='utf-8') as text:
        for batch in pq.ParquetFile(SOURCE).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                full.write(json.dumps(full_record(row, save_media=True), ensure_ascii=False) + '\n')
                if not row['image']:
                    text.write(json.dumps(text_record(row), ensure_ascii=False) + '\n')
    full_temporary.replace(HLE / 'test.jsonl')
    text_temporary.replace(HLE / 'text_only.jsonl')
    report, text_ids = verify_export()
    assert report['source_sha256'] == expected
    ids_path = MANIFESTS / 'hle-text-only-task-ids.txt'
    ids_path.write_text(''.join(value + '\n' for value in text_ids), encoding='utf-8')
    report['text_only_task_ids'] = str(ids_path.relative_to(PROJECT))
    write_json(MANIFESTS / 'hle-jsonl-export.json', report)
    index_path = MANIFESTS / 'core-files.sha256.json'
    index = json.loads(index_path.read_text())
    entries = {entry['path']: entry for entry in index['files']}
    entries.update({entry['path']: entry for entry in report['artifacts']})
    index['files'] = sorted(entries.values(), key=lambda entry: entry['path'])
    write_json(index_path, index)
    print(json.dumps({key: value for key, value in report.items() if key != 'artifacts'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
