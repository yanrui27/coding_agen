#!/usr/bin/env python3
"""Verify fixed datasets and record versions, task IDs and image references."""
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib

import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'datasets'
MANIFESTS = PROJECT / 'archive' / 'manifests'
THIRD_PARTY = PROJECT / 'third_party'
LOCK = json.loads((PROJECT / 'archive/benchmarks.lock.json').read_text())


def load(relative):
    return json.loads((MANIFESTS / relative).read_text())


def revision(directory):
    return subprocess.check_output(['git', '-C', str(THIRD_PARTY / directory), 'rev-parse', 'HEAD'], text=True).strip()


def write_ids(name, values):
    path = MANIFESTS / (name + '-task-ids.txt')
    path.write_text(''.join(value + '\n' for value in values))
    return str(path.relative_to(PROJECT))


def main():
    report = {'datasets': {}}
    downloads = load('core-files.sha256.json')
    expected_paths = {row['path'] for row in downloads['files']}
    actual_paths = {str(path.relative_to(ROOT)) for directory in ROOT.iterdir() if directory.is_dir()
                    for path in directory.rglob('*') if path.is_file()}
    assert actual_paths == expected_paths, (actual_paths - expected_paths, expected_paths - actual_paths)
    for row in downloads['files']:
        path = ROOT / row['path']
        assert not path.is_symlink(), path
        assert path.is_file(), path
        assert path.stat().st_size == row['bytes'], path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], path

    ids = (MANIFESTS / 'swe-hard51-task-ids.txt').read_text().splitlines()
    assert len(ids) == len(set(ids)) == 51
    tasks = ROOT / 'swe-bench-pro-v2-hard51'
    assert len(list(tasks.glob('*/task.toml'))) == 51
    hard_table = pq.read_table(tasks / 'data/test-00000-of-00001.parquet')
    assert hard_table.num_rows == 51
    assert set(hard_table['instance_id'].to_pylist()) == set(ids)
    subset = tasks
    for task_id in ids:
        task = tasks / task_id
        assert (task / 'instruction.md').is_file()
        assert (task / 'tests/test.sh').is_file()
        assert (task / 'environment/Dockerfile').is_file()
        tomllib.loads((task / 'task.toml').read_text())
    report['datasets']['swe_bench_pro_v2_hard'] = {
        'code_commit': revision('swe-bench-pro'),
        'hf_repo': LOCK['datasets']['swe_bench_pro_v2_hard']['hf_repo'],
        'hf_revision': LOCK['datasets']['swe_bench_pro_v2_hard']['hf_revision'],
        'full_v2_tasks': 642, 'full_v2_parquet_retained': False,
        'code_directory': 'third_party/swe-bench-pro',
        'code_checkout': 'sparse: V2 locked-protocol tooling, documentation and release checksums; V1 assets and duplicate task trees excluded',
        'source_task_copies_retained': False,
        'hard_tasks': 51, 'task_ids': write_ids('swe-hard51', ids),
        'hard_directory': str(subset.relative_to(PROJECT))}
    for name in ['patch_replay.py', 'locked_mini_swe.py', 'locked_claude_code.py', 'locked_codex.py']:
        assert (THIRD_PARTY / 'swe-bench-pro/v2/tooling' / name).is_file(), name

    hle_verification = LOCK['datasets']['hle']['source_verification']
    assert hle_verification['verified_byte_identical_to_official_snapshot']
    assert hle_verification['official_repo'] == LOCK['datasets']['hle']['hf_repo']
    assert hle_verification['official_revision'] == LOCK['datasets']['hle']['hf_revision']
    pointer = ('version https://git-lfs.github.com/spec/v1\n'
               f"oid sha256:{hle_verification['sha256']}\nsize {hle_verification['bytes']}\n").encode()
    assert hashlib.sha1(f'blob {len(pointer)}\0'.encode() + pointer).hexdigest() == hle_verification['official_pointer_oid']
    assert (ROOT / 'hle/data/test-00000-of-00001.parquet').stat().st_size == hle_verification['bytes']
    hle = pq.ParquetFile(ROOT / 'hle/data/test-00000-of-00001.parquet')
    assert hle.metadata.num_rows == 2500
    hle_ids = hle.read(columns=['id']).column('id').to_pylist()
    assert len(set(hle_ids)) == 2500
    image_count = sum(bool(value) for batch in hle.iter_batches(batch_size=128, columns=['image'])
                      for value in batch.column(0).to_pylist())
    report['datasets']['hle'] = {
        'code_commit': revision('hle'), 'code_directory': 'third_party/hle',
        'hf_repo': LOCK['datasets']['hle']['hf_repo'],
        'hf_revision': LOCK['datasets']['hle']['hf_revision'],
        'split': 'test', 'tasks': 2500, 'image_tasks': image_count, 'text_only_tasks': 2500-image_count,
        'task_ids': write_ids('hle', hle_ids), 'parquet_sha256': hle_verification['sha256'],
        'official_snapshot_verified': True, 'source_verification': hle_verification}
    if (MANIFESTS / 'hle-jsonl-export.json').is_file():
        from export_hle_jsonl import verify_export
        export = load('hle-jsonl-export.json')
        verified_export, text_ids = verify_export()
        assert verified_export == {key: value for key, value in export.items() if key != 'text_only_task_ids'}
        assert export['source_sha256'] == hle_verification['sha256']
        assert (PROJECT / export['text_only_task_ids']).read_text().splitlines() == text_ids
        report['datasets']['hle']['jsonl_export'] = {
            key: export[key] for key in ('full_jsonl', 'text_only_jsonl', 'full_rows', 'text_only_rows',
                                        'text_only_filter', 'text_only_removed_columns', 'text_only_task_ids',
                                        'media_directory', 'unique_media_files')}
        report['datasets']['hle']['jsonl_export']['verification'] = str(
            (MANIFESTS / 'hle-jsonl-export.json').relative_to(PROJECT))

    asi_metadata = json.loads((ROOT / 'asi-bench-seed42/metadata.json').read_text())
    asi_tasks = sorted(path for path in (ROOT / 'asi-bench-seed42/tasks').iterdir() if path.is_dir())
    assert len(asi_tasks) == asi_metadata['task_count'] == 60
    for task in asi_tasks:
        for level in range(1, 5):
            assert (task / f'prompt_b{level}.md').is_file(), task
        domain, name = task.name.removesuffix('__seed42').split('.', 1)
        assert (THIRD_PARTY / 'asi-bench/tasks' / domain / name / 'task_eval.yaml').is_file(), task
    report['datasets']['asi_bench'] = {
        'code_commit': revision('asi-bench'), 'code_directory': 'third_party/asi-bench',
        'hf_repo': LOCK['datasets']['asi_bench']['hf_repo'],
        'hf_revision': LOCK['datasets']['asi_bench']['hf_revision'],
        'seed': 42, 'tasks': 60, 'prompt_levels': ['b1', 'b2', 'b3', 'b4'], 'task_conditions': 240,
        'private_reference_answers_included': False,
        'tasks_with_inline_inputs_only': [path.name for path in asi_tasks if not (path / 'data').is_dir()],
        'task_ids': write_ids('asi-seed42', [path.name for path in asi_tasks]),
        'code_checkout': 'sparse: runtime, tasks, configuration, tests and guides; large documentation image excluded'}

    registry = load('terminal-bench-2.0-registry.json')
    assert registry['version'] == '2.0' and len(registry['tasks']) == 89
    commits = {entry['git_commit_id'] for entry in registry['tasks']}
    terminal_commit = LOCK['datasets']['terminal_bench']['code_commit']
    assert len(commits) == 1 and terminal_commit in commits
    for entry in registry['tasks']:
        task = ROOT / 'terminal-bench-2.0' / entry['path']
        assert (task / 'instruction.md').is_file(), task
        assert (task / 'tests/test.sh').is_file(), task
        tomllib.loads((task / 'task.toml').read_text())
    report['datasets']['terminal_bench'] = {
        'version': '2.0', 'code_commit': terminal_commit, 'tasks': 89,
        'source_git_metadata_retained': False,
        'task_ids': write_ids('terminal-bench-2.0', [entry['name'] for entry in registry['tasks']]),
        'container_images_prepulled': False}
    (PROJECT / 'archive/benchmarks.lock.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'verified': True, **report}, indent=2), flush=True)


if __name__ == '__main__':
    main()
