#!/usr/bin/env python3
"""Publish live seed summaries and retire the old path after this running job ends."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tb2', ROOT / 'scripts/eval_tb2.py')
tb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tb)


def write_json(path, data):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        temporary = Path(stream.name)
    temporary.replace(path)


def publish():
    for seed in tb.configured_seeds():
        directory = tb.RESULTS / f'seed{seed}'
        progress = directory / 'progress.json'
        if progress.exists():
            for name, row in json.loads(progress.read_text())['tasks'].items():
                if 'attempt' in row:
                    tb.expose_trial_files(directory / name, row['attempt'])
    write_json(tb.RESULTS / 'summary.json', tb.suite_summary(write=False))


def main():
    while True:
        publish()
        state = subprocess.check_output(['systemctl', 'show', 'coding-agent-tb2.service',
                                         '-p', 'ActiveState', '--value'], text=True).strip()
        if state in ('inactive', 'failed'):
            break
        time.sleep(15)
    while subprocess.check_output(['systemctl', 'list-units', 'coding-agent-tb2-s*.scope',
                                   '--state=active,activating,deactivating', '--no-legend', '--plain'], text=True).strip():
        time.sleep(2)
    # The former coordinator used this alias while its task processes were active.
    alias = tb.RESULTS / 'five-seed'
    for seed in tb.configured_seeds():
        path = tb.RESULTS / f'seed{seed}/progress.json'
        if not path.exists():
            continue
        progress = json.loads(path.read_text())
        for row in progress['tasks'].values():
            if row.get('result_path'):
                row['result_path'] = row['result_path'].replace('/terminal-bench-2.0/five-seed/', '/terminal-bench-2.0/')
        write_json(path, progress)
        if progress.get('complete'):
            tb.write_seed_report(seed)
    publish()
    if alias.is_symlink() and alias.resolve() == tb.RESULTS:
        alias.unlink()
    # The seed progress and top-level summary replace the former aggregate progress.
    (tb.RESULTS / 'progress.json').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
