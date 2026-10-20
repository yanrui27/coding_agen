#!/usr/bin/env python3
"""Run a uniform repaired full89, sharing four slots with the old final trials."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'results/Qwen3.6-35B-A3B/terminal-bench-2.0'
LEGACY = BASE/'apptainer-environment-retest17/seed42/progress.json'
TERMINAL = {'passed','failed','timeout','infrastructure_error'}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def legacy_running():
    state = json.loads(LEGACY.read_text())
    rows = list(state['tasks'].values())
    assert all(row['status'] in TERMINAL | {'running'} for row in rows), 'Legacy queue still has pending tasks'
    return sum(row['status'] == 'running' for row in rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=BASE/'apptainer-prefix-isolated-full89')
    parser.add_argument('--service-unit',default='coding-agent-tb2-prefix-isolated89.service')
    parser.add_argument('--scope-prefix',default='coding-agent-tb2-prefix-isolated89')
    args=parser.parse_args()
    output=args.results_dir.resolve()
    os.chdir(ROOT)
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        os.environ.pop(key, None)
    os.environ.update(NO_PROXY='*',no_proxy='*',PIP_PROXY='',PIP_CONFIG_FILE='/dev/null')
    checks=ROOT/'archive/validation/tb2-gap-investigation-20261018'
    integration=json.loads((checks/'harbor-cache-integration.json').read_text())
    assert integration['passed'] and all(r['cached_tokens']==0 and not r['parse_error'] for r in integration['repeats'])
    assert json.loads((checks/'control-protected/install-windows-3.11/receipt.json').read_text())['passed']
    residual=ROOT/'archive/validation/tb2-residual-environment-20261018'
    assert json.loads((residual/'headless-prompt-validation.json').read_text())['runtime_prompt_normalization_passed']
    assert json.loads((residual/'qemu-v2/qemu-startup/receipt.json').read_text())['passed']
    initial=legacy_running()
    assert initial <= 3, 'No free model slot yet'
    command=['bash','run_tb2.sh','--concurrency',str(4-initial),
        '--results-dir',str(output),'--prefix-cache-policy','isolated',
        '--task-manifest','archive/manifests/terminal-bench-2.0-fixed89-priority.txt',
        '--preflight-report','archive/validation/tb2-environment-repair-20261018/rerun-gates.json',
        '--scope-prefix',args.scope_prefix,'--service-unit',args.service_unit]
    child=subprocess.Popen(command)
    def stop(signum, frame):
        if child.poll() is None:
            child.terminate()
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    last=initial
    control=output/'control.json'
    while child.poll() is None:
        running=legacy_running()
        assert running <= last, 'Legacy trial count unexpectedly increased'
        last=running
        if control.exists():
            data=json.loads(control.read_text())
            data.update(concurrency=4-running,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            save(control,data)
            save(output/'shared-slot-allocation.json',{
                'updated_at':data['updated_at'],'legacy_running':running,
                'new_trial_concurrency':4-running,'combined_trial_limit':4,
                'cpus_per_trial':8,'memory_mib_per_trial':32768})
        time.sleep(3)
    raise SystemExit(child.returncode)


if __name__=='__main__':
    main()
