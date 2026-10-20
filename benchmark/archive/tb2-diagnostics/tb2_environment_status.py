#!/usr/bin/env python3
"""Report the fixed 89-task diagnostic composite using latest finished retries."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'results/Qwen3.6-35B-A3B/terminal-bench-2.0'
COHORTS = ('apptainer-environment-retest17',
           'apptainer-environment-public-source-retry1',
           'apptainer-prefix-isolated-full89')
TERMINAL = {'passed', 'failed', 'timeout', 'infrastructure_error'}


def status():
    original=json.loads((BASE/'full-seed42-context-fixed/seed42/progress.json').read_text())
    combined=dict(original['tasks'])
    runs=[]
    for cohort in COHORTS:
        path=BASE/cohort/'seed42/progress.json'
        if not path.exists():
            runs.append({'cohort':cohort,'complete':False,'status':'waiting'})
            continue
        progress=json.loads(path.read_text())
        for task,row in progress['tasks'].items():
            if row['status'] in TERMINAL:
                combined[task]=row
        runs.append({'cohort':cohort,'complete':progress['complete'],
                     'updated_at':progress['updated_at'],
                     'status_counts':progress['status_counts'],
                     'running_tasks':[task for task,row in progress['tasks'].items()
                                      if row['status']=='running']})
    assert set(combined)==set(original['tasks']) and len(combined)==89
    counts=Counter(row['status'] for row in combined.values())
    original_passed=sum(row['status']=='passed' for row in original['tasks'].values())
    uniform_path=BASE/'apptainer-prefix-isolated-full89/seed42/progress.json'
    uniform=json.loads(uniform_path.read_text()) if uniform_path.exists() else {}
    uniform_counts=uniform.get('status_counts',{})
    return {'updated_at':datetime.now(timezone.utc).isoformat(),
            'all_scheduled_reruns_complete':all(run['complete'] for run in runs),
            'baseline':{'passed':original_passed,'total':89,'accuracy_percent':100*original_passed/89},
            'provisional_composite':{'passed':counts['passed'],'total':89,
                                     'accuracy_percent':100*counts['passed']/89,
                                     'status_counts':dict(counts)},
            'cohorts':runs,
            'uniform_repaired_full89':{'complete':uniform.get('complete',False),
                'status_counts':uniform_counts,
                'accuracy_percent':100*uniform_counts.get('passed',0)/89 if uniform.get('complete') else None,
                'resources_per_trial':{'cpus':8,'memory_mib':32768},
                'prefix_cache_policy':'isolated per API request'},
            'note':'Latest finished scheduled attempts replace previous results, regardless of reward. '
                   'Running attempts retain the previous result. Mixed historical resources/revisions; '
                   'not a uniform full rerun or official five-run mean. No-model regrades are excluded.'}


if __name__=='__main__':
    print(json.dumps(status(),ensure_ascii=False,indent=2))
