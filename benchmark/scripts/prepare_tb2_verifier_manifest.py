#!/usr/bin/env python3
"""Inventory phase-separated public source and verifier input caches."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/'.cache/tb2-verifier-assets'


def main():
    tasks = {}
    for task, repo, ref, original in [
        ('build-cython-ext','pyknotid','0.5.3','https://github.com/SPOCKnots/pyknotid.git'),
        ('fix-ocaml-gc','ocaml','tag_purposefully_broken_sweeping_changes','https://github.com/sadiqj/ocaml/'),
    ]:
        source = ASSETS/(repo+'.git')
        try:
            commit = subprocess.check_output(['git','--git-dir='+str(source),'rev-parse',ref+'^{commit}'], text=True, stderr=subprocess.DEVNULL).strip()
        except subprocess.CalledProcessError:
            tasks[task] = {'ready':False}
            continue
        target = '/opt/harbor-verifier-assets/'+repo+'.git'
        tasks[task] = {'ready':True,'upstream_ref':ref,'commit':commit,
            'directories':[{'source':str(source),'target':target}],
            'git_mirrors':[{'original_url':original,'local_url':'file://'+target}],
            'env':{'GIT_CONFIG_COUNT':'2','GIT_CONFIG_KEY_0':'http.proxy','GIT_CONFIG_VALUE_0':'',
                   'GIT_CONFIG_KEY_1':'https.proxy','GIT_CONFIG_VALUE_1':''}}
    c4 = ASSETS/'c4-cache-receipt.json'
    if c4.exists():
        receipt = json.loads(c4.read_text())
        assert receipt['offline_original_fixture_load_passed'] is True
        tasks['reshard-c4-data']={'ready':True,'source_receipt':str(c4),
            'directories':[{'source':receipt['cache_dir'],'target':'/root/.cache/huggingface/datasets'}],
            'env':{'HF_DATASETS_OFFLINE':'1','HF_HUB_OFFLINE':'1'}}
    else:
        tasks['reshard-c4-data']={'ready':False}
    # The instruction explicitly asks the agent to clone this exact public
    # tag. Cache it before the task starts, without cloning into /app or
    # supplying any compatibility fixes. All grader inputs remain phase-gated.
    source = tasks['build-cython-ext']
    public = {}
    if source['ready']:
        target = '/opt/harbor-public-sources/pyknotid.git'
        public['build-cython-ext'] = {
            'ready': True, 'upstream_ref': source['upstream_ref'],
            'commit': source['commit'],
            'reason': 'Exact public repository/tag explicitly requested in instruction.md; no solution changes',
            'directories': [{'source': source['directories'][0]['source'], 'target': target}],
            'git_mirrors': [{'original_url': source['git_mirrors'][0]['original_url'], 'local_url': 'file://'+target}],
        }
    result={'schema_version':2,'delivery_phase':'tasks are verifier-only; agent_public_sources are explicitly requested public source tags',
            'tasks':tasks,'agent_public_sources':public}
    temp = ASSETS/'manifest.json.tmp'
    temp.write_text(json.dumps(result,indent=2)+'\n')
    temp.replace(ASSETS/'manifest.json')
    print(json.dumps({n:r['ready'] for n,r in tasks.items()}))


if __name__=='__main__':
    main()
