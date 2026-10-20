#!/usr/bin/env python3
"""Inventory pinned verifier caches. These host paths are never agent mounts."""
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
    result={'schema_version':1,'delivery_phase':'verifier only, after agent has finished','tasks':tasks}
    temp = ASSETS/'manifest.json.tmp'
    temp.write_text(json.dumps(result,indent=2)+'\n')
    temp.replace(ASSETS/'manifest.json')
    print(json.dumps({n:r['ready'] for n,r in tasks.items()}))


if __name__=='__main__':
    main()
