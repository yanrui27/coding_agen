#!/usr/bin/env python3
"""Prepare the original shard-9 input for datasets' documented offline cache.

The cache remains host-only until the verifier phase. No task code, expected
answer, test assertion, or model-generated output is changed.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
from datasets import BuilderConfig, load_dataset
from datasets.packaged_modules.json.json import Json

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / '.cache/tb2-verifier-assets'


def main():
    receipt = json.loads((ASSETS/'c4-source-receipt.json').read_text())
    source = ASSETS/'c4-train.00009-of-01024.json.gz'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == receipt['sha256']
    builder = Json(cache_dir=str(ASSETS/'c4-build'), dataset_name='c4',
        data_files={'train': [str(source)]})
    builder.download_and_prepare()
    config = {'data_files': {'train': ['en/c4-train.00009-of-01024.json.gz']}}
    config_id = BuilderConfig('default').create_config_id(config_kwargs=config)
    assert config_id == 'default-c33333bd6ac7ada4', config_id
    cache = ASSETS/'c4-datasets-cache'
    destination = cache/'allenai___c4'/config_id/'0.0.0'/receipt['sha256']
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(builder.cache_dir, destination, dirs_exist_ok=True)
    # Exercise exactly the load_dataset request made by the original fixture.
    dataset = load_dataset('allenai/c4', data_files=config['data_files'], split='train', cache_dir=str(cache))
    assert len(dataset) == builder.info.splits['train'].num_examples
    assert set(dataset.column_names) == {'text', 'timestamp', 'url'}
    receipt.update(cache_dir=str(cache), cache_config_id=config_id,
        rows=len(dataset), columns=dataset.column_names,
        offline_original_fixture_load_passed=True,
        arrow_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.glob('*.arrow')})
    (ASSETS/'c4-cache-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
