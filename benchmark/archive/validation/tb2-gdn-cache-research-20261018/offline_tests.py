#!/usr/bin/env python3
"""CPU tests of extracted production lookup methods and explicit race models.

Hash lookup and GPU allocation are fixtures. These tests execute real cache
lookup/control-flow methods, but do not execute the model or CUDA kernels.
They cannot establish which defect caused the observed live-model corruption.
"""
import ast
from collections import namedtuple
from collections.abc import Sequence
import copy
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parent
LOOKUP=Path('v1/core/single_type_kv_cache_manager.py')
COORDINATOR=Path('v1/core/kv_cache_coordinator.py')
OBSERVATIONS={}


class FullAttentionSpec:
    def __init__(self,block_size=4): self.block_size=block_size


class ChunkedLocalAttentionSpec(FullAttentionSpec): pass


class MambaSpec:
    def __init__(self,block_size=4): self.block_size=block_size


class Pool:
    null_block='NULL'
    def __init__(self,entries,hash_block_size=4):
        self.entries=entries
        self.hash_block_size=hash_block_size
    def get_cached_block(self,block_hash,group_ids):
        result=[self.entries.get((gid,block_hash)) for gid in group_ids]
        return result if all(x is not None for x in result) else None


def method(path,source_class,target_class,name):
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==source_class)
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name))
    wrapper=ast.ClassDef(name=target_class,bases=[],keywords=[],body=[fn],decorator_list=[])
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),wrapper],type_ignores=[])
    namespace={'FullAttentionSpec':FullAttentionSpec,'ChunkedLocalAttentionSpec':ChunkedLocalAttentionSpec,
        'MambaSpec':MambaSpec,'Sequence':Sequence,'itertools':itertools,
        'resolve_block_hashes':lambda values,*args,**kwargs:values,
        'cdiv':lambda a,b:(a+b-1)//b}
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    result=namespace[target_class]
    result.supports_fine_grained_hash_lookup=True
    return result


FA=method(ROOT/'baseline'/LOOKUP,'FullAttentionManager','FA','find_longest_cache_hit')
MB=method(ROOT/'baseline'/LOOKUP,'MambaManager','MB','find_longest_cache_hit')
PATCHED_MB=method(ROOT/'candidate'/LOOKUP,'MambaManager','PATCHED_MB','find_longest_cache_hit')
HYBRID=method(ROOT/'baseline'/COORDINATOR,'HybridKVCacheCoordinator','HYBRID','find_longest_cache_hit')


def mamba_lookup(cls,hashes,entries,drop=True,alignment=4,hash_size=4,max_length=8):
    return cls.find_longest_cache_hit(hashes,max_length,[1],Pool(entries,hash_size),MambaSpec(),drop,alignment)


def hybrid_lookup(mamba_cls=MB,mamba_present=True):
    # Both groups are marked for EAGLE by the conservative fallback used by
    # Qwen when no group has an explicit is_eagle_group marker.
    instance=HYBRID()
    Group=namedtuple('Group','spec group_ids manager_cls use_eagle')
    instance.attention_groups=[Group(FullAttentionSpec(),[0],FA,True),Group(MambaSpec(),[1],mamba_cls,True)]
    instance.kv_cache_config=SimpleNamespace(kv_cache_groups=[None,None])
    instance.single_type_managers=[SimpleNamespace(block_size=4),SimpleNamespace(block_size=4)]
    instance.enable_partial_hash_hits=False
    instance._cache_hit_alignment_tokens=4
    instance.hash_block_size=4
    instance.dcp_world_size=instance.pcp_world_size=1
    entries={(0,'h4'):'FA4',(0,'h8'):'FA8'}
    if mamba_present: entries.update({(1,'h4'):'GD4',(1,'h8'):'GD8'})
    instance.block_pool=Pool(entries)
    return instance.find_longest_cache_hit(['h4','h8'],8)


class CacheContracts(unittest.TestCase):
    def test_native_attention_drops_trailing_block(self):
        value=FA.find_longest_cache_hit(['h4','h8'],8,[0],Pool({(0,'h4'):'FA4',(0,'h8'):'FA8'}),FullAttentionSpec(),True,4)
        self.assertEqual(value[1],4)
        OBSERVATIONS['native_attention_hit_after_drop']=value[1]

    def test_mamba_ignores_flag_in_isolation(self):
        entries={(1,'h4'):'GD4',(1,'h8'):'GD8'}
        values=[mamba_lookup(MB,['h4','h8'],entries,drop=drop)[1] for drop in (False,True)]
        self.assertEqual(values,[8,8])
        OBSERVATIONS['isolated_mamba_hits_flag_false_true']=values

    def test_native_coordinator_already_reduces_mamba_bound(self):
        result=hybrid_lookup()
        self.assertEqual(result[1],4)
        self.assertEqual(result[0],(['FA4'],['GD4']))
        OBSERVATIONS['native_coordinated_hit']=result[1]

    def test_extra_mamba_drop_discards_another_valid_checkpoint(self):
        result=hybrid_lookup(PATCHED_MB)
        self.assertEqual(result[1],0)
        OBSERVATIONS['candidate_coordinated_hit']=result[1]
        OBSERVATIONS['candidate_warning']='Extra drop reduces this valid coordinated hit from4 to0; direct flag-only tests are insufficient.'

    def test_zero_gdn_hit_forces_zero_global_hit(self):
        result=hybrid_lookup(mamba_present=False)
        self.assertEqual(result[1],0)
        self.assertEqual(result[0][0],[])
        OBSERVATIONS['no_gdn_snapshot_global_hit']=result[1]

    def test_candidate_direct_fine_grained_lookup(self):
        entries={(1,'h4'):'GD4',(1,'h8'):'GD8'}
        native=mamba_lookup(MB,['h2','h4','h6','h8'],entries,alignment=2,hash_size=2)
        candidate=mamba_lookup(PATCHED_MB,['h2','h4','h6','h8'],entries,alignment=2,hash_size=2)
        self.assertEqual([native[1],candidate[1]],[8,4])
        OBSERVATIONS['fine_grained_direct_native_candidate']=[native[1],candidate[1]]


class StatePrinciples(unittest.TestCase):
    def test_unrestored_recurrent_state_changes_next_step(self):
        # Scalar recurrence is an illustrative state-machine model, NOT GDN.
        step=lambda state,token:state*.7+token
        state=0
        for token in (1,2,3,4): state=step(state,token)
        accepted=state
        temporary=step(step(state,5),6)
        self.assertNotEqual(step(temporary,7),step(accepted,7))
        OBSERVATIONS['illustration_only_rollback_required']=True

    def test_left_overlap_allows_corrupt_interleaving_without_barrier(self):
        original=list(range(64));shift=4;size=60
        expected=original.copy();expected[:size]=original[shift:shift+size]
        # Each lane group loads then stores. Reversing group execution is a
        # valid unsynchronized schedule and destroys a later group's source.
        actual=original.copy()
        for start in reversed(range(0,size,8)):
            count=min(8,size-start)
            chunk=actual[shift+start:shift+start+count]
            actual[start:start+count]=chunk
        self.assertNotEqual(actual,expected)
        # All loads complete before any stores: memmove snapshot semantics.
        buffered=[original[shift+i] for i in range(size)]
        safe=original.copy();safe[:size]=buffered
        self.assertEqual(safe,expected)
        OBSERVATIONS['overlap_simulator_first_mismatch']=next(i for i,(a,b) in enumerate(zip(actual,expected)) if a!=b)
        OBSERVATIONS['gpu_kernel_executed']=False


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'live_model_modified':False,'model_api_calls':0,'cuda_kernels_executed':False,
        'methodology':'AST-extracted real lookup/control-flow methods with fake pre-resolved block hashes and pool. Separate explicit CPU state/race illustrations.',
        'limits':'Does not prove numerical GDN correctness or identify the live defect. No candidate was loaded into the live vLLM process.',
        'observations':OBSERVATIONS,
        'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'baseline'/LOOKUP,ROOT/'candidate'/LOOKUP,ROOT/'baseline'/COORDINATOR]}}
    (ROOT/'offline-test-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
