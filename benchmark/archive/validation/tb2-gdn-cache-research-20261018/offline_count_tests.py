#!/usr/bin/env python3
"""Execute extracted NumPy batch-count logic on CPU; no model/CUDA imports.

The fixture provides a batch permutation and pending-copy schedules explicitly.
This proves behavior under those premises, not that the live incident used them.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parent
LIVE=Path('/mnt/data/rui.yan/venvs/vllm/lib/python3.10/site-packages/vllm/v1/worker')
OBSERVATIONS={}


def find_method(path,cls,name):
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    return next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name)


def build_function(name,args,body):
    node=ast.FunctionDef(name=name,args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=x) for x in args],kwonlyargs=[],kw_defaults=[],defaults=[]),body=copy.deepcopy(body),decorator_list=[])
    ns={'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<extracted-production-count-logic>','exec'),ns)
    return ns[name]


swap_method=find_method(LIVE/'gpu_input_batch.py','InputBatch','swap_states')
swap_statement=next(n for n in swap_method.body if isinstance(n,ast.Assign) and 'num_accepted_tokens_cpu' in ast.unparse(n))
swap_counts=build_function('swap_counts',['self','i1','i2'],[swap_statement])
prepare=find_method(LIVE/'gpu_model_runner.py','GPUModelRunner','_prepare_inputs')
branch=next(n for n in ast.walk(prepare) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='needs_cpu_accepted_counts')
prepare_counts=build_function('prepare_counts',['self','num_reqs','prev_req_id_to_index'],branch.body)


def fixture(values,previous_indices):
    batch=SimpleNamespace(num_accepted_tokens_cpu=np.array(values,dtype=np.int32))
    runner=SimpleNamespace(input_batch=batch,use_async_scheduling=True,
        prev_positions=SimpleNamespace(np=np.array(previous_indices,dtype=np.int64)),
        num_accepted_tokens_event=SimpleNamespace(synchronize=lambda:None),
        num_accepted_tokens=SimpleNamespace(np=np.zeros(len(values),dtype=np.int32),copy_to_gpu=lambda:None))
    return batch,runner


class CountOwnership(unittest.TestCase):
    def test_identity_order_does_not_expose_double_permutation(self):
        batch,runner=fixture([4,1],[0,1])
        prepare_counts(runner,2,{'A':0,'B':1})
        self.assertEqual(runner.num_accepted_tokens.np.tolist(),[4,1])

    def test_real_swap_and_real_prepare_permute_twice(self):
        batch,runner=fixture([4,1],[1,0])
        swap_counts(batch,0,1)
        expected=batch.num_accepted_tokens_cpu.copy()
        prepare_counts(runner,2,{'A':0,'B':1})
        actual=runner.num_accepted_tokens.np.copy()
        self.assertEqual(expected.tolist(),[1,4])
        self.assertEqual(actual.tolist(),[4,1])
        self.assertFalse(np.array_equal(actual,expected))
        OBSERVATIONS['native_permutation_case']={'previous_counts':[4,1],'expected_after_swap':expected.tolist(),'actual_after_prepare':actual.tolist()}

    def test_synchronous_branch_uses_already_current_order(self):
        batch,runner=fixture([4,1],[1,0]);runner.use_async_scheduling=False
        swap_counts(batch,0,1)
        prepare_counts(runner,2,{'A':0,'B':1})
        self.assertEqual(runner.num_accepted_tokens.np.tolist(),[1,4])

    def test_direct_current_order_copy_preserves_request_ownership(self):
        batch,runner=fixture([4,1],[1,0]);swap_counts(batch,0,1)
        # Candidate ownership rule from PR53919, not installed in any library.
        runner.num_accepted_tokens_event.synchronize()
        runner.num_accepted_tokens.np[:2]=batch.num_accepted_tokens_cpu[:2]
        self.assertEqual(runner.num_accepted_tokens.np.tolist(),[1,4])

    def test_delayed_d2h_can_overwrite_host_row_permutation(self):
        gpu_previous_order=np.array([4,1],dtype=np.int32)
        batch,_=fixture([4,1],[1,0]);swap_counts(batch,0,1)
        batch.num_accepted_tokens_cpu[:]=gpu_previous_order
        self.assertEqual(batch.num_accepted_tokens_cpu.tolist(),[4,1])
        self.assertNotEqual(batch.num_accepted_tokens_cpu.tolist(),[1,4])
        safe,_=fixture([0,0],[1,0])
        safe.num_accepted_tokens_cpu[:]=gpu_previous_order
        swap_counts(safe,0,1)
        self.assertEqual(safe.num_accepted_tokens_cpu.tolist(),[1,4])
        OBSERVATIONS['pending_d2h_fixture']={'unsafe':[4,1],'sync_before_swap':[1,4]}


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'live_model_modified':False,'model_api_calls':0,'cuda_kernels_executed':False,
        'methodology':'AST-extracted production accepted-count swap statement and complete guarded prepare-count branch, with NumPy CPU fixtures.',
        'limits':'Batch reorder and pending-copy conditions are fixture inputs. This does not establish that they occurred during the live corruption.',
        'observations':OBSERVATIONS,
        'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [LIVE/'gpu_input_batch.py',LIVE/'gpu_model_runner.py']}}
    (ROOT/'offline-count-test-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
