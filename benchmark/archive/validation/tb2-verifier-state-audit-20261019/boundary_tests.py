#!/usr/bin/env python3
"""Separate acceptance decisions from subsequent state ownership, on CPU only.

AST-extracted production control flow and update algebra with explicit inputs.
No trained weights, server requests, CUDA kernels, or live source mutations.
"""
import ast
import copy
from collections import Counter
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

torch.set_num_threads(1)
OUT = Path(__file__).resolve().parent
SRC = Path('/mnt/data/rui.yan/venvs/vllm/lib/python3.10/site-packages/vllm')
previous = OUT.parent / 'tb2-trigger-audit-20261019/control_flow_tests.py'
spec = importlib.util.spec_from_file_location('previous_trigger_checks', previous)
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)
OBS = {}
SOURCES = set()


def get_node(relative, name, cls=None):
    SOURCES.add(SRC / relative)
    return cf.node(relative, name, cls)


# A candidate is written only under this research directory, never imported by
# the service. It implements the two changes proposed in upstream PR53919.
def make_candidate():
    source_path = SRC / 'v1/worker/gpu_model_runner.py'
    source = source_path.read_text()
    method = get_node('v1/worker/gpu_model_runner.py', '_prepare_inputs', 'GPUModelRunner')
    outer = next(n for n in ast.walk(method) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Name) and n.test.id == 'needs_cpu_accepted_counts')
    gather = next(n for n in outer.body if isinstance(n, ast.If))
    lines = source.splitlines(keepends=True)
    direct = lines[gather.orelse[0].lineno-1:gather.orelse[0].end_lineno]
    direct = [line[4:] for line in direct]
    lines[gather.lineno-1:gather.end_lineno] = direct
    candidate = ''.join(lines)
    anchor = '        # Add the new or resumed requests to the persistent batch.\n'
    assert candidate.count(anchor) == 1
    candidate = candidate.replace(anchor,
        '        # Research candidate only: complete the D2H copy before row writes.\n'
        '        if self.num_accepted_tokens_event is not None:\n'
        '            self.num_accepted_tokens_event.synchronize()\n\n' + anchor, 1)
    candidate = candidate.replace(
        '            # Async mode: condense() reordered indices, use prev_positions mapping\n',
        '            # Counts already follow current rows; do not gather them again.\n', 1)
    path = OUT / 'candidate/v1/worker/gpu_model_runner.py'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(candidate)
    patch = ''.join(difflib.unified_diff(source.splitlines(keepends=True),
        candidate.splitlines(keepends=True),
        fromfile='a/vllm/v1/worker/gpu_model_runner.py',
        tofile='b/vllm/v1/worker/gpu_model_runner.py'))
    (OUT / 'candidate-accepted-count-ownership.patch').write_text(patch)
    cls = next(n for n in ast.parse(candidate).body if isinstance(n, ast.ClassDef)
               and n.name == 'GPUModelRunner')
    prepared = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_prepare_inputs')
    branch = next(n for n in ast.walk(prepared) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Name) and n.test.id == 'needs_cpu_accepted_counts')
    fn = cf.branch_fn('candidate_prepare', ['self', 'num_reqs', 'prev_req_id_to_index'],
                      branch.body, {'np': np})
    update = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_update_states')
    wait_line = next(n.lineno for n in ast.walk(update) if isinstance(n, ast.Call)
                     and ast.unparse(n.func) == 'self.num_accepted_tokens_event.synchronize')
    add_line = next(n.lineno for n in ast.walk(update) if isinstance(n, ast.Call)
                    and ast.unparse(n.func) == 'self.input_batch.add_request')
    assert wait_line < add_line
    OBS['candidate'] = {'path': str(path.relative_to(OUT)),
        'sha256': hashlib.sha256(candidate.encode()).hexdigest(),
        'upstream': 'https://github.com/vllm-project/vllm/pull/53919',
        'changes': ['wait before add/reorder writes', 'read counts directly in current row order'],
        'applied_to_live': False, 'gpu_validated': False}
    return fn


candidate_prepare = make_candidate()
add = get_node('v1/worker/gpu_input_batch.py', 'add_request', 'InputBatch')
reset = next(n for n in ast.walk(add) if isinstance(n, ast.Assign)
             and 'num_accepted_tokens_cpu' in ast.unparse(n))
reset_count = cf.branch_fn('reset_count', ['self', 'req_index'], [reset], {'np': np})


class RowBatch:
    def __init__(self, n, new_rows, accepted, prefix_hit):
        self.req_ids = [f'new{i}' if i in new_rows else f'live{i}' for i in range(n)]
        self.num_accepted_tokens_cpu = np.ones(n, dtype=np.int32)
        self.num_computed_tokens_cpu = np.array(
            [(33728 if prefix_hit else 0) if i in new_rows else 5438 for i in range(n)])
        self.num_prompt_tokens = np.array([35286 if i in new_rows else 5000 for i in range(n)])

    def swap_states(self, i1, i2):
        cf.swap_counts(self, i1, i2)
        self.req_ids[i1], self.req_ids[i2] = self.req_ids[i2], self.req_ids[i1]
        for array in [self.num_computed_tokens_cpu, self.num_prompt_tokens]:
            array[i1], array[i2] = array[i2], array[i1]


def lifecycle(n, new_rows, accepted, timing, prefix_hit, wait_before, direct):
    batch = RowBatch(n, new_rows, accepted, prefix_hit)
    previous_map = {rid:i for i, rid in enumerate(batch.req_ids) if i not in new_rows}
    expected_map = {rid:(1 if i in new_rows else accepted[i]) for i, rid in enumerate(batch.req_ids)}
    # Removed request rows still occur in the previous GPU copy. New admission
    # must reset those rows AFTER the copy is known complete.
    previous_gpu = np.array([2 if i in new_rows else accepted[i] for i in range(n)], dtype=np.int32)
    pending = True
    def land():
        nonlocal pending
        if pending:
            batch.num_accepted_tokens_cpu[:] = previous_gpu
            pending = False
    if timing == 'before_admit' or wait_before:
        land()
    for row in new_rows:
        reset_count(batch, row)
    if timing == 'after_admit':
        land()
    schedule = NS(num_scheduled_tokens={rid:(544 if rid.startswith('new') else 4)
                                      for rid in batch.req_ids})
    changed = cf.reorder(batch, schedule, decode_threshold=4)
    if timing == 'after_reorder':
        land()
    expected = [expected_map[rid] for rid in batch.req_ids]
    runner = NS(input_batch=batch, use_async_scheduling=True,
        prev_positions=NS(np=np.array([previous_map.get(rid, -1) for rid in batch.req_ids])),
        num_accepted_tokens_event=NS(synchronize=land), num_accepted_tokens=cf.Buffer(n))
    (candidate_prepare if direct else cf.prepare_counts)(runner, n, previous_map)
    actual = runner.num_accepted_tokens.np[:n].astype(int).tolist()
    return {'order': batch.req_ids, 'expected': expected, 'actual': actual,
            'row_reordered': changed, 'matches': actual == expected}


class Scalar(int):
    def to(self, dtype):
        return self


class Ptr:
    def __init__(self, values, index=0):
        self.values, self.index = values, int(index)
        self.dtype = NS(element_ty='int32')
    def __add__(self, index):
        return Ptr(self.values, self.index + int(index))
    def __sub__(self, index):
        return Ptr(self.values, self.index - int(index))


class ScalarTL:
    int32, int64 = 'int32', 'int64'
    request_row = 0
    def program_id(self, axis):
        return self.request_row
    def zeros(self, shape, dtype):
        return Scalar(0)
    def load(self, pointer):
        value = pointer.values[pointer.index]
        return Scalar(value) if isinstance(value, (int, np.integer)) else value
    def store(self, pointer, value):
        pointer.values[pointer.index] = int(value)


tl = ScalarTL()
greedy = cf.compile_fn(get_node('v1/sample/rejection_sampler.py', 'rejection_greedy_sample_kernel'), {'tl':tl})
random_accept = cf.compile_fn(get_node('v1/sample/rejection_sampler.py', 'rejection_random_sample_kernel'), {'tl':tl})


def sample_fixture(random_mode, no_draft_probs=False):
    # Request B rejects its first proposal. Request C accepts all3 plus bonus.
    draft = [1,2,3,4,5,6]
    target_ids = [9,2,3,4,5,6]
    output = [-1]*8
    args = dict(output_token_ids_ptr=Ptr(output), cu_num_draft_tokens_ptr=Ptr([3,6]),
        draft_token_ids_ptr=Ptr(draft), bonus_token_ids_ptr=Ptr([8,7]),
        max_spec_len=3, uniform_probs_ptr=Ptr([0.5]*6),
        synthetic_conditional_rates_ptr=None, SYNTHETIC_MODE=False)
    if random_mode:
        vocab = 16
        target_probs, draft_probs = [], []
        for i, token in enumerate(draft):
            p = 0.05 if i == 0 else 0.8
            target = [(1-p)/(vocab-1)]*vocab; target[token] = p
            proposal = [0.5/(vocab-1)]*vocab; proposal[token] = 0.5
            target_probs.extend(target); draft_probs.extend(proposal)
        args.update(draft_probs_ptr=Ptr(draft_probs), target_probs_ptr=Ptr(target_probs),
            recovered_token_ids_ptr=Ptr(target_ids), is_greedy_ptr=Ptr([False,False]),
            vocab_size=vocab, NO_DRAFT_PROBS=no_draft_probs)
    else:
        args.update(target_argmax_ptr=Ptr(target_ids), is_greedy_ptr=None)
    for row in range(2):
        tl.request_row = row
        (random_accept if random_mode else greedy)(**args)
    tl.request_row = 0
    output = torch.tensor(output, dtype=torch.int32).reshape(2,4)
    # Extract the actual statements that derive counts from sampler output.
    update = get_node('v1/worker/gpu_model_runner.py', '_update_states_after_model_execute', 'GPUModelRunner')
    counts_body = [n for n in update.body if isinstance(n, ast.Assign) and (
        any(isinstance(t, ast.Name) and t.id == 'num_reqs' for t in n.targets)
        or 'self.num_accepted_tokens.gpu' in ast.unparse(n))]
    count = cf.branch_fn('count_outputs', ['self', 'output_token_ids'], counts_body, {})
    runner = NS(num_accepted_tokens=NS(gpu=torch.ones(2, dtype=torch.int32)))
    count(runner, output)
    return output.tolist(), runner.num_accepted_tokens.gpu.tolist()


# Extract the numerical update algebra from the actual GDN GPU kernel. GPU
# loads are replaced by CPU tensor clones, before the first store is reached.
gdn_file = 'third_party/flash_linear_attention/ops/fused_sigmoid_gating.py'
gdn = get_node(gdn_file, 'fused_sigmoid_gating_delta_rule_update_kernel')
loop = next(n for n in gdn.body if isinstance(n, ast.For))
algebra = []
for statement in loop.body:
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) and ast.unparse(statement.value.func) == 'tl.store':
        break
    algebra.append(statement)
algebra.append(ast.Return(value=ast.Tuple(elts=[ast.Name(id='b_o',ctx=ast.Load()),
                                               ast.Name(id='b_h',ctx=ast.Load())],ctx=ast.Load())))
math_tl = NS(load=lambda value, **kwargs:value.clone(), float32=torch.float32,
    where=torch.where, log=torch.log, exp=torch.exp, sigmoid=torch.sigmoid,
    rsqrt=torch.rsqrt, sum=lambda value, axis=None:torch.sum(value, dim=axis))
gdn_step = cf.branch_fn('gdn_step', ['b_h','p_q','p_k','p_v','p_b','p_a','p_dt_bias',
    'p_A_log','beta','threshold','scale','USE_QK_L2NORM_IN_KERNEL','IS_KDA'], algebra,
    {'tl':math_tl,'mask_k':True,'mask_v':True})

init = next(n for n in gdn.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'USE_INITIAL_STATE')
continuous = next(n for n in init.body if isinstance(n, ast.If))
spec_decode = next(n for n in continuous.body if isinstance(n, ast.If))
slot_assign = next(n for n in continuous.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'state_idx' for t in n.targets))
selector_body = copy.deepcopy(spec_decode.body + [slot_assign])
selector_body.append(ast.Return(value=ast.Name(id='state_idx',ctx=ast.Load())))
select_slot = cf.branch_fn('select_slot', ['num_accepted_tokens','ssm_state_indices',
    'i_n','stride_indices_seq'], selector_body, {'tl':tl})


class BoundaryChecks(unittest.TestCase):
    def test_acceptance_control_for_greedy_and_random_paths(self):
        rows = []
        for random_mode, no_probs in [(False,False),(True,False),(True,True)]:
            output, counts = sample_fixture(random_mode, no_probs)
            self.assertEqual(output, [[9,-1,-1,-1],[4,5,6,7]])
            self.assertEqual(counts, [1,4])
            rows.append({'random':random_mode,'no_draft_probs':no_probs,'output':output,'counts':counts})
        OBS['acceptance_controls'] = rows

    def test_correct_acceptance_can_be_followed_by_wrong_state_ownership(self):
        _, verified_counts = sample_fixture(True)
        result = lifecycle(3, {0}, {1:verified_counts[0],2:verified_counts[1]},
                           'before_admit', True, False, False)
        # The backend may reverse requests WITHIN the decode region. Check
        # request ownership, not an assumed stable order within that region.
        expected = dict(zip(result['order'],result['expected']))
        actual = dict(zip(result['order'],result['actual']))
        self.assertEqual(expected['live2'],4)
        self.assertEqual(actual['live2'],1)
        self.assertEqual(expected['live1'],1)
        OBS['post_verification_state_witness'] = {
            'before_order':['new','B_rejected_first','C_accepted_all'],
            **result, 'meaning':'Correctly verified C count4 becomes1 after host row handling; this is a synthetic lifecycle, not a captured incident.'}

    def test_candidate_against_batch_size_2_to_4_and_copy_timings(self):
        arms = {'original':(False,False),'wait_only':(True,False),
                'direct_only':(False,True),'both':(True,True)}
        stats = {name:Counter() for name in arms}
        timings = ['before_admit','after_admit','after_reorder']
        cases = 0
        for n in range(2,5):
            for num_new in range(1,n):
                for new_tuple in itertools.combinations(range(n),num_new):
                    new_rows = set(new_tuple)
                    live_rows = [i for i in range(n) if i not in new_rows]
                    for values in itertools.product(range(1,5),repeat=len(live_rows)):
                        accepted = dict(zip(live_rows,values))
                        for prefix_hit,timing in itertools.product([False,True],timings):
                            cases += 1
                            for name,(wait,direct) in arms.items():
                                result = lifecycle(n,new_rows,accepted,timing,prefix_hit,wait,direct)
                                stats[name]['cases'] += 1
                                stats[name]['mismatches'] += not result['matches']
                                if name == 'original' and 'overadvance_witness' not in OBS and any(
                                        a > e for a,e in zip(result['actual'],result['expected'])):
                                    OBS['overadvance_witness'] = {'n':n,'new_rows':sorted(new_rows),
                                        'accepted_by_old_row':accepted,'timing':timing,
                                        'prefix_hit':prefix_hit,**result,
                                        'limits':'Explicit CPU timing fixture; not an observed rejected-draft replay on the live model.'}
        self.assertEqual(stats['both']['mismatches'],0)
        for name in ['original','wait_only','direct_only']:
            self.assertGreater(stats[name]['mismatches'],0)
        OBS['ownership_matrix'] = {'cases_per_arm':cases,'arms':{name:dict(c) for name,c in stats.items()},
            'limits':'Enumerated fixture counts are not real failure rates. Covers2-4 requests, at least1 new and1 live, accepted counts1-4, cold/warm prefill,3 D2H schedules. No real CUDA concurrency.'}

    def test_wrong_selector_changes_actual_gdn_update_algebra(self):
        generator = torch.Generator().manual_seed(1942)
        # One128x128 head uses actual head dimensions, but synthetic q/k/v and
        # gates. No checkpoint weights or vocabulary logits are used.
        features = [(torch.randn(128,generator=generator),torch.randn(128,generator=generator),
                     torch.randn(128,generator=generator)) for _ in range(5)]
        state = torch.zeros(128,128)
        states = {}
        def step(previous, triple):
            q,k,v = triple
            return gdn_step(previous.clone(),q,k,v,torch.tensor(0.2),torch.tensor(0.1),
                torch.tensor(0.),torch.tensor(-1.),1.,20.,128**-0.5,True,False)
        for slot,triple in zip([10,11,12,13],features):
            _,state = step(state,triple)
            states[slot] = state.clone()
        correct_slot = int(select_slot(Ptr([4]),Ptr([10,11,12,13]),0,4))
        wrong_slot = int(select_slot(Ptr([1]),Ptr([10,11,12,13]),0,4))
        self.assertEqual((correct_slot,wrong_slot),(13,10))
        expected,_ = step(states[correct_slot],features[-1])
        actual,_ = step(states[wrong_slot],features[-1])
        max_delta = float((expected-actual).abs().max())
        self.assertGreater(max_delta,1e-6)
        self.assertTrue(torch.isfinite(actual).all())
        OBS['gdn_numeric_witness'] = {'expected_slot':correct_slot,'wrong_slot':wrong_slot,
            'output_max_abs_delta':max_delta,'both_outputs_finite':True,
            'meaning':'The wrong slot changes the GDN output vector without a dimension or NaN error.',
            'limits':'CPU execution of extracted update algebra, synthetic one-head inputs. Not GPU, trained-model generation, token logits, or proof of gibberish.'}


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BoundaryChecks))
    receipt = {'created_at':datetime.now(timezone.utc).isoformat(),
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'model_api_calls':0,'cuda_kernels_executed':False,'live_service_modified':False,
        'methodology':'AST-extracted acceptance kernels, actual count derivation, batch reorder/prepare, candidate branch, GDN slot selector and update algebra; CPU pointer/tensor emulation.',
        'limits':'No actual faulting request state, GPU event ordering, checkpoint weights or model accuracy test. Passing a candidate fixture is not deployment approval.',
        'observations':OBS,'sources':{str(p.relative_to(SRC)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)}}
    (OUT/'boundary-test-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(OBS,indent=2))
    raise SystemExit(not result.wasSuccessful())
