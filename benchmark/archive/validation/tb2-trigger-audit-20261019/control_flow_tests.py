#!/usr/bin/env python3
"""CPU-only trigger checks extracted from the deployed shared source tree.

No vLLM imports, generation calls, CUDA kernels or live mutations. Request
arrival, D2H completion and block-table inputs below are explicit fixtures.
"""
import ast
import copy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np

OUT = Path(__file__).resolve().parent
SRC = Path('/mnt/data/rui.yan/venvs/vllm/lib/python3.10/site-packages/vllm')
OBS = {}
SOURCES = set()


def node(path, name, cls=None):
    path = SRC / path
    SOURCES.add(path)
    tree = ast.parse(path.read_text())
    body = tree.body
    if cls:
        body = next(n.body for n in body if isinstance(n, ast.ClassDef) and n.name == cls)
    return copy.deepcopy(next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name))


def compile_fn(n, env):
    n.decorator_list = []
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, n], type_ignores=[]))
    exec(compile(module, '<extracted-production-control-flow>', 'exec'), env)
    return env[n.name]


def branch_fn(name, args, body, env):
    n = ast.FunctionDef(name=name, args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg=a) for a in args], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=copy.deepcopy(body), decorator_list=[])
    return compile_fn(n, env)


swap = node('v1/worker/gpu_input_batch.py', 'swap_states', 'InputBatch')
swap_stmt = next(n for n in swap.body if isinstance(n, ast.Assign)
                 and 'num_accepted_tokens_cpu' in ast.unparse(n))
swap_counts = branch_fn('swap_counts', ['self', 'i1', 'i2'], [swap_stmt], {'np': np})
prepare = node('v1/worker/gpu_model_runner.py', '_prepare_inputs', 'GPUModelRunner')
count_branch = next(n for n in ast.walk(prepare) if isinstance(n, ast.If)
                    and isinstance(n.test, ast.Name) and n.test.id == 'needs_cpu_accepted_counts')
prepare_counts = branch_fn('prepare_counts', ['self', 'num_reqs', 'prev_req_id_to_index'],
                          count_branch.body, {'np': np})
reorder = compile_fn(node('v1/attention/backends/utils.py',
                         'reorder_batch_to_split_decodes_and_prefills'), {'np': np})


class Buffer:
    def __init__(self, size, dtype=np.int64):
        self.np = np.zeros(size, dtype=dtype)
        self.gpu = self.np

    def copy_to_gpu(self, n=None):
        return self.gpu


class Batch:
    """Only the fields the real reorder algorithm reads, with real count swap."""
    def __init__(self, order=('new', 'live'), accepted=4):
        self.req_ids = list(order)
        self.num_accepted_tokens_cpu = np.array(
            [1 if r == 'new' else accepted for r in order], dtype=np.int32)
        self.num_computed_tokens_cpu = np.array(
            [33728 if r == 'new' else 5438 for r in order], dtype=np.int32)
        self.num_prompt_tokens = np.array(
            [35286 if r == 'new' else 5000 for r in order], dtype=np.int32)

    def swap_states(self, i1, i2):
        swap_counts(self, i1, i2)
        self.req_ids[i1], self.req_ids[i2] = self.req_ids[i2], self.req_ids[i1]
        for a in [self.num_computed_tokens_cpu, self.num_prompt_tokens]:
            a[i1], a[i2] = a[i2], a[i1]


def arrival_fixture(order=('new', 'live'), accepted=4, early=True, async_mode=True):
    batch = Batch(order, accepted)
    previous = {'live': list(order).index('live')}
    schedule = NS(num_scheduled_tokens={'new': 1558, 'live': 4})
    # D2H landing before the real row-reordering algorithm is one valid timing.
    # Delayed landing overwrites the same host buffer in the previous row order.
    previous_counts = batch.num_accepted_tokens_cpu.copy()
    changed = reorder(batch, schedule, decode_threshold=4)
    current_expected = [accepted if r == 'live' else 1 for r in batch.req_ids]
    def synchronize():
        if not early:
            batch.num_accepted_tokens_cpu[:] = previous_counts
    runner = NS(input_batch=batch, use_async_scheduling=async_mode,
        prev_positions=NS(np=np.array([previous.get(r, -1) for r in batch.req_ids])),
        num_accepted_tokens_event=NS(synchronize=synchronize),
        num_accepted_tokens=Buffer(2))
    prepare_counts(runner, 2, previous)
    return batch, current_expected, changed


# Execute the real fused kernel's scalar decision flow with memory operations
# replaced by scalar loads/stores and a copy-plan recorder. This is NOT CUDA.
class Ptr:
    def __init__(self, values, index=0):
        self.values, self.index = values, index
    def __add__(self, offset):
        return Ptr(self.values, self.index + int(offset))


COPIES = []
env = {'tl': NS(program_id=lambda axis: 0,
                load=lambda p: p.values[p.index],
                store=lambda p, v: p.values.__setitem__(p.index, v)),
       '_copy_mamba_state_block': lambda *args: COPIES.append(
           {'source_column': int(args[2]), 'destination_column': int(args[3]),
            'token_bias': int(args[4])})}
post_node = node('v1/worker/mamba_utils.py', 'postprocess_mamba_fused_kernel')
post = compile_fn(post_node, env)


def post_plan(computed, drafts, accepted, block=544):
    scheduled = 1 + drafts
    source = (computed + scheduled + block - 1) // block - 1
    args = {a.arg: None for a in post_node.args.args}
    args.update(num_accepted_tokens_ptr=Ptr([accepted]), mamba_state_idx_ptr=Ptr([source]),
        num_scheduled_tokens_ptr=Ptr([scheduled]), num_computed_tokens_ptr=Ptr([computed]),
        num_draft_tokens_ptr=Ptr([drafts]), num_accepted_tokens_out_ptr=Ptr([accepted]),
        num_reqs=1, block_size=block, COPY_BLOCK_SIZE=1024, CONV_STATE_DIM_FIRST=False,
        HAS_IDX_MAPPING=False, PRECOMPUTED_NEW_COMPUTED=False, TEMPORAL_TILES=16)
    COPIES.clear()
    post(**args)
    return dict(COPIES[0]) if COPIES else None


def has_overlap(plan, width=6):
    # Assume distinct physical pages for distinct live logical columns. Aliased
    # block tables are an additional condition not modelled by this fixture.
    return bool(plan and plan['source_column'] == plan['destination_column']
                and 0 < plan['token_bias'] < width - plan['token_bias'])


pre_env = {'itertools': itertools, 'cdiv': lambda a, b: (a + b - 1) // b,
           '_FusedPrecopy': lambda ctx, state_idx, src_col, token_bias:
               NS(ctx=ctx, state_idx=state_idx, src_col=src_col, token_bias=token_bias)}
for name in ['cleanup_mamba_state_idx', '_resolve_fused_precopy', 'preprocess_mamba']:
    pre = compile_fn(node('v1/worker/mamba_utils.py', name), pre_env)
preprocess = pre_env['preprocess_mamba']


def fused_precopy_plan(batch):
    # Only the live request has existing state. The new request restores its
    # input boundary separately; this fixture focuses on the live request's copy.
    batch.req_ids = ['live']
    live_count = int(batch.num_accepted_tokens_cpu[0])
    batch.num_accepted_tokens_cpu = np.array([live_count], dtype=np.int32)
    requests = {'live': NS(num_computed_tokens=5438)}
    schedule = NS(num_scheduled_tokens={'live': 4}, finished_req_ids=set(),
                  preempted_req_ids=set(), scheduled_cached_reqs=NS(resumed_req_ids=set()))
    ctx = NS(is_initialized=True, mamba_state_idx_buf=Buffer(1),
             precopy_src_col_buf=Buffer(1), precopy_token_bias_buf=Buffer(1))
    plans = []
    def record(**kwargs):
        plans.append({'source_column': int(kwargs['src_col_gpu'][0]),
                      'destination_column': int(kwargs['state_idx_gpu'][0]),
                      'token_bias': int(kwargs['token_bias_gpu'][0])})
    ctx.run_fused_precopy = record
    copy_bufs = NS(mamba_group_ids=[0],
                  mamba_spec=NS(num_speculative_blocks=3, block_size=544), offset=0)
    preprocess(schedule, NS(), NS(enable_prefix_caching=True), {'live': 9},
               batch, requests, {}, (), copy_bufs, align_ctx=ctx)
    return plans[0]


class TriggerChecks(unittest.TestCase):
    def test_real_arrival_reorder_triggers_count_misownership(self):
        batch, expected, changed = arrival_fixture()
        actual = batch.num_accepted_tokens_cpu.tolist()
        self.assertTrue(changed)
        self.assertEqual(batch.req_ids, ['live', 'new'])
        self.assertEqual(expected, [4, 1])
        self.assertEqual(actual, [1, 1])
        OBS['real_new_prefill_reorder'] = {'before_order': ['new', 'live'],
            'after_order': batch.req_ids, 'expected_counts': expected,
            'actual_counts': actual, 'previous_mapping': {'live': 1}}

    def test_wrong_count_reaches_actual_fused_precopy_metadata(self):
        bad, _, _ = arrival_fixture()
        good, _, _ = arrival_fixture(async_mode=False)
        actual, expected = fused_precopy_plan(bad), fused_precopy_plan(good)
        self.assertEqual(expected, {'source_column': 9, 'destination_column': 10, 'token_bias': 3})
        self.assertEqual(actual, {'source_column': 9, 'destination_column': 10, 'token_bias': 0})
        OBS['count_to_state_selection'] = {'expected': expected, 'actual': actual,
            'effect': 'Wrong conv offset and recurrent source slot; actual GPU copy not executed.'}

    def test_late_d2h_timing_masks_double_reorder_in_this_fixture(self):
        batch, expected, _ = arrival_fixture(early=False)
        self.assertEqual(batch.num_accepted_tokens_cpu.tolist(), expected)

    def test_no_row_reorder_preserves_count(self):
        batch, expected, changed = arrival_fixture(order=('live', 'new'))
        self.assertFalse(changed)
        self.assertEqual(batch.num_accepted_tokens_cpu.tolist(), expected)

    def test_equal_counts_mask_ownership_error(self):
        batch, expected, _ = arrival_fixture(accepted=1)
        self.assertEqual(batch.num_accepted_tokens_cpu.tolist(), expected)

    def test_nonasync_branch_preserves_current_rows(self):
        batch, expected, _ = arrival_fixture(async_mode=False)
        self.assertEqual(batch.num_accepted_tokens_cpu.tolist(), expected)

    def test_full_three_draft_window_geometry(self):
        counts = Counter()
        examples = []
        for computed in range(5440, 5984):
            for accepted in range(1, 5):
                plan = post_plan(computed, 3, accepted)
                key = 'no_copy' if not plan else ('same_column' if
                    plan['source_column'] == plan['destination_column'] else 'distinct_columns')
                counts[key] += 1
                self.assertFalse(has_overlap(plan))
                if plan and key == 'same_column': examples.append({'computed': computed,
                    'drafts': 3, 'accepted': accepted, **plan})
        self.assertEqual(counts['same_column'], 1)
        self.assertEqual(examples[0]['token_bias'], 3)
        OBS['mtp3_full_window_geometry'] = {'cases': 544 * 4,
            'counts': dict(counts), 'same_column_examples': examples,
            'overlap_cases': 0,
            'assumptions': 'Correct accepted counts, scheduled=1+drafts, 544-token blocks, unique physical pages per live logical column, width6.'}

    def test_shortened_draft_windows_can_reach_real_overlap(self):
        examples = []
        for drafts in [1, 2]:
            computed = 5440 - (drafts + 1)
            plan = post_plan(computed, drafts, drafts + 1)
            self.assertTrue(has_overlap(plan))
            examples.append({'computed': computed, 'scheduled': drafts + 1,
                'drafts': drafts, 'accepted': drafts + 1, **plan})
        OBS['shortened_draft_overlap_examples'] = examples


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TriggerChecks))
    receipt = {'created_at': datetime.now(timezone.utc).isoformat(),
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'model_api_calls': 0, 'live_service_modified': False, 'cuda_kernels_executed': False,
        'methodology': 'AST-extracted real batch reorder, count swap/prepare, fused precopy metadata and postprocess decision control flow, CPU fixtures.',
        'limits': 'Arrival, D2H timing, block tables and state counters are synthetic. No actual incident per-request GPU state or scheduler trace exists. This narrows trigger conditions but does not establish the live root cause.',
        'observations': OBS,
        'sources': {str(p.relative_to(SRC)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)}}
    (OUT / 'control-flow-test-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(OBS, indent=2))
    raise SystemExit(not result.wasSuccessful())
