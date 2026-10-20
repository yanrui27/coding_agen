#!/usr/bin/env python3
"""Read-only incident and trajectory audit; persists local research artifacts."""
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[4]
BENCH = PROJECT / 'data/benchmark'
RESULTS = BENCH / 'results/Qwen3.6-35B-A3B/terminal-bench-2.0'
RESEARCH = OUT.parent / 'tb2-gdn-cache-research-20261018'
GAP = OUT.parent / 'tb2-gap-investigation-20261018'
LIVE = Path('/mnt/data/rui.yan/venvs/vllm/lib/python3.10/site-packages/vllm')


def load(p):
    return json.loads(p.read_text())


def parse_error(step):
    feedback = '\n'.join(v.get('content') or '' for v in
        (step.get('observation') or {}).get('results', []))
    return 'Previous response had parsing errors:' in feedback


def has_command_json(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get('commands'), list):
            return True
    return False


def cohort_audit(cohort):
    totals = Counter()
    affected = []
    seen = set()
    for path in sorted((RESULTS / cohort / 'seed42').glob('*/attempt*/*/agent/trajectory*.json')):
        try:
            doc = load(path)
        except (ValueError, OSError):
            continue
        totals['trajectory_files'] += 1
        task = path.parents[3].name
        errors, longest, empty_longest, streak, empty_streak = 0, 0, 0, 0, 0
        for step in doc.get('steps', []):
            if step.get('source') != 'agent' or step.get('is_copied_context'):
                continue
            key = task, step.get('timestamp'), step.get('step_id')
            if key in seen:
                continue
            seen.add(key)
            totals['agent_turns'] += 1
            metrics = step.get('metrics') or {}
            prompt = metrics.get('prompt_tokens') or 0
            totals['max_prompt_tokens'] = max(totals['max_prompt_tokens'], prompt)
            totals['turns_at_least_first_failure_length'] += prompt >= 35286
            totals['turns_with_cache_hit'] += bool(metrics.get('cached_tokens'))
            error = parse_error(step)
            empty = error and not step.get('message') and bool(step.get('reasoning_content'))
            errors += error
            totals['parse_errors'] += error
            totals['empty_body_parse_errors'] += empty
            if empty:
                totals['empty_body_with_command_json_in_reasoning'] += has_command_json(step['reasoning_content'])
            streak = streak + 1 if error else 0
            empty_streak = empty_streak + 1 if empty else 0
            longest, empty_longest = max(longest, streak), max(empty_longest, empty_streak)
        totals['max_consecutive_parse_errors'] = max(totals['max_consecutive_parse_errors'], longest)
        totals['max_consecutive_empty_errors'] = max(totals['max_consecutive_empty_errors'], empty_longest)
        if errors:
            affected.append({'task': task, 'trajectory': str(path.relative_to(PROJECT)),
                'parse_errors': errors, 'longest_parse_error_streak': longest})
    return {'totals': dict(totals), 'affected': affected,
        'limits': 'Snapshot of saved trajectories, including running tasks; copied continuation context skipped. Empty body does not by itself imply numerical corruption.'}


def main():
    old = load(RESULTS / 'apptainer-environment-retest17/seed42/progress.json')
    trial = (BENCH / old['tasks']['qemu-alpine-ssh']['result_path']).parent
    result = load(trial / 'result.json')
    steps = [s for s in load(trial / 'agent/trajectory.json')['steps'] if s.get('source') == 'agent']
    bad_index = next(i for i, s in enumerate(steps) if parse_error(s))
    first = steps[bad_index]
    latency = result['agent_result']['metadata']['api_request_times_msec'][bad_index]
    response_time = datetime.fromisoformat(first['timestamp'])
    messages = load(GAP / 'alpine-replay-request.json')['messages']
    first_empty = next(i for i, m in enumerate(messages)
                       if m['role'] == 'assistant' and not m.get('content'))
    first_bad_reasoning = first.get('reasoning_content') or ''
    body = json.dumps(messages, ensure_ascii=False)

    log = PROJECT / 'logs/coding-agent-qwen36-35b-a3b-tp4-1016133720/Qwen3.6-35B-A3B-20261016060735.log'
    # Keep a stable excerpt: the service log itself continues to grow.
    excerpt = []
    for line in log.read_text(errors='replace').splitlines():
        if re.search(r'10-18 12:(?:1[56789]|20):', line):
            excerpt.append(line)
    (OUT / 'first-failure-service-window.log').write_text('\n'.join(excerpt) + '\n')
    metrics = []
    for line in excerpt:
        time = re.search(r'10-18 (\d\d:\d\d:\d\d)', line)
        counts = re.search(r'Running: (\d+) reqs, Waiting: (\d+) reqs', line)
        if time and counts:
            metrics.append({'time_utc': time[1], 'running': int(counts[1]), 'waiting': int(counts[2])})

    checks = {}
    for snapshot in (RESEARCH / 'baseline').rglob('*.py'):
        rel = snapshot.relative_to(RESEARCH / 'baseline')
        checks[str(rel)] = hashlib.sha256(snapshot.read_bytes()).hexdigest() == hashlib.sha256((LIVE / rel).read_bytes()).hexdigest()
    scheduler = (LIVE / 'v1/core/sched/scheduler.py').read_text()
    (OUT / 'scheduler-source-snapshot.py').write_text(scheduler)
    comparison = {
        'shared_source_unchanged_from_prior_snapshots': checks,
        'scheduler_contains_prefill_end_alignment': 'if end < prefill_end:' in scheduler,
        'scheduler_old_last_cache_condition_present': 'next_block_boundary <= last_cache_position' in scheduler,
        'copy_overlap_guard_present': 'is_left_overlap' in (LIVE / 'v1/worker/mamba_utils.py').read_text(),
        'accepted_count_gather_present': 'np.where(new_mask, 0, prev_idx)' in (LIVE / 'v1/worker/gpu_model_runner.py').read_text(),
        'sources': {str(p.relative_to(LIVE)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [LIVE / 'v1/core/sched/scheduler.py', LIVE / 'v1/worker/mamba_utils.py',
                              LIVE / 'v1/worker/gpu_model_runner.py']},
        'runtime_flag_limit': 'Startup has no async override and the shared default enables async with this supported MTP executor. Runtime scheduler configuration not directly obtained: read-only /server_info returned404. No endpoint was enabled.'}
    data = {'created_at': datetime.now(timezone.utc).isoformat(),
        'live_service_modified': False, 'generation_calls_added': 0,
        'original_incident': {
            'trial': str(trial.relative_to(PROJECT)), 'good_replies_before_first_error': bad_index,
            'agent_replies': len(steps), 'parse_errors': sum(map(parse_error, steps)),
            'first_bad_response_time_utc': first['timestamp'],
            'approximate_api_start_utc': (response_time - timedelta(milliseconds=latency)).isoformat(),
            'api_latency_msec': latency,
            'last_good_metrics': steps[bad_index-1].get('metrics'),
            'first_bad_metrics': first.get('metrics'),
            'prefix_block_size_from_startup': 544,
            'last_good_and_first_bad_report_same_cached_length': steps[bad_index-1]['metrics']['cached_tokens'] == first['metrics']['cached_tokens'],
            'first_bad_context_messages': first_empty,
            'saved_request_message_fields': sorted({k for m in messages for k in m}),
            'first_bad_reasoning_text_in_saved_prompt': bool(first_bad_reasoning and first_bad_reasoning in body),
            'replay_evidence': str((GAP / 'cache-comparison/receipt.json').relative_to(PROJECT)),
            'service_concurrency_samples': metrics,
            'log_limit': '10-second aggregate snapshots do not identify per-iteration batch permutations, accepted counts or the originating request. HTTP access logs do not include request IDs.'},
        'code_checks': comparison,
        'old_shared_cohort': cohort_audit('apptainer-environment-retest17'),
        'fresh_isolated_cohort': cohort_audit('apptainer-prefix-isolated-full89'),
        'control_flow_tests': load(OUT / 'control-flow-test-receipt.json'),
        'interpretation': {
            'supported': 'Failure can begin after55 valid command replies, without garbled text in the input; it is not the first prefix hit or a simple35k length threshold. Persistent retry-loop behavior is associated with prefix reuse in the saved incident.',
            'leading_conditional_candidate': 'Arrival/reordering plus accepted-count ownership mismatch can corrupt GDN/conv selection. Actual source reorder+prepare+precopy path reproduces this in CPU fixtures; incident timing itself remains unobserved.',
            'copy_race_qualification': 'For our width6/MTP3 geometry, correctly counted complete3-draft windows have no overlap in2176 enumerated postprocess cases assuming distinct live physical pages. Shortened1/2-draft schedules can reach overlap; source supports truncation but the incident schedule was not logged.',
            'fresh_format_errors': 'Some isolated requests still have empty content because a coherent commands JSON is in reasoning. Do not count these as the prior garbled-state defect without examining raw output.',
            'not_established': 'Unique root cause, actual faulting CUDA copy or live accepted-count permutation; quantitative contribution to TB2 score gap.'}}
    (OUT / 'incident-audit.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'original_incident': {k:v for k,v in data['original_incident'].items() if k != 'service_concurrency_samples'},
        'code_checks': {k:v for k,v in comparison.items() if k not in ['sources','shared_source_unchanged_from_prior_snapshots']},
        'old_totals': data['old_shared_cohort']['totals'],
        'new_totals': data['fresh_isolated_cohort']['totals']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
