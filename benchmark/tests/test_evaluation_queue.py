"""Check full-dataset handoff gates and authoritative fresh-sandbox score handling."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


queue = module('run_eval_queue')
swe = module('eval_swe')


class HandoffTests(unittest.TestCase):
    def test_tb2_requires_all_89_and_coordinator_exit(self):
        full = {'complete': True, 'tasks_per_seed': 89, 'completed_trials': 89}
        self.assertTrue(queue.tb2_finished(full, False))
        self.assertFalse(queue.tb2_finished(full, True))
        self.assertFalse(queue.tb2_finished({**full, 'completed_trials': 88}, False))
        self.assertFalse(queue.tb2_finished({'complete': True, 'tasks_per_seed': 83, 'completed_trials': 83}, False))
        self.assertFalse(queue.tb2_finished({}, False))

    def test_asi_covers_all_levels_and_keeps_latest_attempt_not_best_score(self):
        tasks = [task.removesuffix('__seed42') for task in
                 (ROOT / 'archive/manifests/asi-seed42-task-ids.txt').read_text().splitlines()]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for task in tasks:
                (directory / task).mkdir()
                for level in ['b1', 'b2', 'b3', 'b4']:
                    queue.save(directory / task / f'{level}.json',
                               {'task_id': task, 'instance_id': task + '__seed42', 'prompt_level': level,
                                'attempt': 1, 'agent_output': {'status': 'completed'}})
            result = queue.asi_coverage(directory)
            self.assertTrue(result['execution_complete'])
            self.assertEqual(result['attempted_conditions'], 240)
            self.assertIsNone(result['official_score'])
            self.assertFalse(result['scoring_complete'])
            queue.save(directory / tasks[0] / 'b1__attempt2.json',
                       {'task_id': tasks[0], 'instance_id': tasks[0] + '__seed42', 'prompt_level': 'b1',
                        'attempt': 2, 'agent_output': {'status': 'failed'}})
            queue.save(directory / tasks[0] / 'model_calls.json', [{'unrelated': True}])
            self.assertEqual(queue.asi_coverage(directory)['agent_status_counts']['failed'], 1)
            (directory / tasks[1] / 'b4.json').unlink()
            self.assertFalse(queue.asi_coverage(directory)['execution_complete'])

    def test_missing_or_timed_out_verifier_cannot_be_a_model_pass(self):
        self.assertEqual(swe.score_outcome(1, True, 0), 'passed')
        self.assertEqual(swe.score_outcome(0, True, 1), 'failed')
        self.assertEqual(swe.score_outcome(1, False, 0), 'infrastructure_error')
        self.assertEqual(swe.score_outcome(None, True, 0), 'infrastructure_error')
        self.assertEqual(swe.score_outcome(0, True, 124), 'infrastructure_error')


if __name__ == '__main__':
    unittest.main()
