"""Verify launcher lifecycle and configuration isolation without model calls."""
import contextlib
import concurrent.futures
from collections import Counter
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_asi", ROOT / "scripts/eval_asi.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

from ai4sci_bench.cli import run
from ai4sci_bench.runner.os_sandbox import OSSandbox
from ai4sci_bench.adapters.opencode_cli import OpenCodeCLIAdapter
from asi_agent import server_urls, use_server_pool
import ai4sci_bench.cli as official
import click


class LauncherTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / ".cache/tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="asi-launcher-test-", dir=scratch)
        self.output = Path(self.temp.name) / "results"
        self.defaults = OSSandbox.DEFAULT_CPU_LIMIT, OSSandbox.DEFAULT_MEMORY_LIMIT
        self.args = ["--cpus", "2", "--memory-mib", "1024", "run",
                     "--agent", "opencode_cli", "--agent-config", json.dumps({
                         "model": "fixture", "api_base": "http://fixture.invalid/v1",
                         "api_protocol": "openai", "api_key": "fixture-secret"}),
                     "--output-dir", str(self.output), "--sandbox", "os", "--timeout", "30"]
        self.preflight = patch.object(launcher, "check_environment", return_value={"fixture": True})
        self.preflight.start()
        self.limits = patch.object(launcher, "set_resource_limits")
        self.limits.start()
        self.stdout = io.StringIO()
        self.capture = contextlib.redirect_stdout(self.stdout)
        self.capture.__enter__()

    def tearDown(self):
        self.capture.__exit__(None, None, None)
        self.preflight.stop()
        self.limits.stop()
        OSSandbox.DEFAULT_CPU_LIMIT, OSSandbox.DEFAULT_MEMORY_LIMIT = self.defaults
        self.temp.cleanup()

    def test_preview_redacts_credentials_and_applies_last_override(self):
        self.assertEqual(launcher.main([*self.args, "--timeout", "60", "--print-config"]), 0)
        output = self.stdout.getvalue()
        self.assertNotIn("fixture-secret", output)
        self.assertEqual(json.loads(output)["run_options"]["timeout"], 60)
        self.assertFalse(self.output.exists())

    def test_official_command_resume_and_changed_settings(self):
        with patch.object(run, "callback") as execute:
            self.assertEqual(launcher.main(self.args), 0)
            self.assertEqual(execute.call_args.kwargs["agent"], "opencode_cli")
            self.assertEqual(OSSandbox.DEFAULT_CPU_LIMIT, "2.0")
            self.assertEqual(OSSandbox.DEFAULT_MEMORY_LIMIT, "1024m")
            with self.assertRaisesRegex(ValueError, "Results already exist"):
                launcher.main(self.args)
            self.assertEqual(launcher.main([*self.args, "--resume", str(self.output)]), 0)
            with self.assertRaisesRegex(ValueError, "Run settings changed"):
                launcher.main([*self.args, "--resume", str(self.output), "--timeout", "61"])
            self.assertEqual(execute.call_count, 2)
        record = (self.output / "launcher.json").read_text()
        self.assertNotIn("fixture-secret", record)
        self.assertEqual(json.loads(record)["status"], "outputs_collected")

    def test_interruption_records_failure_without_claiming_success(self):
        with patch.object(run, "callback", side_effect=KeyboardInterrupt):
            with self.assertRaises(click.Abort):
                launcher.main(self.args)
        record = json.loads((self.output / "launcher.json").read_text())
        self.assertEqual(record["status"], "interrupted_or_failed")
        self.assertNotIn("final_score", record)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / ".cache/tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="asi-routing-test-", dir=scratch)
        self.root = Path(self.temp.name)
        self.urls = [f"http://server-{i}.invalid/v1" for i in range(4)]
        with use_server_pool(self.root):
            self.agent = official._build_agent(None, "opencode_cli", {
                "model": "fixture", "api_base": self.urls,
                "api_key": "fixture-secret", "api_protocol": "openai"}, tool_mode="restricted")
        self.agent.setup({"sandbox": "none", "repo_root": str(launcher.SOURCE)})

    def tearDown(self):
        self.temp.cleanup()

    def instance(self, i):
        return SimpleNamespace(run_key=f"fixture-{i}__seed42__b1", task_id=f"fixture-{i}",
                               prompt_level=SimpleNamespace(value="b1"))

    def test_eight_concurrent_attempts_keep_server_and_image_provenance_isolated(self):
        barrier = threading.Barrier(8)

        def solve(child, instance):
            before = json.loads(child._build_api_env()["OPENCODE_CONFIG_CONTENT"])["provider"]["bench"]["options"]["baseURL"]
            barrier.wait(timeout=10)
            self.assertEqual(before, child.api_base)
            child._sandbox_image_identity = "sha256:" + instance.task_id
            return SimpleNamespace(status=SimpleNamespace(value="completed"))

        def worker(i):
            self.agent.solve(self.instance(i))
            return self.agent.api_base, self.agent.sandbox_image_identity

        with patch.object(OpenCodeCLIAdapter, "solve", solve):
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(worker, range(8)))
        self.assertEqual(Counter(url for url, _ in results), Counter({url: 2 for url in self.urls}))
        for i, (_, image) in enumerate(results):
            self.assertEqual(image, f"sha256:fixture-{i}")
        self.assertEqual(self.agent.pool.active, [0, 0, 0, 0])
        raw = (self.root / "endpoint_assignments.jsonl").read_text()
        self.assertNotIn("fixture-secret", raw)
        rows = [json.loads(line) for line in raw.splitlines()]
        self.assertEqual(len(rows), 16)
        for assignment in {row["assignment_id"] for row in rows}:
            events = [row for row in rows if row["assignment_id"] == assignment]
            self.assertEqual([row["event"] for row in events], ["started", "finished"])
            self.assertEqual(events[0]["api_base"], events[1]["api_base"])

    def test_failed_server_attempt_releases_capacity(self):
        with patch.object(OpenCodeCLIAdapter, "solve", side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                self.agent.solve(self.instance(0))
        self.assertEqual(self.agent.pool.active, [0, 0, 0, 0])
        self.assertIsNone(self.agent.sandbox_image_identity)
        rows = [json.loads(line) for line in (self.root / "endpoint_assignments.jsonl").read_text().splitlines()]
        self.assertEqual(rows[-1]["status"], "interrupted_or_failed")

    def test_single_server_compatibility_and_duplicate_validation(self):
        self.assertEqual(server_urls("http://server.invalid/v1/"), ["http://server.invalid/v1"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            server_urls(["http://server.invalid/v1", "http://server.invalid/v1/"])


if __name__ == "__main__":
    unittest.main()
