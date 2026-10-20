"""Local HTTP integration checks; no real model or judge calls."""
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_hle", ROOT / "scripts/eval_hle.py")
hle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hle)


class HLETests(unittest.TestCase):
    def test_concurrent_json_writes_are_atomic(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            hle.write_json(path, {"writer": -1, "payload": "initial"})
            def write(index):
                hle.write_json(path, {"writer": index, "payload": "x" * 1000})
                self.assertIn("writer", json.loads(path.read_text()))
            with ThreadPoolExecutor(max_workers=16) as executor:
                list(executor.map(write, range(64)))
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])

    def test_fixed_subset_matches_parent_and_reproducible_strata(self):
        full, parent = hle.load_dataset("text-only", None)
        rows, metadata = hle.load_dataset("text-only", None, "text-only-10pct")
        self.assertEqual(len(rows), 216)
        self.assertEqual(len({row["id"] for row in rows}), 216)
        self.assertFalse(metadata["complete_scope"])
        self.assertEqual(metadata["available_questions"], 2158)
        self.assertEqual(metadata["selection_seed"], 42)
        self.assertEqual(metadata["subset"], "text-only-10pct")
        self.assertEqual(hle.load_dataset("text-only", 2, "text-only-10pct")[0], rows[:2])
        with self.assertRaises(ValueError):
            hle.load_dataset("full", None, "text-only-10pct")
        spec = importlib.util.spec_from_file_location("prepare_hle_subset", ROOT / "scripts/prepare_hle_subset.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        base = hle.read_records(ROOT / "results/Qwen3.6-35B-A3B/hle/predictions.jsonl")
        lengths = {task: record["prompt_tokens"] for task, record in base.items()}
        selected, strata, _ = builder.select(full, lengths)
        self.assertEqual([r["id"] for r in selected], metadata["task_ids"])
        # Reference answers and parent ordering cannot influence selected membership.
        changed = [{**row, "answer": "unused", "rationale": "unused"} for row in reversed(full)]
        again, _, _ = builder.select(changed, lengths)
        self.assertEqual({r["id"] for r in again}, set(metadata["task_ids"]))
        for category in {r["category"] for r in full}:
            for answer_type in {r["answer_type"] for r in full}:
                expected = sum(r["category"] == category and r["answer_type"] == answer_type for r in rows)
                self.assertEqual(sum(s["subset_count"] for s in strata
                    if s["category"] == category and s["answer_type"] == answer_type), expected)
        self.assertEqual(parent["selected_questions"], 2158)

    def test_generation_does_not_expose_reference_fields(self):
        row = {"question": "Which option?", "answer": "SECRET_REFERENCE", "rationale": "SECRET_RATIONALE", "image": ""}
        text = json.dumps(hle.question_messages(row, "SYSTEM"))
        self.assertNotIn("SECRET_REFERENCE", text)
        self.assertNotIn("SECRET_RATIONALE", text)
        self.assertNotIn("tools", text)

    def test_model_directory_cannot_escape_results(self):
        self.assertEqual(hle.model_directory("org/model"), "org__model")
        self.assertNotIn("/", hle.model_directory("../../model"))

    def test_verified_replica_can_join_a_running_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "replicas.json"
            registry = {"created_at": "fixture", "model": "fixture-model", "new_replica_count": 1,
                "replicas": [{"name": "replica", "ready": False, "base_url": "http://replica.test/v1"}]}
            hle.write_json(path, registry)
            pool = hle.EndpointPool(["http://primary.test/v1"], "UNUSED", 5, 0, 1, path, "fixture-model")
            identity = dict(pool.registry_configuration)
            self.assertEqual(pool.workers, 2)
            with ThreadPoolExecutor(max_workers=1) as executor:
                with pool.lease() as primary:
                    self.assertEqual(primary.base, "http://primary.test/v1")
                    def take_replica():
                        with pool.lease() as replica:
                            return replica.base
                    future = executor.submit(take_replica)
                    registry["replicas"][0]["ready"] = True
                    hle.write_json(path, registry)
                    self.assertEqual(future.result(timeout=5), "http://replica.test/v1")
            self.assertEqual(pool.registry_configuration, identity)
            pool.refresh()
            self.assertEqual(pool.slots.qsize(), 2, "Refreshing must not add duplicate concurrency slots")
            replicas_only = hle.EndpointPool([], "UNUSED", 5, 0, 32, path, "fixture-model")
            self.assertEqual(replicas_only.workers, 32)
            with replicas_only.lease() as replica:
                self.assertEqual(replica.base, "http://replica.test/v1")
            self.assertEqual(replicas_only.bases, ["http://replica.test/v1"])
            args = hle.parser().parse_args(["generate", "--replica-manifest", str(path), "--replicas-only"])
            configuration = hle.generation_configuration(args, {}, {}, hle.EndpointPool([], "UNUSED", 5, 0, 32, path, "fixture-model"))
            self.assertIsNone(configuration["base_url"])
            self.assertEqual(configuration["base_urls"], [])
            self.assertTrue(configuration["replicas_only"])
            self.assertEqual(configuration["total_workers"], 32)

    def test_multi_endpoint_generation_keeps_task_on_one_replica_and_resumes(self):
        servers, threads, states = [], [], []

        def handler_for(state):
            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_POST(self):
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    question = json.dumps(body["messages"])
                    if body.get("chat_template_kwargs", {}).get("enable_thinking") is not True:
                        state["thinking_mismatches"] += 1
                    if self.path == "/tokenize":
                        state["tokenized"].add(question)
                        data = {"count": 20, "max_model_len": 262144}
                    else:
                        self.assert_on_same_replica = question in state["tokenized"]
                        if not self.assert_on_same_replica:
                            state["mismatches"] += 1
                        state["active"] += 1
                        state["peak"] = max(state["peak"], state["active"])
                        state["generation"] += 1
                        time.sleep(0.04)
                        data = {"choices": [{"finish_reason": "stop", "message": {"content": "Answer: A"}}],
                            "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23}}
                        state["active"] -= 1
                    encoded = json.dumps(data).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
            return Handler

        try:
            for _ in range(2):
                state = {"tokenized": set(), "mismatches": 0, "thinking_mismatches": 0,
                    "active": 0, "peak": 0, "generation": 0}
                server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(state))
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                states.append(state)
                servers.append(server)
                threads.append(thread)
            bases = [f"http://127.0.0.1:{server.server_port}/v1" for server in servers]
            with tempfile.TemporaryDirectory() as temporary:
                command = ["generate", "--model", "replica-fixture", "--results-root", temporary,
                    "--workers", "1", "--limit", "8", "--base-url", *bases]
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(hle.main(command), 0)
                    self.assertEqual(hle.main(command), 0)
                folder = Path(temporary) / "replica-fixture/hle"
                predictions = hle.read_records(folder / "predictions.jsonl")
                self.assertEqual(len(predictions), 8)
                self.assertEqual({row["base_url"] for row in predictions.values()}, set(bases))
                self.assertTrue(all(row["runner_sha256"] for row in predictions.values()))
                self.assertEqual(sum(state["generation"] for state in states), 8, "Resume must not regenerate saved answers")
                self.assertTrue(all(state["generation"] > 0 and state["peak"] == 1 and state["mismatches"] == 0 for state in states))
                self.assertTrue(all(state["thinking_mismatches"] == 0 for state in states),
                    "Default generation and tokenization must both enable thinking")
                configuration = json.loads((folder / "run.json").read_text())["configuration"]
                self.assertEqual(configuration["total_workers"], 2)
                self.assertEqual(configuration["workers_per_endpoint"], 1)
                self.assertEqual(configuration["base_urls"], bases)
                self.assertEqual(configuration["thinking"], "on")
                with self.assertRaises(ValueError):
                    hle.EndpointPool([bases[0], bases[0] + "/"], "UNUSED", 5, 0, 1)
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()

    def test_http_pipeline_resume_judge_failures_and_denominator(self):
        state = {"generation": 0, "judge": 0, "malformed_judge": True, "requests": []}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                state["requests"].append((self.path, body))
                status = 200
                if self.path == "/tokenize":
                    data = {"count": 262140, "max_model_len": 262144}
                elif body["model"] == "fixture-model":
                    state["generation"] += 1
                    if state["generation"] == 2:
                        if state.get("reasoning_only_failure"):
                            data = {"choices": [{"finish_reason": "length",
                                "message": {"content": None, "reasoning": "Incomplete reasoning"}}],
                                "usage": {"prompt_tokens": 262140, "completion_tokens": 4, "total_tokens": 262144}}
                        else:
                            status, data = 503, {"error": "controlled generation failure"}
                    else:
                        data = {"choices": [{"finish_reason": "stop", "message": {"content": "Answer: A\nConfidence: 90%"}}],
                            "usage": {"prompt_tokens": 262140, "completion_tokens": 4, "total_tokens": 262144}}
                else:
                    state["judge"] += 1
                    answer = {"extracted_final_answer": "A", "reasoning": "Fixture equivalence", "correct": "yes", "confidence": 90, "strict": True}
                    content = "invalid JSON" if state["malformed_judge"] else json.dumps(answer)
                    data = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
                encoded = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                common = ["--model", "fixture-model", "--results-root", temporary, "--workers", "1", "--timeout", "5", "--retries", "0"]
                base = f"http://127.0.0.1:{server.server_port}/v1"
                generate = ["generate", *common, "--base-url", base, "--limit", "2"]
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(hle.main(generate), 2)
                    self.assertEqual(hle.main(generate), 2)
                self.assertEqual(state["generation"], 2, "Resume must not silently retry failed tasks")
                folder = Path(temporary) / "fixture-model/hle"
                summary = json.loads((folder / "summary.json").read_text())
                self.assertEqual(summary["expected"], 2)
                self.assertEqual(summary["generation_failed"], 1)
                self.assertIsNone(summary["accuracy_percent"], "Ungraded runs must not report zero as a score")
                self.assertFalse(summary["complete_dataset_scope"])
                requests = [body for path, body in state["requests"] if path.endswith("completions") and body["model"] == "fixture-model"]
                self.assertTrue(all(body["max_tokens"] == 4 for body in requests))
                self.assertTrue(all(body["temperature"] == 1.0 and body["top_p"] == 1.0 and body["top_k"] == 20 for body in requests))
                judge = ["judge", *common, "--judge-base-url", base, "--judge-model", "fixture-judge"]
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(hle.main(judge), 2)
                self.assertIsNone(json.loads((folder / "summary.json").read_text())["accuracy_percent"])
                state["malformed_judge"] = False
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(hle.main([*judge, "--retry-failed"]), 0)
                summary = json.loads((folder / "summary.json").read_text())
                self.assertEqual(summary["accuracy_percent"], 50.0)
                self.assertEqual(summary["accuracy_denominator"], 2, "Generation failures must remain in the denominator")
                self.assertEqual(len(list((folder / "raw/judge").glob("*.json"))), 2, "All judge attempts must retain their raw responses")
                with self.assertRaises(ValueError):
                    with redirect_stdout(io.StringIO()):
                        hle.main([*generate, "--temperature", "0"])
                state["generation"] = 0
                pipeline = ["run", "--model", "fixture-model", "--results-root", str(Path(temporary) / "pipeline"),
                    "--workers", "1", "--timeout", "5", "--retries", "0", "--limit", "2", "--base-url", base,
                    "--judge-base-url", base, "--judge-model", "fixture-judge", "--judge-api-key-env", "FIXTURE_JUDGE_KEY"]
                with patch.dict(os.environ, {"FIXTURE_JUDGE_KEY": "local-test-only"}), redirect_stdout(io.StringIO()):
                    self.assertEqual(hle.main(pipeline), 0)
                final = json.loads((Path(temporary) / "pipeline/fixture-model/hle/summary.json").read_text())
                self.assertEqual(final["accuracy_percent"], 50.0)
                self.assertEqual(final["generation_failed"], 1)
                full_result = Path(temporary) / "pipeline/fixture-model/hle/summary.json"
                full_hash = hle.digest(full_result)
                state["generation"] = 0
                subset_pipeline = [*pipeline, "--subset", "text-only-10pct"]
                with patch.dict(os.environ, {"FIXTURE_JUDGE_KEY": "local-test-only"}), redirect_stdout(io.StringIO()):
                    self.assertEqual(hle.main(subset_pipeline), 0)
                    count = state["generation"]
                    self.assertEqual(hle.main(subset_pipeline), 0)
                self.assertEqual(state["generation"], count, "Subset runs must also resume saved answers")
                subset_folder = Path(temporary) / "pipeline/fixture-model/hle-text-only-10pct"
                subset_summary = json.loads((subset_folder / "summary.json").read_text())
                self.assertEqual(subset_summary["subset"], "text-only-10pct")
                self.assertFalse(subset_summary["complete_dataset_scope"])
                self.assertEqual(subset_summary["accuracy_denominator"], 2)
                self.assertEqual(subset_summary["accuracy_percent"], 50.0)
                subset_rows, _ = hle.load_dataset("text-only", 2, "text-only-10pct")
                self.assertEqual(set(hle.read_records(subset_folder / "predictions.jsonl")), {r["id"] for r in subset_rows})
                self.assertEqual(hle.digest(full_result), full_hash, "Subset output must not overwrite the full benchmark")
                # Exercise the actual shell entrypoint with a local-only fixture key,
                # CLI overrides, explicit full selection, failure counts, and process-level resume.
                environment = dict(os.environ)
                environment["VIDEO_JUDGE_KEY"] = "local-wrapper-test-only"
                wrapper_root = Path(temporary) / "wrapper"
                options = ["--model", "fixture-model", "--base-url", base,
                    "--results-root", str(wrapper_root), "--workers", "1", "--limit", "2",
                    "--timeout", "5", "--retries", "0", "--judge-base-url", base,
                    "--judge-model", "fixture-judge"]
                state["generation"] = 0
                state["reasoning_only_failure"] = True
                wrapper = ["bash", str(ROOT / "run_hle.sh"), *options]
                for _ in range(2):
                    result = subprocess.run(wrapper, env=environment, capture_output=True, text=True, timeout=20)
                    self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(state["generation"], 2, "The entrypoint must resume without silently retrying failures")
                quick_folder = wrapper_root / "fixture-model/hle-text-only-10pct"
                quick_summary = json.loads((quick_folder / "summary.json").read_text())
                self.assertEqual(quick_summary["accuracy_percent"], 50.0)
                self.assertEqual(quick_summary["generation_failed"], 1)
                self.assertEqual(quick_summary["truncated"], 1, "A reasoning-only truncation must be counted")
                self.assertEqual(quick_summary["judge"]["base_url"], base)
                self.assertEqual(quick_summary["judge"]["model"], "fixture-judge")
                self.assertEqual(quick_summary["judge"]["workers"], 1)
                quick_config = json.loads((quick_folder / "run.json").read_text())["configuration"]
                self.assertEqual(quick_config["base_url"], base, "CLI endpoints must replace all default endpoints")
                self.assertEqual(quick_config["thinking"], "on")
                self.assertEqual(set(hle.read_records(quick_folder / "predictions.jsonl")), {r["id"] for r in subset_rows})
                state["generation"] = 0
                full_wrapper = ["bash", str(ROOT / "run_hle.sh"), "--full", *options]
                result = subprocess.run(full_wrapper, env=environment, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                full_folder = wrapper_root / "fixture-model/hle"
                full_rows, _ = hle.load_dataset("text-only", 2)
                self.assertEqual(set(hle.read_records(full_folder / "predictions.jsonl")), {r["id"] for r in full_rows})
                self.assertNotIn("subset", json.loads((full_folder / "run.json").read_text())["configuration"]["dataset"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
