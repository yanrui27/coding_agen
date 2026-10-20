#!/usr/bin/env python3
"""Stream local teacher snapshots into fixed-schema, auditable SWIFT JSONL.

No model inference, network access, or modification of the source snapshots.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import io
import json
import re
import tarfile
import time
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.json as paj
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "math": "Nemotron-SFT-Math-v4",
    "science": "Nemotron-SFT-Science-v2",
    "pdbthink": "pdbthink-glm53-teacher-traces",
    "proofs": "Nemotron-Math-Proofs-v3-SFT",
    "mindforge": "MindForge-27B-Training-Trajectories",
    "rts": "recursive-task-synthesis-glm-5.3-rollouts",
    "rlmath": "rlmath-kimi-k3-trajectories",
    "naturebench": "NatureBench-traces",
}
AGENT_DATASETS = {"mindforge", "rts", "rlmath", "naturebench"}
# Every row is constructed with these exact fields/types. In particular tools is
# a JSON STRING, avoiding Arrow's incompatible nested dynamic tool schemas.
ROW_SCHEMA = pa.schema([
    ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    ("tools", pa.string()), ("id", pa.string()), ("dataset", pa.string()),
    ("task_type", pa.string()), ("license", pa.string()),
])
SELF_EVAL = re.compile(r"self[\s_\-–]*evaluation", re.I)
BOX_SCORE = re.compile(r"\\boxed\s*\{\s*(0(?:\.0+)?|0\.5|1(?:\.0+)?)\s*\}")


class Reject(ValueError):
    pass


def dump(value):
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)


def sha(value):
    if not isinstance(value, bytes):
        value = str(value).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def group_hash(text):
    # Normalization affects grouping only; never rewrite source problem text.
    return sha(" ".join(str(text).split()))


def self_score(content):
    sections = list(SELF_EVAL.finditer(content))
    if not sections:
        return None
    scores = BOX_SCORE.findall(content[sections[-1].end():])
    return float(scores[-1]) if scores else None


def text(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        raise Reject("non_text_message")
    return value


def thinking(reasoning, answer):
    reasoning, answer = text(reasoning), text(answer)
    if not reasoning.strip():
        return answer
    # Idempotent for upstream exports which already wrapped the same thought.
    if answer.lstrip().startswith("<think>") and "</think>" in answer:
        return answer
    return f"<think>\n{reasoning}\n</think>\n\n{answer}"


def normalize_tools(raw):
    if not raw:
        return [], set()
    if isinstance(raw, str):
        raw = orjson.loads(raw)
    if not isinstance(raw, list):
        raise Reject("invalid_tools")
    result, names = [], set()
    for tool in raw:
        tool = dict(tool)
        function = dict(tool.get("function") or tool)
        params = function.get("parameters", {})
        if isinstance(params, str):
            params = orjson.loads(params)
        if not isinstance(params, dict) or not function.get("name"):
            raise Reject("invalid_tool_schema")
        function["parameters"] = params
        result.append({"type": "function", "function": function})
        names.add(function["name"])
    return result, names


def normalize_messages(raw, names, terminal=False):
    result, pending, call_names = [], {}, []
    seen_user = False
    for msg in raw:
        role = msg.get("role")
        content = text(msg.get("content"))
        if role == "assistant":
            if not seen_user or pending:
                raise Reject("invalid_assistant_order")
            content = thinking(msg.get("reasoning_content"), content)
            calls = msg.get("tool_calls") or []
            if content:
                result.append({"role": "assistant", "content": content})
            for i, call in enumerate(calls):
                function = call.get("function", {})
                name = function.get("name")
                if name not in names:
                    raise Reject("undefined_tool")
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = orjson.loads(arguments)
                    except orjson.JSONDecodeError:
                        raise Reject("invalid_tool_arguments")
                if not isinstance(arguments, dict):
                    raise Reject("invalid_tool_arguments")
                call_id = call.get("id") or f"unnamed_{i}"
                if call_id in pending:
                    raise Reject("duplicate_call_id")
                pending[call_id] = name
                call_names.append(name)
                result.append({"role": "tool_call", "content": dump({"name": name, "arguments": arguments}).decode()})
            if not content and not calls:
                raise Reject("empty_assistant")
        elif role == "tool":
            if not pending:
                raise Reject("orphan_tool_response")
            call_id = msg.get("tool_call_id")
            if call_id not in pending:
                # Some releases omit IDs but keep strict call/response order.
                if call_id or len(pending) != 1:
                    raise Reject("unmatched_tool_response")
                call_id = next(iter(pending))
            del pending[call_id]
            result.append({"role": "tool", "content": content})
        elif role in ("system", "user"):
            if pending:
                raise Reject("unfinished_tool_round")
            if role == "system" and result:
                raise Reject("misplaced_system")
            seen_user |= role == "user"
            result.append({"role": role, "content": content})
        else:
            raise Reject("unknown_role")
    if not result or not result[-1]["content"].strip():
        raise Reject("incomplete_final_answer")
    if pending and not (terminal and result[-1]["role"] == "tool_call"):
        raise Reject("unfinished_tool_round")
    if result[-1]["role"] != "assistant" and not terminal:
        raise Reject("incomplete_final_answer")
    return result, call_names


def atif_messages(steps):
    """Map a Terminus command batch to one call and its aggregated observation."""
    if not steps or not any(c["function_name"] == "mark_task_complete" for c in steps[-1].get("tool_calls", [])):
        raise Reject("no_explicit_completion")
    messages = [{"role": "system", "content": "Solve the task in a Linux terminal. Use terminal_batch to send ordered commands, inspect observations, and mark task_complete only after finishing."}]
    for step in steps:
        if step["source"] == "user":
            content = text(step.get("message"))
            if "Task Description:" in content:
                content = content.split("Task Description:", 1)[1].lstrip()
            messages.append({"role": "user", "content": content})
        elif step["source"] == "agent":
            calls = step.get("tool_calls") or []
            if any(c["function_name"] not in ("bash_command", "mark_task_complete") for c in calls):
                raise Reject("unknown_terminal_action")
            args = {"commands": [c["arguments"] for c in calls if c["function_name"] == "bash_command"],
                    "task_complete": any(c["function_name"] == "mark_task_complete" for c in calls)}
            call_id = f"batch_{step['step_id']}"
            message = {"role": "assistant", "content": text(step.get("message")),
                       "reasoning_content": step.get("reasoning_content")}
            if calls:
                message["tool_calls"] = [{"id": call_id, "function": {"name": "terminal_batch", "arguments": args}}]
            messages.append(message)
            observations = (step.get("observation") or {}).get("results") or []
            if observations:
                observation = "\n\n".join(text(o.get("content")) for o in observations)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": observation} if calls else
                                {"role": "user", "content": observation})
            elif calls and not args["task_complete"]:
                raise Reject("missing_terminal_observation")
        else:
            raise Reject("unknown_atif_source")
    tools = [{"type": "function", "function": {"name": "terminal_batch",
        "description": "Execute an ordered batch of terminal keystrokes with per-command wait durations; optionally mark the task complete after the batch.",
        "parameters": {"type": "object", "properties": {
            "commands": {"type": "array", "items": {"type": "object", "properties": {
                "keystrokes": {"type": "string"}, "duration": {"type": "number"}}, "required": ["keystrokes"]}},
            "task_complete": {"type": "boolean"}}, "required": ["commands", "task_complete"]}}}]
    return messages, tools


def claude_messages(path):
    messages, signatures, schemas = [], set(), {}
    def content_text(value):
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            if any(v.get("type") != "text" for v in value):
                raise Reject("non_text_media")
            return "\n".join(text(v.get("text")) for v in value)
        raise Reject("non_text_content")
    def value_type(value):
        return "boolean" if isinstance(value, bool) else "number" if isinstance(value, (int, float)) else \
            "object" if isinstance(value, dict) else "array" if isinstance(value, list) else "null" if value is None else "string"
    with path.open("rb") as stream:
        for line in stream:
            try:
                record = orjson.loads(line)
            except orjson.JSONDecodeError:
                raise Reject("corrupt_transcript")
            if record.get("isSidechain") or record.get("isApiErrorMessage"):
                continue
            message = record.get("message")
            if not isinstance(message, dict) or message.get("role") not in ("user", "assistant", "system"):
                continue
            signature = record.get("uuid") or sha(dump(message))
            if signature in signatures:
                continue
            signatures.add(signature)
            blocks = message.get("content")
            if isinstance(blocks, str):
                blocks = [{"type": "text", "text": blocks}]
            if not isinstance(blocks, list):
                raise Reject("invalid_claude_content")
            if message["role"] == "assistant":
                current = {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": []}
                for block in blocks:
                    kind = block.get("type")
                    if kind == "text":
                        current["content"] += text(block.get("text")) + "\n"
                    elif kind == "thinking":
                        current["reasoning_content"] += text(block.get("thinking")) + "\n"
                    elif kind == "tool_use":
                        name, arguments = block["name"], block["input"]
                        if not isinstance(arguments, dict):
                            raise Reject("invalid_tool_arguments")
                        schema = schemas.setdefault(name, {})
                        for key, value in arguments.items():
                            schema.setdefault(key, set()).add(value_type(value))
                        current["tool_calls"].append({"id": block["id"], "function": {"name": name, "arguments": arguments}})
                    elif kind == "redacted_thinking":
                        raise Reject("redacted_reasoning")
                    else:
                        raise Reject("unsupported_assistant_block")
                if messages and messages[-1]["role"] == "assistant":
                    for key in ("content", "reasoning_content"):
                        messages[-1][key] += current[key]
                    messages[-1]["tool_calls"].extend(current["tool_calls"])
                else:
                    messages.append(current)
            else:
                texts = []
                for block in blocks:
                    if block.get("type") == "tool_result":
                        messages.append({"role": "tool", "tool_call_id": block["tool_use_id"],
                                         "content": content_text(block.get("content", ""))})
                    elif block.get("type") == "text":
                        texts.append(text(block.get("text")))
                    else:
                        raise Reject("non_text_media")
                if texts:
                    messages.append({"role": message["role"], "content": "\n".join(texts)})
    tools = [{"type": "function", "function": {"name": name, "description": f"Source harness tool: {name}.",
        "parameters": {"type": "object", "properties": {key: {"type": sorted(types) if len(types)>1 else next(iter(types))}
            for key, types in properties.items()}, "additionalProperties": True}}} for name, properties in sorted(schemas.items())]
    return messages, tools


def agent_examples(raw_root, key, counts, include_failed=False):
    base = raw_root / DATASETS[key]
    if key == "mindforge":
        path = base / "frontier.surgical.messages_only.jsonl"
        for number, row in iter_rows(path):
            counts["source_rows"] += 1
            yield row, {"id": f"mindforge:{number}", "teacher": "GLM-5.2", "source_file": str(path), "source_row": number,
                        "group": group_hash(next(m["content"] for m in row["messages"] if m["role"] == "user")), "native": True}
    elif key == "rlmath":
        path = base / "data/train.parquet"
        for number, row in iter_rows(path):
            counts["source_rows"] += 1
            if (not include_failed and row["reward"] != 1) or row["exception_type"] or row["agent_timed_out"]:
                counts["rejected:unsuccessful_or_interrupted"] += 1
                continue
            yield {"steps": orjson.loads(row["steps"])}, {"id": row["trial_name"], "group": row["task_name"],
                "teacher": "Kimi-K3", "source_file": str(path), "source_row": number, "reward": row["reward"]}
    elif key == "rts":
        rows = pq.read_table(base / "logical_attempts.parquet").to_pylist()
        counts["source_rows"] = len(rows)
        wanted = collections.defaultdict(dict)
        for row in rows:
            selected = ("agent/trajectory.json" not in (row["missing_artifacts"] or [])) if include_failed else \
                       (row["reward"] == 1 and not row["missing_artifacts"])
            if selected and not row["exception_type"] and not row["summarization_count"]:
                wanted[row["shard"]][row["member_prefix"]] = row
        counts["complete_candidates" if include_failed else "success_candidates"] = sum(map(len, wanted.values()))
        def read_shard(item):
            shard, selected = item
            found = []
            with (base / shard).open("rb", buffering=1024*1024) as stream, tarfile.open(fileobj=stream) as archive:
                for member in archive:
                    prefix = member.name.split("/", 1)[0]
                    if prefix not in selected or not member.name.endswith("/agent/trajectory.json"):
                        continue
                    row = selected[prefix]
                    found.append((archive.extractfile(member).read(), {"id": row["execution_id"], "group": row["task_id"], "family": row["task_group_id"],
                        "teacher": "GLM-5.3", "source_file": str(base / shard), "source_member": member.name,
                        "source_row": row["attempt"], "reward": row["reward"], "missing_artifacts": row["missing_artifacts"]}))
            return shard, found
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for shard, found in pool.map(read_shard, sorted(wanted.items())):
                for payload, provenance in found:
                    try:
                        trajectory = orjson.loads(payload)
                    except orjson.JSONDecodeError:
                        counts["rejected:corrupt_trajectory"] += 1
                        continue
                    yield trajectory, provenance
                print(json.dumps({"dataset": key, "shard": shard, **counts}), flush=True)
    else:
        rows = pq.read_table(base / "metadata/index.parquet").to_pylist()
        counts["source_rows"] = len(rows)
        for row in rows:
            if row["model"] not in ("kimi-k3", "glm-5.3"):
                continue
            if not include_failed and (row["judge_valid"] is not True or (row["effective_improvement"] or 0) <= 0):
                continue
            path = base / "trajectories" / f"{row['harness']}__{row['model']}" / row["case_id"]
            if not row["has_result"] or not row["has_transcript"]:
                counts["rejected:missing_artifacts"] += 1
                continue
            result = orjson.loads((path / "result.json").read_bytes())
            if result.get("status") != "success" or result.get("returncode") != 0:
                counts["rejected:interrupted"] += 1
                continue
            counts["complete_candidates" if include_failed else "success_candidates"] += 1
            yield {"claude_path": path / "transcript.jsonl"}, {"id": f"{row['model']}:{row['case_id']}", "group": row["case_id"],
                "teacher": "Kimi-K3" if row["model"] == "kimi-k3" else "GLM-5.3", "source_file": str(path / "transcript.jsonl"),
                "source_row": 1, "effective_improvement": row["effective_improvement"], "judge_valid": row["judge_valid"]}


def process_agent_dataset(key, options):
    args = argparse.Namespace(**options)
    include_failed = getattr(args, "agent_all", False) and key != "mindforge"
    output_key = key + "_all" if include_failed else key
    report_path = Path(args.log_root) / "reports" / f"{output_key}.json"
    target = Path(args.output) / f"{output_key}.train.jsonl"
    if report_path.exists() and args.resume:
        report = json.loads(report_path.read_text())
        assert all(Path(f["path"]).stat().st_size == f["bytes"] for f in report["files"])
        return report
    if target.exists() or report_path.exists():
        raise RuntimeError(f"Refusing to overwrite {key}; use --resume or a fresh output")
    started = time.time()
    counts, candidates = collections.Counter(), {}
    for original, provenance in agent_examples(Path(args.raw_root), key, counts, include_failed):
        counts["candidate_rows"] += 1
        try:
            if key == "mindforge":
                tools, names = normalize_tools(original["tools"])
                messages, calls = [], []
                for msg in original["messages"]:
                    role, content = msg["role"], text(msg.get("content"))
                    if role not in ("system", "user", "assistant", "tool_call", "tool"):
                        raise Reject("unknown_role")
                    if role == "tool_call":
                        call = orjson.loads(content)
                        if call["name"] not in names or not isinstance(call["arguments"], dict):
                            raise Reject("invalid_native_tool_call")
                        calls.append(call["name"])
                    messages.append({"role": role, "content": content})
                if messages[-1]["role"] != "tool_call" or not re.search(r'echo\s+["\x27]?COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT["\x27]?\s*$',
                    orjson.loads(messages[-1]["content"])["arguments"].get("command", "")):
                    raise Reject("missing_completion_sentinel")
            else:
                raw, definitions = claude_messages(original["claude_path"]) if key == "naturebench" else atif_messages(original["steps"])
                tools, names = normalize_tools(definitions)
                messages, calls = normalize_messages(raw, names, terminal=True)
            content_hash = sha(dump({"messages": messages, "tools": tools}))
            row = {"messages": messages, "tools": dump(tools).decode(), "id": provenance["id"],
                   "dataset": DATASETS[key], "task_type": "scientific_execution" if key == "naturebench" else "agent_execution",
                   "license": "MIT" if key == "mindforge" else "See original dataset terms" if key == "rlmath" else "CC-BY-4.0"}
            payload = dump(row)
            # Different successful solutions to the same task remain useful SFT
            # examples. Deduplicate normalized content below, not task identity.
            if provenance["id"] in candidates:
                raise Reject("duplicate_execution_id")
            success = ((provenance.get("reward") == 1 and not provenance.get("missing_artifacts")) if key != "naturebench" else
                       (provenance.get("judge_valid") is True and (provenance.get("effective_improvement") or 0) > 0))
            provenance.update({"split": "train", "group_sha256": group_hash(provenance.pop("group")),
                "content_sha256": content_hash, "route": output_key,
                "quality": "upstream_sft" if key == "mindforge" else "verified_success" if success else "completed_not_strict_success",
                "with_thinking": any("<think>" in m["content"] for m in messages if m["role"] == "assistant"), "tool_calls": calls})
            candidates[provenance["id"]] = payload, row, provenance
        except (Reject, KeyError, TypeError, orjson.JSONDecodeError) as error:
            counts[f"rejected:{error if isinstance(error, Reject) else type(error).__name__}"] += 1
    writer = JsonlWriter(Path(args.output), output_key)
    provenance_dir = Path(args.log_root) / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    seen, groups, teachers, tool_calls = set(), set(), collections.Counter(), collections.Counter()
    with (provenance_dir / f"{output_key}.jsonl").open("wb") as stream:
        for _, row, provenance in sorted(candidates.values(), key=lambda item: item[1]["id"]):
            if provenance["content_sha256"] in seen:
                counts["rejected:duplicate_content"] += 1
                continue
            seen.add(provenance["content_sha256"])
            groups.add(provenance["group_sha256"])
            writer.write(row)
            stream.write(dump(provenance) + b"\n")
            counts["kept_rows"] += 1
            counts["strict_success_rows"] += provenance["quality"] == "verified_success"
            counts["with_thinking"] += provenance["with_thinking"]
            counts["with_tools"] += bool(provenance["tool_calls"])
            teachers[provenance["teacher"]] += 1
            tool_calls.update(provenance["tool_calls"])
    # MindForge's messages-only release uses one shared prompt and carries no
    # instance IDs; that prompt hash cannot establish distinct task coverage.
    if key != "mindforge":
        counts["unique_tasks"] = len(groups)
    report = {"dataset": DATASETS[key], "options": options, "elapsed_seconds": round(time.time()-started, 2),
              "counts": dict(counts), "teachers": dict(teachers), "tool_calls": dict(tool_calls), "files": writer.close()}
    if include_failed:
        report["variant"] = "all"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"dataset": key, **counts}), flush=True)
    return report


def sources(raw_root, key):
    base = raw_root / DATASETS[key]
    if key == "math":
        return sorted((base / "data").glob("*.parquet"))
    if key == "pdbthink":
        return sorted((base / "sft").glob("*.parquet"))
    return sorted((base / "data" if key == "proofs" else base).glob("*.jsonl"))


def iter_rows(path):
    if path.suffix == ".parquet":
        index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                index += 1
                yield index, row
    else:
        with path.open("rb", buffering=8 * 1024 * 1024) as stream:
            for index, line in enumerate(stream, 1):
                if line.strip():
                    yield index, orjson.loads(line)


class JsonlWriter:
    def __init__(self, base, dataset):
        self.path, self.dataset = base / f"{dataset}.train.jsonl", dataset
        self.stream = self.path.open("wb", buffering=8*1024*1024)
        self.checksum = hashlib.sha256()
        self.bytes = self.rows = 0

    def write(self, row):
        payload = dump(row) + b"\n"
        self.stream.write(payload)
        self.bytes += len(payload)
        self.rows += 1
        self.checksum.update(payload)

    def close(self):
        self.stream.close()
        return [{"path":str(self.path),"dataset":self.dataset,"split":"train",
                 "bytes":self.bytes,"rows":self.rows,"sha256":self.checksum.hexdigest()}]


def process_dataset(key, options):
    if key in AGENT_DATASETS:
        return process_agent_dataset(key, options)
    args = argparse.Namespace(**options)
    start = time.time()
    out = Path(args.output)
    report_path = Path(args.log_root) / "reports" / f"{key}.json"
    if report_path.exists() or list(out.glob(f"{key}*.jsonl")):
        if report_path.exists() and args.resume:
            previous = json.loads(report_path.read_text())
            if {k:v for k,v in previous["options"].items() if k != "resume"} != {k:v for k,v in options.items() if k != "resume"}:
                raise RuntimeError(f"{key}: existing report uses different options")
            for record in previous["files"]:
                if Path(record["path"]).stat().st_size != record["bytes"]:
                    raise RuntimeError(f"Changed or incomplete file: {record['path']}")
            return previous
        raise RuntimeError(f"Refusing to overwrite {key}; use --resume or fresh --output and --log-root")
    out.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    counters, teachers, topics, categories, scores, tools_count = [collections.Counter() for _ in range(6)]
    writers, seen, groups = {}, set(), set()
    provenance_dir = Path(args.log_root) / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    provenance = (provenance_dir / f"{key}.jsonl").open("wb", buffering=8*1024*1024)
    files = sources(Path(args.raw_root), key)
    if not files:
        raise RuntimeError(f"No source files for {key}")
    for file in files:
        for line, original in iter_rows(file):
            counters["raw_rows"] += 1
            md = original.get("metadata") or {}
            md = md if isinstance(md, dict) else {}
            category = str(md.get("data_type") or original.get("subset") or key)
            categories[category] += 1
            teacher = str(md.get("generation_model") or {
                "math": "DeepSeek-V4-Pro High", "proofs": "DeepSeek-V4-Pro Max", "pdbthink": "GLM-5.3"
            }.get(key, "unknown"))
            teachers[teacher] += 1
            topic = str(md.get("topic") or original.get("family") or "")
            topics[topic] += 1
            score = None
            try:
                if key == "science" and re.sub(r"[^a-z0-9]", "", teacher.lower()) != "deepseekv4pro":
                    raise Reject("teacher_not_selected")
                if key == "pdbthink":
                    if original.get("reward") != 1 or original.get("finish_reason") != "stop" or original.get("tool_violation"):
                        raise Reject("pdb_quality_filter")
                    raw_msgs = [dict(m) for m in original["messages"]]
                    if raw_msgs[-1]["role"] != "assistant":
                        raise Reject("incomplete_final_answer")
                    raw_msgs[-1]["content"] = original["answer"]
                    raw_msgs[-1]["reasoning_content"] = original.get("reasoning")
                else:
                    raw_msgs = original["messages"]
                if key == "proofs":
                    if category not in ("proof", "refinement", "verification", "meta_verification"):
                        raise Reject("unknown_proof_role")
                    if category in ("proof", "refinement"):
                        score = self_score(text(raw_msgs[-1].get("content")))
                        scores[f"{category}:{score}"] += 1
                        if score != 1.0:
                            raise Reject("solver_self_score_not_1")
                tools, names = normalize_tools(original.get("tools"))
                msgs, calls = normalize_messages(raw_msgs, names)
                route = category if key == "proofs" else "main"
                content_hash = sha(dump({"messages": msgs, "tools": tools}))
                if content_hash in seen:
                    raise Reject("duplicate_content")
                seen.add(content_hash)
                first_user = next(m["content"] for m in msgs if m["role"] == "user")
                # Retain source groups for provenance; every row is training data.
                if key == "pdbthink":
                    group = group_hash("pdb:" + str(original["source_group"]))
                elif key == "proofs":
                    group = group_hash(md.get("problem") or first_user)
                else:
                    group = group_hash(original.get("problem") or first_user)
                groups.add(group)
                identifier = str(original.get("uuid") or original.get("task_id") or content_hash)
                license_value = original.get("license") or "See original dataset license"
                if not isinstance(license_value, str):
                    license_value = dump(license_value).decode()
                row = {"messages": msgs, "tools": dump(tools).decode(), "id": identifier,
                       "dataset": DATASETS[key], "task_type": category, "license": license_value}
                # Fixed scalar/message types avoid dynamic Arrow schemas.
                if not all(isinstance(row[k], str) for k in row if k != "messages"):
                    raise Reject("invalid_scalar_type")
                pool = "proofs_critics" if route in ("verification", "meta_verification") else key
                if pool not in writers:
                    writers[pool] = JsonlWriter(out, pool)
                writer = writers[pool]
                writer.write(row)
                provenance.write(dump({"id": identifier, "content_sha256": content_hash, "group_sha256": group,
                    "split": "train", "route": route, "source_file": str(file), "source_row": line,
                    "teacher": teacher, "topic": topic, "self_score": score,
                    "quality": "teacher_self_score_1_not_independently_verified" if score == 1 else
                        ("critic_target" if key == "proofs" else "upstream_filtered")}) + b"\n")
                counters["kept_rows"] += 1
                counters["split:train"] += 1
                counters[f"route:{route}"] += 1
                counters["with_tools"] += bool(calls)
                counters["with_thinking"] += any(m["role"] == "assistant" and "<think>" in m["content"] for m in msgs)
                tools_count.update(calls)
            except Reject as error:
                counters[f"rejected:{error}"] += 1
            if counters["raw_rows"] % 10000 == 0:
                print(json.dumps({"dataset": key, "elapsed_s": round(time.time()-start), **counters}), flush=True)
        print(json.dumps({"dataset": key, "finished_file": file.name, **counters}), flush=True)
    provenance.close()
    output_files = [record for writer in writers.values() for record in writer.close()]
    report = {"dataset": DATASETS[key], "options": options, "elapsed_seconds": round(time.time()-start, 2),
              "counts": dict(counters), "teachers": dict(teachers), "topics": dict(topics),
              "raw_categories": dict(categories), "self_scores": dict(scores), "tool_calls": dict(tools_count),
              "groups": len(groups), "source_files": [str(p) for p in files], "files": output_files}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report



def write_manifest(log_root, reports):
    log_root = Path(log_root)
    files = sorted((f for report in reports for f in report["files"]), key=lambda f: f["path"])
    path = log_root / "manifest.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    source = Path(reports[0]["options"]["raw_root"]) / "manifest.json"
    result = {"format": "ms-swift messages + serialized tools", "layout": "flat per-dataset training-only JSONL",
              "files": files, "options": {k:v for k,v in reports[0]["options"].items() if k != "resume"},
              "source_snapshot_manifest": str(source),
              "source_snapshot_manifest_sha256": sha(source.read_bytes()) if source.exists() else None}
    if files == previous.get("files"):
        result["validation"] = previous.get("validation", {})
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    return result


class ChecksumReader(io.RawIOBase):
    def __init__(self, stream):
        self.stream, self.checksum = stream, hashlib.sha256()

    def readable(self):
        return True

    def read(self, size=-1):
        block = self.stream.read(size)
        self.checksum.update(block)
        return block


def check_file(record):
    path = Path(record["path"])
    assert path.stat().st_size == record["bytes"], path
    with path.open("rb") as stream:
        checked = ChecksumReader(stream)
        reader = paj.open_json(checked, read_options=paj.ReadOptions(block_size=32*1024*1024, use_threads=False),
            parse_options=paj.ParseOptions(explicit_schema=ROW_SCHEMA, unexpected_field_behavior="error"))
        rows = 0
        for batch in reader:
            batch.validate(full=True)
            assert batch.schema == ROW_SCHEMA, path
            rows += batch.num_rows
        assert rows == record["rows"] and checked.checksum.hexdigest() == record["sha256"], path
    return rows


def check_data(log_root, workers):
    path = Path(log_root)/"manifest.json"
    manifest = json.loads(path.read_text())
    groups, count = set(), 0
    for file in sorted((Path(log_root)/"provenance").glob("*.jsonl")):
        with file.open("rb") as stream:
            for line in stream:
                row = orjson.loads(line)
                assert row["split"] == "train", "Unexpected non-training sample"
                groups.add(row["group_sha256"])
                count += 1
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(check_file, record) for record in manifest["files"]]
        total = 0
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            total += future.result()
            print(f"Checked {i}/{len(futures)} files", flush=True)
    assert total == count, (total, count)
    manifest["validation"] = {"status":"passed", "method":"full Arrow parse, SHA256, row counts and training-only provenance audit",
                              "rows":total,"bytes":sum(f["bytes"] for f in manifest["files"]),
                              "files":len(manifest["files"]),"groups":len(groups)}
    path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(manifest["validation"],indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=str(ROOT.parent / "data/training"))
    parser.add_argument("--output", default=str(ROOT / "data"))
    parser.add_argument("--log-root", default=str(ROOT.parent / "logs/sft"))
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--agent-all", action="store_true", help="Keep complete Agent trajectories regardless of task success")
    parser.add_argument("--check", action="store_true", help="Validate existing output; do not regenerate data")
    args = parser.parse_args()
    if args.agent_all and not set(args.datasets) <= AGENT_DATASETS:
        parser.error("--agent-all requires an explicit --datasets selection containing only Agent datasets")
    if args.check:
        check_data(args.log_root, args.workers)
        return
    options = vars(args).copy()
    for key in ("workers", "datasets", "check"):
        options.pop(key)
    if not args.agent_all:
        options.pop("agent_all")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        reports = list(pool.map(process_dataset, args.datasets, [options]*len(args.datasets)))
    # Keep other completed datasets indexed when rerunning only one dataset.
    indexed = {(r["dataset"], r.get("variant", "default")):r for r in reports}
    for path in (Path(args.log_root)/"reports").glob("*.json"):
        report = json.loads(path.read_text())
        if not all(key in report for key in ("options", "dataset", "files")):
            continue
        if report["options"]["output"] == args.output:
            indexed.setdefault((report["dataset"], report.get("variant", "default")),report)
    result = write_manifest(args.log_root, list(indexed.values()))
    print(json.dumps({"complete":True,"files":len(result["files"]),"rows":sum(f["rows"] for f in result["files"])}),flush=True)


if __name__ == "__main__":
    main()
