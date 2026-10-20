#!/usr/bin/env python3
"""Rebuild the fixed, stratified 216-question HLE text-only diagnostic set locally."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "text-only-10pct"
SEED = 42
SIZE = 216
MANIFEST = "archive/manifests/hle-text-only-10pct-seed42.json"
DATA = "datasets/hle/text_only_10pct_seed42.jsonl"
LENGTH_BINS = ("0-50%", "50-90%", "90-99%", "99-100%")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def allocate(counts, size):
    """Largest-remainder integer quotas, with deterministic ties."""
    total = sum(counts.values())
    quotas = {key: size * count // total for key, count in counts.items()}
    order = sorted(counts, key=lambda key: (-(size * counts[key] % total), key))
    for key in order[:size - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def select(rows, prompt_tokens):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["category"], row["answer_type"])].append(row)
    lengths = {}
    ordered = sorted(rows, key=lambda row: (prompt_tokens[row["id"]], row["id"]))
    for rank, row in enumerate(ordered):
        lengths[row["id"]] = next(index for index, upper in enumerate((50, 90, 99, 100))
            if 100 * rank < upper * len(rows))
    cells = {(group, bucket): [r for r in group_rows if lengths[r["id"]] == bucket]
        for group, group_rows in groups.items() for bucket in range(4)}
    group_quotas = allocate({g: len(v) for g, v in groups.items()}, SIZE)
    length_quotas = allocate(Counter(lengths.values()), SIZE)
    quotas = {}
    for group in sorted(groups):
        quotas.update({(group, b): n for b, n in allocate(
            {b: len(cells[group, b]) for b in range(4)}, group_quotas[group]).items()})
    # Preserve both sets of margins by transferring one place within a subject/type group.
    while True:
        totals = {b: sum(quotas[g, b] for g in groups) for b in range(4)}
        over = [b for b in range(4) if totals[b] > length_quotas[b]]
        under = [b for b in range(4) if totals[b] < length_quotas[b]]
        if not over:
            break
        choices = []
        for group in sorted(groups):
            for donor in over:
                for recipient in under:
                    if not quotas[group, donor] or quotas[group, recipient] >= len(cells[group, recipient]):
                        continue
                    def error(bucket, change):
                        return ((quotas[group, bucket] + change) * len(rows)
                            - SIZE * len(cells[group, bucket])) ** 2
                    cost = error(donor, -1) + error(recipient, 1) - error(donor, 0) - error(recipient, 0)
                    choices.append((cost, group, donor, recipient))
        if not choices:
            raise ValueError("Cannot meet the fixed subject/type and length quotas")
        _, group, donor, recipient = min(choices)
        quotas[group, donor] -= 1
        quotas[group, recipient] += 1
    chosen = set()
    strata = []
    for (group, bucket), cell in sorted(cells.items()):
        rank = lambda row: (hashlib.sha256(
            f"hle-text-only-10pct-v1|{SEED}|{row['id']}".encode()).hexdigest(), row["id"])
        chosen.update(row["id"] for row in sorted(cell, key=rank)[:quotas[group, bucket]])
        strata.append({"category": group[0], "answer_type": group[1], "length_bin": LENGTH_BINS[bucket],
            "full_count": len(cell), "subset_count": quotas[group, bucket]})
    assert len(chosen) == SIZE
    return [row for row in rows if row["id"] in chosen], strata, lengths


def distribution(full, subset):
    a, b = Counter(full), Counter(subset)
    comparisons = [{"value": key, "full_count": a[key], "subset_count": b[key],
        "full_percent": 100 * a[key] / len(full), "subset_percent": 100 * b[key] / len(subset),
        "difference_percentage_points": 100 * (b[key] / len(subset) - a[key] / len(full))}
        for key in sorted(a)]
    return {"values": comparisons, "max_absolute_difference_percentage_points":
        max(abs(row["difference_percentage_points"]) for row in comparisons)}


def numeric(values):
    values = sorted(values)
    return {"mean": sum(values) / len(values), **{
        key: values[max(0, math.ceil(q * len(values)) - 1)]
        for key, q in [("p25", .25), ("median", .5), ("p75", .75), ("p90", .9), ("p99", .99), ("max", 1)]}}


def store(path, value):
    path = ROOT / path
    data = value if isinstance(value, bytes) else (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Refusing to change an existing fixed subset artifact: {path}")
        return
    path.write_bytes(data)


def main():
    source = ROOT / "datasets/hle/text_only.jsonl"
    records = source.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in records]
    pinned = json.loads((ROOT / "archive/manifests/core-files.sha256.json").read_text())
    expected = next(r["sha256"] for r in pinned["files"] if r["path"] == "hle/text_only.jsonl")
    assert len(rows) == 2158 and sha(source) == expected
    assert len({r["id"] for r in rows}) == len(rows) and not any(r.get("image") for r in rows)
    base = ROOT / "results/Qwen3.6-35B-A3B/hle"
    config = json.loads((base / "run.json").read_text())["configuration"]
    summary = json.loads((base / "summary.json").read_text())
    predictions = {r["id"]: r for r in map(json.loads, (base / "predictions.jsonl").open())}
    grades = {r["id"]: r for r in map(json.loads, (base / "grades.jsonl").open())}
    assert summary["scoring_complete"] and set(predictions) == {r["id"] for r in rows}
    tokens = {task: row["prompt_tokens"] for task, row in predictions.items()}
    selected, strata, lengths = select(rows, tokens)
    ids = {r["id"] for r in selected}
    payload = b"".join(line for row, line in zip(rows, records) if row["id"] in ids)
    checks = {key: distribution([r[key] for r in rows], [r[key] for r in selected])
        for key in ["category", "answer_type"]}
    checks["category_answer_type"] = distribution(
        [r["category"] + " / " + r["answer_type"] for r in rows],
        [r["category"] + " / " + r["answer_type"] for r in selected])
    checks["prompt_length_bin"] = distribution(
        [LENGTH_BINS[lengths[r["id"]]] for r in rows], [LENGTH_BINS[lengths[r["id"]]] for r in selected])
    for field in ["prompt_tokens", "completion_tokens"]:
        def value(row):
            p = predictions[row["id"]]
            return p["prompt_tokens"] if field == "prompt_tokens" else p.get("usage", {}).get(field, 0)
        checks[field] = {"full": numeric([value(r) for r in rows]), "subset": numeric([value(r) for r in selected])}
    manifest = {"name": NAME, "version": 1, "seed": SEED, "size": SIZE,
        "fraction": SIZE / len(rows), "parent_size": len(rows), "parent_scope": "text-only",
        "repo": config["dataset"]["repo"], "revision": config["dataset"]["revision"], "split": "test",
        "parent_path": str(source.relative_to(ROOT)), "parent_sha256": sha(source),
        "path": DATA, "sha256": hashlib.sha256(payload).hexdigest(),
        "method": "Proportional subject/type quotas and global prompt-length rank bins (0-50/50-90/90-99/99-100%); largest remainders, deterministic quota balancing, then SHA256(seed,id) ranking within each cell; preserve parent order.",
        "selection_fields": ["category", "answer_type", "Base prompt_tokens", "id", "seed"],
        "outcome_fields_used_for_selection": [],
        "length_features_source": {"model": config["model"], "path": str((base / "predictions.jsonl").relative_to(ROOT)),
            "sha256": sha(base / "predictions.jsonl"), "note": "Existing Base /tokenize counts; lengths include the fixed official system prompt and chat template."},
        "task_ids": [r["id"] for r in selected], "strata": strata, "representativeness": checks,
        "use": "Fixed quick diagnostic subset. Report separately from the full HLE text-only score. Existing Base outcomes are inspected only after selection."}
    store(DATA, payload)
    store(MANIFEST, manifest)
    per_question = []
    for row in selected:
        task = row["id"]; pred = predictions[task]; grade = grades.get(task, {})
        if pred["status"] == "ok":
            assert grade["status"] == "ok" and grade["prediction_sha256"] == hashlib.sha256(pred["response"].encode()).hexdigest()
        per_question.append({"id": task, "generation_status": pred["status"],
            "correct": pred["status"] == "ok" and grade.get("judge_response", {}).get("correct") == "yes",
            "prediction_sha256": hashlib.sha256(pred.get("response", "").encode()).hexdigest()})
    correct = sum(row["correct"] for row in per_question)
    baseline = {"benchmark": "HLE", "scope": "text-only", "subset": NAME, "model": config["model"],
        "tools": False, "expected": SIZE, "correct": correct, "accuracy_percent": 100 * correct / SIZE,
        "generation_failed": sum(row["generation_status"] != "ok" for row in per_question),
        "complete_dataset_scope": False, "scoring_complete": True,
        "full_text_only_accuracy_percent": summary["accuracy_percent"],
        "difference_from_full_percentage_points": 100 * correct / SIZE - summary["accuracy_percent"],
        "subset_manifest": MANIFEST, "subset_manifest_sha256": sha(ROOT / MANIFEST),
        "source": {"directory": str(base.relative_to(ROOT)),
            "files_sha256": {name: sha(base / name) for name in ["run.json", "judge.json", "predictions.jsonl", "grades.jsonl"]}},
        "generation_configuration": {k: v for k, v in config.items() if k not in ["dataset", "official_protocol"]},
        "judge": json.loads((base / "judge.json").read_text())["configuration"],
        "per_question": per_question,
        "note": "Derived from the completed full Base run; no inference or judging was repeated. The subset was selected without correctness, answers, or reasoning-token counts. Full-run raw outputs remain in the source directory. This diagnostic score is not the full HLE score."}
    store("results/Qwen3.6-35B-A3B/hle-text-only-10pct/baseline.json", baseline)
    print(json.dumps({"size": SIZE, "data": DATA, "manifest": MANIFEST,
        "baseline_correct": correct, "baseline_accuracy_percent": baseline["accuracy_percent"],
        "full_accuracy_percent": summary["accuracy_percent"],
        "maximum_category_difference_percentage_points": checks["category"]["max_absolute_difference_percentage_points"],
        "maximum_type_difference_percentage_points": checks["answer_type"]["max_absolute_difference_percentage_points"],
        "length_bin_distribution": checks["prompt_length_bin"], "prompt_tokens": checks["prompt_tokens"],
        "completion_tokens": checks["completion_tokens"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
