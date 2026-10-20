"""Read existing TB2 records and collect evidence without changing run results."""
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmark"
FULL = BENCH / "results/Qwen3.6-35B-A3B/terminal-bench-2.0/full-seed42-context-fixed"
OUT = FULL / "provenance/failure-root-cause-audit-20261018"


def duration(phase):
    if not phase or not phase.get("finished_at") or not phase.get("started_at"):
        return None
    return round((datetime.fromisoformat(phase["finished_at"]) - datetime.fromisoformat(phase["started_at"])).total_seconds(), 3)


def main():
    progress = json.loads((FULL / "seed42/progress.json").read_text())
    rows = []
    for name, row in progress["tasks"].items():
        path = BENCH / row["result_path"]
        result = json.loads(path.read_text())
        trial = path.parent
        trajectory = trial / "agent/trajectory.json"
        steps = json.loads(trajectory.read_text()).get("steps", []) if trajectory.exists() else []
        agent_steps = [s for s in steps if s.get("source") == "agent"]
        usage = result.get("agent_result") or row.get("agent_usage") or {}
        times = usage.get("metadata", {}).get("api_request_times_msec", [])
        ctrf = trial / "verifier/ctrf.json"
        grading = json.loads(ctrf.read_text()).get("results", {}) if ctrf.exists() else {}
        logs = {}
        for filename in ("trial.log", "exception.txt", "verifier/test-stdout.txt"):
            p = trial / filename
            if p.exists():
                logs[filename] = p.read_text(errors="replace")
        rows.append({
            "task": name, "status": row["status"], "origin": row.get("result_origin"),
            "trial": str(trial.relative_to(ROOT)), "endpoint": row.get("base_url"),
            "exception": result.get("exception_info"),
            "phase_seconds": {k: duration(result.get(k)) for k in ("environment_setup", "agent_setup", "agent_execution", "verifier")},
            "steps": len(agent_steps), "output_tokens": usage.get("n_output_tokens"),
            "api_seconds_recorded_completed_calls": round(sum(times)/1000, 3),
            "longest_api_seconds": round(max(times, default=0)/1000, 3),
            "grader_summary": grading.get("summary"),
            "grader_failures": [t for t in grading.get("tests", []) if t.get("status") not in ("passed", "skipped")],
            "last_agent_steps": [{k: s[k] for k in ("step_id", "timestamp", "message", "tool_calls", "observation", "metrics") if k in s} for s in agent_steps[-4:]],
            "logs": logs,
        })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw-evidence.json").write_text(json.dumps({"source_progress": str((FULL / "seed42/progress.json").relative_to(ROOT)), "tasks": rows}, ensure_ascii=False, indent=2) + "\n")
    print(OUT / "raw-evidence.json")
    for r in rows:
        if r["status"] != "failed":
            continue
        print("\nTASK", r["task"], r["grader_summary"])
        for t in r["grader_failures"]:
            print(t["name"], "\n", t.get("trace", "")[-1500:])


if __name__ == "__main__":
    main()
