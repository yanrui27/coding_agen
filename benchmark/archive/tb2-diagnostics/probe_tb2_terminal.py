"""No-model reproduction of terminal recording backpressure, isolated tmux sockets."""
import hashlib
import json
import os
import shlex
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "benchmark/results/Qwen3.6-35B-A3B/terminal-bench-2.0/full-seed42-context-fixed/provenance/failure-root-cause-audit-20261018"
LIB = ROOT / "benchmark/.cache/tb2-audit-tools/asciinema-2.2.0"


def run_case(recording, size, chunked=False):
    with tempfile.TemporaryDirectory(prefix="tb2-terminal-audit-") as temp:
        base = Path(temp)
        socket = "tb2-audit-" + uuid.uuid4().hex
        env = os.environ.copy()
        env.update(PYTHONPATH=str(LIB), TERM="xterm-256color", SHELL="/bin/bash", ASCIINEMA_CONFIG_HOME=str(base / "config"))

        def tmux(*args, check=True):
            return subprocess.run(["tmux", "-L", socket, "-f", "/dev/null", *args], env=env, capture_output=True, text=True, check=check, timeout=5)

        command = "bash --noprofile --norc"
        if recording:
            command = f"python3 -m asciinema rec -q --stdin -c 'bash --noprofile --norc' {shlex.quote(str(base / 'recording.cast'))}"
        tmux("new-session", "-d", "-s", "audit", "-x", "160", "-y", "40", command)
        try:
            # Wait for the inner shell; sending during recorder startup can lose
            # the leading command and falsely make later comments look successful.
            deadline = time.monotonic() + 8
            while "bash-5.2#" not in tmux("capture-pane", "-p", "-t", "audit").stdout:
                if time.monotonic() > deadline:
                    raise RuntimeError("inner bash prompt did not become ready")
                time.sleep(0.1)
            # Only write comments to a task-specific temporary file. No model commands run.
            body = ("# harmless input for PTY pressure probe " + "x" * 90 + "\n") * (size // 131)
            dest = base / "payload.txt"
            marker = base / "complete"
            payload = f"cat > {shlex.quote(str(dest))} << 'AUDITEOF'\n{body}AUDITEOF\nprintf DONE > {shlex.quote(str(marker))}\n"
            staged = base / "input"
            staged.write_text(payload)
            chunks = [payload]
            if chunked:
                chunks = [payload[i:i+1024] for i in range(0, len(payload), 1024)]
            for chunk in chunks:
                staged.write_text(chunk)
                tmux("load-buffer", str(staged))
                # Normal paste converts newlines to CR, matching Harbor.
                tmux("paste-buffer", "-d", "-t", "audit")
                if chunked:
                    time.sleep(0.03)
            time.sleep(2)
            complete = marker.exists()
            intact = dest.exists() and dest.read_bytes() == body.encode()
            screen = tmux("capture-pane", "-p", "-t", "audit").stdout
            recovered = complete
            if not complete:
                tmux("send-keys", "-t", "audit", "C-c")
                tmux("send-keys", "-t", "audit", f"printf RECOVERED > {shlex.quote(str(base / 'recovered'))}\n")
                time.sleep(1)
                recovered = (base / "recovered").exists()
            pane_pid = int(tmux("display-message", "-p", "-t", "audit", "#{pane_pid}").stdout.strip())
            descendants = []
            pending = [pane_pid]
            while pending:
                pid = pending.pop()
                try:
                    children = [int(p) for p in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
                    pending.extend(children)
                    descendants.append({"pid": pid, "wchan": Path(f"/proc/{pid}/wchan").read_text(), "command": Path(f"/proc/{pid}/cmdline").read_text().replace("\x00", " ")[:160]})
                except (OSError, ValueError):
                    pass
            return {"recording": recording, "chunked_1024_bytes": chunked, "asciinema_version": "2.2.0" if recording else None, "payload_bytes": len(payload.encode()), "expected_body_bytes": len(body.encode()), "written_bytes": dest.stat().st_size if dest.exists() else None, "complete": complete, "payload_intact": intact, "ctrl_c_recovery": recovered, "pane_tail": screen[-500:], "process_waits": descendants}
        finally:
            # Scope cleanup exclusively to this probe's uniquely named tmux session.
            tmux("kill-server", check=False)
            for process in reversed(locals().get("descendants", [])):
                try:
                    os.kill(process["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass


def main():
    rows = []
    for size in (4096, 35000, 70000):
        for recording in (False, True):
            row = run_case(recording, size)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    for size in (35000, 70000):
        row = run_case(True, size, chunked=True)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    receipt = {"model_calls": 0, "scope": "synthetic harmless heredoc, same tmux paste-buffer delivery as Harbor; no benchmark solution is executed or changed", "asciinema_wheel_sha256": hashlib.sha256(next(LIB.parent.glob("asciinema-2.2.0*.whl")).read_bytes()).hexdigest(), "cases": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "terminal-recording-probe.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
