"""Replay the saved headless-terminal implementation in two network namespaces."""
import hashlib
import importlib.util
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "benchmark/results/Qwen3.6-35B-A3B/terminal-bench-2.0/full-seed42-context-fixed/provenance/failure-root-cause-audit-20261018"


def worker():
    rows = json.loads((OUT / "raw-evidence.json").read_text())["tasks"]
    row = next(r for r in rows if r["task"] == "headless-terminal")
    trajectory = json.loads((ROOT / row["trial"] / "agent/trajectory.json").read_text())
    writes = [c["arguments"]["keystrokes"] for s in trajectory["steps"] for c in s.get("tool_calls", []) if c.get("arguments", {}).get("keystrokes", "").startswith("cat > /app/headless_terminal.py << 'PYEOF'\n")]
    code = writes[-1].split("\n", 1)[1].rsplit("\nPYEOF", 1)[0]
    with tempfile.TemporaryDirectory(prefix="tb2-headless-audit-") as temp:
        base = Path(temp)
        (base / "headless_terminal.py").write_text(code)
        (base / "base_terminal.py").write_bytes((ROOT / "benchmark/datasets/terminal-bench-2.0/headless-terminal/environment/base_terminal.py").read_bytes())
        (base / "index.html").write_text("Hello, world!\n")
        sys.path.insert(0, str(base))
        spec = importlib.util.spec_from_file_location("headless_terminal", base / "headless_terminal.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        terminal = module.HeadlessTerminal()
        try:
            terminal.send_keystrokes(f"{shlex.quote(sys.executable)} -m http.server 8000 --directory {shlex.quote(temp)} > {shlex.quote(str(base / 'server.log'))} 2>&1 & echo $! > {shlex.quote(str(base / 'server.pid'))}")
            terminal.send_keystrokes("\n", wait_sec=5)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                with opener.open("http://localhost:8000", timeout=3) as response:
                    status, content = response.status, response.read().decode()
            except urllib.error.HTTPError as exc:
                status, content = exc.code, exc.read().decode()
            return {"implementation_sha256": hashlib.sha256(code.encode()).hexdigest(), "http_status": status, "body_matches": content.strip() == "Hello, world!", "server_log": (base / "server.log").read_text()[-1500:] if (base / "server.log").exists() else None}
        finally:
            if (base / "server.pid").exists():
                pid = int((base / "server.pid").read_text())
                try:
                    if temp in Path(f"/proc/{pid}/cmdline").read_text():
                        os.kill(pid, signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    pass
            terminal.close(force=True)


def main():
    if "--worker" in sys.argv:
        print(json.dumps(worker()))
        return
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    cases = []
    for isolated in (False, True):
        command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
        if isolated:
            command = ["unshare", "--net", "bash", "-c", "ip link set lo up\nexec " + " ".join(shlex.quote(c) for c in command)]
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30, check=True)
        row = json.loads(result.stdout)
        row["isolated_network_namespace"] = isolated
        cases.append(row)
        print(json.dumps(row), flush=True)
    receipt = {"model_calls": 0, "source": "last saved headless_terminal.py heredoc in original trajectory; implementation unchanged", "probe_differences_from_official_fixture": ["Python executable resolved using sys.executable", "test index and source stored in a temporary directory", "stdlib urllib used to inspect HTTP status/body"], "cases": cases, "full_official_regrade": False}
    (OUT / "headless-network-probe.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
