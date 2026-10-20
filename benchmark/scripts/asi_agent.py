"""Assign whole ASI attempts to model servers while using the official agent."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit
from uuid import uuid4

from ai4sci_bench.adapters.opencode_cli import OpenCodeCLIAdapter

ASSIGNMENT = "least-active server at attempt start; server fixed throughout the attempt"


def server_urls(api_base):
    values = [api_base] if isinstance(api_base, str) else api_base
    if not isinstance(values, list) or not values or any(not isinstance(value, str) for value in values):
        raise ValueError("api_base must be a URL or a nonempty list of URLs")
    urls = [value.rstrip("/") for value in values]
    for url in urls:
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Model server URLs must be HTTP(S) URLs without credentials, queries or fragments")
    if len(set(urls)) != len(urls):
        raise ValueError("api_base contains duplicate server URLs")
    return urls


class ServerPool:
    def __init__(self, urls, log_path=None):
        self.urls = server_urls(urls)
        self.active = [0] * len(self.urls)
        self.cursor = 0
        self.lock = threading.Lock()
        self.log_path = Path(log_path) if log_path else None

    def acquire(self):
        with self.lock:
            index = min(range(len(self.urls)), key=lambda i: (self.active[i], (i - self.cursor) % len(self.urls)))
            self.active[index] += 1
            self.cursor = (index + 1) % len(self.urls)
            return index, self.urls[index]

    def release(self, index):
        with self.lock:
            self.active[index] -= 1

    def record(self, **event):
        if self.log_path:
            event["timestamp"] = datetime.now(timezone.utc).isoformat()
            with self.lock, self.log_path.open("a") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")


class MultiServerOpenCodeCLIAdapter(OpenCodeCLIAdapter):
    def __init__(self, *, api_base, endpoint_log, **config):
        self.pool = ServerPool(api_base, endpoint_log)
        self.local = threading.local()
        self.child_config = config.copy()
        super().__init__(api_base=self.pool.urls[0], **config)

    @property
    def api_base(self):
        return getattr(self.local, "api_base", self.primary_api_base)

    @api_base.setter
    def api_base(self, value):
        self.primary_api_base = value

    @property
    def sandbox_image_identity(self):
        return getattr(self.local, "image_identity", None)

    def setup(self, config):
        self.setup_config = config.copy()
        super().setup(config)

    def solve(self, instance):
        index, url = self.pool.acquire()
        self.local.api_base = url
        self.local.image_identity = None
        child = None
        status = "interrupted_or_failed"
        event = {"assignment_id": uuid4().hex, "run_key": instance.run_key,
                 "task_id": instance.task_id, "prompt_level": instance.prompt_level.value,
                 "api_base": url}
        try:
            self.pool.record(**event, event="started")
            child = OpenCodeCLIAdapter(api_base=url, **self.child_config)
            child.setup(self.setup_config.copy())
            output = child.solve(instance)
            status = output.status.value
            return output
        finally:
            try:
                if child:
                    self.local.image_identity = child.sandbox_image_identity
                    child.teardown()
            finally:
                self.pool.release(index)
                self.pool.record(**event, event="finished", status=status)


@contextmanager
def use_server_pool(output_dir):
    import ai4sci_bench.cli as official

    original = official._build_agent

    def build(agent_cmd, agent_name, agent_config, allow_external_tools=False, tool_mode=None):
        if agent_cmd or agent_name != "opencode_cli":
            return original(agent_cmd, agent_name, agent_config, allow_external_tools, tool_mode)
        config = agent_config.copy()
        config.setdefault("allow_external_tools", allow_external_tools)
        if tool_mode is not None:
            config.setdefault("tool_mode", tool_mode)
        return MultiServerOpenCodeCLIAdapter(endpoint_log=Path(output_dir) / "endpoint_assignments.jsonl", **config)

    official._build_agent = build
    try:
        yield
    finally:
        official._build_agent = original
