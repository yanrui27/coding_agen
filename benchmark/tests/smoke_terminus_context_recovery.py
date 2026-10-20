#!/usr/bin/env python3
"""Check installed Terminus-2 recovery against a real, already-running model.

Run after sourcing scripts/harbor_env.sh. This sends synthetic prompts only;
it does not launch a sandbox or execute benchmark commands.
"""
import argparse
import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.llms.base import ContextLengthExceededError
from harbor.llms.chat import Chat

ROOT = Path(__file__).resolve().parents[1]
MARKER = "CONTEXT_RECOVERY_OK"


async def check_case(args, output, forced_short):
    name = "forced-short-summary" if forced_short else "normal-summary"
    logs = output / name
    logs.mkdir(parents=True, exist_ok=True)
    agent = Terminus2(
        logs_dir=logs,
        model_name=f"openai/{args.model}",
        api_base=args.base_url,
        temperature=1.0,
        model_info={"max_input_tokens": 262144, "max_output_tokens": 81920,
                    "input_cost_per_token": 0, "output_cost_per_token": 0,
                    "litellm_provider": "openai", "mode": "chat"},
        llm_kwargs={"api_key": "EMPTY", "timeout": 90, "max_retries": 0},
        llm_call_kwargs={"max_tokens": 81920, "top_p": 0.95, "seed": 42,
                         "extra_body": {"top_k": 20,
                                        "chat_template_kwargs": {"enable_thinking": True}}},
    )
    initial = (
        "You operate a terminal. Reply as a JSON object containing analysis, plan, "
        "commands (an empty array), and task_complete (false). Keep replies concise."
    )
    task = (
        f"This is a connectivity test. Include {MARKER} in analysis. "
        "Do not run commands. Return the required JSON object."
    )
    chat = Chat(agent._llm)
    chat._messages = [
        {"role": "system", "content": initial},
        {"role": "user", "content": " context" * 300000},
        {"role": "assistant", "content": "Padding history received."},
    ]
    session = AsyncMock(spec=TmuxSession)
    session.capture_pane.return_value = "Terminal idle. No commands executed."
    requests = []
    original_call = agent._llm.call

    async def traced_call(prompt, **kwargs):
        entry = {"history_messages": len(kwargs.get("message_history", [])),
                 "input_characters": len(prompt) + sum(len(m.get("content", ""))
                     for m in kwargs.get("message_history", [])),
                 "max_tokens": kwargs.get("max_tokens"),
                 "thinking": kwargs.get("extra_body", {}).get("chat_template_kwargs", {})}
        requests.append(entry)
        started = time.monotonic()
        try:
            response = await original_call(prompt, **kwargs)
            if response.usage:
                entry["usage"] = response.usage.model_dump()
            return response
        except Exception as error:
            entry["error_type"] = type(error).__name__
            raise
        finally:
            entry["elapsed_seconds"] = round(time.monotonic() - started, 3)

    agent._llm.call = traced_call
    with patch.object(agent, "_summarize", AsyncMock(side_effect=ContextLengthExceededError(
            "Forced full-summary failure to test the short fallback"))) if forced_short else patch.object(
            agent, "_summarize", wraps=agent._summarize):
        response = await asyncio.wait_for(
            agent._query_llm(chat, task, task, session), timeout=240
        )
    parsed = agent._parser.parse_response(response.content)
    assert requests[0].get("error_type") == "ContextLengthExceededError", requests[0]
    assert not parsed.error, parsed.error
    assert MARKER in response.content, response.content
    assert requests[-1]["history_messages"] == (1 if forced_short else 3)
    assert chat.messages[0]["content"] == initial
    assert all(r["max_tokens"] == 81920 for r in requests)
    result = {"case": name, "passed": True, "requests": requests,
              "reply": response.content, "forced_summary_failure": forced_short,
              "real_initial_context_error": True, "sandbox_commands_executed": 0}
    (logs / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"{name}: passed, {len(requests)} backend calls", flush=True)
    return result


async def run(args):
    # A real overflow is safe only after confirming the server's configured cap.
    with httpx.Client(trust_env=False, timeout=15) as client:
        response = client.get(args.base_url.rstrip("/") + "/models")
        response.raise_for_status()
        model = next(m for m in response.json()["data"] if m["id"] == args.model)
        assert model.get("max_model_len") == 262144, model
    output = ROOT / "archive/validation/terminus-context-recovery"
    output.mkdir(parents=True, exist_ok=True)
    results = [await check_case(args, output, forced_short) for forced_short in (False, True)]
    source = ROOT / ".venv-harbor/lib/python3.12/site-packages/harbor/agents/terminus_2/terminus_2.py"
    receipt = {"checked_at": datetime.now(timezone.utc).isoformat(), "model": args.model,
               "base_url": args.base_url, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "thinking": True, "context_limit": 262144, "max_tokens": 81920,
               "benchmark_evaluation": False, "passed": True, "cases": results}
    (output / "live-smoke.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="Qwen3.6-35B-A3B")
    asyncio.run(run(parser.parse_args()))
