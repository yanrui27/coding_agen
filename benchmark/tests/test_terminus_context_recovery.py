"""Harbor 0.24.0 backport regression tests, using real Chat request history.

Adapted from upstream PR #3310, head 194356575255097b0974fdcf49c7868858b981c1.
Version adaptation: options model replaces legacy private option attributes;
only native 0.24.0 output-truncation handling is tested, without fork policies.
"""

import copy
import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

import harbor.agents.terminus_2.terminus_2 as installed_module

module = installed_module
candidate = os.environ.get("TB2_CONTEXT_AGENT_SOURCE")
if candidate:
    spec = importlib.util.spec_from_file_location(
        "harbor.agents.terminus_2.context_recovery_candidate", candidate
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
Terminus2 = module.Terminus2
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.llms.base import (
    BaseLLM,
    ContextLengthExceededError,
    LLMResponse,
    OutputLengthExceededError,
)
from harbor.llms.chat import Chat
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import Step


@pytest.fixture
def recovery_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Terminus2:
    model = Mock(spec=BaseLLM)
    model.get_model_context_limit.return_value = 20000
    model.call = AsyncMock()
    monkeypatch.setattr(Terminus2, "_init_llm", lambda self, **kwargs: model)
    agent = Terminus2(logs_dir=tmp_path, model_name="gpt-4")
    monkeypatch.setattr(agent, "_count_total_tokens", lambda chat: 12000)
    return agent


def history_chat(agent: Terminus2) -> Chat:
    chat = Chat(agent._llm)
    chat._messages = [
        {"role": "system", "content": "Follow the terminal response format."},
        {"role": "user", "content": "old history " * 1000},
        {"role": "assistant", "content": "old reasoning " * 1000},
    ]
    chat._last_response_id = "old-response-chain"
    return chat


def mock_llm(agent: Terminus2) -> Mock:
    return cast(Mock, agent._llm)


@pytest.mark.parametrize("short_summary_fails", [False, True])
async def test_fallback_replaces_history(
    recovery_agent: Terminus2,
    monkeypatch: pytest.MonkeyPatch,
    short_summary_fails: bool,
) -> None:
    agent = recovery_agent
    chat = history_chat(agent)
    first_message = copy.deepcopy(chat.messages[0])
    session = AsyncMock(spec=TmuxSession)
    session.capture_pane.return_value = "CURRENT TERMINAL STATE"
    monkeypatch.setattr(
        agent, "_summarize", AsyncMock(side_effect=ContextLengthExceededError())
    )
    requests = []

    async def call(prompt: str, **kwargs: Any) -> LLMResponse:
        requests.append((prompt, copy.deepcopy(kwargs)))
        if len(requests) == 1:
            raise ContextLengthExceededError()
        if prompt.startswith("Briefly continue"):
            if short_summary_fails:
                raise ContextLengthExceededError()
            return LLMResponse(content="Inspect the saved implementation.")
        # Emulate a server that still rejects retained full history, even if
        # the local token estimate claims there is enough room.
        if len(kwargs.get("message_history", [])) > 1:
            raise ContextLengthExceededError()
        return LLMResponse(content='{"commands": [], "task_complete": false}')

    mock_llm(agent).call.side_effect = call
    result = await agent._query_llm(chat, "continue", "ORIGINAL TASK", session)

    assert result.content.startswith('{"commands"')
    assert len(requests) == 3
    handoff, resumed_kwargs = requests[-1]
    assert resumed_kwargs["message_history"] == [first_message]
    assert resumed_kwargs["previous_response_id"] is None
    assert "ORIGINAL TASK" in handoff
    assert agent._pending_handoff_prompt == handoff
    assert chat.messages[0] == first_message
    if short_summary_fails:
        assert "CURRENT TERMINAL STATE" in handoff
        assert agent._pending_subagent_refs is None
    else:
        assert agent._pending_subagent_refs


async def test_unrecoverable_context_error_is_not_a_fabricated_response(
    recovery_agent: Terminus2, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = recovery_agent
    session = AsyncMock(spec=TmuxSession)
    session.capture_pane.return_value = "screen"
    monkeypatch.setattr(
        agent, "_summarize", AsyncMock(side_effect=ContextLengthExceededError())
    )
    mock_llm(agent).call.side_effect = ContextLengthExceededError()

    with pytest.raises(ContextLengthExceededError):
        await agent._query_llm(history_chat(agent), "continue", "task", session)
    assert mock_llm(agent).call.await_count == 3


async def test_disabled_summarization_does_not_retry(
    recovery_agent: Terminus2, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = recovery_agent
    agent.options.enable_summarize = False
    summarize = AsyncMock()
    monkeypatch.setattr(agent, "_summarize", summarize)
    mock_llm(agent).call.side_effect = ContextLengthExceededError()

    with pytest.raises(ContextLengthExceededError):
        await agent._query_llm(history_chat(agent), "continue")
    assert mock_llm(agent).call.await_count == 1
    summarize.assert_not_awaited()


@pytest.mark.parametrize("proactive", [False, True])
@pytest.mark.parametrize("answer_size", [7, 30000])
async def test_full_summary_fits_generated_text_into_answers_request(
    recovery_agent: Terminus2,
    monkeypatch: pytest.MonkeyPatch,
    proactive: bool,
    answer_size: int,
) -> None:
    agent = recovery_agent
    mock_llm(agent).get_model_context_limit.return_value = 60000
    agent.options.proactive_summarization_threshold = 21000
    chat = Chat(agent._llm)
    chat._messages = [
        {"role": "system", "content": "terminal protocol"},
        *[
            {"role": role, "content": "x" * 10000}
            for role in ("user", "assistant", "user", "assistant")
        ],
    ]
    session = AsyncMock(spec=TmuxSession)
    session.capture_pane.return_value = "current terminal"

    def count_tokens(model: str, messages: list[dict[str, Any]]) -> int:
        return sum(len(message["content"]) for message in messages)

    monkeypatch.setattr(
        module, "token_counter", count_tokens
    )
    monkeypatch.setattr(
        agent, "_count_total_tokens", lambda chat: count_tokens("", chat.messages)
    )
    requests = []

    async def call(prompt: str, **kwargs: Any) -> LLMResponse:
        history = copy.deepcopy(kwargs.get("message_history", []))
        requests.append((prompt, history))
        if not proactive and len(requests) == 1:
            raise ContextLengthExceededError()
        size = count_tokens("", history) + len(prompt)
        if size > 60000:
            raise ContextLengthExceededError(f"request contains {size} tokens")
        if prompt.startswith("You are about to hand off"):
            return LLMResponse(content="s" * 25000)
        if prompt.startswith("You are picking up work"):
            return LLMResponse(content="q" * 12000)
        if prompt.startswith("The next agent has a few questions"):
            return LLMResponse(content="a" * answer_size)
        if prompt.startswith("Briefly continue"):
            return LLMResponse(content="Inspect the saved implementation.")
        return LLMResponse(content='{"commands": [], "task_complete": false}')

    mock_llm(agent).call.side_effect = call
    if proactive:
        handoff = await agent._check_proactive_summarization(chat, "task", session)
        assert handoff is not None
        prompt, refs = handoff
        assert refs is not None and len(refs) == (1 if answer_size == 30000 else 3)
        await chat.chat(prompt)
    else:
        await agent._query_llm(chat, "continue", "task", session)
        assert agent._pending_subagent_refs is not None
        assert len(agent._pending_subagent_refs) == (1 if answer_size == 30000 else 3)

    summary_history = next(h for p, h in requests if p.startswith("You are about"))
    answer_history = next(h for p, h in requests if p.startswith("The next agent"))
    assert len(summary_history) == 5
    # Only system + summary prompt + summary response fit alongside questions.
    assert len(answer_history) == 3
    assert answer_history[0] == summary_history[0]
    assert len(requests) == (4 if proactive else 5) + (answer_size == 30000)
    assert agent._summarization_count == 1


@pytest.mark.parametrize("failures", [1, 3])
async def test_recovered_transport_retries_preserve_handoff(
    recovery_agent: Terminus2,
    monkeypatch: pytest.MonkeyPatch,
    failures: int,
) -> None:
    agent = recovery_agent
    chat = history_chat(agent)
    session = AsyncMock(spec=TmuxSession)
    handoff = "Here are the answers: preserve the completed implementation."

    async def summarize(*args: Any) -> tuple[str, None]:
        chat._messages = chat.messages[:1]
        chat.reset_response_chain()
        return handoff, None

    summary = AsyncMock(side_effect=summarize)
    monkeypatch.setattr(agent, "_summarize_with_fallback", summary)
    requests = []

    async def call(prompt: str, **kwargs: Any) -> LLMResponse:
        requests.append((prompt, copy.deepcopy(kwargs)))
        if len(requests) == 1:
            raise ContextLengthExceededError()
        if len(requests) <= 1 + failures:
            raise RuntimeError("transient provider error")
        return LLMResponse(content="continued")

    mock_llm(agent).call.side_effect = call
    if failures == 3:
        with pytest.raises(RuntimeError, match="transient provider"):
            await agent._query_llm(chat, "original prompt", "task", session)
    else:
        response = await agent._query_llm(chat, "original prompt", "task", session)
        assert response.content == "continued"
    assert len(requests) == 1 + min(failures + 1, 3)
    assert all(p == handoff for p, _ in requests[1:])
    assert all(k == requests[1][1] for _, k in requests[1:])
    summary.assert_awaited_once()


@pytest.mark.parametrize("salvage", [False, True])
async def test_recovered_output_uses_native_truncation_handler(
    recovery_agent: Terminus2, monkeypatch: pytest.MonkeyPatch, salvage: bool
) -> None:
    agent = recovery_agent
    chat = history_chat(agent)
    session = AsyncMock(spec=TmuxSession)
    handoff = "Here are the answers from the prior agent."
    mock_llm(agent).get_model_output_limit.return_value = 1024
    if salvage:
        agent._parser = Mock()
        agent._parser.salvage_truncated_response.return_value = (
            LLMResponse(content="salvaged"), False
        )

    async def summarize(*args: Any) -> tuple[str, None]:
        chat._messages = chat.messages[:1]
        chat.reset_response_chain()
        return handoff, None

    monkeypatch.setattr(agent, "_summarize_with_fallback", summarize)
    requests = []

    async def call(prompt: str, **kwargs: Any) -> LLMResponse:
        requests.append((prompt, copy.deepcopy(kwargs)))
        if len(requests) == 1:
            raise ContextLengthExceededError()
        if len(requests) == 2:
            raise OutputLengthExceededError("truncated", truncated_response="partial")
        return LLMResponse(content="continued")

    mock_llm(agent).call.side_effect = call
    response = await agent._query_llm(chat, "original prompt", "task", session)
    assert response.content == ("salvaged" if salvage else "continued")
    assert len(requests) == (2 if salvage else 3)
    if not salvage:
        assert {"role": "user", "content": handoff} in requests[-1][1]["message_history"]
        assert requests[-1][0].startswith("ERROR!!")


async def test_proactive_failure_also_replaces_history(
    recovery_agent: Terminus2, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = recovery_agent
    agent.options.proactive_summarization_threshold = 9000
    chat = history_chat(agent)
    session = AsyncMock(spec=TmuxSession)
    session.capture_pane.return_value = "screen"
    monkeypatch.setattr(
        agent, "_summarize", AsyncMock(side_effect=ContextLengthExceededError())
    )
    mock_llm(agent).call.return_value = LLMResponse(content="next steps")

    result = await agent._check_proactive_summarization(chat, "task", session)

    assert result is not None
    assert len(chat.messages) == 1
    assert chat._last_response_id is None
    assert mock_llm(agent).call.await_count == 1


def test_unwind_never_removes_initial_message(
    recovery_agent: Terminus2, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = history_chat(recovery_agent)
    chat._messages = chat.messages[:2]
    initial = copy.deepcopy(chat.messages[0])
    monkeypatch.setattr(recovery_agent, "_count_total_tokens", lambda chat: 25000)

    recovery_agent._unwind_messages_to_free_tokens(chat)

    assert chat.messages == [initial]
    assert chat._last_response_id is None


async def test_cancellation_does_not_start_fallback(
    recovery_agent: Terminus2, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = recovery_agent
    session = AsyncMock(spec=TmuxSession)
    monkeypatch.setattr(
        agent, "_summarize", AsyncMock(side_effect=asyncio.CancelledError)
    )

    with pytest.raises(asyncio.CancelledError):
        await agent._summarize_with_fallback(history_chat(agent), "task", session)
    mock_llm(agent).call.assert_not_awaited()
    session.capture_pane.assert_not_awaited()


@pytest.mark.parametrize("linear_history", [False, True])
async def test_terminal_fallback_records_compaction_boundary(
    recovery_agent: Terminus2,
    monkeypatch: pytest.MonkeyPatch,
    linear_history: bool,
) -> None:
    agent = recovery_agent
    chat = history_chat(agent)
    agent._chat = chat
    agent._context = AgentContext()
    agent._max_episodes = 1
    agent.options.trajectory_config = {"linear_history": linear_history}
    agent._trajectory_steps = [Step(step_id=1, source="system", message="protocol")]
    session = AsyncMock(spec=TmuxSession)
    session.is_session_alive.return_value = True
    session.capture_pane.return_value = "screen"
    session.get_incremental_output.return_value = "screen"
    agent._session = session

    async def failed_summary(*args: Any) -> None:
        agent._summarization_count += 1
        raise ContextLengthExceededError()

    monkeypatch.setattr(agent, "_summarize", failed_summary)
    mock_llm(agent).call.side_effect = [
        ContextLengthExceededError(),
        ContextLengthExceededError(),
        LLMResponse(
            content='{"analysis": "Continue", "plan": "Inspect", "commands": [], "task_complete": false}'
        ),
    ]
    dump = Mock()
    monkeypatch.setattr(agent, "_dump_trajectory_with_continuation_index", dump)
    monkeypatch.setattr(agent, "_dump_trajectory", Mock())

    await agent._run_agent_loop("continue", chat, "task")

    assert agent._pending_handoff_prompt is None
    if linear_history:
        dump.assert_called_once_with(0)
        assert agent._session_id.endswith("-cont-1")
        assert len(agent._trajectory_steps) == 3
    else:
        boundaries = [
            s
            for s in agent._trajectory_steps
            if s.extra and "context_management" in s.extra
        ]
        assert len(boundaries) == 1
        assert boundaries[0].extra is not None
        assert boundaries[0].extra["context_management"]["boundary"] == "replace"
        assert boundaries[0].observation is None
    assert any(
        "Current state: screen" in str(s.message) for s in agent._trajectory_steps
    )


@pytest.mark.parametrize("output_key", ["max_tokens", "max_completion_tokens"])
def test_summary_request_reserves_output_and_prompt(
    recovery_agent: Terminus2, monkeypatch: pytest.MonkeyPatch, output_key: str
) -> None:
    """Reproduce the 64K-free / 80K-requested boundary with deterministic counts."""
    agent = recovery_agent
    mock_llm(agent).get_model_context_limit.return_value = 262144
    agent._llm_call_kwargs = {output_key: 81920, "top_p": 0.95}
    original_kwargs = copy.deepcopy(agent._llm_call_kwargs)

    def count_tokens(model: str, messages: list[dict[str, Any]]) -> int:
        return sum(len(message["content"]) for message in messages)

    monkeypatch.setattr(module, "token_counter", count_tokens)
    monkeypatch.setattr(agent, "_count_total_tokens", lambda chat: count_tokens("", chat.messages))
    chat = Chat(agent._llm)
    chat._messages = [
        {"role": "system", "content": "terminal protocol"},
        {"role": "user", "content": "u" * 80000},
        {"role": "assistant", "content": "a" * 80000},
        {"role": "user", "content": "u" * 18000},
        {"role": "assistant", "content": "a" * 18000},
    ]
    prompt = [{"role": "user", "content": "summary prompt" * 700}]
    assert count_tokens("", chat.messages + prompt) + 81920 > 262144
    agent._unwind_messages_to_free_tokens(chat, extra_messages=prompt)
    assert count_tokens("", chat.messages + prompt) + 81920 <= 262144
    assert len(chat.messages) == 3
    assert chat.messages[0]["content"] == "terminal protocol"
    assert agent._llm_call_kwargs == original_kwargs


async def test_proactive_summary_accounts_for_80k_output(
    recovery_agent: Terminus2, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = recovery_agent
    mock_llm(agent).get_model_context_limit.return_value = 262144
    agent._llm_call_kwargs = {"max_tokens": 81920}
    monkeypatch.setattr(agent, "_count_total_tokens", lambda chat: 197889)
    summarize = AsyncMock(return_value=("handoff", None))
    monkeypatch.setattr(agent, "_summarize_with_fallback", summarize)
    result = await agent._check_proactive_summarization(
        history_chat(agent), "task", AsyncMock(spec=TmuxSession)
    )
    assert result == ("handoff", None)
    summarize.assert_awaited_once()


async def test_zero_proactive_threshold_remains_disabled(
    recovery_agent: Terminus2, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = recovery_agent
    agent.options.proactive_summarization_threshold = 0
    agent._llm_call_kwargs = {"max_tokens": 81920}
    summarize = AsyncMock()
    monkeypatch.setattr(agent, "_summarize_with_fallback", summarize)
    assert await agent._check_proactive_summarization(
        history_chat(agent), "task", AsyncMock(spec=TmuxSession)
    ) is None
    summarize.assert_not_awaited()


async def test_generated_handoff_includes_output_reservation(
    recovery_agent: Terminus2, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = recovery_agent
    mock_llm(agent).get_model_context_limit.return_value = 60000
    agent._llm_call_kwargs = {"max_tokens": 15000}
    chat = history_chat(agent)
    session = AsyncMock(spec=TmuxSession)
    session.capture_pane.return_value = "screen"

    def count_tokens(model: str, messages: list[dict[str, Any]]) -> int:
        return sum(len(message["content"]) for message in messages)

    monkeypatch.setattr(module, "token_counter", count_tokens)
    monkeypatch.setattr(agent, "_count_total_tokens", lambda chat: count_tokens("", chat.messages))
    requests = []

    async def call(prompt: str, **kwargs: Any) -> LLMResponse:
        requests.append(prompt)
        assert count_tokens("", kwargs.get("message_history", [])) + len(prompt) + 15000 <= 60000
        if prompt.startswith("You are about to hand off"):
            return LLMResponse(content="s" * 21000)
        if prompt.startswith("You are picking up work"):
            return LLMResponse(content="q" * 9000)
        if prompt.startswith("The next agent has a few questions"):
            return LLMResponse(content="a" * 19000)
        return LLMResponse(content="Inspect the saved implementation.")

    mock_llm(agent).call.side_effect = call
    handoff, refs = await agent._summarize_with_fallback(chat, "TASK", session)
    assert len(requests) == 4
    assert refs is not None and len(refs) == 1
    assert "Current state: screen" in handoff
    assert len(chat.messages) == 1


async def test_short_summary_cancellation_is_propagated(
    recovery_agent: Terminus2, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = recovery_agent
    monkeypatch.setattr(agent, "_summarize", AsyncMock(side_effect=ContextLengthExceededError()))
    mock_llm(agent).call.side_effect = asyncio.CancelledError()
    session = AsyncMock(spec=TmuxSession)
    session.capture_pane.return_value = "screen"
    with pytest.raises(asyncio.CancelledError):
        await agent._summarize_with_fallback(history_chat(agent), "task", session)
    assert mock_llm(agent).call.await_count == 1
