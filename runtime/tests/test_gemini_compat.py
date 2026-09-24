"""Gemini 3 tool-call thought signatures and sampling temperature.

Gemini 3 returns ``extra_content.google.thought_signature`` on tool calls and
rejects the next request with HTTP 400 if that field is not echoed back.
Google also recommends temperature 1.0 for Gemini 3+; lower values loop.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from app.llm import LLM, effective_temperature, is_gemini_3_plus
from app.schema import Function, Message, ToolCall


class _SdkFunction(BaseModel):
    """Stand-in for an OpenAI SDK model: unknown fields live in model_extra."""

    model_config = ConfigDict(extra="allow")

    name: str
    arguments: str


class _SdkToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str = "function"
    function: _SdkFunction


def _signature(value: str = "sig-abc") -> dict:
    return {"google": {"thought_signature": value}}


def test_tool_call_extra_content_is_optional_and_omitted_when_absent():
    call = ToolCall(id="c1", function=Function(name="bash", arguments="{}"))
    assert call.extra_content is None

    message = Message.from_tool_calls([call], content="done")
    dumped = message.to_dict()
    assert dumped["role"] == "assistant"
    assert dumped["content"] == "done"
    assert dumped["tool_calls"] == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{}"},
        }
    ]
    assert "extra_content" not in dumped["tool_calls"][0]


def test_from_tool_calls_copies_extra_content_attribute():
    call = ToolCall(
        id="c1",
        function=Function(name="bash", arguments='{"command":"ls"}'),
        extra_content=_signature("from-attr"),
    )
    message = Message.from_tool_calls([call], content="")
    assert message.tool_calls[0].extra_content == _signature("from-attr")
    assert message.to_dict()["tool_calls"][0]["extra_content"] == _signature(
        "from-attr"
    )


def test_from_tool_calls_reads_openai_sdk_model_extra():
    """OpenAI SDK objects do not declare extra_content; it sits in model_extra."""
    call = _SdkToolCall.model_validate(
        {
            "id": "call_sdk",
            "type": "function",
            "function": {"name": "bash", "arguments": "{}"},
            "extra_content": _signature("from-sdk"),
        }
    )
    assert "extra_content" not in _SdkToolCall.model_fields
    assert call.model_extra["extra_content"] == _signature("from-sdk")

    message = Message.from_tool_calls([call], content="calling bash")
    tool_call = message.to_dict()["tool_calls"][0]
    assert tool_call["extra_content"] == _signature("from-sdk")
    assert tool_call["function"] == {"name": "bash", "arguments": "{}"}

    # The payload actually handed to the client keeps the signature.
    formatted = LLM.format_messages([message])
    assert (
        formatted[0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"]
        == "from-sdk"
    )


def test_model_extra_is_used_when_extra_content_is_not_an_attribute():
    function = Function(name="read", arguments='{"path":"README.md"}')
    call = SimpleNamespace(
        id="call_raw",
        type="function",
        function=function,
        model_extra={"extra_content": _signature("raw-only")},
    )
    message = Message.from_tool_calls([call])
    assert message.to_dict()["tool_calls"][0]["extra_content"] == _signature("raw-only")


def test_parallel_calls_keep_only_the_signatures_that_were_returned():
    signed = SimpleNamespace(
        id="call_a",
        function=Function(name="bash", arguments="{}"),
        model_extra={"extra_content": _signature("only-first")},
    )
    unsigned = ToolCall(id="call_b", function=Function(name="bash", arguments="{}"))
    message = Message.from_tool_calls([signed, unsigned], content=None)
    calls = message.to_dict()["tool_calls"]
    assert calls[0]["extra_content"] == _signature("only-first")
    assert "extra_content" not in calls[1]
    assert "content" not in message.to_dict()


def test_real_openai_sdk_tool_call_keeps_thought_signature():
    from openai.types.chat import ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion_message_tool_call import (
        Function as SdkFunction,
    )

    call = ChatCompletionMessageToolCall.model_validate(
        {
            "id": "call_live_sdk",
            "type": "function",
            "function": SdkFunction(name="bash", arguments="{}").model_dump(),
            "extra_content": _signature("live-sdk"),
        }
    )
    stored = getattr(call, "extra_content", None)
    if not isinstance(stored, dict):
        stored = (call.model_extra or {}).get("extra_content")
    assert stored == _signature("live-sdk")

    message = Message.from_tool_calls([call], content="ok")
    assert message.to_dict()["tool_calls"][0]["extra_content"] == _signature("live-sdk")


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gemini-3", True),
        ("gemini-3-pro", True),
        ("gemini-3-pro-preview", True),
        ("gemini-3.1-flash-lite", True),
        ("gemini-3.5-flash", True),
        ("gemini-3.8-flash", True),
        ("GEMINI-3-PRO", True),
        ("google/gemini-3-pro-preview", True),
        ("gemini/gemini-3.1-pro-preview", True),
        ("models/gemini-3.0-pro", True),
        ("gemini-4-flash", True),
        ("gemini-10-pro", True),
        ("gemini-2.5-pro", False),
        ("gemini-2.0-flash", False),
        ("gemini-1.5-pro", False),
        ("gemini-pro", False),
        ("gemini-flash", False),
        ("gpt-4o", False),
        ("claude-sonnet-5", False),
        ("", False),
    ],
)
def test_is_gemini_3_plus(model, expected):
    assert is_gemini_3_plus(model) is expected


def test_gemini_3_temperature_is_one_unless_keep_flag(monkeypatch):
    monkeypatch.delenv("LLM_KEEP_TEMPERATURE", raising=False)
    assert effective_temperature("gemini-3-pro-preview", 0.0) == 1.0
    assert effective_temperature("google/gemini-3.1-flash", 0.2) == 1.0
    assert effective_temperature("gemini-2.5-pro", 0.0) == 0.0
    assert effective_temperature("gpt-4o", 0.3) == 0.3
    # Already at the recommended value: no change, and not a special case.
    assert effective_temperature("gemini-3-pro", 1.0) == 1.0

    monkeypatch.setenv("LLM_KEEP_TEMPERATURE", "true")
    assert effective_temperature("gemini-3-pro-preview", 0.0) == 0.0
    assert effective_temperature("gemini-3.1-flash", 0.4) == 0.4

    monkeypatch.setenv("LLM_KEEP_TEMPERATURE", "1")
    assert effective_temperature("gemini-4-pro", 0.2) == 0.2

    monkeypatch.setenv("LLM_KEEP_TEMPERATURE", "false")
    assert effective_temperature("gemini-3.5-flash", 0.0) == 1.0


def _make_llm(model: str, temperature: float) -> LLM:
    from app.config import LLMSettings

    LLM._instances.pop("default", None)
    settings = LLMSettings(
        model=model,
        base_url="http://localhost",
        api_key="test",
        max_tokens=128,
        temperature=temperature,
        api_type="openai",
        api_version="",
    )
    return LLM("default", {"default": settings})


def test_sampling_params_follow_gemini_recommendation(monkeypatch):
    monkeypatch.delenv("LLM_KEEP_TEMPERATURE", raising=False)
    monkeypatch.delenv("LLM_REASONING_MODEL", raising=False)
    gemini = _make_llm("gemini-3.1-pro-preview", 0.0)
    assert gemini._sampling_params(None) == {"max_tokens": 128, "temperature": 1.0}
    # An explicit per-call temperature is still overridden for Gemini 3+.
    assert gemini._sampling_params(0.2)["temperature"] == 1.0

    monkeypatch.setenv("LLM_KEEP_TEMPERATURE", "true")
    assert gemini._sampling_params(None)["temperature"] == 0.0
    assert gemini._sampling_params(0.2)["temperature"] == 0.2

    monkeypatch.delenv("LLM_KEEP_TEMPERATURE", raising=False)
    older = _make_llm("gemini-2.5-pro", 0.0)
    assert older._sampling_params(None) == {"max_tokens": 128, "temperature": 0.0}
    gpt = _make_llm("gpt-4o", 0.2)
    assert gpt._sampling_params(0.4)["temperature"] == 0.4

    reasoning = _make_llm("gpt-5", 0.0)
    assert reasoning._sampling_params(0.2) == {"max_completion_tokens": 128}
    LLM._instances.clear()


def test_ask_tool_sends_temperature_one_for_gemini_3(monkeypatch):
    monkeypatch.delenv("LLM_KEEP_TEMPERATURE", raising=False)
    llm = _make_llm("gemini-3-pro-preview", 0.0)
    captured = {}

    class Completions:
        async def create(self, **params):
            captured.update(params)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok", tool_calls=None)
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
            )

    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    async def go():
        return await llm.ask_tool([Message.user_message("list the files")])

    asyncio.run(go())
    assert captured["temperature"] == 1.0
    assert captured["model"] == "gemini-3-pro-preview"
    assert "max_tokens" in captured
    LLM._instances.clear()
