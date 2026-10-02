from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.llm import LLM, ThinkTagFilter, strip_think_tags


def test_think_tag_filter_handles_tags_split_across_chunks():
    stream_filter = ThinkTagFilter()
    chunks = ["Visible <thi", "nk>private", " scratch</thi", "nk> answer"]
    visible = "".join(stream_filter.feed(chunk) for chunk in chunks) + stream_filter.finish()
    assert visible == "Visible  answer"
    assert "private" not in visible
    assert strip_think_tags("A<think>hidden</think>B") == "AB"
    assert strip_think_tags("A<think>unfinished private block") == "A"


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        self._index = 0
        return self

    async def __anext__(self):
        if self._index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self._index]
        self._index += 1
        return chunk


class FakeCompletions:
    def __init__(self, chunks):
        self.chunks = chunks
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeStream(self.chunks)


def _chunk(*, content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
        usage=None,
    )


@pytest.mark.asyncio
async def test_ask_stream_emits_tokens_and_returns_complete_text():
    completion_client = FakeCompletions([
        SimpleNamespace(choices=[]),
        _chunk(content="First "),
        _chunk(content="answer", finish_reason="stop"),
    ])
    llm = object.__new__(LLM)
    llm.model = "qwen2.5-coder-7b"
    llm.max_tokens = 256
    llm.temperature = 0.1
    llm.max_input_tokens = None
    llm.total_input_tokens = 0
    llm.total_completion_tokens = 0
    llm.tokenizer = SimpleNamespace(encode=lambda text: list(text))
    llm.token_counter = SimpleNamespace(count_message_tokens=lambda messages: 5)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completion_client))
    llm.check_token_limit = lambda _: True
    llm.get_limit_error_message = lambda _: "token limit"
    llm.count_tokens = lambda text: len(text)
    llm.count_message_tokens = lambda messages: 5
    llm.update_token_count = lambda input_tokens, completion_tokens=0: None

    visible = []
    answer = await llm.ask(
        messages=[{"role": "user", "content": "Answer"}],
        stream=True,
        on_token=visible.append,
    )

    assert completion_client.kwargs["stream"] is True
    assert visible == ["First ", "answer"]
    assert answer == "First answer"


@pytest.mark.asyncio
async def test_ask_ollama_json_response_without_usage_metadata():
    class Completion:
        kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"capability_ids":[167]}'), finish_reason="stop")],
                usage=None,
            )

    completion = Completion()
    llm = object.__new__(LLM)
    llm.model = "deepseek-r1:7b"
    llm.max_tokens = 2048
    llm.temperature = 0.1
    llm.max_input_tokens = None
    llm.total_input_tokens = 0
    llm.total_completion_tokens = 0
    llm.tokenizer = SimpleNamespace(encode=lambda text: list(text))
    llm.token_counter = SimpleNamespace(count_message_tokens=lambda messages: 5)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completion))
    llm.check_token_limit = lambda _: True
    llm.count_tokens = lambda text: len(text)
    llm.count_message_tokens = lambda messages: 5
    recorded = []
    llm.update_token_count = lambda prompt, completion: recorded.append((prompt, completion))

    result = await llm.ask(
        [{"role": "user", "content": "Hello"}],
        stream=False,
        response_format={"type": "json_object"},
    )
    assert result == '{"capability_ids":[167]}'
    assert completion.kwargs["response_format"] == {"type": "json_object"}
    assert recorded == [(5, len(result))]


@pytest.mark.asyncio
async def test_ask_tool_stream_reassembles_function_name_and_arguments():
    first = SimpleNamespace(
        index=0,
        id="call_1",
        function=SimpleNamespace(name="platform_", arguments='{"action":"nav'),
    )
    second = SimpleNamespace(
        index=0,
        id=None,
        function=SimpleNamespace(name="browser", arguments='igate","url":"https://example.com"}'),
    )
    completion_client = FakeCompletions([
        _chunk(content="Checking "),
        _chunk(content="the site.", tool_calls=[first]),
        _chunk(tool_calls=[second], finish_reason="tool_calls"),
    ])

    llm = object.__new__(LLM)
    llm.model = "qwen2.5-coder-7b"
    llm.max_tokens = 256
    llm.temperature = 0.1
    llm.max_input_tokens = None
    llm.total_input_tokens = 0
    llm.total_completion_tokens = 0
    llm.tokenizer = SimpleNamespace(encode=lambda text: list(text))
    llm.token_counter = SimpleNamespace(count_message_tokens=lambda messages: 5)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completion_client))

    llm.check_token_limit = lambda _: True
    llm.get_limit_error_message = lambda _: "token limit"
    llm.count_tokens = lambda text: len(text)
    llm.count_message_tokens = lambda messages: 5
    llm.update_token_count = lambda input_tokens, completion_tokens=0: None

    visible = []
    response = await llm.ask_tool(
        messages=[{"role": "user", "content": "Open the example site"}],
        tools=[{"type": "function", "function": {"name": "platform_browser"}}],
        on_token=visible.append,
    )

    assert completion_client.kwargs["stream"] is True
    assert "".join(visible) == "Checking the site."
    assert response.content == "Checking the site."
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].function.name == "platform_browser"
    assert response.tool_calls[0].function.arguments == '{"action":"navigate","url":"https://example.com"}'
