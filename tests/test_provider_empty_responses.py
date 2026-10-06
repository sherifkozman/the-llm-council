"""Offline regressions for provider answer/diagnostic normalization (#65/#67)."""

import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import pytest

from llm_council.providers.base import (
    ErrorType,
    GenerateRequest,
    StructuredOutputConfig,
    classify_error,
    ensure_text_response,
)
from llm_council.providers.openrouter import OpenRouterProvider
from llm_council.providers.vertex import VertexAIProvider


@pytest.fixture
def genai_types():
    return pytest.importorskip("google.genai.types")


@pytest.fixture
def vertex():
    return VertexAIProvider(project="offline-test", default_model="gemini-3.1-pro-preview")


def test_vertex_function_only_retains_calls_and_finish_reason(genai_types, vertex):
    raw = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    parts=[
                        genai_types.Part(
                            function_call=genai_types.FunctionCall(
                                id="call-1", name="inspect", args={"path": "synthetic.py"}
                            )
                        ),
                        genai_types.Part(function_call=genai_types.FunctionCall(name="finish")),
                    ]
                ),
                finish_reason="STOP",
            )
        ]
    )

    response = vertex._parse_response(raw)

    assert response.text is None
    assert response.content is None
    assert response.tool_calls == [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "inspect", "arguments": {"path": "synthetic.py"}},
        },
        {"id": None, "type": "function", "function": {"name": "finish", "arguments": {}}},
    ]
    assert response.finish_reason == "STOP"
    assert response.raw is raw


def test_vertex_mixed_parts_preserve_only_answer_text_without_sdk_accessor(
    genai_types, vertex, caplog
):
    raw = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    parts=[
                        genai_types.Part(text="private-thought-sentinel", thought=True),
                        genai_types.Part(text="Visible "),
                        genai_types.Part(
                            function_call=genai_types.FunctionCall(name="inspect", args={})
                        ),
                        genai_types.Part(
                            inline_data=genai_types.Blob(data=b"image", mime_type="image/png")
                        ),
                        genai_types.Part(text="answer."),
                    ]
                ),
                finish_reason="MAX_TOKENS",
            ),
            genai_types.Candidate(
                content=genai_types.Content(
                    parts=[
                        genai_types.Part(text="another candidate must not be joined"),
                    ]
                )
            ),
        ]
    )
    with patch.object(type(raw), "text", new_callable=PropertyMock) as accessor:
        accessor.side_effect = AssertionError("SDK convenience .text must not be accessed")
        response = vertex._parse_response(raw)

    assert accessor.call_count == 0
    assert response.text == response.content == "Visible answer."
    assert response.tool_calls == [
        {"id": None, "type": "function", "function": {"name": "inspect", "arguments": {}}}
    ]
    assert response.finish_reason == "MAX_TOKENS"
    assert not caplog.records


@pytest.mark.parametrize(
    "parts",
    [
        [{"text": "private-thought-sentinel", "thought": True}],
        [{"text": " \n\t "}],
        [{}],
        [],
        None,
    ],
)
def test_vertex_non_answers_stay_empty_with_finish_reason(genai_types, vertex, parts):
    raw = genai_types.GenerateContentResponse(
        candidates=[genai_types.Candidate(content={"parts": parts}, finish_reason="MAX_TOKENS")]
    )

    response = vertex._parse_response(raw)

    assert response.text is None
    assert response.content is None
    assert response.tool_calls is None
    assert response.finish_reason == "MAX_TOKENS"


@pytest.mark.parametrize("candidates", [None, [], [{"finish_reason": "SAFETY"}]])
def test_vertex_empty_candidates_do_not_crash(genai_types, vertex, candidates):
    raw = genai_types.GenerateContentResponse(candidates=candidates)
    response = vertex._parse_response(raw)
    assert response.text is None
    assert response.tool_calls is None
    assert response.finish_reason == ("SAFETY" if candidates else None)


@pytest.mark.asyncio
async def test_vertex_request_remains_tools_free(genai_types, vertex):
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(
        return_value=genai_types.GenerateContentResponse(candidates=[])
    )
    with patch.object(vertex, "_get_gemini_client", return_value=client):
        await vertex.generate(GenerateRequest(prompt="Synthetic diagnostic test"))

    kwargs = client.aio.models.generate_content.await_args.kwargs
    assert kwargs["contents"] == "Synthetic diagnostic test"
    assert (
        not {"tools", "tool_config", "automatic_function_calling"} & (kwargs["config"] or {}).keys()
    )
    assert client.aio.models.generate_content.await_count == 1


@pytest.mark.parametrize("content", [None, "", " \n\t "])
@pytest.mark.parametrize("finish_reason", ["length", "stop"])
def test_openrouter_reasoning_only_is_not_an_answer(content, finish_reason, caplog):
    provider = OpenRouterProvider(api_key="offline-test")
    raw = {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "content": content,
                    "reasoning": "private-reasoning-sentinel",
                    "reasoning_details": [
                        {"type": "reasoning.text", "text": "private-detail-sentinel"}
                    ],
                },
            }
        ]
    }

    response = provider._parse_response(raw)

    assert response.text is None
    assert response.content is None
    assert response.tool_calls is None
    assert response.finish_reason == finish_reason
    assert "private-" not in repr(response.model_dump(exclude={"raw"}))
    assert not caplog.records


@pytest.mark.parametrize("choices", [None, [], [None], ["bad"], {}, "bad"])
def test_openrouter_malformed_choices_normalize_to_empty(choices):
    response = OpenRouterProvider(api_key="offline-test")._parse_response({"choices": choices})
    assert response.text is None
    assert response.tool_calls is None
    assert response.finish_reason is None


@pytest.mark.parametrize("message", [None, "bad", [], {"content": {"private": "sentinel"}}])
def test_openrouter_malformed_message_preserves_finish_reason(message):
    response = OpenRouterProvider(api_key="offline-test")._parse_response(
        {"choices": [{"message": message, "finish_reason": "length"}]}
    )
    assert response.text is None
    assert response.content is None
    assert response.finish_reason == "length"


@pytest.mark.asyncio
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("field", ["reasoning", "reasoning_content", "reasoning_details"])
@pytest.mark.parametrize("content", [None, "", " \n ", '{"result":"FINAL"}'])
async def test_openrouter_generation_accepts_only_final_channel(structured, field, content):
    reasoning = '{"result": "private-reasoning-sentinel"}'
    value = (
        [{"type": "reasoning.text", "text": reasoning}]
        if field == "reasoning_details"
        else reasoning
    )
    raw = {"choices": [{"message": {"content": content, field: value}, "finish_reason": "length"}]}

    def handle(request):
        body = json.loads(request.content)
        assert ("response_format" in body) is structured
        return httpx.Response(200, json=raw)

    schema = (
        StructuredOutputConfig(
            json_schema={
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
                "additionalProperties": False,
            }
        )
        if structured
        else None
    )
    provider = OpenRouterProvider(api_key="offline-test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with patch.object(provider, "_get_client", new_callable=AsyncMock, return_value=client):
            response = await provider.generate(
                GenerateRequest(prompt="Synthetic test", structured_output=schema)
            )

    assert response.finish_reason == "length"
    assert "private-reasoning-sentinel" not in repr(response.model_dump(exclude={"raw"}))
    if content and content.strip():
        assert response.text == response.content == content
        ensure_text_response(response, provider="openrouter", phase="synthesis")
    else:
        assert response.text is None
        assert response.content is None
        with pytest.raises(RuntimeError, match="empty_response") as caught:
            ensure_text_response(response, provider="openrouter", phase="synthesis")
        assert classify_error(str(caught.value)) == ErrorType.EMPTY_RESPONSE
        assert "finish_reason=length" in str(caught.value)
        assert "private-reasoning-sentinel" not in str(caught.value)


def test_openrouter_non_json_reasoning_is_never_recovered():
    response = OpenRouterProvider(api_key="offline-test")._parse_response(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": " \n ",
                        "reasoning": "private-reasoning-sentinel",
                    },
                }
            ]
        },
    )
    assert response.text is None
    assert response.finish_reason == "length"


def test_openrouter_retains_answer_whitespace_and_tool_calls():
    call = {"id": "call-1", "type": "function", "function": {"name": "inspect", "arguments": "{}"}}
    response = OpenRouterProvider(api_key="offline-test")._parse_response(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": " Answer \n",
                        "tool_calls": [call],
                        "reasoning": "private-sentinel",
                    },
                }
            ]
        }
    )
    assert response.text == response.content == " Answer \n"
    assert response.tool_calls == [call]
    assert response.finish_reason == "tool_calls"


def test_openrouter_request_remains_tools_free_unless_explicit():
    provider = OpenRouterProvider(api_key="offline-test")
    body = provider._build_request_body(GenerateRequest(prompt="Synthetic test"))
    assert "tools" not in body
    assert "tool_choice" not in body
    tool = {"type": "function", "function": {"name": "inspect", "parameters": {"type": "object"}}}
    body = provider._build_request_body(GenerateRequest(prompt="Synthetic test", tools=[tool]))
    assert body["tools"] == [tool]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parts",
    [
        None,
        [{"function_call": {"name": "inspect", "args": {"path": "synthetic.py"}}}],
        [{"text": "private-thought-sentinel", "thought": True}],
    ],
)
async def test_vertex_stream_retains_nontext_terminal_chunks(genai_types, vertex, parts):
    raw = genai_types.GenerateContentResponse(
        candidates=[genai_types.Candidate(content={"parts": parts}, finish_reason="MAX_TOKENS")]
    )

    async def stream_chunks():
        yield raw

    client = MagicMock()
    client.aio.models.generate_content_stream = AsyncMock(return_value=stream_chunks())
    chunks = [chunk async for chunk in vertex._generate_stream(client, "gemini", "test", {})]

    assert len(chunks) == 1
    assert chunks[0].text is None
    assert chunks[0].content is None
    assert chunks[0].finish_reason == "MAX_TOKENS"
    expected_calls = (
        [
            {
                "id": None,
                "type": "function",
                "function": {"name": "inspect", "arguments": {"path": "synthetic.py"}},
            }
        ]
        if parts and "function_call" in parts[0]
        else None
    )
    assert chunks[0].tool_calls == expected_calls


@pytest.mark.asyncio
async def test_vertex_stream_preserves_text_separators_without_sdk_text(
    genai_types, vertex, caplog
):
    async def stream_chunks():
        for part in [
            {"text": "private-thought-sentinel", "thought": True},
            {"text": "Visible"},
            {"text": " "},
            {"text": "answer"},
            {"function_call": {"name": "inspect", "args": {}}},
        ]:
            yield genai_types.GenerateContentResponse(
                candidates=[genai_types.Candidate(content={"parts": [part]})]
            )
        yield genai_types.GenerateContentResponse(
            candidates=[genai_types.Candidate(finish_reason="STOP")],
            usage_metadata={
                "prompt_token_count": 10,
                "candidates_token_count": 5,
                "total_token_count": 15,
            },
        )

    client = MagicMock()
    client.aio.models.generate_content_stream = AsyncMock(return_value=stream_chunks())
    with patch.object(
        genai_types.GenerateContentResponse, "text", new_callable=PropertyMock
    ) as text:
        text.side_effect = AssertionError("SDK convenience .text must not be accessed")
        with patch.object(vertex, "_get_gemini_client", return_value=client):
            stream = await vertex.generate(GenerateRequest(prompt="Synthetic test", stream=True))
            chunks = [chunk async for chunk in stream]

    assert "".join(chunk.text or "" for chunk in chunks) == "Visible answer"
    assert any(chunk.tool_calls for chunk in chunks)
    assert chunks[-1].finish_reason == "STOP"
    assert chunks[-1].usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert text.call_count == 0
    assert not caplog.records
    config = client.aio.models.generate_content_stream.await_args.kwargs["config"] or {}
    assert not {"tools", "tool_config", "automatic_function_calling"} & config.keys()


@pytest.mark.parametrize("choices", [None, [], [None], ["bad"], {}, "bad"])
def test_openrouter_stream_malformed_choices_are_empty(choices):
    response = OpenRouterProvider(api_key="offline-test")._parse_stream_chunk({"choices": choices})
    assert response.text is None
    assert response.tool_calls is None
    assert response.finish_reason is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_delta", [None, {}, [], "bad", {"content": {"private": "sentinel"}}]
)
async def test_openrouter_stream_preserves_terminal_metadata_and_not_reasoning(
    terminal_delta, caplog
):
    call = {
        "index": 0,
        "id": "call-1",
        "type": "function",
        "function": {"name": "inspect", "arguments": "{}"},
    }
    payloads = [
        {
            "choices": [
                {
                    "delta": {
                        "reasoning": "private-reasoning-sentinel",
                        "reasoning_details": [{"text": "private-detail-sentinel"}],
                    }
                }
            ]
        },
        {"choices": [{"delta": {"content": "Visible"}}]},
        {"choices": [{"delta": {"content": " "}}]},
        {"choices": [{"delta": {"content": "answer"}}]},
        {"choices": [{"delta": {"tool_calls": [call]}}]},
        {"choices": [{"delta": terminal_delta, "finish_reason": "length"}]},
    ]
    sse = (
        "\n\n".join("data: " + json.dumps(payload) for payload in payloads) + "\n\ndata: [DONE]\n\n"
    )

    def handle(request):
        body = json.loads(request.content)
        assert body["stream"] is True
        assert "tools" not in body
        assert "tool_choice" not in body
        return httpx.Response(200, text=sse, headers={"Content-Type": "text/event-stream"})

    provider = OpenRouterProvider(api_key="offline-test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with patch.object(provider, "_get_client", new_callable=AsyncMock, return_value=client):
            stream = await provider.generate(GenerateRequest(prompt="Synthetic test", stream=True))
            chunks = [chunk async for chunk in stream]

    assert "".join(chunk.text or "" for chunk in chunks) == "Visible answer"
    assert chunks[-1].finish_reason == "length"
    assert chunks[-1].text is None
    assert chunks[-2].tool_calls == [call]
    assert "private-" not in repr([chunk.model_dump(exclude={"raw"}) for chunk in chunks])
    assert not caplog.records
