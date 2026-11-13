"""Exercise optional SDK wire contracts without provider traffic or credentials."""

from __future__ import annotations

import importlib
import json
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace

import httpx
import pytest

from llm_council.providers.anthropic import (
    AnthropicProvider,
    _beta_messages,
    _is_structured_output_schema_error,
)
from llm_council.providers.base import (
    GenerateRequest,
    PromptCacheConfig,
    StructuredOutputConfig,
)
from llm_council.providers.openai import OpenAIProvider


def test_anthropic_beta_features_fail_clearly_when_sdk_has_no_beta_messages():
    with pytest.raises(RuntimeError, match="beta messages"):
        _beta_messages(object())


def test_anthropic_structured_schema_fallback_recognizes_sdk_v2_error_path():
    assert _is_structured_output_schema_error(
        "Invalid output_config.format.schema: unsupported JSON Schema field"
    )


def import_optional(module_name, distribution):
    try:
        version(distribution)
    except PackageNotFoundError:
        pytest.skip(f"{distribution} is not installed")
    # An installed but broken SDK is a compatibility failure, not a skip.
    return importlib.import_module(module_name)


@pytest.mark.parametrize("provider_name", ["openai", "anthropic"])
async def test_sdk_serializes_basic_request_and_parses_response(provider_name):
    sdk = import_optional(provider_name, provider_name)
    transport_module = httpx
    if provider_name == "anthropic" and int(version("anthropic").split(".")[0]) >= 2:
        transport_module = importlib.import_module("httpx2")
    captured = []

    def respond(request):
        captured.append((request.url.path, json.loads(request.content)))
        if provider_name == "openai":
            payload = {
                "id": "synthetic",
                "object": "chat.completion",
                "created": 0,
                "model": "synthetic-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "OK"},
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        else:
            payload = {
                "id": "synthetic",
                "type": "message",
                "role": "assistant",
                "model": "synthetic-model",
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 2, "output_tokens": 1},
            }
        return transport_module.Response(200, json=payload)

    async with transport_module.AsyncClient(
        transport=transport_module.MockTransport(respond)
    ) as http:
        if provider_name == "openai":
            adapter = OpenAIProvider(api_key="synthetic")
            adapter._client = sdk.AsyncOpenAI(api_key="synthetic", http_client=http, max_retries=0)
            path = "/v1/chat/completions"
        else:
            adapter = AnthropicProvider(api_key="synthetic")
            adapter._client = sdk.AsyncAnthropic(
                api_key="synthetic", http_client=http, max_retries=0
            )
            path = "/v1/messages"
        response = await adapter.generate(
            GenerateRequest(model="synthetic-model", prompt="hello", max_tokens=30)
        )

    assert response.text == "OK"
    assert captured == [
        (
            path,
            {
                "model": "synthetic-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 30,
            },
        )
    ]


async def test_anthropic_serializes_council_temperature_and_reasoning_together():
    """Anthropic reasoning requests omit incompatible temperature in the wire body."""
    sdk = import_optional("anthropic", "anthropic")
    transport_module = httpx
    if int(version("anthropic").split(".")[0]) >= 2:
        transport_module = importlib.import_module("httpx2")
    from llm_council.providers.base import ReasoningConfig
    from llm_council.providers.compiler import compile_request_for_provider

    captured = []

    def respond(request):
        captured.append((request.url.path, json.loads(request.content)))
        return transport_module.Response(
            200,
            json={
                "id": "synthetic",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        )

    async with transport_module.AsyncClient(
        transport=transport_module.MockTransport(respond), trust_env=False
    ) as http:
        adapter = AnthropicProvider(api_key="synthetic")
        adapter._client = sdk.AsyncAnthropic(api_key="synthetic", http_client=http, max_retries=0)
        compiled = compile_request_for_provider(
            "anthropic",
            GenerateRequest(
                model="claude-opus-4-6",
                prompt="hello",
                max_tokens=30,
                temperature=0.2,
                reasoning=ReasoningConfig(enabled=True, effort="high"),
            ),
        )
        response = await adapter.generate(compiled.request)

    assert response.text == "OK"
    assert len(captured) == 1
    path, body = captured[0]
    assert path == "/v1/messages"
    assert body["model"] == "claude-opus-4-6"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["max_tokens"] == 30
    assert body["thinking"] == {"type": "adaptive"}
    assert "temperature" not in body


async def test_anthropic_rejects_known_unsupported_model_temperature():
    """Direct adapter calls must fail clearly for unsupported model sampling."""
    sdk = import_optional("anthropic", "anthropic")
    transport_module = httpx
    if int(version("anthropic").split(".")[0]) >= 2:
        transport_module = importlib.import_module("httpx2")
    async with transport_module.AsyncClient(
        transport=transport_module.MockTransport(
            lambda request: (_ for _ in ()).throw(AssertionError("must not send"))
        ),
        trust_env=False,
    ) as http:
        adapter = AnthropicProvider(api_key="synthetic")
        adapter._client = sdk.AsyncAnthropic(api_key="synthetic", http_client=http, max_retries=0)
        with pytest.raises(ValueError, match="sampling parameters|temperature"):
            await adapter.generate(
                GenerateRequest(
                    model="claude-sonnet-5-5",
                    prompt="hello",
                    temperature=0.3,
                )
            )


async def test_anthropic_doctor_probes_the_configured_model():
    """Doctor must use the provider's configured model in its synthetic API call."""
    sdk = import_optional("anthropic", "anthropic")
    transport_module = httpx
    if int(version("anthropic").split(".")[0]) >= 2:
        transport_module = importlib.import_module("httpx2")
    captured = []

    def respond(request):
        captured.append((request.url.path, json.loads(request.content)))
        return transport_module.Response(
            200,
            json={
                "id": "synthetic",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        )

    async with transport_module.AsyncClient(
        transport=transport_module.MockTransport(respond), trust_env=False
    ) as http:
        adapter = AnthropicProvider(api_key="synthetic", default_model="claude-opus-4-6")
        adapter._client = sdk.AsyncAnthropic(api_key="synthetic", http_client=http, max_retries=0)
        result = await adapter.doctor()

    assert result.ok is True
    assert captured == [
        (
            "/v1/messages",
            {
                "model": "claude-opus-4-6",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
    ]


async def test_anthropic_preserves_cache_control_and_applies_sdk_sampling_policy():
    """SDK 2's 2026 API default rejects sampling parameters for every model."""
    sdk = import_optional("anthropic", "anthropic")
    from llm_council.providers.compiler import compile_request_for_provider

    transport_module = httpx
    if int(version("anthropic").split(".")[0]) >= 2:
        transport_module = importlib.import_module("httpx2")
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return transport_module.Response(
            200,
            json={
                "id": "synthetic",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        )

    async with transport_module.AsyncClient(
        transport=transport_module.MockTransport(respond), trust_env=False
    ) as http:
        adapter = AnthropicProvider(api_key="synthetic")
        adapter._client = sdk.AsyncAnthropic(api_key="synthetic", http_client=http, max_retries=0)
        compiled = compile_request_for_provider(
            "anthropic",
            GenerateRequest(
                model="claude-opus-4-6",
                prompt="hello",
                max_tokens=30,
                temperature=0.3,
                prompt_cache=PromptCacheConfig(ttl="1h"),
            ),
        )
        response = await adapter.generate(compiled.request)

    assert response.text == "OK"
    if int(version("anthropic").split(".")[0]) >= 2:
        assert "temperature" not in captured[0]
        assert any(
            decision.option == "temperature" and decision.action == "dropped"
            for decision in compiled.decisions
        )
    else:
        assert captured[0]["temperature"] == 0.3
        assert not any(decision.option == "temperature" for decision in compiled.decisions)
    assert captured[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


async def test_anthropic_sdk_maps_structured_output_to_supported_beta_parameter():
    """Use the structured-output shape exposed by the installed SDK signature."""
    sdk = import_optional("anthropic", "anthropic")
    transport_module = httpx
    if int(version("anthropic").split(".")[0]) >= 2:
        transport_module = importlib.import_module("httpx2")
    captured = []

    def respond(request):
        captured.append((request.headers.get("anthropic-beta"), json.loads(request.content)))
        return transport_module.Response(
            200,
            json={
                "id": "synthetic",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [{"type": "text", "text": '{"ok":true}'}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        )

    async with transport_module.AsyncClient(
        transport=transport_module.MockTransport(respond), trust_env=False
    ) as http:
        adapter = AnthropicProvider(api_key="synthetic")
        adapter._client = sdk.AsyncAnthropic(api_key="synthetic", http_client=http, max_retries=0)
        response = await adapter.generate(
            GenerateRequest(
                model="claude-opus-4-6",
                prompt="hello",
                max_tokens=30,
                structured_output=StructuredOutputConfig(
                    json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}}
                ),
            )
        )

    assert response.text == '{"ok":true}'
    beta_header, body = captured[0]
    assert "structured-outputs-2025-11-13" in beta_header
    output_config = {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "additionalProperties": False,
                "required": ["ok"],
            },
        }
    }
    if "output_config" in body:
        assert body["output_config"] == output_config
    else:
        assert body["output_format"] == output_config["format"]


async def test_anthropic_beta_stream_adapts_structured_output_keyword():
    """The streaming path must apply the same SDK signature adaptation."""
    sdk = import_optional("anthropic", "anthropic")
    client = sdk.AsyncAnthropic(api_key="synthetic")
    captured = {}

    class SyntheticStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get_final_message(self):
            return SimpleNamespace(
                content=[], usage=None, model="synthetic", stop_reason="end_turn"
            )

        @property
        def text_stream(self):
            async def chunks():
                yield "OK"

            return chunks()

    def synthetic_stream(*, output_config, betas, **kwargs):
        captured.update(output_config=output_config, betas=betas, **kwargs)
        return SyntheticStream()

    client.beta.messages.stream = synthetic_stream
    adapter = AnthropicProvider(api_key="synthetic")
    chunks = [
        chunk
        async for chunk in adapter._generate_stream(
            client,
            {
                "model": "claude-opus-4-6",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 30,
                "output_format": {
                    "type": "json_schema",
                    "schema": {"type": "object"},
                },
            },
            use_beta=True,
        )
    ]

    assert [chunk.text for chunk in chunks if chunk.text] == ["OK"]
    assert chunks[-1].finish_reason == "end_turn"
    assert captured["output_config"] == {
        "format": {"type": "json_schema", "schema": {"type": "object"}}
    }
    assert "output_format" not in captured
    assert captured["betas"] == ["structured-outputs-2025-11-13"]


def test_google_sdk_constructs_gemini_and_vertex_clients_with_retry_options():
    import_optional("google.genai", "google-genai")
    credentials = import_optional("google.auth.credentials", "google-auth").AnonymousCredentials()
    from llm_council.providers.gemini import GeminiProvider
    from llm_council.providers.vertex import VertexAIProvider

    gemini = GeminiProvider(api_key="synthetic")._build_client(timeout_ms=1000)
    vertex = VertexAIProvider(project="synthetic", credentials=credentials)._build_gemini_client(
        timeout_ms=1000
    )
    assert gemini is not None
    assert vertex is not None


def test_google_sdk_preserves_requested_high_thinking_level():
    types = import_optional("google.genai.types", "google-genai")
    config = types.GenerateContentConfig(thinking_config={"thinking_level": "HIGH"})
    assert config.thinking_config.thinking_level == "HIGH"


@pytest.mark.parametrize("provider_name", ["gemini", "vertex"])
async def test_google_sdk_serializes_requested_thinking_without_provider_traffic(
    provider_name, monkeypatch
):
    genai = import_optional("google.genai", "google-genai")
    credentials_module = import_optional("google.auth.credentials", "google-auth")
    from llm_council.providers.base import ReasoningConfig
    from llm_council.providers.gemini import GeminiProvider
    from llm_council.providers.vertex import VertexAIProvider

    class SyntheticCredentials(credentials_module.Credentials):
        def __init__(self):
            super().__init__()
            self.token = "synthetic"

        def refresh(self, request):
            raise AssertionError("synthetic credentials must not refresh")

    captured = []

    def respond(request):
        captured.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "OK"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 2,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 3,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False) as http:
        if provider_name == "gemini":
            monkeypatch.delenv("GOOGLE_GENAI_API_VERSION", raising=False)
            adapter = GeminiProvider(api_key="synthetic")
            client_factory = genai.Client

            def client_with_mock_transport(*, http_options, **kwargs):
                options_with_transport = http_options.model_copy(
                    update={"httpx_async_client": http}
                )
                return client_factory(http_options=options_with_transport, **kwargs)

            monkeypatch.setattr(genai, "Client", client_with_mock_transport)
            adapter._client = adapter._build_client(timeout_ms=1000)
            path = "/v1beta/models/gemini-3-pro:generateContent"
        else:
            options = genai.types.HttpOptions(httpx_async_client=http)
            credentials = SyntheticCredentials()
            adapter = VertexAIProvider(
                project="synthetic", location="us-central1", credentials=credentials
            )
            adapter._gemini_client = genai.Client(
                vertexai=True,
                project="synthetic",
                location="us-central1",
                credentials=credentials,
                http_options=options,
            )
            path = (
                "/v1beta1/projects/synthetic/locations/us-central1/"
                "publishers/google/models/gemini-3-pro:generateContent"
            )
        response = await adapter.generate(
            GenerateRequest(
                model="gemini-3-pro",
                prompt="hello",
                max_tokens=30,
                reasoning=ReasoningConfig(enabled=True, thinking_level="high"),
            )
        )

    assert response.text == "OK"
    assert captured == [
        (
            path,
            {
                "contents": [{"parts": [{"text": "hello"}], "role": "user"}],
                "generationConfig": {
                    "maxOutputTokens": 30,
                    "thinkingConfig": {"thinking_level": "HIGH"},
                },
            },
        )
    ]


async def test_vertex_anthropic_sdk_serializes_basic_request():
    sdk = import_optional("anthropic", "anthropic")
    transport_module = httpx
    if int(version("anthropic").split(".")[0]) >= 2:
        transport_module = importlib.import_module("httpx2")
    from llm_council.providers.vertex import VertexAIProvider

    captured = []

    def respond(request):
        captured.append((request.url.path, json.loads(request.content)))
        return transport_module.Response(
            200,
            json={
                "id": "synthetic",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet@20241022",
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        )

    async with transport_module.AsyncClient(
        transport=transport_module.MockTransport(respond), trust_env=False
    ) as http:
        adapter = VertexAIProvider(project="synthetic", location="us-central1")
        adapter._claude_client = sdk.AsyncAnthropicVertex(
            project_id="synthetic",
            region="us-central1",
            access_token="synthetic",
            http_client=http,
            max_retries=0,
        )
        response = await adapter.generate(
            GenerateRequest(model="claude-3-5-sonnet@20241022", prompt="hello", max_tokens=30)
        )

    assert response.text == "OK"
    assert len(captured) == 1
    path, body = captured[0]
    assert path == (
        "/v1/projects/synthetic/locations/us-central1/publishers/anthropic/"
        "models/claude-3-5-sonnet@20241022:rawPredict"
    )
    assert isinstance(body.pop("anthropic_version"), str)
    beta = body.pop("anthropic_beta", [])
    assert isinstance(beta, list) and all(isinstance(value, str) for value in beta)
    assert body == {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 30}


async def test_vertex_anthropic_sdk_serializes_structured_manual_thinking():
    """Vertex Claude keeps manual thinking and adapts structured output for the SDK."""
    sdk = import_optional("anthropic", "anthropic")
    transport_module = httpx
    if int(version("anthropic").split(".")[0]) >= 2:
        transport_module = importlib.import_module("httpx2")
    from llm_council.providers.base import ReasoningConfig
    from llm_council.providers.vertex import VertexAIProvider

    captured = []

    def respond(request):
        captured.append((request.url.path, json.loads(request.content)))
        return transport_module.Response(
            200,
            json={
                "id": "synthetic",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        )

    async with transport_module.AsyncClient(
        transport=transport_module.MockTransport(respond), trust_env=False
    ) as http:
        adapter = VertexAIProvider(project="synthetic", location="us-central1")
        adapter._claude_client = sdk.AsyncAnthropicVertex(
            project_id="synthetic",
            region="us-central1",
            access_token="synthetic",
            http_client=http,
            max_retries=0,
        )
        response = await adapter.generate(
            GenerateRequest(
                model="claude-opus-4-6",
                prompt="hello",
                max_tokens=4096,
                temperature=0.2,
                reasoning=ReasoningConfig(enabled=True, budget_tokens=2048),
                structured_output=StructuredOutputConfig(
                    json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}}
                ),
            )
        )

    assert response.text == "OK"
    path, body = captured[0]
    assert path.endswith(":rawPredict")
    assert body["max_tokens"] == 4096
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "temperature" not in body
    if "output_config" in body:
        assert body["output_config"]["format"]["schema"]["required"] == ["ok"]
    else:
        assert body["output_format"]["schema"]["required"] == ["ok"]


async def test_vertex_anthropic_rejects_publisher_qualified_model_temperature():
    """Vertex policy checks normalize publisher paths before checking model support."""
    sdk = import_optional("anthropic", "anthropic")
    transport_module = httpx
    if int(version("anthropic").split(".")[0]) >= 2:
        transport_module = importlib.import_module("httpx2")
    from llm_council.providers.vertex import VertexAIProvider

    async with transport_module.AsyncClient(
        transport=transport_module.MockTransport(
            lambda request: (_ for _ in ()).throw(AssertionError("must not send"))
        ),
        trust_env=False,
    ) as http:
        adapter = VertexAIProvider(project="synthetic", location="us-central1")
        adapter._claude_client = sdk.AsyncAnthropicVertex(
            project_id="synthetic",
            region="us-central1",
            access_token="synthetic",
            http_client=http,
            max_retries=0,
        )
        with pytest.raises(ValueError, match="sampling parameters|non-default temperature"):
            await adapter.generate(
                GenerateRequest(
                    model="publishers/anthropic/models/claude-opus-4-7@20260301",
                    prompt="hello",
                    temperature=0.3,
                )
            )


async def test_openrouter_serializes_basic_request():
    from llm_council.providers.openrouter import OpenRouterProvider

    captured = []

    def respond(request):
        captured.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "id": "synthetic",
                "model": "synthetic-model",
                "choices": [{"finish_reason": "stop", "message": {"content": "OK"}}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False) as http:
        adapter = OpenRouterProvider(api_key="synthetic", http_client=http)
        response = await adapter.generate(
            GenerateRequest(model="synthetic-model", prompt="hello", max_tokens=30)
        )

    assert response.text == "OK"
    assert captured == [
        (
            "/api/v1/chat/completions",
            {
                "model": "synthetic-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 30,
            },
        )
    ]


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6.1-sol"])
async def test_openai_gpt6_serializes_reasoning_request(model):
    sdk = import_optional("openai", "openai")
    from llm_council.providers.base import ReasoningConfig
    from llm_council.providers.compiler import compile_request_for_provider

    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "synthetic",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "{}"},
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False) as http:
        adapter = OpenAIProvider(api_key="synthetic")
        adapter._client = sdk.AsyncOpenAI(api_key="synthetic", http_client=http, max_retries=0)
        compiled = compile_request_for_provider(
            "openai",
            GenerateRequest(
                model=model,
                prompt="hello",
                max_tokens=4096,
                temperature=0.2,
                top_p=0.9,
                reasoning=ReasoningConfig(enabled=True, effort="high"),
                structured_output=StructuredOutputConfig(
                    name="result", json_schema={"type": "object", "properties": {}}
                ),
            ),
        )
        await adapter.generate(compiled.request)
    body = captured[0]
    assert body["max_completion_tokens"] == 4096
    assert body["reasoning_effort"] == "high"
    assert body["response_format"]["type"] == "json_schema"
    assert "temperature" not in body and "top_p" not in body and "max_tokens" not in body


@pytest.mark.parametrize("model", ["gemini-4-argon", "gemini-4-carbon", "gemini-4-hydrogen"])
async def test_vertex_gemini4_serializes_schema_and_reasoning(model):
    genai = import_optional("google.genai", "google-genai")
    credentials_module = import_optional("google.auth.credentials", "google-auth")
    from llm_council.providers.base import ReasoningConfig
    from llm_council.providers.compiler import compile_request_for_provider
    from llm_council.providers.vertex import VertexAIProvider

    class SyntheticCredentials(credentials_module.Credentials):
        def __init__(self):
            super().__init__()
            self.token = "synthetic"

        def refresh(self, request):
            raise AssertionError("synthetic credentials must not refresh")

    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "{}"}]},
                        "finishReason": "STOP",
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False) as http:
        credentials = SyntheticCredentials()
        adapter = VertexAIProvider(
            project="synthetic", location="us-central1", credentials=credentials
        )
        adapter._gemini_client = genai.Client(
            vertexai=True,
            project="synthetic",
            location="us-central1",
            credentials=credentials,
            http_options=genai.types.HttpOptions(httpx_async_client=http),
        )
        compiled = compile_request_for_provider(
            "vertex-ai",
            GenerateRequest(
                model=model,
                prompt="hello",
                max_tokens=16000,
                reasoning=ReasoningConfig(enabled=True, thinking_level="high"),
                structured_output=StructuredOutputConfig(
                    name="result", json_schema={"type": "object", "properties": {}}
                ),
            ),
        )
        await adapter.generate(compiled.request)
    config = captured[0]["generationConfig"]
    assert config["maxOutputTokens"] == 16000
    assert config["thinkingConfig"]["thinking_level"] == "HIGH"
    assert config["responseMimeType"] == "application/json"
    assert "responseSchema" in config


def test_google_sdk_finish_reason_normalizes_to_wire_value():
    types = import_optional("google.genai.types", "google-genai")
    from types import SimpleNamespace

    from llm_council.providers.gemini import GeminiProvider

    response = SimpleNamespace(
        text="partial",
        candidates=[SimpleNamespace(finish_reason=types.FinishReason.MAX_TOKENS)],
        usage_metadata=None,
    )
    assert (
        GeminiProvider(api_key="synthetic")._parse_response(response).finish_reason == "MAX_TOKENS"
    )
