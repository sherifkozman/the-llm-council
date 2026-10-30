"""Exercise optional SDK wire contracts without provider traffic or credentials."""

from __future__ import annotations

import importlib
import json
from importlib.metadata import PackageNotFoundError, version

import httpx
import pytest

from llm_council.providers.anthropic import AnthropicProvider
from llm_council.providers.base import GenerateRequest
from llm_council.providers.openai import OpenAIProvider


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
async def test_google_sdk_serializes_requested_thinking_without_provider_traffic(provider_name):
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
        options = genai.types.HttpOptions(httpx_async_client=http)
        if provider_name == "gemini":
            adapter = GeminiProvider(api_key="synthetic")
            adapter._client = genai.Client(api_key="synthetic", http_options=options)
            path = "/v1beta/models/gemini-3-pro:generateContent"
        else:
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
