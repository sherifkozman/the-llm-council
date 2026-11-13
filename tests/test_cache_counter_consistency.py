"""Provider cache counters must be usable evidence, not impossible totals."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from llm_council.engine.orchestrator import Orchestrator, OrchestratorConfig
from llm_council.providers.gemini import GeminiProvider
from llm_council.providers.openai import OpenAIProvider
from llm_council.providers.openrouter import OpenRouterProvider
from llm_council.providers.vertex import VertexAIProvider


def parsed_response(provider, cached):
    if provider == "openrouter":
        return OpenRouterProvider.__new__(OpenRouterProvider)._parse_response(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                    "prompt_tokens_details": {"cached_tokens": cached},
                },
            }
        )
    if provider == "openai":
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        )
        choice = SimpleNamespace(
            message=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop"
        )
        return OpenAIProvider.__new__(OpenAIProvider)._parse_response(
            SimpleNamespace(choices=[choice], usage=usage, model="synthetic")
        )
    metadata = SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=4,
        total_token_count=14,
        cached_content_token_count=cached,
    )
    part = SimpleNamespace(text="ok", thought=False, function_call=None)
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]), finish_reason=None)
    response = SimpleNamespace(text="ok", candidates=[candidate], usage_metadata=metadata)
    cls = GeminiProvider if provider == "gemini" else VertexAIProvider
    return cls.__new__(cls)._parse_response(response)


@pytest.mark.parametrize("provider", ["openai", "openrouter", "gemini", "vertex"])
@pytest.mark.parametrize("cached", [999, -1, "invalid", True])
def test_invalid_cache_read_is_not_observed_or_verified(provider, cached):
    result = parsed_response(provider, cached)
    assert result.usage_verified is False
    assert "cache_read_tokens" not in (result.usage or {})
    orch = Orchestrator([], OrchestratorConfig(enable_artifacts=False))
    orch._execution_plan = {}
    orch._record_prompt_cache_observation(
        phase="draft", provider_name=provider, requested=True, forwarded=True, usage=result.usage
    )
    obs = orch._execution_plan["prompt_cache_observations"]["draft"][0]
    assert obs["cache_read_observed"] is False


@pytest.mark.parametrize("provider", ["openai", "openrouter", "gemini", "vertex"])
def test_reported_zero_cache_read_remains_distinct_from_missing(provider):
    result = parsed_response(provider, 0)
    assert result.usage_verified is True
    assert result.usage["cache_read_tokens"] == 0


def test_openrouter_nested_cache_write_is_reported():
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "prompt_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 3},
    }
    result = OpenRouterProvider.__new__(OpenRouterProvider)._parse_response(
        {"choices": [], "usage": usage}
    )
    assert result.usage_verified is True
    assert result.usage["cache_read_tokens"] == 2
    assert result.usage["cache_creation_tokens"] == 3


def test_openrouter_malformed_cache_details_are_not_verified():
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "prompt_tokens_details": "invalid",
    }
    result = OpenRouterProvider.__new__(OpenRouterProvider)._parse_response(
        {"choices": [], "usage": usage}
    )
    assert result.usage_verified is False


@pytest.mark.parametrize("nested,legacy", [(3, 4), (-1, None), ("invalid", None), (True, None)])
def test_openrouter_invalid_or_conflicting_cache_write_is_not_verified(nested, legacy):
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "prompt_tokens_details": {"cache_write_tokens": nested},
    }
    if legacy is not None:
        usage["cache_write_tokens"] = legacy
    result = OpenRouterProvider.__new__(OpenRouterProvider)._parse_response(
        {"choices": [], "usage": usage}
    )
    assert result.usage_verified is False
    assert "cache_creation_tokens" not in (result.usage or {})


@pytest.mark.asyncio
async def test_openai_stream_invalid_cache_count_is_not_verified():
    async def chunks():
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
            prompt_tokens_details=SimpleNamespace(cached_tokens=999),
        )
        yield SimpleNamespace(choices=[], usage=usage)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=chunks())))
    )
    result = [
        chunk async for chunk in OpenAIProvider.__new__(OpenAIProvider)._generate_stream(client, {})
    ]
    assert result[0].usage_invalid is True
    assert result[0].usage_verified is False
    assert "cache_read_tokens" not in (result[0].usage or {})
