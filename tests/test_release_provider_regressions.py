from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_council.engine.orchestrator import Orchestrator, OrchestratorConfig
from llm_council.providers.base import GenerateRequest, ReasoningConfig, StructuredOutputConfig
from llm_council.providers.compiler import compile_request_for_provider
from llm_council.providers.openai import OpenAIProvider
from llm_council.providers.vertex import VertexAIProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6.1-sol"])
async def test_openai_gpt6_high_reasoning_wire_parameters(model):
    provider = OpenAIProvider(api_key="test")
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(message=MagicMock(content="{}", tool_calls=None), finish_reason="stop")
            ],
            usage=MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model=model,
        )
    )
    provider._client = client
    request = GenerateRequest(
        prompt="test",
        model=model,
        max_tokens=5000,
        temperature=0.2,
        top_p=0.8,
        reasoning=ReasoningConfig(enabled=True, effort="high"),
        structured_output=StructuredOutputConfig(
            name="test", json_schema={"type": "object", "properties": {}}
        ),
    )
    compiled = compile_request_for_provider("openai", request)
    await provider.generate(compiled.request)
    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["max_completion_tokens"] == 5000
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["response_format"]["type"] == "json_schema"
    assert "temperature" not in kwargs and "top_p" not in kwargs and "max_tokens" not in kwargs


@pytest.mark.asyncio
async def test_vertex_gemini4_numeric_budget_is_not_sent():
    provider = VertexAIProvider(project="test", default_model="gemini-4-argon")
    client = MagicMock()
    provider._gemini_client = client
    client.aio.models.generate_content = AsyncMock(
        return_value=SimpleNamespace(
            text="ok",
            candidates=[],
            usage_metadata=SimpleNamespace(
                prompt_token_count=1, candidates_token_count=1, total_token_count=2
            ),
        )
    )
    with pytest.raises(ValueError, match="thinking_level"):
        await provider.generate(
            GenerateRequest(
                prompt="test",
                model="gemini-4-argon",
                reasoning=ReasoningConfig(enabled=True, budget_tokens=16384),
            )
        )
    client.aio.models.generate_content.assert_not_awaited()


def test_unabridged_prompt_used_when_within_budget():
    orch = Orchestrator([], OrchestratorConfig())
    orch._execution_plan = {}
    prompt, metadata = orch._select_prompt_profile(
        provider_name="vertex-ai",
        phase="synthesis",
        system_prompt="system",
        prompt_builder=lambda p: "full evidence" if p.get("draft_limit") is None else "trimmed",
    )
    assert prompt == "full evidence"
    assert metadata["compacted"] is False


def test_gemini4_structured_output_is_preserved():
    request = GenerateRequest(
        prompt="test",
        model="gemini-4-argon",
        structured_output=StructuredOutputConfig(
            name="test", json_schema={"type": "object", "properties": {}}
        ),
    )
    compiled = compile_request_for_provider("vertex-ai", request)
    assert compiled.request.structured_output is not None
    assert VertexAIProvider(project="test")._model_supports_structured_output("gemini-4-argon")


def test_vertex_gemini4_default_output_budget_preserves_explicit_override():
    orch = Orchestrator(["vertex-ai"], OrchestratorConfig())
    orch._resolved_model_overrides = {"vertex-ai": "gemini-4-argon"}
    orch._reasoning = ReasoningConfig(enabled=True, thinking_level="high")
    assert orch._phase_max_tokens(4000, phase="draft", provider_name="vertex-ai") >= 16000
    assert orch._phase_max_tokens(8000, phase="synthesis", provider_name="vertex-ai") >= 16000
    orch._config.max_tokens = 5000
    assert orch._phase_max_tokens(4000, phase="draft", provider_name="vertex-ai") == 5000


@pytest.mark.parametrize("provider", ["vertex-ai", "gemini"])
def test_hydrogen_reasoning_budget_respects_explicit_limits(provider):
    orch = Orchestrator([provider], OrchestratorConfig())
    orch._resolved_model_overrides = {provider: "gemini-4-hydrogen"}
    orch._reasoning = ReasoningConfig(enabled=True, thinking_level="high")
    assert orch._phase_max_tokens(4000, phase="draft", provider_name=provider) == 16000
    assert orch._phase_max_tokens(2000, phase="critique", provider_name=provider) == 16000
    assert orch._phase_max_tokens(8000, phase="synthesis", provider_name=provider) == 16000
    orch._config.max_tokens = 5000
    assert orch._phase_max_tokens(4000, phase="draft", provider_name=provider) == 5000

    configured = Orchestrator([provider], OrchestratorConfig(max_draft_tokens=1234))
    configured._resolved_model_overrides = {provider: "gemini-4-hydrogen"}
    configured._reasoning = ReasoningConfig(enabled=True, thinking_level="high")
    assert configured._phase_max_tokens(1234, phase="draft", provider_name=provider) == 1234

    bounded = Orchestrator([provider], OrchestratorConfig(runtime_profile="bounded"))
    bounded._resolved_model_overrides = {provider: "gemini-4-hydrogen"}
    bounded._reasoning = ReasoningConfig(enabled=True, thinking_level="high")
    assert bounded._phase_max_tokens(4000, phase="draft", provider_name=provider) == 2000


def test_openai_gpt54_high_reasoning_is_preserved_and_sampling_removed():
    request = GenerateRequest(
        prompt="test",
        model="gpt-5.4",
        temperature=0.2,
        top_p=0.8,
        reasoning=ReasoningConfig(enabled=True, effort="high"),
    )
    compiled = compile_request_for_provider("openai", request).request
    assert compiled.reasoning is not None and compiled.reasoning.effort == "high"
    assert compiled.temperature is None and compiled.top_p is None


def test_vertex_reports_thinking_usage_for_budget_diagnosis():
    provider = VertexAIProvider(project="test")
    response = SimpleNamespace(
        candidates=[],
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=20,
            thoughts_token_count=500,
            total_token_count=620,
        ),
    )
    parsed = provider._parse_response(response)
    assert parsed.usage["reasoning_tokens"] == 500


def test_vertex_thinking_tokens_are_counted_in_output_usage():
    orch = Orchestrator(["vertex-ai"], OrchestratorConfig())
    orch._record_usage(
        "vertex-ai", {"prompt_tokens": 100, "completion_tokens": 20, "reasoning_tokens": 500}
    )
    assert orch._output_tokens["vertex-ai"] == 520


def test_execution_plan_reports_provider_specific_output_budgets():
    orch = Orchestrator(
        ["vertex-ai"],
        OrchestratorConfig(
            model_overrides={"vertex-ai": "gemini-4-argon"}, enable_health_check=False
        ),
    )
    orch._prepare_run("critic")
    budgets = orch._execution_plan["phase_token_budgets_by_provider"]["vertex-ai"]
    assert budgets["draft"] == 16000
    assert budgets["synthesis"] == 16000


@pytest.mark.asyncio
async def test_vertex_gemini4_rejects_unsupported_minimal_thinking():
    provider = VertexAIProvider(project="test", default_model="gemini-4-argon")
    provider._gemini_client = MagicMock()
    provider._gemini_client.aio.models.generate_content = AsyncMock(
        return_value=SimpleNamespace(candidates=[], usage_metadata=None)
    )
    with pytest.raises(ValueError, match="minimal"):
        await provider.generate(
            GenerateRequest(
                prompt="test",
                model="gemini-4-argon",
                reasoning=ReasoningConfig(enabled=True, thinking_level="minimal"),
            )
        )


def test_vertex_carbon_uses_schema_and_reasoning_budget():
    request = GenerateRequest(
        prompt="test",
        model="gemini-4-carbon",
        structured_output=StructuredOutputConfig(
            name="test", json_schema={"type": "object", "properties": {}}
        ),
    )
    assert compile_request_for_provider("vertex-ai", request).request.structured_output is not None
    orch = Orchestrator(["vertex-ai"], OrchestratorConfig())
    orch._resolved_model_overrides = {"vertex-ai": "gemini-4-carbon"}
    orch._reasoning = ReasoningConfig(enabled=True, thinking_level="high")
    assert orch._phase_max_tokens(8000, phase="synthesis", provider_name="vertex-ai") == 16000


def test_manifest_retains_only_bounded_attempt_usage():
    import json

    orch = Orchestrator([], OrchestratorConfig())
    orch._execution_plan = {
        "provider_attempts": [
            {
                "provider": "vertex-ai",
                "phase": "draft",
                "status": "failed",
                "finish_reason": "max_tokens",
                "usage": {"completion_tokens": 0, "reasoning_tokens": 4000, "extra_private": 42},
            }
        ]
    }
    orch._store_phase_artifact = MagicMock()
    orch._store_execution_manifest("settled", "degraded")
    manifest = json.loads(orch._store_phase_artifact.call_args.args[0])
    attempt = manifest["provider_attempts"][0]
    assert attempt["finish_reason"] == "max_tokens"
    assert attempt["usage"] == {"completion_tokens": 0, "reasoning_tokens": 4000}


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["gemini", "vertex"])
@pytest.mark.parametrize("model", ["gemini-4-argon", "gemini-4-carbon"])
@pytest.mark.parametrize(
    "reasoning",
    [
        ReasoningConfig(enabled=True, thinking_level="minimal"),
        ReasoningConfig(enabled=True, budget_tokens=16384),
    ],
)
async def test_gemini4_invalid_thinking_rejected_before_cache(provider_name, model, reasoning):
    from llm_council.providers.gemini import GeminiProvider

    provider = (
        GeminiProvider(api_key="test", default_model=model)
        if provider_name == "gemini"
        else VertexAIProvider(project="test", default_model=model)
    )
    if provider_name == "gemini":
        provider._client = MagicMock()
    else:
        provider._gemini_client = MagicMock()
    provider._prepare_cached_content = AsyncMock()
    with pytest.raises(ValueError, match="thinking"):
        await provider.generate(GenerateRequest(prompt="test", model=model, reasoning=reasoning))
    provider._prepare_cached_content.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["draft", "critique", "synthesis"])
async def test_nonempty_truncated_response_marks_execution_degraded(phase):
    from llm_council.engine.orchestrator import CouncilResult
    from llm_council.providers.base import GenerateResponse

    orch = Orchestrator([], OrchestratorConfig())
    orch._execution_plan = {"required_phases": [], "providers": []}
    adapter = MagicMock()
    adapter.resolve_model.return_value = "gemini-4-argon"
    adapter.generate = AsyncMock(
        return_value=GenerateResponse(text="partial", finish_reason="MAX_TOKENS")
    )
    await orch._call_provider("vertex-ai", adapter, GenerateRequest(prompt="test"), phase=phase)
    result = CouncilResult(success=True, execution_plan=orch._execution_plan)
    assert result.execution_status == "degraded"
    assert result.degradation_report["output_warnings"][0]["phase"] == phase


def test_direct_gemini_sdk_finish_reason_enum_normalized():
    from enum import Enum

    from llm_council.providers.gemini import GeminiProvider

    class FinishReason(Enum):
        MAX_TOKENS = "MAX_TOKENS"

    provider = GeminiProvider(api_key="test")
    response = SimpleNamespace(
        text="partial",
        candidates=[SimpleNamespace(finish_reason=FinishReason.MAX_TOKENS)],
        usage_metadata=None,
    )
    assert provider._parse_response(response).finish_reason == "MAX_TOKENS"


@pytest.mark.parametrize(
    "model", ["models/gemini-4-carbon", "publishers/google/models/gemini-4-carbon"]
)
def test_qualified_gemini4_names_support_schema_and_budget(model):
    from llm_council.providers.gemini import GeminiProvider

    request = GenerateRequest(
        prompt="test",
        model=model,
        structured_output=StructuredOutputConfig(
            name="test", json_schema={"type": "object", "properties": {}}
        ),
    )
    assert compile_request_for_provider("vertex-ai", request).request.structured_output is not None
    assert VertexAIProvider(project="test")._model_supports_structured_output(model)
    assert GeminiProvider(api_key="test")._model_supports_structured_output(model)
    orch = Orchestrator(["vertex-ai"], OrchestratorConfig())
    orch._resolved_model_overrides = {"vertex-ai": model}
    orch._reasoning = ReasoningConfig(enabled=True, thinking_level="high")
    assert orch._phase_max_tokens(8000, phase="synthesis", provider_name="vertex-ai") == 16000


def test_direct_gemini_thinking_usage_is_reported_and_counted():
    from llm_council.providers.gemini import GeminiProvider

    provider = GeminiProvider(api_key="test")
    response = SimpleNamespace(
        text="ok",
        candidates=[],
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=20,
            thoughts_token_count=500,
            total_token_count=620,
        ),
    )
    usage = provider._parse_response(response).usage
    assert usage["reasoning_tokens"] == 500
    orch = Orchestrator(["gemini"], OrchestratorConfig())
    orch._record_usage("gemini", usage)
    assert orch._output_tokens["gemini"] == 520
