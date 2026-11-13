"""Cost estimates must expose missing inputs rather than imply free usage."""

from llm_council.engine.orchestrator import Orchestrator, OrchestratorConfig


def make_orchestrator(**kwargs):
    return Orchestrator(providers=[], config=OrchestratorConfig(enable_artifacts=False, **kwargs))


def test_missing_prices_do_not_report_zero_cost():
    orch = make_orchestrator()
    orch._record_usage("example", {"input_tokens": 1000, "output_tokens": 100})
    estimate = orch._build_cost_estimate()
    assert estimate.estimated_cost_usd is None
    assert estimate.incomplete_reasons
    assert estimate.total_input_tokens == 1000


def test_known_zero_rates_and_complete_usage_are_explicitly_zero():
    orch = make_orchestrator(cost_per_1k_input={"example": 0}, cost_per_1k_output={"example": 0})
    orch._record_usage("example", {"input_tokens": 1000, "output_tokens": 100}, usage_verified=True)
    estimate = orch._build_cost_estimate()
    assert estimate.estimated_cost_usd is None
    assert estimate.reported_usage_estimate_usd == 0


def test_missing_usage_or_failed_started_attempt_makes_total_unknown():
    orch = make_orchestrator(cost_per_1k_input={"example": 1}, cost_per_1k_output={"example": 2})
    orch._record_usage("example", None)
    assert orch._build_cost_estimate().estimated_cost_usd is None
    orch = make_orchestrator(cost_per_1k_input={"example": 1}, cost_per_1k_output={"example": 2})
    orch._cost_started_calls["example"] = 1
    assert orch._build_cost_estimate().estimated_cost_usd is None


def test_partial_usage_does_not_imply_zero_for_missing_direction():
    orch = make_orchestrator(cost_per_1k_input={"example": 1}, cost_per_1k_output={"example": 2})
    orch._record_usage("example", {"input_tokens": 1000})
    assert orch._build_cost_estimate().estimated_cost_usd is None


def test_invalid_primary_alias_cannot_create_negative_usage():
    orch = make_orchestrator(cost_per_1k_input={"example": 1}, cost_per_1k_output={"example": 2})
    orch._record_usage("example", {"prompt_tokens": -100, "input_tokens": 100, "output_tokens": 50})
    estimate = orch._build_cost_estimate()
    assert estimate.estimated_cost_usd is None
    assert estimate.total_input_tokens == 100
    assert estimate.incomplete_reasons


def test_invalid_rates_do_not_create_nonfinite_estimate():
    orch = make_orchestrator(
        cost_per_1k_input={"example": float("nan")}, cost_per_1k_output={"example": 2}
    )
    orch._record_usage("example", {"input_tokens": 1000, "output_tokens": 100})
    assert orch._build_cost_estimate().estimated_cost_usd is None


def test_started_attempts_and_per_provider_token_counts_are_visible():
    orch = make_orchestrator()
    orch._cost_started_calls["example"] = 2
    orch._record_usage("example", {"input_tokens": 1000, "output_tokens": 100})
    estimate = orch._build_cost_estimate()
    assert estimate.started_provider_calls == {"example": 2}
    assert estimate.provider_input_tokens == {"example": 1000}
    assert estimate.provider_output_tokens == {"example": 100}


def test_council_forwards_explicit_rate_maps():
    from llm_council.council import Council
    from llm_council.protocol.types import CouncilConfig

    config = CouncilConfig(
        providers=["example"], cost_per_1k_input={"example": 1}, cost_per_1k_output={"example": 2}
    )
    council = Council(config=config)
    built = council._build_orchestrator_config(config)
    assert built.cost_per_1k_input == {"example": 1}
    assert built.cost_per_1k_output == {"example": 2}


async def test_router_follow_up_combines_both_stages():
    from unittest.mock import AsyncMock, MagicMock

    from llm_council.council import Council
    from llm_council.engine.orchestrator import CostEstimate, CouncilResult

    council = Council(providers=["example"])
    router = CouncilResult(
        success=True,
        output={"subagent_to_run": "planner"},
        cost_estimate=CostEstimate(
            total_input_tokens=100,
            total_output_tokens=10,
            tokens=110,
            provider_input_tokens={"example": 100},
            provider_output_tokens={"example": 10},
            provider_calls={"example": 1},
            started_provider_calls={"example": 1},
        ),
    )
    child = CouncilResult(
        success=True,
        output={},
        cost_estimate=CostEstimate(
            total_input_tokens=200,
            total_output_tokens=20,
            tokens=220,
            provider_input_tokens={"example": 200},
            provider_output_tokens={"example": 20},
            provider_calls={"example": 1},
            started_provider_calls={"example": 1},
        ),
    )
    council._build_orchestrator = MagicMock(
        return_value=MagicMock(run=AsyncMock(return_value=child))
    )
    result = await council._run_router_follow_up("test", router)
    assert result.cost_estimate.total_input_tokens == 300
    assert result.cost_estimate.total_output_tokens == 30
    assert result.cost_estimate.provider_calls == {"example": 2}
    assert result.cost_estimate.estimated_cost_usd is None


def test_missing_gemini_token_field_does_not_create_reported_usage_estimate():
    from types import SimpleNamespace

    from llm_council.providers.gemini import GeminiProvider

    adapter = GeminiProvider.__new__(GeminiProvider)
    response = adapter._parse_response(
        SimpleNamespace(
            text="ok",
            candidates=[],
            usage_metadata=SimpleNamespace(prompt_token_count=None, candidates_token_count=10),
        )
    )
    orch = make_orchestrator(cost_per_1k_input={"gemini": 1}, cost_per_1k_output={"gemini": 1})
    orch._record_usage("gemini", response.usage, usage_verified=response.usage_verified)
    estimate = orch._build_cost_estimate()
    assert estimate.reported_usage_estimate_usd is None
    assert estimate.provider_usage_verified == {"gemini": False}
    assert any("usage" in reason for reason in estimate.incomplete_reasons)


def test_gemini_reported_zero_is_verified():
    from types import SimpleNamespace

    from llm_council.providers.gemini import GeminiProvider

    adapter = GeminiProvider.__new__(GeminiProvider)
    response = adapter._parse_response(
        SimpleNamespace(
            text="ok",
            candidates=[],
            usage_metadata=SimpleNamespace(
                prompt_token_count=0, candidates_token_count=10, total_token_count=10
            ),
        )
    )
    assert response.usage_verified is True


async def test_stream_usage_keeps_final_chunk_provenance():
    from unittest.mock import AsyncMock, MagicMock

    from llm_council.providers.base import GenerateRequest, GenerateResponse

    async def stream():
        yield GenerateResponse(text="ok")
        yield GenerateResponse(
            usage={"prompt_tokens": 4, "completion_tokens": 3}, usage_verified=True
        )

    orch = make_orchestrator(cost_per_1k_input={"example": 1}, cost_per_1k_output={"example": 1})
    adapter = MagicMock()
    adapter.resolve_model.return_value = "example-model"
    adapter.generate = AsyncMock(return_value=stream())
    await orch._call_provider("example", adapter, GenerateRequest(prompt="test"))
    assert orch._build_cost_estimate().reported_usage_estimate_usd == 0.007


def test_missing_router_stage_cannot_verify_aggregate_provider_usage():
    from llm_council.council import _combine_cost_estimates
    from llm_council.engine.orchestrator import CostEstimate

    observed = CostEstimate(provider_calls={"gemini": 1}, provider_usage_verified={"gemini": True})
    result = _combine_cost_estimates(observed, None)
    assert result.provider_usage_verified == {"gemini": False}
    assert result.reported_usage_estimate_usd is None


async def test_gemini_stream_emits_usage_metadata_without_text():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from llm_council.providers.gemini import GeminiProvider

    async def source():
        yield SimpleNamespace(text="hello", candidates=[], usage_metadata=None)
        yield SimpleNamespace(
            text=None,
            candidates=[],
            usage_metadata=SimpleNamespace(
                prompt_token_count=4, candidates_token_count=3, total_token_count=7
            ),
        )

    adapter = GeminiProvider.__new__(GeminiProvider)
    client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content_stream=AsyncMock(return_value=source()))
        )
    )
    chunks = [item async for item in adapter._generate_stream(client, "model", "prompt", {})]
    assert chunks[0].text == "hello"
    assert chunks[-1].usage["prompt_tokens"] == 4
    assert chunks[-1].usage_verified is True


def test_gemini_inconsistent_total_is_not_verified():
    from types import SimpleNamespace

    from llm_council.providers.gemini import GeminiProvider

    adapter = GeminiProvider.__new__(GeminiProvider)
    response = adapter._parse_response(
        SimpleNamespace(
            text="ok",
            candidates=[],
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=10,
                total_token_count=120,
                thoughts_token_count=None,
            ),
        )
    )
    assert response.usage_verified is False


def test_vertex_gemini_verifies_only_reconciled_source_counts():
    from types import SimpleNamespace

    from llm_council.providers.vertex import VertexAIProvider

    provider = VertexAIProvider.__new__(VertexAIProvider)
    good = SimpleNamespace(
        candidates=[],
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=4,
            thoughts_token_count=3,
            total_token_count=17,
        ),
    )
    bad = SimpleNamespace(
        candidates=[],
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=None,
            thoughts_token_count=None,
            total_token_count=17,
        ),
    )
    assert provider._parse_response(good).usage_verified is True
    assert provider._parse_response(bad).usage_verified is False


def test_openrouter_verifies_reconciled_token_counts_and_rejects_missing():
    from llm_council.providers.openrouter import OpenRouterProvider

    provider = OpenRouterProvider.__new__(OpenRouterProvider)
    good = {
        "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }
    bad = {"choices": [], "usage": {"prompt_tokens": 10, "total_tokens": 14}}
    assert provider._parse_response(good).usage_verified is True
    assert provider._parse_response(bad).usage_verified is False


def test_openai_verifies_reconciled_token_counts_and_rejects_missing():
    from types import SimpleNamespace

    from llm_council.providers.openai import OpenAIProvider

    provider = OpenAIProvider.__new__(OpenAIProvider)
    choice = SimpleNamespace(
        message=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop"
    )
    good = SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14),
        model="example",
    )
    bad = SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=None, total_tokens=14),
        model="example",
    )
    assert provider._parse_response(good).usage_verified is True
    assert provider._parse_response(bad).usage_verified is False


def test_anthropic_adapters_verify_cache_and_output_counts():
    from types import SimpleNamespace

    from llm_council.providers.anthropic import AnthropicProvider
    from llm_council.providers.vertex import VertexAIProvider

    good_usage = SimpleNamespace(
        input_tokens=10, output_tokens=5, cache_read_input_tokens=20, cache_creation_input_tokens=7
    )
    missing_usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    for cls, method in (
        (AnthropicProvider, "_parse_response"),
        (VertexAIProvider, "_parse_claude_response"),
    ):
        provider = cls.__new__(cls)

        def make(usage):
            return SimpleNamespace(content=[], usage=usage, model="test", stop_reason="end_turn")

        assert getattr(provider, method)(make(good_usage)).usage_verified is True
        assert getattr(provider, method)(make(missing_usage)).usage_verified is False


def test_codex_cached_tokens_are_not_double_counted():
    import json

    from llm_council.providers.cli.codex import _ingest_codex_stdout_line, _LiveCodexState

    state = _LiveCodexState()
    _ingest_codex_stdout_line(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 10},
            }
        ),
        state,
    )
    assert state.usage["prompt_tokens"] == 100
    assert state.usage["cache_read_tokens"] == 80
    assert state.usage["total_tokens"] == 110
    assert state.usage_verified is True


def test_openrouter_stream_preserves_usage_only_terminal_chunk():
    from llm_council.providers.openrouter import OpenRouterProvider

    provider = OpenRouterProvider.__new__(OpenRouterProvider)
    response = provider._parse_stream_chunk(
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}
    )
    assert response.usage["prompt_tokens"] == 10
    assert response.usage_verified is True


def test_claude_cli_includes_cache_creation_in_prompt_tokens():
    from llm_council.providers.cli.claude_code import _normalize_usage

    usage = _normalize_usage(
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 7,
        }
    )
    assert usage["prompt_tokens"] == 37
    assert usage["cache_read_tokens"] == 20
    assert usage["cache_creation_tokens"] == 7


def test_gemini_cli_requires_explicit_consistent_stats():
    from llm_council.providers.cli.gemini_cli import _usage_stats_verified

    assert _usage_stats_verified({"inputTokenCount": 5, "outputTokenCount": 3}) is True
    assert _usage_stats_verified({"inputTokenCount": 5}) is False
    assert (
        _usage_stats_verified({"inputTokenCount": 5, "outputTokenCount": 3, "totalTokenCount": 9})
        is False
    )
    assert _usage_stats_verified({"inputTokenCount": True, "outputTokenCount": 3}) is False


def test_openrouter_stream_keeps_text_when_usage_is_malformed():
    from llm_council.providers.openrouter import OpenRouterProvider

    provider = OpenRouterProvider.__new__(OpenRouterProvider)
    for usage in ("invalid", {"prompt_tokens": "bad", "completion_tokens": 4, "total_tokens": 14}):
        chunk = provider._parse_stream_chunk(
            {"choices": [{"delta": {"content": "hello"}}], "usage": usage}
        )
        assert chunk.text == "hello"
        assert chunk.usage_verified is False


def test_codex_rejects_second_turn_instead_of_reusing_prior_usage():
    import json

    from llm_council.providers.cli.codex import _ingest_codex_stdout_line, _LiveCodexState

    state = _LiveCodexState()
    for event in [
        {"type": "turn.started"},
        {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 10}},
        {"type": "turn.started"},
    ]:
        _ingest_codex_stdout_line(json.dumps(event), state)
    assert state.protocol_error


async def test_stream_interim_usage_followed_by_output_is_unverified():
    from unittest.mock import AsyncMock, MagicMock

    from llm_council.providers.base import GenerateRequest, GenerateResponse

    async def stream():
        yield GenerateResponse(
            text="part", usage={"prompt_tokens": 4, "completion_tokens": 1}, usage_verified=True
        )
        yield GenerateResponse(text=" more")

    orch = make_orchestrator(cost_per_1k_input={"example": 1}, cost_per_1k_output={"example": 1})
    adapter = MagicMock()
    adapter.resolve_model.return_value = "example-model"
    adapter.generate = AsyncMock(return_value=stream())
    await orch._call_provider("example", adapter, GenerateRequest(prompt="test"))
    assert orch._build_cost_estimate().reported_usage_estimate_usd is None


async def test_openai_stream_usage_only_chunk_is_retained():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from llm_council.providers.openai import OpenAIProvider

    async def source():
        yield SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14),
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=source())))
    )
    provider = OpenAIProvider.__new__(OpenAIProvider)
    chunks = [chunk async for chunk in provider._generate_stream(client, {"stream": True})]
    assert chunks[0].usage["prompt_tokens"] == 10
    assert chunks[0].usage_verified is True


async def test_anthropic_streams_emit_final_usage_without_duplicating_text():
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from llm_council.providers.anthropic import AnthropicProvider
    from llm_council.providers.vertex import VertexAIProvider

    async def texts():
        yield "hello"

    final = SimpleNamespace(
        content=[],
        model="test",
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )

    @asynccontextmanager
    async def stream_context(**kwargs):
        yield SimpleNamespace(text_stream=texts(), get_final_message=AsyncMock(return_value=final))

    client = SimpleNamespace(messages=SimpleNamespace(stream=stream_context))
    for cls, method in (
        (AnthropicProvider, "_generate_stream"),
        (VertexAIProvider, "_generate_claude_stream"),
    ):
        provider = cls.__new__(cls)
        chunks = [chunk async for chunk in getattr(provider, method)(client, {})]
        assert [chunk.text for chunk in chunks if chunk.text] == ["hello"]
        assert chunks[-1].usage["prompt_tokens"] == 10
        assert chunks[-1].usage_verified is True


def test_openai_stream_usage_option_respects_sdk_signature():
    from llm_council.providers.openai import _supports_stream_usage_option

    def old_sdk(*, stream):
        pass

    def new_sdk(*, stream, stream_options=None):
        pass

    assert _supports_stream_usage_option(old_sdk) is False
    assert _supports_stream_usage_option(new_sdk) is True


async def test_anthropic_streams_support_sdk_without_final_message_method():
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from llm_council.providers.anthropic import AnthropicProvider
    from llm_council.providers.vertex import VertexAIProvider

    async def texts():
        yield "hello"

    @asynccontextmanager
    async def stream_context(**kwargs):
        yield SimpleNamespace(text_stream=texts())

    client = SimpleNamespace(messages=SimpleNamespace(stream=stream_context))
    for cls, method in (
        (AnthropicProvider, "_generate_stream"),
        (VertexAIProvider, "_generate_claude_stream"),
    ):
        chunks = [chunk async for chunk in getattr(cls.__new__(cls), method)(client, {})]
        assert [chunk.text for chunk in chunks] == ["hello"]
        assert chunks[-1].usage_verified is False


def test_codex_rejects_duplicate_terminal_event():
    import json

    from llm_council.providers.cli.codex import _ingest_codex_stdout_line, _LiveCodexState

    state = _LiveCodexState()
    for event in [
        {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 10}},
        {"type": "turn.completed"},
    ]:
        _ingest_codex_stdout_line(json.dumps(event), state)
    assert state.protocol_error


async def test_invalid_later_openrouter_usage_invalidates_earlier_counts():
    from unittest.mock import AsyncMock, MagicMock

    from llm_council.providers.base import GenerateRequest
    from llm_council.providers.openrouter import OpenRouterProvider

    parser = OpenRouterProvider.__new__(OpenRouterProvider)

    async def stream():
        yield parser._parse_stream_chunk(
            {
                "choices": [{"delta": {"content": "ok"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            }
        )
        yield parser._parse_stream_chunk({"choices": [], "usage": {"prompt_tokens": "bad"}})

    orch = make_orchestrator(cost_per_1k_input={"example": 1}, cost_per_1k_output={"example": 1})
    adapter = MagicMock()
    adapter.resolve_model.return_value = "example-model"
    adapter.generate = AsyncMock(return_value=stream())
    await orch._call_provider("example", adapter, GenerateRequest(prompt="test"))
    assert orch._build_cost_estimate().reported_usage_estimate_usd is None


async def test_invalid_later_openai_usage_invalidates_earlier_counts():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from llm_council.providers.base import GenerateRequest
    from llm_council.providers.openai import OpenAIProvider

    async def source():
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"), finish_reason=None)],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6),
        )
        yield SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=None, completion_tokens=3, total_tokens=7),
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=source())))
    )
    provider = OpenAIProvider.__new__(OpenAIProvider)
    orch = make_orchestrator(cost_per_1k_input={"example": 1}, cost_per_1k_output={"example": 1})
    adapter = MagicMock()
    adapter.resolve_model.return_value = "example-model"
    adapter.generate = AsyncMock(return_value=provider._generate_stream(client, {"stream": True}))
    await orch._call_provider("example", adapter, GenerateRequest(prompt="test"))
    assert orch._build_cost_estimate().reported_usage_estimate_usd is None


async def test_gemini_stream_keeps_text_when_usage_is_malformed():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from llm_council.providers.gemini import GeminiProvider

    async def source():
        yield SimpleNamespace(
            text="hello",
            candidates=[],
            usage_metadata=SimpleNamespace(
                prompt_token_count="bad", candidates_token_count=1, total_token_count=2
            ),
        )

    client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content_stream=AsyncMock(return_value=source()))
        )
    )
    provider = GeminiProvider.__new__(GeminiProvider)
    chunks = [chunk async for chunk in provider._generate_stream(client, "model", "prompt", {})]
    assert chunks[0].text == "hello"
    assert chunks[0].usage_invalid is True


def test_codex_rejects_duplicate_turn_start():
    import json

    from llm_council.providers.cli.codex import _ingest_codex_stdout_line, _LiveCodexState

    state = _LiveCodexState()
    for event in [{"type": "turn.started"}, {"type": "turn.started"}]:
        _ingest_codex_stdout_line(json.dumps(event), state)
    assert state.protocol_error
