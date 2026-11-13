"""Council exposes automatic caching only through explicit opt-in."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_council.council import Council
from llm_council.engine.orchestrator import Orchestrator, OrchestratorConfig
from llm_council.protocol.types import CouncilConfig
from llm_council.providers.base import GenerateRequest, GenerateResponse, PromptCacheConfig


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "enabled,provider,expected",
    [(False, "anthropic", False), (True, "anthropic", True), (True, "gemini", False)],
)
async def test_auto_cache_is_opt_in_and_respects_provider_support(enabled, provider, expected):
    orch = Orchestrator([], OrchestratorConfig(enable_artifacts=False, prompt_cache_auto=enabled))
    orch._execution_plan = {}
    adapter = MagicMock()
    adapter.resolve_model.return_value = "example"
    adapter.generate = AsyncMock(return_value=GenerateResponse(text="done"))
    await orch._call_provider(provider, adapter, GenerateRequest(prompt="synthetic"))
    request = adapter.generate.await_args.args[0]
    assert bool(request.prompt_cache) is expected
    if enabled:
        observations = orch._execution_plan["prompt_cache_observations"]["unknown"][0]
        assert observations["requested"] is True
        assert observations["forwarded"] is expected
    else:
        assert "prompt_cache_observations" not in orch._execution_plan


@pytest.mark.asyncio
async def test_explicit_disabled_cache_overrides_council_opt_in():
    orch = Orchestrator([], OrchestratorConfig(enable_artifacts=False, prompt_cache_auto=True))
    adapter = MagicMock()
    adapter.resolve_model.return_value = "example"
    adapter.generate = AsyncMock(return_value=GenerateResponse(text="done"))
    await orch._call_provider(
        "anthropic",
        adapter,
        GenerateRequest(prompt="synthetic", prompt_cache=PromptCacheConfig(enabled=False)),
    )
    assert adapter.generate.await_args.args[0].prompt_cache is None


def test_public_config_forwards_cache_opt_in():
    config = CouncilConfig(providers=["openrouter"], prompt_cache_auto=True)
    council = Council(config=config)
    assert council._build_orchestrator_config(config).prompt_cache_auto is True


def test_cli_dry_run_shows_automatic_cache_opt_in():
    import json
    from unittest.mock import patch

    from typer.testing import CliRunner

    from llm_council.cli.main import app

    with patch("llm_council.cli.main._load_config_defaults", return_value={}):
        result = CliRunner().invoke(
            app, ["run", "critic", "synthetic", "--dry-run", "--json", "--prompt-cache-auto"]
        )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["prompt_cache_auto"] is True


def test_cli_run_forwards_cache_opt_in():
    from unittest.mock import patch

    from typer.testing import CliRunner

    from llm_council.cli.main import app
    from llm_council.engine.orchestrator import CouncilResult

    with (
        patch("llm_council.Council") as council_class,
        patch("llm_council.cli.main._execute_run", return_value=CouncilResult(success=True)),
        patch("llm_council.cli.main._load_config_defaults", return_value={}),
        patch("llm_council.cli.main._load_provider_configs", return_value={}),
    ):
        result = CliRunner().invoke(
            app, ["run", "critic", "synthetic", "--json", "--prompt-cache-auto"]
        )
    assert result.exit_code == 0
    assert council_class.call_args.kwargs["config"].prompt_cache_auto is True


def test_cache_observation_distinguishes_write_zero_and_read():
    orch = Orchestrator([], OrchestratorConfig(enable_artifacts=False))
    orch._execution_plan = {}
    for usage in (
        {"cache_creation_tokens": 12},
        {"cache_read_tokens": 0},
        {"cache_read_tokens": 7},
    ):
        orch._record_prompt_cache_observation(
            phase="draft", provider_name="anthropic", requested=True, forwarded=True, usage=usage
        )
    write, zero, read = orch._execution_plan["prompt_cache_observations"]["draft"]
    assert write["cache_read_observed"] is False
    assert write["cache_write_observed"] is True
    assert zero["metrics_available"] is True
    assert zero["observed"] is False
    assert read["cache_read_observed"] is True


def test_anthropic_reported_zero_survives_to_cache_observation():
    from types import SimpleNamespace

    from llm_council.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider.__new__(AnthropicProvider)
    response = provider._parse_response(
        SimpleNamespace(
            content=[],
            model="synthetic",
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
        )
    )
    orch = Orchestrator([], OrchestratorConfig(enable_artifacts=False))
    orch._execution_plan = {}
    orch._record_prompt_cache_observation(
        phase="draft",
        provider_name="anthropic",
        requested=True,
        forwarded=True,
        usage=response.usage,
    )
    obs = orch._execution_plan["prompt_cache_observations"]["draft"][0]
    assert obs["metrics_available"] is True
    assert obs["cache_read_observed"] is False
    assert obs["cache_write_observed"] is False


def test_openrouter_cache_zero_is_not_fabricated_when_absent():
    from llm_council.providers.openrouter import OpenRouterProvider

    provider = OpenRouterProvider.__new__(OpenRouterProvider)
    base = {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}
    without = provider._parse_response(base)
    assert "cache_read_tokens" not in without.usage
    with_zero = provider._parse_response(
        {"choices": [], "usage": {**base["usage"], "prompt_tokens_details": {"cached_tokens": 0}}}
    )
    assert with_zero.usage["cache_read_tokens"] == 0


def test_current_default_routes_do_not_forward_automatic_cache():
    from llm_council.providers.compiler import compile_request_for_provider

    for provider, model in [
        ("vertex-ai", "gemini-4-argon"),
        ("openrouter", "z-ai/glm-5.2"),
        ("claude", "claude-opus-5-5"),
    ]:
        request = GenerateRequest(
            model=model, prompt="synthetic", prompt_cache=PromptCacheConfig(mode="auto")
        )
        compiled = compile_request_for_provider(provider, request)
        assert compiled.request.prompt_cache is None
