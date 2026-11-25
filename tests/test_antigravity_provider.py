"""Antigravity's headless protocol and supplementary-draft policy."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm_council.engine.orchestrator import Orchestrator
from llm_council.providers.base import GenerateRequest, ReasoningConfig, StructuredOutputConfig

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def transcript(**updates):
    result = {
        "status": "SUCCESS",
        "structured_output": {"ok": True},
        "usage": {
            "input_tokens": 10350,
            "output_tokens": 1093,
            "thinking_tokens": 1075,
            "cache_read_tokens": 0,
            "total_tokens": 11443,
        },
        **updates,
    }
    return (json.dumps({"event": "result", "result": result}) + "\n").encode()


@pytest.mark.asyncio
async def test_agy_adc_schema_and_usage(tmp_path, monkeypatch):
    from llm_council.providers.cli.antigravity import AntigravityCLIProvider

    adc = tmp_path / "adc.json"
    adc.touch()
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(adc))
    monkeypatch.setenv("GEMINI_API_KEY", "not-for-the-child")
    monkeypatch.setenv("OPENAI_API_KEY", "not-for-the-child")
    process = Mock(returncode=0, communicate=AsyncMock(return_value=(transcript(), b"")))
    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as spawn:
        spawn.return_value = process
        provider = AntigravityCLIProvider(cli_path="agy")
        output = await provider.generate(
            GenerateRequest(
                prompt="synthetic",
                structured_output=StructuredOutputConfig(name="result", json_schema=SCHEMA),
                reasoning=ReasoningConfig(enabled=True, effort="high"),
                timeout_seconds=12,
            )
        )
    args, kwargs = spawn.call_args
    assert "--input-format" in args and "--json-schema" in args and "high" in args
    assert "--mode" in args and "plan" in args and "--sandbox" in args
    assert "synthetic" not in args
    assert kwargs["env"]["AGY_ADC_AUTH"] == "true"
    assert kwargs["env"]["GOOGLE_APPLICATION_CREDENTIALS"] == str(adc)
    assert kwargs["cwd"] == kwargs["env"]["HOME"]
    assert "GEMINI_API_KEY" not in kwargs["env"] and "OPENAI_API_KEY" not in kwargs["env"]
    assert json.loads(process.communicate.call_args.args[0])["message"]["content"] == "synthetic"
    assert output.content == {"ok": True}
    assert output.usage_verified
    assert output.usage == {
        "prompt_tokens": 10350,
        "completion_tokens": 18,
        "reasoning_tokens": 1075,
        "total_tokens": 11443,
        "cache_read_tokens": 0,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["ERROR", "CANCELED", "INTERRUPTED", "RUNNING"])
async def test_agy_rejects_incomplete_results(tmp_path, monkeypatch, status):
    from llm_council.providers.cli.antigravity import AntigravityCLIProvider

    adc = tmp_path / "adc.json"
    adc.touch()
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(adc))
    process = Mock(
        returncode=0, communicate=AsyncMock(return_value=(transcript(status=status), b""))
    )
    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as spawn:
        spawn.return_value = process
        with pytest.raises(RuntimeError, match=status):
            await AntigravityCLIProvider(cli_path="agy").generate(GenerateRequest(prompt="x"))


@pytest.mark.asyncio
async def test_agy_is_never_a_critique_or_synthesis_candidate():
    sentinel = Mock(supports=AsyncMock(return_value=True))
    orch = object.__new__(Orchestrator)
    orch._providers = {"antigravity-cli": sentinel, "vertex-ai": sentinel}
    orch._provider_init_errors = {}
    assert [name for name, _ in await orch._candidate_providers_for_phase("critique")] == [
        "vertex-ai"
    ]
    assert [name for name, _ in await orch._candidate_providers_for_phase("synthesis")] == [
        "vertex-ai"
    ]


def test_agy_does_not_become_default_or_final_fallback():
    orch = object.__new__(Orchestrator)
    orch._config = Mock(provider_configs={"agy": {}, "vertex-ai": {}})
    orch._execution_plan = None
    with patch.object(orch, "_validate_response") as validate:
        result, attempts = orch._fallback_synthesis_from_drafts(
            {"antigravity-cli": '{"ok":true}'}, TimeoutError("synthesis")
        )
        validate.assert_not_called()
    assert not result.ok and attempts == 0
    assert orch._fallback_provider_candidates() == ["vertex-ai"]


def test_agy_usage_requires_reconciling_unambiguous_counters():
    from llm_council.providers.cli.antigravity import _normalized_usage

    assert (
        _normalized_usage(
            {"input_tokens": 20, "output_tokens": 5, "thinking_tokens": 3, "total_tokens": 28}
        )
        is None
    )
    assert (
        _normalized_usage(
            {"input_tokens": 20, "output_tokens": 5, "thinking_tokens": 6, "total_tokens": 25}
        )
        is None
    )


@pytest.mark.parametrize("cached", [None, True, -1, "7", 21])
def test_agy_rejects_invalid_explicit_cache_usage(cached):
    from llm_council.providers.cli.antigravity import _normalized_usage

    assert (
        _normalized_usage(
            {
                "input_tokens": 20,
                "output_tokens": 5,
                "thinking_tokens": 3,
                "total_tokens": 25,
                "cache_read_tokens": cached,
            }
        )
        is None
    )


@pytest.mark.asyncio
async def test_agy_timeout_terminates_owned_process(tmp_path, monkeypatch):
    import asyncio

    from llm_council.providers.cli.antigravity import AntigravityCLIProvider

    adc = tmp_path / "adc.json"
    adc.touch()
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(adc))
    process = Mock(communicate=AsyncMock(side_effect=asyncio.TimeoutError))
    with (
        patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)),
        patch(
            "llm_council.providers.cli.antigravity.terminate_process_tree", new_callable=AsyncMock
        ) as terminate,
    ):
        with pytest.raises(TimeoutError, match="Antigravity CLI timed out"):
            await AntigravityCLIProvider(cli_path="agy").generate(GenerateRequest(prompt="x"))
        terminate.assert_awaited_once_with(process)
