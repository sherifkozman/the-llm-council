"""Regressions for review evidence loss and empty provider responses (#63/65/67)."""

import json
from contextlib import asynccontextmanager
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest
from jsonschema import validate

from llm_council.engine.orchestrator import Orchestrator, OrchestratorConfig
from llm_council.providers.base import (
    DoctorResult,
    GenerateResponse,
    ProviderAdapter,
    ProviderCapabilities,
)
from llm_council.schemas import load_schema


class ReviewProvider(ProviderAdapter):
    name: ClassVar[str] = "fixture"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(structured_output=True)

    def __init__(self, response=None, *, stream=False, files_reviewed=4):
        self.response = response
        self.stream = stream
        self.files_reviewed = files_reviewed
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        if self.response is not None:
            if self.stream:

                async def chunks():
                    yield GenerateResponse(text="")
                    yield self.response

                return chunks()
            return self.response
        if request.messages[0].content.startswith("You are the synthesizer"):
            verdict = {
                "review_summary": "Synthetic source review found a retry defect.",
                "review_type": "code_quality",
                "verdict": "request_changes",
                "confidence": 80,
                "scope": {
                    "files_reviewed": self.files_reviewed,
                    "lines_changed": 0,
                    "components_affected": ["retry"],
                },
                "issues": [
                    {
                        "severity": "high",
                        "category": "logic",
                        "location": {"file": "file-0.py"},
                        "description": "The retry loop never stops after exhaustion.",
                    }
                ],
                "recommendations": [{"priority": "must_fix", "recommendation": "Bound retries."}],
                "blocking_issues": [],
                "reasoning": "The supplied draft evidence identifies an unbounded retry loop which must terminate.",
            }
            validate(verdict, load_schema("reviewer"))
            return GenerateResponse(text=json.dumps(verdict))
        if request.messages[0].content.startswith("You are an adversarial"):
            return GenerateResponse(text="Critique confirms the retry defect.")
        return GenerateResponse(
            text="DRAFT_SENTINEL: The retry loop never stops after exhaustion.\n" * 240
        )

    async def supports(self, capability):
        return capability == "structured_output"

    async def doctor(self):
        return DoctorResult(ok=True, message="offline fixture")


@pytest.fixture(autouse=True)
def offline_slots(monkeypatch, tmp_path):
    @asynccontextmanager
    async def slot(*args, **kwargs):
        yield 0.0

    monkeypatch.setattr("llm_council.engine.orchestrator.provider_call_slot", slot)
    monkeypatch.chdir(tmp_path)


def orchestrator(providers, **config):
    registry = MagicMock()
    registry.get_provider.side_effect = lambda name, **kwargs: providers[name]
    with patch("llm_council.engine.orchestrator.get_registry", return_value=registry):
        return Orchestrator(
            list(providers),
            OrchestratorConfig(
                mode="review",
                enable_artifacts=False,
                enable_health_check=False,
                **config,
            ),
        )


def large_python_context():
    return "\n\n".join(
        f"=== FILE: file-{i}.py ===\nSOURCE_SENTINEL\n"
        + "# source line\n" * 950
        + f"=== END: file-{i}.py ==="
        for i in range(4)
    )


@pytest.mark.asyncio
async def test_large_python_review_keeps_drafts_when_critique_is_skipped():
    providers = {name: ReviewProvider() for name in ("codex", "claude")}
    result = await orchestrator(providers, system_context=large_python_context()).run(
        "Review supplied source files for retry defects.", "critic"
    )
    synthesis = [
        r
        for p in providers.values()
        for r in p.requests
        if r.messages[0].content.startswith("You are the synthesizer")
    ]
    assert synthesis
    assert not result.critique  # The triggering whole-file/no-critique combination.
    assert all("DRAFT_SENTINEL" in r.messages[1].content for r in synthesis)
    assert result.success, result.validation_errors
    assert result.validation_errors is None
    assert not result.execution_plan.get("degraded_output")
    assert result.output["issues"]
    decisions = result.execution_plan["phase_prompt_compaction"]["synthesis"]
    assert all(not d["over_budget"] for d in decisions)
    assert all(not d["profile"].get("omit_drafts") for d in decisions)
    metric = result.execution_plan["phase_prompt_metrics"]["synthesis"][0]
    assert 0 < metric["evidence_pack_chars"] <= metric["phase_prompt_chars"]
    assert metric["evidence_pack_chars"] < metric["raw_source_chars"]


@pytest.mark.asyncio
async def test_synthesis_never_calls_provider_when_no_evidence_profile_fits(monkeypatch):
    provider = ReviewProvider()
    orch = orchestrator({"openai": provider})
    orch._task = "Review source."
    orch._subagent_name = "critic"
    orch._prepare_run("critic")
    original = orch._phase_budget_status
    monkeypatch.setattr(
        orch,
        "_phase_budget_status",
        lambda *args: {
            **original(*args),
            "over_budget": True,
        },
    )
    with pytest.raises(RuntimeError, match="budget"):
        await orch._run_synthesis({"openai": "short draft"}, "")
    assert not provider.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["openrouter", "vertex-ai"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("policy", [False, True])
async def test_empty_response_fails_with_phase_and_finish_reason(name, stream, policy):
    provider = ReviewProvider(GenerateResponse(text=" \n", finish_reason="length"), stream=stream)
    result = await orchestrator({name: provider}, enable_graceful_degradation=policy).run(
        "Review supplied source.", "critic"
    )
    assert not result.success
    assert result.execution_status == "failed"
    assert len(provider.requests) == 1  # Empty output is not a transient transport retry.
    assert "empty_response" in result.provider_errors[name]
    assert "length" in result.provider_errors[name]
    if policy:
        failures = result.degradation_report["failures"]
        assert failures[0]["phase"] == "draft"
        assert failures[0]["error_type"] == "empty_response"


@pytest.mark.asyncio
async def test_mixed_chunked_and_whole_drafts_do_not_restore_raw_context(monkeypatch):
    providers = {name: ReviewProvider() for name in ("openai", "codex")}
    orch = orchestrator(providers, system_context=large_python_context(), runtime_profile="bounded")
    original = orch._draft_chunk_plan
    monkeypatch.setattr(
        orch,
        "_draft_chunk_plan",
        lambda name, *args: original(name, *args) if name == "openai" else None,
    )
    result = await orch.run("Review supplied source for retry defects.", "critic")
    assert result.success
    assert orch._draft_handoffs["openai"]["findings"]
    assert result.critique
    assert result.validation_errors is None
    downstream = [
        r
        for p in providers.values()
        for r in p.requests
        if r.messages[0].content.startswith(("You are the synthesizer", "You are an adversarial"))
    ]
    assert downstream
    assert all("=== FILE:" not in r.messages[1].content for r in downstream)


@pytest.mark.asyncio
async def test_empty_provider_does_not_erase_successful_peer():
    providers = {
        "openai": ReviewProvider(),
        "openrouter": ReviewProvider(
            GenerateResponse(text=None, finish_reason="STOP", tool_calls=[{"private": "SECRET"}])
        ),
    }
    result = await orchestrator(providers).run("Review supplied source.", "critic")
    assert result.success
    assert result.execution_status == "degraded"
    assert result.drafts["openai"]
    assert not result.drafts["openrouter"]
    assert "tool_calls_present=true" in result.provider_errors["openrouter"]
    assert "SECRET" not in json.dumps(result.model_dump())


def test_reviewer_explicit_zero_files_is_invalid_for_supplied_files():
    orch = orchestrator({"openai": ReviewProvider()}, system_context=large_python_context())
    orch._task = "Review supplied files."
    orch._prepare_run("critic")
    orch._config.enable_schema_validation = False
    result = orch._validate_response('{"scope":{"files_reviewed":0},"issues":[],"confidence":100}')
    assert not result.ok
    assert any("files_reviewed" in error for error in result.errors)


def test_reviewer_fallback_cannot_approve_without_findings():
    orch = orchestrator({"openai": ReviewProvider()})
    orch._schema_name = "reviewer"
    assert orch._fallback_synthesis_from_evidence({}, "", ["No evidence"]) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("override,expected", [(None, 900), (8000, 8000), (500, 500)])
async def test_chunked_drafts_honor_explicit_output_budget(override, expected):
    provider = ReviewProvider()
    orch = orchestrator(
        {"openrouter": provider},
        system_context=large_python_context(),
        runtime_profile="bounded",
        max_tokens=override,
    )
    orch._task = "Review these supplied files."
    orch._subagent_name = "critic"
    orch._prepare_run("critic")
    await orch._generate_draft("openrouter", provider)
    assert len(provider.requests) > 1
    assert all(request.max_tokens == expected for request in provider.requests)


def test_exception_diagnostics_respect_suppressed_secret_context():
    orch = orchestrator({"openai": ReviewProvider()})
    try:
        try:
            raise RuntimeError("private-secret-sentinel")
        except RuntimeError:
            raise RuntimeError("sanitized provider failure") from None
    except RuntimeError as error:
        assert orch._format_exception_chain(error) == "sanitized provider failure"


def test_exception_diagnostics_terminate_on_cyclic_causes():
    orch = orchestrator({"openai": ReviewProvider()})

    class OnceRenderedError(RuntimeError):
        renders = 0

        def __str__(self):
            self.renders += 1
            assert self.renders == 1, "diagnostics followed a cyclic cause"
            return super().__str__()

    error = OnceRenderedError("provider failure")
    error.__cause__ = error
    assert orch._format_exception_chain(error) == "provider failure"
