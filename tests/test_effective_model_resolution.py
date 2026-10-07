"""Engine compilation must use the same model as the adapter's SDK payload."""

import json
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from llm_council.engine.orchestrator import Orchestrator, OrchestratorConfig
from llm_council.providers.base import (
    DoctorResult,
    GenerateRequest,
    GenerateResponse,
    ProviderAdapter,
    ProviderCapabilities,
    ReasoningConfig,
    StructuredOutputConfig,
)
from llm_council.providers.registry import get_registry
from llm_council.storage.artifacts import ArtifactStore


@pytest.fixture
def engine(monkeypatch):
    @asynccontextmanager
    async def slot(*args, **kwargs):
        yield 0.0

    monkeypatch.setattr("llm_council.engine.orchestrator.provider_call_slot", slot)
    result = Orchestrator(
        [],
        OrchestratorConfig(
            enable_artifacts=False,
            enable_health_check=False,
            enable_graceful_degradation=False,
        ),
    )
    result._execution_plan = {}
    return result


@pytest.fixture
def sdk(monkeypatch):
    async def create(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"result":"ok"}', tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=None,
            model="synthetic-reported-snapshot",
        )

    create_call = AsyncMock(side_effect=create)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_call)))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=lambda **kwargs: client))
    return create_call


def synthesis_request(model=None):
    return GenerateRequest(
        prompt="Return the synthesis result.",
        model=model,
        reasoning=ReasoningConfig(enabled=True, effort="high"),
        structured_output=StructuredOutputConfig(
            json_schema={"type": "object", "properties": {"result": {"type": "string"}}},
            name="synthesis",
            strict=True,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["builtin", "environment", "constructor"])
async def test_default_model_controls_reach_real_openai_sdk(engine, sdk, monkeypatch, selection):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    kwargs = {"api_key": "synthetic-not-a-real-key"}
    expected = "gpt-5.4"
    if selection == "environment":
        monkeypatch.setenv("OPENAI_MODEL", "o3-mini")
        expected = "o3-mini"
    elif selection == "constructor":
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
        kwargs["default_model"] = "o4-mini"
        expected = "o4-mini"
    adapter = get_registry().get_provider("openai", **kwargs)
    # Resolution must use the instantiated adapter, not read environment again.
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
    request = synthesis_request()
    before = request.model_dump()

    await engine._call_provider("openai", adapter, request, phase="synthesis")

    wire = sdk.call_args.kwargs
    assert wire["model"] == expected
    assert wire.get("reasoning_effort") == (None if selection == "builtin" else "high")
    assert wire.get("response_format", {}).get("type") == "json_schema"
    assert wire["response_format"]["json_schema"]["schema"]["required"] == ["result"]
    assert request.model_dump() == before
    compilation = engine._execution_plan["phase_request_compilation"]["synthesis"][0]
    assert compilation["model"] == wire["model"]
    dropped = [d for d in compilation["decisions"] if d["action"] == "dropped"]
    if selection == "builtin":
        # Preserve the existing o-series-only reasoning policy; do not add capabilities here.
        assert dropped == [
            {
                "option": "reasoning",
                "action": "dropped",
                "detail": "gpt-5.4 does not support reasoning controls",
            }
        ]
    else:
        assert dropped == []
    attempt = engine._execution_plan["provider_attempts"][0]
    assert attempt["requested_model"] is None
    assert attempt["adapter_resolved_model"] == wire["model"]
    assert attempt["adapter_reported_model"] == "synthetic-reported-snapshot"


@pytest.mark.asyncio
async def test_explicit_unsupported_model_wins_over_adapter_default(engine, sdk):
    adapter = get_registry().get_provider(
        "openai", api_key="synthetic-not-a-real-key", default_model="gpt-5.4"
    )
    request = synthesis_request("gpt-4")
    before = request.model_dump()

    await engine._call_provider("openai", adapter, request, phase="synthesis")

    wire = sdk.call_args.kwargs
    assert wire["model"] == "gpt-4"
    assert "reasoning_effort" not in wire
    assert "response_format" not in wire
    compilation = engine._execution_plan["phase_request_compilation"]["synthesis"][0]
    assert compilation["model"] == wire["model"]
    assert {d["option"] for d in compilation["decisions"] if d["action"] == "dropped"} == {
        "reasoning",
        "structured_output",
    }
    attempt = engine._execution_plan["provider_attempts"][0]
    assert attempt["requested_model"] == attempt["adapter_resolved_model"] == "gpt-4"
    assert request.model_dump() == before


class ExternalProvider(ProviderAdapter):
    name = "external-test-provider"
    capabilities = ProviderCapabilities()

    async def generate(self, request):
        self.received = request
        return GenerateResponse(text="ok")

    async def supports(self, capability):
        return False

    async def doctor(self):
        return DoctorResult(ok=True, message="synthetic")


@pytest.mark.asyncio
@pytest.mark.parametrize("default", [None, 42])
async def test_external_adapter_unknown_default_preserves_request(engine, default):
    adapter = ExternalProvider()
    adapter._default_model = default
    request = GenerateRequest(prompt="test")
    await engine._call_provider(adapter.name, adapter, request, phase="draft")
    assert adapter.received.model is None
    assert request.model is None
    assert engine._execution_plan["provider_attempts"][0]["adapter_resolved_model"] is None


@pytest.mark.asyncio
async def test_resolved_identity_survives_sdk_failure(engine, sdk):
    adapter = get_registry().get_provider(
        "openai", api_key="synthetic-not-a-real-key", default_model="gpt-5.4"
    )
    sdk.side_effect = RuntimeError("synthetic SDK failure")
    with pytest.raises(RuntimeError, match="synthetic SDK failure"):
        await engine._call_provider("openai", adapter, synthesis_request(), phase="synthesis")
    attempt = engine._execution_plan["provider_attempts"][0]
    assert attempt["requested_model"] is None
    assert attempt["adapter_resolved_model"] == sdk.call_args.kwargs["model"] == "gpt-5.4"
    assert "adapter_reported_model" not in attempt


@pytest.mark.parametrize(
    "name",
    ["openai", "anthropic", "gemini", "vertex-ai", "openrouter", "codex", "claude", "gemini-cli"],
)
def test_all_builtin_resolvers_preserve_instantiated_default_and_explicit_selection(name):
    adapter = get_registry().get_provider(name, default_model="selected-default")
    assert adapter.resolve_model(None) == "selected-default"
    assert adapter.resolve_model("explicit-model") == "explicit-model"
    assert adapter.resolve_model(None) == "selected-default"


@pytest.mark.asyncio
async def test_execution_manifest_retains_each_model_identity(engine, sdk, tmp_path):
    adapter = get_registry().get_provider(
        "openai", api_key="synthetic-not-a-real-key", default_model="gpt-5.4"
    )
    store = ArtifactStore(artifact_dir=tmp_path / "artifacts", db_path=tmp_path / "ledger.db")
    engine._artifact_store = store
    engine._run_id = store.create_run("synthetic", "synthetic").run_id

    await engine._call_provider("openai", adapter, synthesis_request(), phase="synthesis")
    engine._store_execution_manifest("settled", "completed")

    artifacts = store.get_run_artifacts(engine._run_id)
    manifest = json.loads(store.get_artifact_content(artifacts[0].artifact_id))
    attempt = manifest["provider_attempts"][0]
    assert attempt["requested_model"] is None
    assert attempt["adapter_resolved_model"] == sdk.call_args.kwargs["model"] == "gpt-5.4"
    assert attempt["adapter_reported_model"] == "synthetic-reported-snapshot"
