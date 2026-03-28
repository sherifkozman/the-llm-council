"""Tests for the Orchestrator engine."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_council.config.models import ModelPack, get_model_for_pack
from llm_council.engine.orchestrator import (
    CostEstimate,
    CouncilResult,
    Orchestrator,
    OrchestratorConfig,
    ValidationResult,
)
from llm_council.protocol.types import ReasoningProfile, RuntimeProfile
from llm_council.providers.base import (
    DoctorResult,
    GenerateRequest,
    GenerateResponse,
    ProviderAdapter,
    ProviderCapabilities,
)


class CaptureProvider(ProviderAdapter):
    """Provider stub that captures the last request for assertions."""

    name: ClassVar[str] = "capture"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        streaming=False,
        tool_use=False,
        structured_output=True,
        multimodal=False,
        max_tokens=4096,
    )

    def __init__(self) -> None:
        self.last_request: GenerateRequest | None = None

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        self.last_request = request
        return GenerateResponse(text='{"ok": true}')

    async def supports(self, capability: str) -> bool:
        return capability == "structured_output"

    async def doctor(self) -> DoctorResult:
        return DoctorResult(ok=True, message="ok")


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OrchestratorConfig()
        assert config.timeout == 120
        assert config.max_retries == 3
        assert config.enable_schema_validation is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = OrchestratorConfig(
            timeout=60,
            max_retries=5,
            max_draft_tokens=1000,
            draft_temperature=0.5,
        )
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.max_draft_tokens == 1000
        assert config.draft_temperature == 0.5

    def test_config_validation(self):
        """Test config validation constraints."""
        # Valid range
        config = OrchestratorConfig(timeout=600)
        assert config.timeout == 600

        # Test min/max for max_retries
        config = OrchestratorConfig(max_retries=10)
        assert config.max_retries == 10


class TestCostEstimate:
    """Tests for CostEstimate model."""

    def test_basic_cost_estimate(self):
        """Test basic cost estimate creation."""
        estimate = CostEstimate(
            provider_calls={"openrouter": 3},
            tokens=1500,
            total_input_tokens=1000,
            total_output_tokens=500,
            estimated_cost_usd=0.0015,
        )
        assert estimate.tokens == 1500
        assert estimate.estimated_cost_usd == 0.0015

    def test_default_cost_estimate(self):
        """Test default cost estimate values."""
        estimate = CostEstimate()
        assert estimate.tokens == 0
        assert estimate.estimated_cost_usd == 0.0


class TestCouncilResult:
    """Tests for CouncilResult model."""

    def test_successful_result(self):
        """Test successful council result."""
        result = CouncilResult(
            success=True,
            output={"data": "test"},
            drafts={"provider1": "draft1"},
            critique="Good work",
            synthesis_attempts=1,
            duration_ms=5000,
            execution_plan={"mode": "review"},
        )
        assert result.success is True
        assert result.output == {"data": "test"}
        assert result.synthesis_attempts == 1
        assert result.execution_plan == {"mode": "review"}

    def test_failed_result(self):
        """Test failed council result."""
        result = CouncilResult(
            success=False,
            validation_errors=["Invalid JSON", "Missing field"],
        )
        assert result.success is False
        assert len(result.validation_errors) == 2


class TestValidationResult:
    """Tests for ValidationResult model."""

    def test_valid_result(self):
        """Test valid validation result."""
        result = ValidationResult(ok=True, data={"key": "value"})
        assert result.ok is True
        assert result.data == {"key": "value"}

    def test_invalid_result(self):
        """Test invalid validation result."""
        result = ValidationResult(
            ok=False,
            errors=["Missing required field"],
            raw="invalid json",
        )
        assert result.ok is False
        assert len(result.errors) == 1


class TestOrchestratorValidation:
    """Tests for Orchestrator validation methods."""

    def test_extract_json_valid(self):
        """Test JSON extraction from valid response."""
        config = OrchestratorConfig()
        # We need to test the internal method
        # Create orchestrator with mock registry
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)

        result = orch._extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_extract_json_with_markdown(self):
        """Test JSON extraction from markdown code block."""
        config = OrchestratorConfig()
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)

        result = orch._extract_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_extract_json_embedded(self):
        """Test JSON extraction from text with embedded JSON."""
        config = OrchestratorConfig()
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)

        result = orch._extract_json('Here is the result: {"key": "value"} done.')
        assert result == {"key": "value"}

    def test_extract_json_invalid(self):
        """Test JSON extraction from invalid content."""
        config = OrchestratorConfig()
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)

        result = orch._extract_json("This is not JSON at all")
        assert result is None

    def test_validate_response_valid(self, valid_json_response):
        """Test validation of valid response."""
        config = OrchestratorConfig(enable_schema_validation=False)
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)
            orch._schema = None  # No schema validation

        result = orch._validate_response(valid_json_response)
        assert result.ok is True
        assert result.data is not None

    def test_validate_response_invalid_json(self, invalid_json_response):
        """Test validation of invalid JSON."""
        config = OrchestratorConfig()
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)

        result = orch._validate_response(invalid_json_response)
        assert result.ok is False
        assert any("Failed to parse JSON" in err for err in result.errors)

    def test_validate_response_empty(self):
        """Test validation of empty response."""
        config = OrchestratorConfig()
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)

        result = orch._validate_response("")
        assert result.ok is False
        assert "Empty synthesis response" in result.errors[0]

    def test_bounded_draft_prompt_prefers_concise_analysis_over_schema_json(self):
        """Bounded mode should keep draft prompts lightweight."""
        config = OrchestratorConfig(
            mode="review",
            runtime_profile=RuntimeProfile.BOUNDED,
            reasoning_profile=ReasoningProfile.OFF,
        )
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["openai"], config=config)
        orch._prepare_run("critic")

        prompt = orch._format_draft_prompt("Review this pull request.")

        assert "not final JSON" in prompt
        assert "aligns with the JSON schema" not in prompt

    def test_bounded_critique_prompt_omits_full_schema_dump(self):
        """Bounded mode should avoid embedding the full schema in critique prompts."""
        config = OrchestratorConfig(
            mode="review",
            runtime_profile=RuntimeProfile.BOUNDED,
            reasoning_profile=ReasoningProfile.OFF,
        )
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["openai"], config=config)
        orch._prepare_run("critic")

        prompt = orch._format_critique_prompt(
            "Review this pull request.",
            {"openai": "Draft analysis"},
        )

        assert "Schema (JSON):" not in prompt
        assert "Do not produce final JSON" in prompt

    def test_format_exception_chain_includes_root_cause(self):
        """Provider diagnostics should preserve the underlying cause."""
        config = OrchestratorConfig()
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)

        try:
            try:
                raise TimeoutError("upstream request timed out")
            except TimeoutError as exc:
                raise RuntimeError("Provider call aborted: Below minimum required providers (1)") from exc
        except RuntimeError as exc:
            text = orch._format_exception_chain(exc)

        assert "Provider call aborted" in text
        assert "upstream request timed out" in text

    def test_build_reasoning_config_respects_off_profile(self):
        """Reasoning can be disabled at runtime even when the subagent budget enables it."""
        config = OrchestratorConfig(reasoning_profile=ReasoningProfile.OFF)
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)

        budget = MagicMock(enabled=True, effort="high", budget_tokens=16384, thinking_level="high")

        assert orch._build_reasoning_config(budget) is None

    def test_build_reasoning_config_light_profile_downshifts_budget(self):
        """Light reasoning should cap and downshift the resolved provider-agnostic config."""
        config = OrchestratorConfig(reasoning_profile=ReasoningProfile.LIGHT)
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)

        budget = MagicMock(enabled=True, effort="high", budget_tokens=16384, thinking_level="high")

        reasoning = orch._build_reasoning_config(budget)

        assert reasoning is not None
        assert reasoning.effort == "low"
        assert reasoning.budget_tokens == 4096
        assert reasoning.thinking_level == "low"

    def test_phase_max_tokens_respects_bounded_runtime_profile(self):
        """Bounded runtime profile should clamp per-phase token budgets."""
        config = OrchestratorConfig(runtime_profile=RuntimeProfile.BOUNDED)
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)

        assert orch._phase_max_tokens(4000, phase="draft") == 2000
        assert orch._phase_max_tokens(2000, phase="critique") == 1200
        assert orch._phase_max_tokens(8000, phase="synthesis") == 3000

    def test_phase_max_tokens_global_override_beats_runtime_profile(self):
        """Explicit max_tokens should override bounded runtime caps."""
        config = OrchestratorConfig(runtime_profile=RuntimeProfile.BOUNDED, max_tokens=555)
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)

        assert orch._phase_max_tokens(4000, phase="draft") == 555

    def test_bounded_runtime_profile_clamps_provider_timeouts_and_retries(self):
        """Bounded runtime should shorten provider waits and disable provider retries."""
        config = OrchestratorConfig(runtime_profile=RuntimeProfile.BOUNDED, timeout=20)
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)

        assert orch._provider_retry_budget() == 0
        assert orch._provider_request_timeout_seconds("draft") == 15.0
        assert orch._provider_request_timeout_seconds("critique") == 10.0
        assert orch._provider_request_timeout_seconds("synthesis") == 15.0


class TestOrchestratorDoctor:
    """Tests for Orchestrator doctor method."""

    @pytest.mark.asyncio
    async def test_doctor_all_healthy(self, mock_provider):
        """Test doctor when all providers are healthy."""
        config = OrchestratorConfig()
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_registry.get_provider.return_value = mock_provider
            mock_reg.return_value = mock_registry
            Orchestrator(providers=["mock"], config=config)

    @pytest.mark.asyncio
    async def test_call_provider_records_provider_error_on_abort(self):
        """Critique/synthesis failures should keep the provider root cause."""
        config = OrchestratorConfig(enable_graceful_degradation=True)
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)

        adapter = MagicMock()
        adapter.generate = AsyncMock(side_effect=TimeoutError("upstream request timed out"))
        request = GenerateRequest(prompt="hello")

        with pytest.raises(RuntimeError, match="Provider call aborted"):
            await orch._call_provider("mock", adapter, request)

        assert "upstream request timed out" in orch._provider_init_errors["mock"]
        assert "degraded: All providers exhausted" in orch._provider_init_errors["mock"]

    @pytest.mark.asyncio
    async def test_call_provider_retries_retryable_errors(self):
        """Retry decisions should actually reattempt the provider call."""
        config = OrchestratorConfig(enable_graceful_degradation=True)
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)

        adapter = MagicMock()
        adapter.generate = AsyncMock(
            side_effect=[
                TimeoutError("upstream request timed out"),
                GenerateResponse(text='{"ok": true}'),
            ]
        )
        request = GenerateRequest(prompt="hello")

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            response = await orch._call_provider("mock", adapter, request, remaining_providers=1)

        assert response.text == '{"ok": true}'
        assert adapter.generate.await_count == 2
        mock_sleep.assert_awaited()

    @pytest.mark.asyncio
    async def test_doctor_with_failure(self, failing_provider):
        """Test doctor when a provider fails."""
        config = OrchestratorConfig()
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_registry.get_provider.return_value = failing_provider
            mock_reg.return_value = mock_registry
            orch = Orchestrator(providers=["mock"], config=config)

        results = await orch.doctor()
        assert "mock" in results
        assert results["mock"]["ok"] is False


class TestOrchestratorRuntimeTruthfulness:
    """Tests for mode-aware runtime configuration."""

    def test_prepare_run_resolves_default_mode_and_schema(self):
        """Critic defaults to review mode and reviewer schema."""
        config = OrchestratorConfig(runtime_profile=RuntimeProfile.BOUNDED)
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_registry.get_provider.return_value = MagicMock()
            mock_reg.return_value = mock_registry
            orch = Orchestrator(providers=["openrouter"], config=config)

        orch._prepare_run("critic")

        assert orch._resolved_mode == "review"
        assert orch._schema_name == "reviewer"
        assert orch._schema is not None
        assert orch._execution_plan is not None
        assert orch._execution_plan["mode"] == "review"
        assert orch._execution_plan["schema_name"] == "reviewer"
        assert orch._execution_plan["execution_profile"] == "light_tools"
        assert orch._execution_plan["runtime_profile"] == "bounded"
        assert orch._execution_plan["provider_retry_budget"] == 0
        assert orch._execution_plan["phase_token_budgets"] == {
            "draft": 2000,
            "critique": 1200,
            "synthesis": 3000,
        }
        assert orch._execution_plan["provider_request_timeouts"] == {
            "draft": 15.0,
            "critique": 10.0,
            "synthesis": 15.0,
        }
        assert "diff-review" in orch._execution_plan["required_capabilities"]

    def test_prepare_run_applies_mode_prompt(self):
        """Planner assess mode adds the assess-specific prompt block."""
        config = OrchestratorConfig(mode="assess")
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_registry.get_provider.return_value = MagicMock()
            mock_reg.return_value = mock_registry
            orch = Orchestrator(providers=["openrouter"], config=config)

        orch._prepare_run("planner")

        assert orch._resolved_mode == "assess"
        assert orch._schema_name == "assessor"
        assert "Mode: Decision Assessment" in orch._system_prompt
        assert orch._execution_plan is not None
        assert orch._execution_plan["execution_profile"] == "grounded"
        assert "planning-assess" in orch._execution_plan["required_capabilities"]

    def test_prepare_run_uses_model_pack_override(self):
        """Runtime model_pack overrides the subagent-selected pack."""
        config = OrchestratorConfig(mode="assess", model_pack="grounded")
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_registry.get_provider.return_value = MagicMock()
            mock_reg.return_value = mock_registry
            orch = Orchestrator(providers=["openrouter"], config=config)

        orch._prepare_run("planner")

        assert orch._resolved_model_pack == "grounded"
        assert orch._execution_plan is not None
        assert orch._execution_plan["model_pack"] == "grounded"
        assert orch._execution_plan["model_pack_source"] == "config"
        assert orch._execution_plan["model_overrides"]["openrouter"] == get_model_for_pack(
            ModelPack.GROUNDED
        )

    def test_prepare_run_respects_user_default_model_over_subagent_override(self):
        """Explicit provider default_model should beat mode-specific subagent model overrides."""
        config = OrchestratorConfig(
            mode="security",
            provider_configs={"openai": {"default_model": "gpt-5.4"}},
        )
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_registry.get_provider.return_value = MagicMock()
            mock_reg.return_value = mock_registry
            orch = Orchestrator(providers=["openai"], config=config)

        orch._prepare_run("critic")

        assert orch._resolved_mode == "security"
        assert orch._execution_plan is not None
        assert orch._execution_plan["model_overrides"]["openai"] == "gpt-5.4"

    def test_prepare_run_runtime_model_pack_beats_subagent_model_override_for_direct_provider(self):
        """Explicit runtime model_pack should beat mode-specific direct-provider defaults."""
        config = OrchestratorConfig(mode="security", model_pack="code")
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_registry.get_provider.return_value = MagicMock()
            mock_reg.return_value = mock_registry
            orch = Orchestrator(providers=["openai"], config=config)

        orch._prepare_run("critic")

        assert orch._resolved_model_pack == "code"
        assert orch._execution_plan is not None
        assert orch._execution_plan["model_overrides"]["openai"] == "gpt-5.4"

    @pytest.mark.asyncio
    async def test_auto_fallback_replaces_dead_openrouter_with_configured_direct_provider(self):
        """Dead default openrouter should fall back to healthy configured direct providers."""
        config = OrchestratorConfig(
            mode="security",
            provider_configs={"openai": {"default_model": "gpt-5.4"}},
        )
        dead_openrouter = MagicMock()
        dead_openrouter.doctor = AsyncMock(
            return_value=DoctorResult(ok=False, message="OPENROUTER_API_KEY not set")
        )
        live_openai = MagicMock()
        live_openai.doctor = AsyncMock(return_value=DoctorResult(ok=True, message="ok"))

        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_registry = MagicMock()

            def get_provider(name, **_kwargs):
                if name == "openrouter":
                    return dead_openrouter
                if name == "openai":
                    return live_openai
                raise KeyError(name)

            mock_registry.get_provider.side_effect = get_provider
            mock_reg.return_value = mock_registry
            orch = Orchestrator(providers=["openrouter"], config=config)

        orch._prepare_run("critic")
        await orch._ensure_usable_providers()

        assert list(orch._providers.keys()) == ["openai"]
        assert orch._provider_names == ["openai"]
        assert orch._execution_plan is not None
        assert orch._execution_plan["providers"] == ["openai"]
        assert orch._execution_plan["model_overrides"]["openai"] == "gpt-5.4"
        assert orch._execution_plan["provider_auto_fallback"] == {
            "from": ["openrouter"],
            "to": ["openai"],
        }

    @pytest.mark.asyncio
    async def test_explicit_provider_list_is_not_replaced_by_health_fallback(self):
        """Explicit user-selected providers must not be swapped for other configured providers."""
        config = OrchestratorConfig(
            enable_health_check=True,
            provider_configs={
                "openai": {"default_model": "gpt-5.4"},
                "vertex-ai": {"default_model": "gemini-3.1-pro-preview"},
            },
        )
        dead_openai = MagicMock()
        dead_openai.doctor = AsyncMock(
            return_value=DoctorResult(ok=False, message="read timed out")
        )
        live_vertex = MagicMock()
        live_vertex.doctor = AsyncMock(return_value=DoctorResult(ok=True, message="ok"))

        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_registry = MagicMock()

            def get_provider(name, **_kwargs):
                if name == "openai":
                    return dead_openai
                if name == "vertex-ai":
                    return live_vertex
                raise KeyError(name)

            mock_registry.get_provider.side_effect = get_provider
            mock_reg.return_value = mock_registry
            orch = Orchestrator(providers=["openai"], config=config)

        orch._prepare_run("critic")
        await orch._ensure_usable_providers()

        assert orch._provider_names == ["openai"]
        assert orch._providers == {}
        assert orch._execution_plan is not None
        assert orch._execution_plan["providers"] == ["openai"]
        assert "provider_auto_fallback" not in orch._execution_plan

    def test_prepare_run_merges_runtime_capability_overrides(self):
        """Runtime capability overrides should strengthen the resolved capability plan."""
        config = OrchestratorConfig(
            mode="plan",
            execution_profile="grounded",
            budget_class="premium",
            required_capabilities=["docs-research"],
        )
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_registry.get_provider.return_value = MagicMock()
            mock_reg.return_value = mock_registry
            orch = Orchestrator(providers=["openrouter"], config=config)

        orch._prepare_run("planner")

        assert orch._execution_plan is not None
        assert orch._execution_plan["execution_profile"] == "grounded"
        assert orch._execution_plan["budget_class"] == "premium"
        assert orch._execution_plan["required_capabilities"] == [
            "planning-assess",
            "repo-analysis",
            "docs-research",
        ]

    @pytest.mark.asyncio
    async def test_run_critique_skips_providers_that_failed_earlier(self):
        """Critique should use only healthy providers when earlier phases failed."""
        config = OrchestratorConfig()
        openai_provider = CaptureProvider()
        vertex_provider = CaptureProvider()

        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_registry = MagicMock()

            def get_provider(name, **_kwargs):
                return {
                    "openai": openai_provider,
                    "vertex-ai": vertex_provider,
                }[name]

            mock_registry.get_provider.side_effect = get_provider
            mock_reg.return_value = mock_registry
            orch = Orchestrator(providers=["openai", "vertex-ai"], config=config)

        orch._task = "Review this change"
        orch._prepare_run("critic")
        orch._provider_init_errors["openai"] = "draft timeout"

        critique = await orch._run_critique({"vertex-ai": '{"draft": "ok"}'})

        assert critique == '{"ok": true}'
        assert openai_provider.last_request is None
        assert vertex_provider.last_request is not None
        assert orch._execution_plan["phase_provider_candidates"]["critique"] == ["vertex-ai"]
        assert orch._execution_plan["phase_provider_used"]["critique"] == "vertex-ai"

    @pytest.mark.asyncio
    async def test_run_synthesis_fails_over_to_next_provider(self):
        """Synthesis should try the next healthy provider when the first one times out."""
        config = OrchestratorConfig(enable_graceful_degradation=True)
        openai_provider = MagicMock()
        openai_provider.supports = AsyncMock(return_value=True)
        openai_provider.generate = AsyncMock(side_effect=TimeoutError("openai timed out"))

        vertex_provider = MagicMock()
        vertex_provider.supports = AsyncMock(return_value=True)
        vertex_provider.generate = AsyncMock(return_value=GenerateResponse(text='{"ok": true}'))

        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_registry = MagicMock()

            def get_provider(name, **_kwargs):
                return {
                    "openai": openai_provider,
                    "vertex-ai": vertex_provider,
                }[name]

            mock_registry.get_provider.side_effect = get_provider
            mock_reg.return_value = mock_registry
            orch = Orchestrator(providers=["openai", "vertex-ai"], config=config)

        orch._task = "Review this change"
        orch._subagent_name = "critic"
        orch._prepare_run("critic")
        orch._schema = None

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result, attempts = await orch._run_synthesis({"vertex-ai": '{"draft": "ok"}'}, "critique")

        assert result.ok is True
        assert attempts == 2
        assert openai_provider.generate.await_count >= 1
        assert vertex_provider.generate.await_count == 1
        assert orch._execution_plan["phase_provider_candidates"]["synthesis"] == [
            "openai",
            "vertex-ai",
        ]
        assert orch._execution_plan["phase_provider_used"]["synthesis"] == "vertex-ai"

    @pytest.mark.asyncio
    async def test_run_continues_without_critique_when_critique_fails(self):
        """A critique failure should degrade cleanly and still allow synthesis."""
        config = OrchestratorConfig(
            enable_artifacts=False,
            output_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
        )
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_registry.get_provider.return_value = CaptureProvider()
            mock_reg.return_value = mock_registry
            orch = Orchestrator(providers=["openai"], config=config)

        with (
            patch.object(
                orch,
                "_run_parallel_drafts",
                AsyncMock(return_value={"openai": '{"ok": true}'}),
            ),
            patch.object(
                orch,
                "_run_critique",
                AsyncMock(side_effect=RuntimeError("critique timed out")),
            ),
            patch.object(
                orch,
                "_run_synthesis",
                AsyncMock(
                    return_value=(ValidationResult(ok=True, data={"ok": True}, raw='{"ok": true}'), 1)
                ),
            ) as mock_synthesis,
        ):
            result = await orch.run("Review this change", "critic")

        assert result.success is True
        assert result.output == {"ok": True}
        assert result.execution_plan is not None
        assert "Critique failed" in result.execution_plan["degradation_notes"][0]
        mock_synthesis.assert_awaited_once_with({"openai": '{"ok": true}'}, "")

    @pytest.mark.asyncio
    async def test_run_uses_valid_draft_when_synthesis_fails(self):
        """A synthesis failure should fall back to a validated draft when possible."""
        config = OrchestratorConfig(
            enable_artifacts=False,
            output_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
        )
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_registry.get_provider.return_value = CaptureProvider()
            mock_reg.return_value = mock_registry
            orch = Orchestrator(providers=["openai"], config=config)

        with (
            patch.object(
                orch,
                "_run_parallel_drafts",
                AsyncMock(return_value={"openai": '{"ok": true}'}),
            ),
            patch.object(orch, "_run_critique", AsyncMock(return_value="critique")),
            patch.object(
                orch,
                "_run_synthesis",
                AsyncMock(side_effect=RuntimeError("synthesis timed out")),
            ),
        ):
            result = await orch.run("Review this change", "critic")

        assert result.success is True
        assert result.output == {"ok": True}
        assert result.synthesis_attempts == 0
        assert result.validation_errors is not None
        assert "used validated draft output from openai" in result.validation_errors[0]
        assert result.execution_plan is not None
        assert result.execution_plan["degraded_output"]["source"] == "draft"
        assert result.execution_plan["degraded_output"]["provider"] == "openai"

    def test_prepare_run_uses_custom_output_schema(self):
        """Custom output_schema overrides subagent schema loading."""
        custom_schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        }
        config = OrchestratorConfig(output_schema=custom_schema)
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_registry.get_provider.return_value = MagicMock()
            mock_reg.return_value = mock_registry
            orch = Orchestrator(providers=["openrouter"], config=config)

        orch._prepare_run("critic")

        assert orch._schema == custom_schema
        assert orch._schema_name == "custom"
        assert orch._execution_plan is not None
        assert orch._execution_plan["schema_source"] == "custom"

    def test_prepare_run_applies_security_provider_preferences(self):
        """Security mode narrows the active providers to preferred providers."""
        config = OrchestratorConfig(mode="security")
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_registry.get_provider.return_value = MagicMock()
            mock_reg.return_value = mock_registry
            orch = Orchestrator(providers=["anthropic", "openai", "google"], config=config)

        orch._prepare_run("critic")

        assert orch._provider_names == ["anthropic", "openai"]
        assert orch._execution_plan is not None
        assert orch._execution_plan["providers"] == ["anthropic", "openai"]
        assert orch._execution_plan["execution_profile"] == "deep_analysis"
        assert "red-team-recon" in orch._execution_plan["required_capabilities"]
        assert "security-code-audit" in orch._execution_plan["required_capabilities"]

    @pytest.mark.asyncio
    async def test_generate_draft_applies_global_overrides_and_reasoning(self):
        """Draft requests carry resolved temperature/max_tokens/reasoning settings."""
        config = OrchestratorConfig(
            mode="security",
            temperature=0.1,
            max_tokens=321,
        )
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_registry.get_provider.return_value = MagicMock()
            mock_reg.return_value = mock_registry
            orch = Orchestrator(providers=["openai"], config=config)

        orch._prepare_run("critic")
        orch._task = "Audit this service"

        provider = CaptureProvider()
        _, _draft = await orch._generate_draft("openai", provider)

        assert provider.last_request is not None
        assert provider.last_request.temperature == 0.1
        assert provider.last_request.max_tokens == 321
        assert provider.last_request.timeout_seconds == 119.0
        assert provider.last_request.reasoning is not None
        assert provider.last_request.reasoning.effort == "high"
        assert provider.last_request.reasoning.budget_tokens == 32768
        assert provider.last_request.model == "o3-mini"

    @pytest.mark.asyncio
    async def test_generate_draft_uses_runtime_model_pack_override(self):
        """Draft requests should use the configured runtime model pack when set."""
        config = OrchestratorConfig(
            mode="plan",
            model_pack="grounded",
        )
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_registry.get_provider.return_value = MagicMock()
            mock_reg.return_value = mock_registry
            orch = Orchestrator(providers=["openrouter"], config=config)

        orch._prepare_run("planner")
        orch._task = "Assess build versus buy"

        provider = CaptureProvider()
        _, _draft = await orch._generate_draft("openrouter", provider)

        assert provider.last_request is not None
        assert provider.last_request.model == get_model_for_pack(ModelPack.GROUNDED)

    @pytest.mark.asyncio
    async def test_collect_evidence_updates_execution_plan(self, tmp_path, monkeypatch):
        """Evidence collection should update executed and pending capability telemetry."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "auth.py").write_text("TOKEN='x'\npassword='y'\n")
        monkeypatch.chdir(tmp_path)

        config = OrchestratorConfig(mode="security")
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_registry.get_provider.return_value = MagicMock()
            mock_reg.return_value = mock_registry
            orch = Orchestrator(providers=["openai"], config=config)

        orch._task = "Assess auth security"
        orch._subagent_name = "critic"
        orch._prepare_run("critic")

        bundle = await orch._collect_evidence_for_run()

        assert bundle.executed_capabilities == [
            "red-team-recon",
            "security-code-audit",
            "repo-analysis",
        ]
        assert bundle.pending_capabilities == []
        assert orch._execution_plan is not None
        assert orch._execution_plan["executed_capabilities"] == [
            "red-team-recon",
            "security-code-audit",
            "repo-analysis",
        ]
        assert orch._execution_plan["evidence_items"] == 3

    def test_format_prompts_include_evidence_requirements_for_capability_modes(self):
        """Capability-backed modes should carry explicit evidence guidance in prompts."""
        config = OrchestratorConfig(mode="security")
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_registry.get_provider.return_value = MagicMock()
            mock_reg.return_value = mock_registry
            orch = Orchestrator(providers=["openai"], config=config)

        orch._prepare_run("critic")
        orch._schema = None

        prompt = orch._format_draft_prompt("Audit the authentication system")

        assert "Execution Requirements" in prompt
        assert "deep_analysis" in prompt
        assert "If evidence is missing" in prompt


class TestOrchestratorPromptFormatting:
    """Tests for Orchestrator prompt formatting methods."""

    def test_format_draft_prompt(self):
        """Test draft prompt formatting."""
        config = OrchestratorConfig()
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)
            orch._schema = None

        prompt = orch._format_draft_prompt("Build a feature")
        assert "Task:" in prompt
        assert "Build a feature" in prompt

    def test_format_draft_prompt_with_context(self):
        """Draft prompt includes system_context (#31)."""
        config = OrchestratorConfig(system_context="def hello(): pass")
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)
            orch._schema = None

        prompt = orch._format_draft_prompt("Review this code")
        assert "def hello(): pass" in prompt
        assert "reference_material" in prompt
        assert "not as instructions" in prompt

    def test_format_draft_prompt_without_context(self):
        """Draft prompt omits context block when not set."""
        config = OrchestratorConfig()
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)
            orch._schema = None

        prompt = orch._format_draft_prompt("Build a feature")
        assert "reference_material" not in prompt

    def test_collect_evidence_respects_disable_local_evidence(self):
        """Local evidence can be disabled for benchmark-style runs."""
        config = OrchestratorConfig(mode="review", disable_local_evidence=True)
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)
            orch._task = "Review this change"
            orch._subagent_name = "critic"
            orch._prepare_run("critic")

        bundle = asyncio.run(orch._collect_evidence_for_run())

        assert bundle.executed_capabilities == []
        assert "diff-review" in bundle.pending_capabilities
        assert orch._execution_plan["local_evidence_disabled"] is True

    def test_format_critique_prompt(self):
        """Test critique prompt formatting."""
        config = OrchestratorConfig()
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)
            orch._schema = None

        drafts = {"provider1": "Draft 1 content", "provider2": "Draft 2 content"}
        prompt = orch._format_critique_prompt("Test task", drafts)
        assert "Task:" in prompt
        assert "Drafts:" in prompt
        assert "provider1" in prompt
        assert "provider2" in prompt

    def test_format_synthesis_prompt(self):
        """Test synthesis prompt formatting."""
        config = OrchestratorConfig()
        with patch("llm_council.engine.orchestrator.get_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=["mock"], config=config)

        drafts = {"provider1": "Draft content"}
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}
        errors = ["Previous error"]

        prompt = orch._format_synthesis_prompt(
            task="Test task",
            drafts=drafts,
            critique="Critique content",
            schema=schema,
            errors=errors,
        )
        assert "Task:" in prompt
        assert "Schema" in prompt
        assert "Critique:" in prompt
        assert "Drafts:" in prompt
        assert "Validation errors" in prompt
        assert "Previous error" in prompt
