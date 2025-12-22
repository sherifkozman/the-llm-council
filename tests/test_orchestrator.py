"""Tests for the Orchestrator engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_council.engine.orchestrator import (
    CostEstimate,
    CouncilResult,
    Orchestrator,
    OrchestratorConfig,
    ValidationResult,
)


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
        )
        assert result.success is True
        assert result.output == {"data": "test"}
        assert result.synthesis_attempts == 1

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
            orch = Orchestrator(providers=["mock"], config=config)

        results = await orch.doctor()
        assert "mock" in results
        assert results["mock"]["ok"] is True

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
