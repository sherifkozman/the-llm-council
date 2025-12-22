"""Tests for the Council facade class."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from llm_council import Council, CouncilConfig, CouncilResult


class TestCouncilInit:
    """Tests for Council initialization."""

    def test_default_init(self):
        """Test Council with default options."""
        with patch("llm_council.council.Orchestrator"):
            council = Council()
            assert council.providers == ["openrouter"]

    def test_init_with_providers(self):
        """Test Council with custom providers."""
        with patch("llm_council.council.Orchestrator"):
            council = Council(providers=["anthropic", "openai"])
            assert council.providers == ["anthropic", "openai"]

    def test_init_with_config(self):
        """Test Council with config object."""
        config = CouncilConfig(
            providers=["google"],
            timeout=60,
            max_retries=5,
        )
        with patch("llm_council.council.Orchestrator"):
            council = Council(config=config)
            assert council.providers == ["google"]
            assert council.config.timeout == 60


class TestCouncilRun:
    """Tests for Council.run() method."""

    @pytest.mark.asyncio
    async def test_run_basic(self):
        """Test basic council run."""
        mock_result = CouncilResult(
            success=True,
            output={"result": "test"},
            drafts={"mock": "draft"},
            synthesis_attempts=1,
            duration_ms=1000,
        )

        with patch("llm_council.council.Orchestrator") as mock_orch_class:
            mock_orch = AsyncMock()
            mock_orch.run.return_value = mock_result
            mock_orch_class.return_value = mock_orch

            council = Council(providers=["mock"])
            result = await council.run(task="Test task", subagent="router")

            assert result.success is True
            assert result.output == {"result": "test"}
            mock_orch.run.assert_called_once_with(task="Test task", subagent="router")

    @pytest.mark.asyncio
    async def test_run_with_subagent(self):
        """Test council run with specific subagent."""
        mock_result = CouncilResult(success=True, output={})

        with patch("llm_council.council.Orchestrator") as mock_orch_class:
            mock_orch = AsyncMock()
            mock_orch.run.return_value = mock_result
            mock_orch_class.return_value = mock_orch

            council = Council(providers=["mock"])
            await council.run(task="Implement feature", subagent="implementer")

            mock_orch.run.assert_called_with(task="Implement feature", subagent="implementer")

    @pytest.mark.asyncio
    async def test_run_failure(self):
        """Test council run that fails."""
        mock_result = CouncilResult(
            success=False,
            validation_errors=["Schema validation failed"],
        )

        with patch("llm_council.council.Orchestrator") as mock_orch_class:
            mock_orch = AsyncMock()
            mock_orch.run.return_value = mock_result
            mock_orch_class.return_value = mock_orch

            council = Council(providers=["mock"])
            result = await council.run(task="Bad task", subagent="router")

            assert result.success is False
            assert "Schema validation failed" in result.validation_errors


class TestCouncilDoctor:
    """Tests for Council.doctor() method."""

    @pytest.mark.asyncio
    async def test_doctor(self):
        """Test doctor health check."""
        mock_health = {
            "mock": {"ok": True, "message": "Healthy", "latency_ms": 50},
        }

        with patch("llm_council.council.Orchestrator") as mock_orch_class:
            mock_orch = AsyncMock()
            mock_orch.doctor.return_value = mock_health
            mock_orch_class.return_value = mock_orch

            council = Council(providers=["mock"])
            result = await council.doctor()

            assert "mock" in result
            assert result["mock"]["ok"] is True


class TestCouncilAvailableSubagents:
    """Tests for Council.available_subagents() method."""

    def test_available_subagents(self):
        """Test listing available subagents."""
        subagents = Council.available_subagents()
        assert isinstance(subagents, list)
        assert "router" in subagents
        assert "planner" in subagents
        assert "implementer" in subagents
        assert "reviewer" in subagents
        assert "architect" in subagents

    def test_subagent_count(self):
        """Test that we have expected number of subagents."""
        subagents = Council.available_subagents()
        assert len(subagents) >= 10  # At least 10 subagents


class TestCouncilConfig:
    """Tests for CouncilConfig model."""

    def test_default_config(self):
        """Test default config values."""
        config = CouncilConfig()
        assert config.providers == ["openrouter"]
        assert config.timeout == 120
        assert config.max_retries == 3

    def test_custom_config(self):
        """Test custom config values."""
        config = CouncilConfig(
            providers=["anthropic", "openai"],
            timeout=60,
            max_retries=5,
        )
        assert config.providers == ["anthropic", "openai"]
        assert config.timeout == 60
        assert config.max_retries == 5
