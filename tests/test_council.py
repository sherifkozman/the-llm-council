"""Tests for the Council facade class."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from llm_council import Council, CouncilConfig, CouncilResult
from llm_council.protocol.types import ReasoningProfile, RuntimeProfile


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
            providers=["gemini"],
            timeout=60,
            max_retries=5,
        )
        with patch("llm_council.council.Orchestrator"):
            council = Council(config=config)
            assert council.providers == ["gemini"]
            assert council.config.timeout == 60

    def test_init_provider_argument_overrides_config_providers(self):
        """Explicit providers should become the effective provider list."""
        config = CouncilConfig(providers=["gemini"])
        with patch("llm_council.council.Orchestrator"):
            council = Council(providers=["openai"], config=config)

        assert council.providers == ["openai"]

    def test_init_forwards_runtime_truthfulness_fields(self):
        """Council forwards mode and request override fields to the orchestrator."""
        config = CouncilConfig(
            providers=["openrouter"],
            mode="security",
            model_pack="grounded",
            model_overrides={"openai": "gpt-5.4"},
            execution_profile="deep_analysis",
            budget_class="premium",
            required_capabilities=["security-audit"],
            disable_local_evidence=True,
            temperature=0.1,
            max_tokens=777,
            runtime_profile=RuntimeProfile.BOUNDED,
            reasoning_profile=ReasoningProfile.LIGHT,
            output_schema={"type": "object"},
            system_context="repo context",
        )

        with patch("llm_council.council.Orchestrator") as mock_orch_class:
            council = Council(config=config)

            orch_config = mock_orch_class.call_args.kwargs["config"]
            assert orch_config.mode == "security"
            assert orch_config.model_pack == "grounded"
            assert orch_config.model_overrides == {"openai": "gpt-5.4"}
            assert orch_config.execution_profile == "deep_analysis"
            assert orch_config.budget_class == "premium"
            assert orch_config.required_capabilities == ["security-audit"]
            assert orch_config.disable_local_evidence is True
            assert orch_config.temperature == 0.1
            assert orch_config.max_tokens == 777
            assert orch_config.runtime_profile == RuntimeProfile.BOUNDED
            assert orch_config.reasoning_profile == ReasoningProfile.LIGHT
            assert orch_config.output_schema == {"type": "object"}
            assert orch_config.system_context == "repo context"
            assert council.config.follow_router is False


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

    @pytest.mark.asyncio
    async def test_run_with_follow_router_executes_routed_subagent(self):
        """A router run can be followed by the selected subagent and mode."""
        router_result = CouncilResult(
            success=True,
            output={
                "task_type": "planning",
                "risk_level": "high",
                "subagent_to_run": "planner",
                "mode": "assess",
                "reasoning": "Assessment is the best fit for this decision task.",
                "model_pack": "deep_reasoner",
                "execution_profile": "grounded",
                "budget_class": "premium",
                "required_capabilities": ["planning-assess", "docs-research"],
            },
            execution_plan={"mode": None, "execution_profile": "prompt_only"},
        )
        routed_result = CouncilResult(
            success=True,
            output={"recommendation": "proceed"},
            execution_plan={"mode": "assess", "execution_profile": "grounded"},
        )

        with patch("llm_council.council.Orchestrator") as mock_orch_class:
            router_orch = AsyncMock()
            router_orch.run.return_value = router_result
            routed_orch = AsyncMock()
            routed_orch.run.return_value = routed_result
            mock_orch_class.side_effect = [router_orch, routed_orch]

            council = Council(
                config=CouncilConfig(
                    providers=["openrouter"],
                    follow_router=True,
                )
            )
            result = await council.run(task="Should we build or buy SSO?", subagent="router")

            router_orch.run.assert_awaited_once_with(
                task="Should we build or buy SSO?",
                subagent="router",
            )
            routed_orch.run.assert_awaited_once_with(
                task="Should we build or buy SSO?",
                subagent="planner",
            )
            assert mock_orch_class.call_args_list[1].kwargs["config"].mode == "assess"
            assert mock_orch_class.call_args_list[1].kwargs["config"].model_pack == "deep_reasoner"
            assert (
                mock_orch_class.call_args_list[1].kwargs["config"].execution_profile == "grounded"
            )
            assert mock_orch_class.call_args_list[1].kwargs["config"].budget_class == "premium"
            assert mock_orch_class.call_args_list[1].kwargs["config"].required_capabilities == [
                "planning-assess",
                "docs-research",
            ]
            assert result.routed is True
            assert result.routing_decision is not None
            assert result.routing_decision["subagent_to_run"] == "planner"
            assert result.execution_plan is not None
            assert result.execution_plan["routed_via_router"] is True
            assert result.execution_plan["routing_subagent"] == "planner"

    @pytest.mark.asyncio
    async def test_run_with_follow_router_requires_router_subagent(self):
        """follow_router is invalid when the initial subagent is not router."""
        with patch("llm_council.council.Orchestrator"):
            council = Council(config=CouncilConfig(providers=["openrouter"], follow_router=True))

        with pytest.raises(ValueError, match="follow_router"):
            await council.run(task="Plan this work", subagent="planner")

    @pytest.mark.asyncio
    async def test_run_with_follow_router_returns_router_result_when_no_followup(self):
        """If the router does not pick a follow-up subagent, the router result is returned."""
        router_result = CouncilResult(
            success=True,
            output={
                "task_type": "shipping",
                "risk_level": "low",
                "subagent_to_run": "router",
                "reasoning": "No follow-up subagent required.",
            },
        )

        with patch("llm_council.council.Orchestrator") as mock_orch_class:
            router_orch = AsyncMock()
            router_orch.run.return_value = router_result
            mock_orch_class.return_value = router_orch

            council = Council(config=CouncilConfig(providers=["openrouter"], follow_router=True))
            result = await council.run(task="Classify this task", subagent="router")

            assert result is router_result
            assert mock_orch_class.call_count == 1


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
