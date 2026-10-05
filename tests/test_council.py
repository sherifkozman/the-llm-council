"""Tests for the Council facade class."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from llm_council import Council, CouncilConfig, CouncilResult
from llm_council.engine.orchestrator import _CouncilRunCancelled
from llm_council.protocol.types import ReasoningProfile, RuntimeProfile
from llm_council.providers.registry import ProviderRegistry
from llm_council.storage.artifacts import ArtifactStore


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


@pytest.mark.asyncio
@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize(
    "requested,expected",
    [([], "completed"), (["diff-review"], "degraded"), (["repo-analysis"], "completed")],
)
async def test_whole_file_capability_requirements_provenance_and_ledger(
    mock_registry: ProviderRegistry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    routed: bool,
    requested: list[str],
    expected: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "retry client.py"
    content = "def review_me():\n    return 0\n"
    source.write_text(content)
    file_metadata = {
        "path": str(source),
        "status": "included",
        "original_chars": len(content),
        "retained_chars": len(content),
        "truncated": False,
    }
    store = ArtifactStore(artifact_dir=tmp_path / "artifacts", db_path=tmp_path / "ledger.db")
    with (
        patch("llm_council.engine.orchestrator.get_registry", return_value=mock_registry),
        patch("llm_council.engine.orchestrator.get_store", return_value=store),
    ):
        council = Council(
            config=CouncilConfig(
                providers=["mock"],
                mode="review",
                required_capabilities=[] if routed else requested,
                enable_artifact_store=True,
                enable_health_check=False,
                enable_graceful_degradation=False,
                output_schema={"type": "object"},
                system_context=f"=== FILE: {source} ===\n{content}=== END: {source} ===",
                context_metadata={"files": [file_metadata]},
            )
        )
        expected_rows = {}
        if routed:
            router_id = store.create_run(subagent="router", task="Review the supplied file").run_id
            store.complete_run(router_id, status="completed")
            expected_rows[router_id] = "completed"
            router_result = CouncilResult(
                success=True,
                output={
                    "subagent_to_run": "critic",
                    "mode": "review",
                    "required_capabilities": requested,
                },
                execution_plan={"required_phases": []},
                run_id=router_id,
            )
            with patch.object(council._orchestrator, "run", AsyncMock(return_value=router_result)):
                result = await council.run("Review the supplied file", "router", follow_router=True)
        else:
            result = await council.run("Review the supplied file", "critic")

    assert result.success is True
    assert result.execution_status == expected
    assert result.output == {"result": "mock response"}
    assert result.drafts == {"mock": '{"result": "mock response"}'}
    assert result.critique == '{"result": "mock response"}'
    assert result.provider_errors is None
    plan = result.execution_plan
    assert plan is not None
    assert plan["requested_capabilities"] == requested
    assert plan["required_capabilities"] == ["diff-review", "repo-analysis"]
    assert plan["executed_capabilities"] == ["repo-analysis"]
    assert plan["pending_capabilities"] == ["diff-review"]
    assert plan["context_preparation"]["files"] == [file_metadata]
    if routed:
        assert plan["routing_required_capabilities"] == requested
        assert plan["routed_execution_status"] == expected
        assert result.routing_execution_plan is not None
        assert result.routing_execution_plan["run_id"] == router_id
        assert result.routing_execution_plan["execution_status"] == "completed"
    assert result.run_id is not None
    expected_rows[result.run_id] = expected
    with store._get_conn() as conn:
        assert dict(conn.execute("SELECT run_id, status FROM runs").fetchall()) == expected_rows


@pytest.mark.asyncio
@pytest.mark.parametrize("router_degraded", [False, True])
@pytest.mark.parametrize("follow_status", ["completed", "degraded", "failed", "cancelled"])
async def test_routing_preserves_execution_status_and_ledger(
    mock_registry, tmp_path, router_degraded, follow_status
):
    store = ArtifactStore(artifact_dir=tmp_path / "artifacts", db_path=tmp_path / "ledger.db")
    with patch("llm_council.engine.orchestrator.get_registry", return_value=mock_registry):
        council = Council(config=CouncilConfig(providers=["mock"], enable_artifact_store=False))
        routed = council._build_orchestrator(council.config)
    routed._artifact_store = store
    routed._run_id = store.create_run(subagent="planner", task="synthetic").run_id
    router_id = store.create_run(subagent="router", task="synthetic").run_id
    router = CouncilResult(
        success=True,
        output={"subagent_to_run": "planner", "mode": "plan"},
        execution_plan={"required_phases": [], "degraded_output": router_degraded},
        run_id=router_id,
        provider_errors={"mock": "lost router draft"} if router_degraded else None,
    )
    store.complete_run(router_id, status=router.execution_status)
    fallback = {"source": "evidence_fallback", "provider": "mock", "reason": "schema failure"}
    follow = CouncilResult(
        success=follow_status in {"completed", "degraded"},
        execution_status=follow_status,
        execution_plan={
            "required_phases": [],
            "degraded_output": fallback if follow_status == "degraded" else False,
        },
        output={"recommendation": "review"},
    )
    with (
        patch.object(council, "_build_orchestrator", return_value=routed),
        patch.object(routed, "_run", AsyncMock(return_value=follow)),
    ):
        result = await council._run_router_follow_up("synthetic", router)
    expected = "degraded" if router_degraded and follow_status == "completed" else follow_status
    assert result.execution_status == expected
    assert result.routing_execution_plan["execution_status"] == router.execution_status
    assert result.routing_execution_plan["run_id"] == router_id
    assert result.routing_execution_plan["provider_errors"] == router.provider_errors
    assert result.execution_plan["routed_execution_status"] == follow_status
    if follow_status == "degraded":
        assert result.execution_plan["degraded_output"] == fallback
    with store._get_conn() as conn:
        row = conn.execute("SELECT status FROM runs WHERE run_id = ?", (result.run_id,)).fetchone()
        router_row = conn.execute(
            "SELECT status FROM runs WHERE run_id = ?", (router_id,)
        ).fetchone()
    assert row[0] == expected
    assert router_row[0] == router.execution_status


@pytest.mark.asyncio
async def test_routed_task_cancellation_retains_router_evidence(mock_registry, tmp_path):
    store = ArtifactStore(artifact_dir=tmp_path / "artifacts", db_path=tmp_path / "ledger.db")
    with patch("llm_council.engine.orchestrator.get_registry", return_value=mock_registry):
        council = Council(config=CouncilConfig(providers=["mock"], enable_artifact_store=False))
        child = council._build_orchestrator(council.config)
    child._artifact_store = store
    child._run_id = store.create_run(subagent="planner", task="synthetic").run_id
    router_id = store.create_run(subagent="router", task="synthetic").run_id
    store.complete_run(router_id, status="degraded")
    router = CouncilResult(
        success=True,
        output={"subagent_to_run": "planner", "mode": "plan"},
        execution_plan={"required_phases": [], "degraded_output": True},
        provider_errors={"mock": "missing draft"},
        run_id=router_id,
    )
    started = asyncio.Event()
    captured = []

    async def blocked_run(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    async def caller():
        try:
            await council._run_router_follow_up("synthetic", router)
        except _CouncilRunCancelled as exc:
            captured.append(exc.result)
            raise

    with (
        patch.object(council, "_build_orchestrator", return_value=child),
        patch.object(child, "_run", blocked_run),
    ):
        task = asyncio.create_task(caller())
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    result = captured[0]
    assert result.execution_status == "cancelled"
    assert result.routed is True
    assert result.routing_decision == router.output
    assert result.routing_execution_plan["run_id"] == router_id
    assert result.routing_execution_plan["execution_status"] == "degraded"
    assert result.routing_execution_plan["provider_errors"] == router.provider_errors
    assert result.execution_plan["routed_execution_status"] == "cancelled"
    with store._get_conn() as conn:
        rows = dict(conn.execute("SELECT run_id, status FROM runs").fetchall())
    assert rows == {router_id: "degraded", child._run_id: "cancelled"}


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
