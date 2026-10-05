"""Provider-free execution outcome and attempt deadline regressions."""

import asyncio
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from llm_council.engine.degradation import DegradationPolicy
from llm_council.engine.orchestrator import CouncilResult, Orchestrator, OrchestratorConfig
from llm_council.protocol.types import ReasoningProfile
from llm_council.providers.base import GenerateRequest, GenerateResponse
from llm_council.providers.compiler import compile_request_for_provider
from llm_council.providers.registry import ProviderRegistry
from llm_council.storage.artifacts import ArtifactStore
from llm_council.subagents import ReasoningBudget


@pytest.mark.parametrize("new_plan", [False, True])
@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({}, "completed"),
        ({"success": False}, "failed"),
        ({"execution_status": "cancelled"}, "cancelled"),
        ({"drafts": {"a": "answer", "b": ""}}, "degraded"),
        ({"critique": ""}, "degraded"),
        (
            {
                "provider_errors": {"a": "earlier timeout"},
                "degradation_report": {"events": [{"action": "retry"}]},
            },
            "completed",
        ),
        ({"execution_plan": {"degraded_output": {"source": "draft"}}}, "degraded"),
        ({"execution_plan": {"provider_auto_fallback": True}}, "degraded"),
        ({"execution_plan": {"cleanup_incomplete": True}}, "degraded"),
        ({"execution_plan": {"context_preparation": {"files": [{"truncated": True}]}}}, "degraded"),
        (
            {"execution_plan": {"context_preparation": {"files": [{"status": "missing"}]}}},
            "degraded",
        ),
        (
            {
                "execution_plan": {
                    "context_preparation": {"files": [{"status": "skipped_total_limit"}]}
                }
            },
            "degraded",
        ),
        (
            {"critique": "", "execution_plan": {"required_phases": ["draft", "synthesis"]}},
            "completed",
        ),
    ],
)
def test_status_uses_final_coverage_not_retry_history(
    updates: dict[str, Any], expected: str, new_plan: bool
) -> None:
    fields: dict[str, Any] = {
        "success": True,
        "output": {"verdict": "request_changes"},
        "drafts": {"a": "answer", "b": "answer"},
        "critique": "review",
        "execution_plan": {
            "providers": ["a", "b"],
            "required_phases": ["draft", "critique", "synthesis"],
        },
    }
    fields.update(updates)
    if new_plan:
        fields["execution_plan"].update(
            requested_capabilities=[], pending_capabilities=["diff-review"]
        )
    result = CouncilResult(**fields)
    assert result.model_dump()["execution_status"] == expected
    assert result.success is (expected not in {"failed", "cancelled"})


@pytest.mark.parametrize(
    "requested_fields,pending,expected",
    [
        ({}, ["diff-review"], "degraded"),
        ({"requested_capabilities": []}, ["diff-review"], "completed"),
        ({"requested_capabilities": ["diff-review"]}, ["diff-review"], "degraded"),
        ({"requested_capabilities": ["repo-analysis"]}, ["diff-review"], "completed"),
        ({"requested_capabilities": ["diff-review"]}, [], "completed"),
    ],
)
def test_pending_capability_status_preserves_legacy_contract(
    requested_fields: dict[str, list[str]], pending: list[str], expected: str
) -> None:
    plan = {
        "required_phases": [],
        "required_capabilities": ["diff-review", "repo-analysis"],
        "executed_capabilities": ["repo-analysis"],
        "pending_capabilities": pending,
        **requested_fields,
    }
    result = CouncilResult(success=True, output={}, execution_plan=plan)
    assert result.execution_status == expected
    assert result.execution_plan == plan


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "subagent,mode,profile,capabilities",
    [
        (
            "critic",
            "security",
            "deep_analysis",
            ["red-team-recon", "security-code-audit", "repo-analysis"],
        ),
        ("researcher", None, "grounded", ["docs-research"]),
    ],
)
def test_default_capability_guidance_is_distinct_from_explicit_requirements(
    mock_registry: ProviderRegistry,
    explicit: bool,
    subagent: str,
    mode: str | None,
    profile: str,
    capabilities: list[str],
) -> None:
    requested = capabilities[:1] if explicit else []
    runner = engine(mock_registry, mode=mode, required_capabilities=requested)
    runner._prepare_run(subagent)
    plan = runner._execution_plan
    assert plan is not None
    assert plan["requested_capabilities"] == requested
    assert plan["requested_capabilities"] is not runner._config.required_capabilities
    assert plan["required_capabilities"] == capabilities
    assert plan["pending_capabilities"] == capabilities
    assert plan["execution_profile"] == profile
    result = CouncilResult(
        success=True,
        output={},
        drafts=dict.fromkeys(plan["selected_providers"], "draft"),
        critique="complete critique",
        execution_plan=plan,
    )
    assert result.execution_status == ("degraded" if explicit else "completed")
    assert result.execution_plan == plan


@pytest.mark.asyncio
async def test_recovered_attempt_restores_completed_coverage(mock_registry, monkeypatch):
    runner = engine(mock_registry)
    runner._degradation_policy = DegradationPolicy(max_retries=1)
    monkeypatch.setattr(DegradationPolicy, "BASE_RETRY_DELAY_MS", 0)
    adapter = mock_registry.get_provider("mock")
    requests = []

    async def generate(request):
        requests.append(request)
        if len(requests) == 1:
            await asyncio.sleep(0.15)
        await asyncio.sleep(0.02)
        return GenerateResponse(text='{"verdict": "request_changes"}')

    with (
        patch.object(mock_registry, "get_provider", return_value=adapter),
        patch.object(adapter, "generate", generate),
        patch.object(runner, "_provider_request_timeout_seconds", return_value=0.08),
    ):
        result = await runner.run("test", "planner")
    assert result.success is True
    assert result.execution_status == "completed"
    assert result.output == {"verdict": "request_changes"}
    assert len(requests) == 4
    assert all(0.04 < request.timeout_seconds <= 0.08 for request in requests)
    assert result.degradation_report["total_retries"] == 1


@pytest.mark.asyncio
async def test_multi_model_selected_drafts_are_model_ids(mock_registry):
    with patch(
        "llm_council.providers.openrouter.create_openrouter_for_model",
        side_effect=lambda _: mock_registry.get_provider("mock"),
    ):
        runner = Orchestrator(
            ["openrouter"],
            OrchestratorConfig(
                models=["first/model", "second/model"],
                enable_artifacts=False,
                enable_health_check=False,
                output_schema={"type": "object"},
            ),
        )
        result = await runner.run("test", "planner")
    assert sorted(result.drafts) == ["first/model", "second/model"]
    assert result.success is True
    assert result.execution_status == "completed"


@pytest.mark.asyncio
async def test_unsettled_attempt_does_not_retry_or_hide_cleanup(mock_registry, monkeypatch):
    runner = engine(mock_registry)
    runner._execution_plan = {}
    runner._degradation_policy = DegradationPolicy(max_retries=1)
    monkeypatch.setattr(DegradationPolicy, "BASE_RETRY_DELAY_MS", 0)
    adapter = runner._providers["mock"]
    release = asyncio.Event()
    pending = []
    original_wait = asyncio.wait

    async def bounded_wait(tasks, *, timeout, **kwargs):
        return await original_wait(tasks, timeout=0.01 if timeout == 5.0 else timeout, **kwargs)

    async def generate(request):
        pending.append(asyncio.current_task())
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
            return GenerateResponse(text="late")

    try:
        with (
            patch.object(adapter, "generate", generate),
            patch("asyncio.wait", bounded_wait),
            pytest.raises(TimeoutError),
        ):
            await runner._call_provider(
                "mock", adapter, GenerateRequest(prompt="test", timeout_seconds=0.02)
            )
        assert len(pending) == 1
        assert runner._execution_plan["cleanup_incomplete"] is True
    finally:
        release.set()
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_cleanup_exception_preserves_chain_and_prevents_retry(
    mock_registry: ProviderRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = engine(mock_registry)
    runner._execution_plan = {}
    runner._degradation_policy = DegradationPolicy(max_retries=1)
    monkeypatch.setattr(DegradationPolicy, "BASE_RETRY_DELAY_MS", 0)
    adapter = runner._providers["mock"]
    attempts: list[GenerateRequest] = []

    async def generate(request: GenerateRequest) -> GenerateResponse:
        attempts.append(request)
        if len(attempts) > 1:
            return GenerateResponse(text="must not recover by retrying")
        try:
            await asyncio.Event().wait()
            raise AssertionError("provider resumed without cancellation")
        except asyncio.CancelledError:
            raise RuntimeError("synthetic resource still owned") from OSError("cleanup root cause")

    with patch.object(adapter, "generate", generate), pytest.raises(TimeoutError) as caught:
        await asyncio.wait_for(
            runner._call_provider(
                "mock", adapter, GenerateRequest(prompt="test", timeout_seconds=0.02)
            ),
            timeout=1,
        )
    assert len(attempts) == 1
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert isinstance(caught.value.__cause__.__cause__, OSError)
    assert runner._execution_plan["cleanup_incomplete"] is True
    diagnostic = runner._provider_init_errors["mock"]
    assert "Provider attempt deadline expired" in diagnostic
    assert "synthetic resource still owned" in diagnostic
    assert "cleanup root cause" in diagnostic
    result = CouncilResult(
        success=True,
        output={"verdict": "request_changes"},
        critique="complete",
        execution_plan=runner._execution_plan,
    )
    assert result.execution_status == "degraded"


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_raises", [False, True])
async def test_repeated_cancellation_keeps_cleanup_owned_until_settled(
    mock_registry: ProviderRegistry, cleanup_raises: bool
) -> None:
    runner = engine(mock_registry)
    runner._execution_plan = {}
    adapter = runner._providers["mock"]
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    pending: list[asyncio.Task[GenerateResponse]] = []
    cancellations: list[asyncio.CancelledError] = []

    async def generate(request: GenerateRequest) -> GenerateResponse:
        task = asyncio.current_task()
        assert task is not None
        pending.append(task)
        entered.set()
        try:
            await asyncio.Event().wait()
            raise AssertionError("provider resumed without cancellation")
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            if cleanup_raises:
                raise RuntimeError("cleanup failed after cancellation") from OSError("cleanup root")
            return GenerateResponse(text="late response must not defeat cancellation")

    async def call() -> GenerateResponse:
        try:
            return await runner._call_provider(
                "mock", adapter, GenerateRequest(prompt="test", timeout_seconds=10)
            )
        except asyncio.CancelledError as exc:
            # Inspect before Python 3.10 Task propagation can lose the cause.
            cancellations.append(exc)
            raise

    with patch.object(adapter, "generate", generate):
        outer = asyncio.create_task(call())
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            outer.cancel("first cancellation")
            await asyncio.wait_for(cleaning.wait(), timeout=1)
            outer.cancel("second cancellation")
            await asyncio.sleep(0.01)
            assert not outer.done(), "repeated cancellation abandoned the owned attempt"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(outer, timeout=1)
            assert len(pending) == 1 and pending[0].done()
            assert len(cancellations) == 1
            assert cancellations[0].args == ("first cancellation",)
            if cleanup_raises:
                assert runner._execution_plan["cleanup_incomplete"] is True
                assert "cleanup root" in runner._provider_init_errors["mock"]
                assert isinstance(cancellations[0].__cause__, RuntimeError)
            else:
                assert not runner._execution_plan.get("cleanup_incomplete")
                assert "mock" not in runner._provider_init_errors
        finally:
            release.set()
            await asyncio.wait_for(
                asyncio.gather(outer, *pending, return_exceptions=True), timeout=1
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["timeout", "external", "repeated", "adapter"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_native_shaped_cancellation_retains_cleanup_outcome(
    mock_registry: ProviderRegistry,
    monkeypatch: pytest.MonkeyPatch,
    trigger: str,
    cleanup_fails: bool,
) -> None:
    runner = engine(mock_registry)
    runner._execution_plan = {}
    runner._degradation_policy = DegradationPolicy(max_retries=1)
    monkeypatch.setattr(DegradationPolicy, "BASE_RETRY_DELAY_MS", 0)
    adapter = runner._providers["mock"]
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    pending: list[asyncio.Task[GenerateResponse]] = []
    observed_errors: list[BaseException] = []

    def consume_task_cancellation(task: asyncio.Task[GenerateResponse]) -> None:
        # A first Task inspection may consume its original cancellation cause.
        # Engine correctness must not depend on inspecting it first or later.
        if task.cancelled():
            try:
                task.exception()
            except asyncio.CancelledError:
                pass

    async def generate(request: GenerateRequest) -> GenerateResponse:
        task = asyncio.current_task()
        assert task is not None
        pending.append(task)
        task.add_done_callback(consume_task_cancellation)
        if len(pending) > 1:
            return GenerateResponse(text="retry only after clean cancellation")
        entered.set()
        try:
            if trigger == "adapter":
                raise asyncio.CancelledError("native adapter cancellation")
            await asyncio.Event().wait()
            raise AssertionError("provider resumed without cancellation")
        except asyncio.CancelledError as cancelled:
            cleaning.set()
            await release.wait()
            if cleanup_fails:
                try:
                    raise RuntimeError(
                        "native cleanup incomplete: resource still owned"
                    ) from OSError("native cleanup root cause")
                except RuntimeError as cleanup_error:
                    raise cancelled from cleanup_error
            raise

    async def call() -> GenerateResponse:
        try:
            return await runner._call_provider(
                "mock",
                adapter,
                GenerateRequest(
                    prompt="test", timeout_seconds=0.02 if trigger == "timeout" else 10
                ),
            )
        except BaseException as exc:
            observed_errors.append(exc)
            raise

    with patch.object(adapter, "generate", generate):
        outer = asyncio.create_task(call())
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            if trigger in {"external", "repeated"}:
                outer.cancel("first library cancellation")
            await asyncio.wait_for(cleaning.wait(), timeout=1)
            if trigger == "repeated":
                outer.cancel("second library cancellation")
                await asyncio.sleep(0.01)
                assert not outer.done(), "repeated cancellation abandoned native cleanup"
            release.set()
            if trigger == "timeout" and not cleanup_fails:
                response = await asyncio.wait_for(outer, timeout=1)
                assert response.text == "retry only after clean cancellation"
                assert len(pending) == 2
            else:
                expected = TimeoutError if trigger == "timeout" else asyncio.CancelledError
                with pytest.raises(expected):
                    await asyncio.wait_for(outer, timeout=1)
                assert len(pending) == 1
                assert len(observed_errors) == 1
            assert all(task.done() for task in pending)
            if cleanup_fails:
                assert runner._execution_plan["cleanup_incomplete"] is True
                diagnostic = runner._provider_init_errors["mock"]
                assert "native cleanup incomplete: resource still owned" in diagnostic
                assert "native cleanup root cause" in diagnostic
                error = observed_errors[0]
                assert isinstance(error.__cause__, RuntimeError)
                assert isinstance(error.__cause__.__cause__, OSError)
                if trigger in {"external", "repeated"}:
                    assert error.args == ("first library cancellation",)
            else:
                assert not runner._execution_plan.get("cleanup_incomplete")
                assert "mock" not in runner._provider_init_errors
            result = CouncilResult(
                success=True,
                output={"verdict": "request_changes"},
                critique="complete",
                execution_plan=runner._execution_plan,
            )
            assert result.execution_status == ("degraded" if cleanup_fails else "completed")
        finally:
            release.set()
            if not outer.done():
                outer.cancel()
            await asyncio.wait_for(
                asyncio.gather(outer, *pending, return_exceptions=True), timeout=1
            )


@pytest.mark.asyncio
async def test_repeated_cancellation_keeps_original_cleanup_deadline(
    mock_registry: ProviderRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = engine(mock_registry)
    runner._execution_plan = {}
    adapter = runner._providers["mock"]
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    waiting = asyncio.Event()
    release = asyncio.Event()
    pending: list[asyncio.Task[GenerateResponse]] = []
    allowances: list[float] = []
    now = [100.0]
    original_wait = asyncio.wait

    async def observe_wait(
        tasks: set[asyncio.Task[GenerateResponse]], *, timeout: float, **kwargs: Any
    ) -> tuple[set[asyncio.Task[GenerateResponse]], set[asyncio.Task[GenerateResponse]]]:
        if timeout > 1:
            allowances.append(timeout)
            waiting.set()
        return await original_wait(tasks, timeout=timeout, **kwargs)

    async def generate(request: GenerateRequest) -> GenerateResponse:
        task = asyncio.current_task()
        assert task is not None
        pending.append(task)
        entered.set()
        try:
            await asyncio.Event().wait()
            raise AssertionError("provider resumed without cancellation")
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            return GenerateResponse(text="late")

    monkeypatch.setattr(
        "llm_council.engine.orchestrator.time", SimpleNamespace(monotonic=lambda: now[0])
    )
    with patch.object(adapter, "generate", generate), patch("asyncio.wait", observe_wait):
        outer = asyncio.create_task(
            runner._call_provider(
                "mock", adapter, GenerateRequest(prompt="test", timeout_seconds=1)
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            outer.cancel()
            await asyncio.wait_for(cleaning.wait(), timeout=1)
            await asyncio.wait_for(waiting.wait(), timeout=1)
            waiting.clear()
            now[0] = 103.0
            outer.cancel()
            await asyncio.sleep(0.01)
            assert not outer.done(), "second cancellation abandoned cleanup"
            await asyncio.wait_for(waiting.wait(), timeout=1)
            assert allowances == [5.0, 2.0], "cleanup budget was restarted"
            now[0] = 106.0
            outer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(outer, timeout=1)
            assert runner._execution_plan["cleanup_incomplete"] is True
            assert "cleanup" in runner._provider_init_errors["mock"]
            assert len(pending) == 1 and not pending[0].done()
        finally:
            release.set()
            await asyncio.wait_for(
                asyncio.gather(outer, *pending, return_exceptions=True), timeout=1
            )


@pytest.mark.parametrize(
    "profile,effort",
    [
        (ReasoningProfile.DEFAULT, "high"),
        (ReasoningProfile.LIGHT, "low"),
        (ReasoningProfile.OFF, "none"),
    ],
)
@pytest.mark.parametrize("provider,model", [("codex", "gpt-5.4"), ("claude", "claude-opus-5")])
def test_reasoning_profile_reaches_provider_compilation(
    mock_registry: ProviderRegistry,
    profile: ReasoningProfile,
    effort: str,
    provider: str,
    model: str,
) -> None:
    runner = engine(mock_registry, reasoning_profile=profile)
    budget = ReasoningBudget(
        enabled=True, effort="high", budget_tokens=16384, thinking_level="high"
    )
    compiled = compile_request_for_provider(
        provider,
        GenerateRequest(
            model=model, prompt="test", reasoning=runner._build_reasoning_config(budget)
        ),
    )
    control = compiled.to_dict()["reasoning_control"]
    if profile == ReasoningProfile.OFF and provider == "claude":
        assert control == {"status": "uncontrolled"}
        assert compiled.request.reasoning is None
        assert any(
            decision.option == "reasoning"
            and decision.action == "dropped"
            and "not off" in decision.detail
            for decision in compiled.decisions
        )
    else:
        assert control["status"] == "requested"
        assert control["effort"] == effort
        assert compiled.request.reasoning is not None
        assert compiled.request.reasoning.enabled is (profile != ReasoningProfile.OFF)


def engine(mock_registry: ProviderRegistry, **kwargs: Any) -> Orchestrator:
    with patch("llm_council.engine.orchestrator.get_registry", return_value=mock_registry):
        return Orchestrator(
            ["mock"],
            OrchestratorConfig(
                enable_artifacts=False,
                enable_health_check=False,
                enable_graceful_degradation=False,
                output_schema={"type": "object"},
                **kwargs,
            ),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["health", "draft"])
async def test_cancel_keeps_result_and_ledger_terminal(mock_registry, tmp_path, phase):
    runner = engine(mock_registry)
    store = ArtifactStore(artifact_dir=tmp_path / "artifacts", db_path=tmp_path / "ledger.db")
    runner._artifact_store = store
    entered = asyncio.Event()

    async def block():
        entered.set()
        await asyncio.Event().wait()

    target = "_ensure_usable_providers" if phase == "health" else "_run_parallel_drafts"
    results = []

    async def run():
        try:
            return await runner.run("test", "planner")
        except asyncio.CancelledError as exc:
            results.append(exc.result)
            raise

    with patch.object(runner, target, block):
        task = asyncio.create_task(run())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(results) == 1
    result = results[0]
    assert result.execution_status == "cancelled"
    assert result.success is False
    assert result.run_id == runner._run_id
    with store._get_conn() as conn:
        row = conn.execute("SELECT status FROM runs WHERE run_id = ?", (result.run_id,)).fetchone()
    assert row[0] == "cancelled"


@pytest.mark.asyncio
async def test_early_failure_completes_ledger(mock_registry, tmp_path):
    runner = engine(mock_registry)
    runner._artifact_store = ArtifactStore(
        artifact_dir=tmp_path / "artifacts", db_path=tmp_path / "ledger.db"
    )
    with patch.object(
        runner, "_run_parallel_drafts", AsyncMock(side_effect=RuntimeError("draft failed"))
    ):
        result = await runner.run("test", "planner")
    assert result.success is False
    assert result.execution_status == "failed"
    assert result.run_id == runner._run_id
    with runner._artifact_store._get_conn() as conn:
        row = conn.execute("SELECT status FROM runs WHERE run_id = ?", (result.run_id,)).fetchone()
    assert row[0] == "failed"


@pytest.mark.asyncio
async def test_queue_time_is_passed_as_actual_remainder(mock_registry):
    runner = engine(mock_registry)
    received = []

    @asynccontextmanager
    async def slot(*args, **kwargs):
        await asyncio.sleep(0.04)
        yield 0  # The clock, not a queue diagnostic, is the deadline authority.

    async def generate(request):
        received.append(request.timeout_seconds)
        return GenerateResponse(text="done")

    adapter = runner._providers["mock"]
    with (
        patch("llm_council.engine.orchestrator.provider_call_slot", slot),
        patch.object(adapter, "generate", generate),
    ):
        result = await runner._call_provider(
            "mock", adapter, GenerateRequest(prompt="test", timeout_seconds=0.12)
        )
    assert result.text == "done"
    assert len(received) == 1
    assert 0 < received[0] < 0.1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "queue_delay,generate_delay,stream_delay", [(0.15, 0, 0), (0, 0.15, 0), (0, 0.07, 0.07)]
)
async def test_queue_generate_and_stream_share_one_deadline(
    mock_registry, queue_delay, generate_delay, stream_delay
):
    runner = engine(mock_registry)
    calls = []
    closed = []

    @asynccontextmanager
    async def slot(*args, **kwargs):
        await asyncio.sleep(queue_delay)
        yield 0

    async def stream():
        try:
            await asyncio.sleep(stream_delay)
            yield GenerateResponse(text="late")
        finally:
            closed.append(True)

    async def generate(request):
        calls.append(request)
        await asyncio.sleep(generate_delay)
        return stream()

    adapter = runner._providers["mock"]
    started = time.monotonic()
    with (
        patch("llm_council.engine.orchestrator.provider_call_slot", slot),
        patch.object(adapter, "generate", generate),
        pytest.raises(TimeoutError),
    ):
        await runner._call_provider(
            "mock", adapter, GenerateRequest(prompt="test", timeout_seconds=0.1)
        )
    assert time.monotonic() - started < 0.3
    assert len(calls) == (0 if queue_delay else 1)
    if stream_delay:
        assert closed == [True]
