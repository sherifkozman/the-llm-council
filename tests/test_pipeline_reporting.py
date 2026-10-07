"""Provider-free regressions for the October 6 reporting audit."""

import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_council.engine.degradation import DegradationPolicy
from llm_council.engine.orchestrator import CouncilResult, Orchestrator, OrchestratorConfig
from llm_council.providers.base import GenerateRequest, GenerateResponse, Message
from llm_council.schemas import load_schema
from llm_council.storage.artifacts import ArtifactStore


@pytest.fixture(autouse=True)
def isolate_reporting(monkeypatch, tmp_path):
    @asynccontextmanager
    async def slot(*args, **kwargs):
        yield 0.0

    monkeypatch.setattr("llm_council.engine.orchestrator.provider_call_slot", slot)
    monkeypatch.setattr(DegradationPolicy, "BASE_RETRY_DELAY_MS", 0)
    monkeypatch.chdir(tmp_path)


def runner():
    with patch("llm_council.engine.orchestrator.get_registry", return_value=MagicMock()):
        result = Orchestrator(
            [],
            OrchestratorConfig(
                enable_artifacts=False,
                enable_health_check=False,
                enable_graceful_degradation=False,
            ),
        )
    adapter = MagicMock()
    adapter.supports = AsyncMock(return_value=False)
    adapter.generate = AsyncMock(return_value=GenerateResponse(text='{"ok":true}'))

    def prepare(subagent):
        result._subagent_config = {}
        result._schema = {"type": "object"}
        result._schema_name = "custom"
        result._providers = {"synthetic": adapter}
        result._provider_names = ["synthetic"]
        result._execution_plan = {
            "selected_providers": ["synthetic"],
            "required_phases": ["draft", "critique", "synthesis"],
        }

    result._prepare_run = prepare
    result._ensure_usable_providers = AsyncMock()
    result._run_parallel_drafts = AsyncMock(return_value={"synthetic": '{"ok":true}'})
    result._run_critique = AsyncMock(return_value="Synthetic critique")
    return result, adapter


@pytest.mark.asyncio
async def test_failed_draft_records_only_executed_retry():
    orch, _ = runner()
    orch._degradation_policy = DegradationPolicy(max_retries=2)
    adapters = {name: MagicMock() for name in ("bad", "peer1", "peer2")}
    adapters["bad"].generate = AsyncMock(side_effect=RuntimeError("Synthetic failure"))
    for name in ("peer1", "peer2"):
        adapters[name].generate = AsyncMock(return_value=GenerateResponse(text="draft"))
    orch._providers = adapters
    orch._task = "synthetic"
    orch._subagent_config = {}
    orch._execution_plan = {}

    async def draft(name, adapter):
        response = await orch._call_provider(
            name,
            adapter,
            GenerateRequest(messages=[Message(role="user", content="synthetic")]),
            phase="draft",
        )
        return name, response.text

    orch._generate_draft = draft
    drafts = await Orchestrator._run_parallel_drafts(orch)
    assert drafts["peer1"] == "draft"
    assert adapters["bad"].generate.await_count == 2
    assert orch._degradation_policy.get_report().total_retries == 1


@pytest.mark.asyncio
async def test_failed_synthesis_preserves_attempts_and_timing():
    orch, adapter = runner()
    adapter.generate.side_effect = RuntimeError("Synthetic synthesis failure")
    result = await orch.run("synthetic", "synthetic")
    assert adapter.generate.await_count == 2
    assert result.output == {"ok": True}
    assert result.synthesis_attempts == 2
    assert "synthesis" in [timing.phase for timing in result.phase_timings]
    assert result.execution_status == "degraded"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["store_artifact", "complete_run"])
async def test_persistence_failure_is_returned_without_erasing_output(tmp_path, operation):
    orch, _ = runner()
    store = ArtifactStore(artifact_dir=tmp_path / "artifacts", db_path=tmp_path / "ledger.db")
    orch._artifact_store = store
    with patch.object(store, operation, side_effect=OSError("PRIVATE DETAIL")):
        result = await orch.run("synthetic", "synthetic")
    assert result.output == {"ok": True}
    assert result.execution_status == "degraded"
    errors = result.execution_plan["persistence"]["errors"]
    assert any(item["operation"] == operation for item in errors)
    assert "PRIVATE DETAIL" not in str(result.model_dump())


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["critique", "synthesis"])
async def test_cancellation_preserves_completed_work(phase):
    orch, adapter = runner()
    if phase == "critique":
        orch._run_critique.side_effect = asyncio.CancelledError()
    else:
        adapter.generate.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError) as raised:
        await orch.run("synthetic", "synthetic")
    result = raised.value.result
    assert result.execution_status == "cancelled"
    assert result.drafts == {"synthetic": '{"ok":true}'}
    assert result.critique == ("Synthetic critique" if phase == "synthesis" else "")
    assert phase in [timing.phase for timing in result.phase_timings]
    assert result.synthesis_attempts == (1 if phase == "synthesis" else 0)


def test_partial_source_selection_cannot_report_completed():
    orch, _ = runner()
    orch._resolved_mode = "review"
    orch._subagent_name = "critic"
    orch._task = "List every section number in the file."
    text = "\n".join(f"## Section {i}\n" + "synthetic body " * 40 for i in range(1, 28))
    orch._config.system_context = f"=== FILE: doc.txt ===\n{text}\n=== END: doc.txt ==="
    orch._execution_plan = {"required_phases": []}
    orch._prepare_reference_context()
    result = CouncilResult(success=True, output={}, execution_plan=orch._execution_plan)
    coverage = result.execution_plan["context_preparation"]["coverage"][0]
    assert coverage["delivered_source_chars"] < coverage["source_chars"]
    assert coverage["complete"] is False
    assert result.execution_status == "degraded"
    assert any(
        w["kind"] == "source_selection_loss" for w in result.degradation_report["context_warnings"]
    )


def test_slice_span_reports_actual_excerpt_limit():
    orch, _ = runner()
    orch._task = "Review retry defect."
    text = "## Retry\n" + "retry " * 600
    rendered, slices, _ = orch._slice_markdown_block("doc.md", text)
    assert "excerpt truncated" in rendered
    start, end = slices[0]["retained_char_span"]
    assert end - start == 1800


@pytest.mark.parametrize("multiple_sections", [False, True])
def test_slice_span_excludes_whitespace_removed_at_cutoff(multiple_sections):
    orch, _ = runner()
    orch._task = "Review retry defect."
    text = "## Retry\n" + "x" * 1781 + " " * 20 + "tail"
    if multiple_sections:
        text += "\n## Other\nshort body"
    rendered, slices, _ = orch._slice_markdown_block("doc.md", text)
    selected = next(item for item in slices if item["anchor"] == "retry")
    start, end = selected["retained_char_span"]
    assert end - start == 1790
    assert text[start:end] == text.strip()[:1800].rstrip()
    assert f"<quoted_evidence>\n{text[start:end]}\n... [excerpt truncated]" in rendered


def test_first_prompt_profile_reports_compaction():
    orch, _ = runner()
    orch._execution_plan = {}
    draft = "synthetic finding " * 150
    _, decision = orch._select_prompt_profile(
        provider_name="openai",
        phase="critique",
        system_prompt="Review drafts",
        prompt_builder=lambda profile: orch._compact_text(draft, profile.get("draft_limit")),
    )
    assert decision["profile_index"] == 0
    assert decision["compacted"] is True
    assert decision["original_prompt_chars"] > decision["delivered_prompt_chars"]
    assert orch._execution_plan["phase_prompt_compaction"]["critique"]


@pytest.mark.asyncio
async def test_invalid_synthesis_preserves_handoff_evidence_without_manufacturing_issues():
    orch, adapter = runner()
    orch._prepare_run("critic")
    orch._task = "Review source evidence."
    orch._schema = load_schema("reviewer")
    orch._schema_name = "reviewer"
    orch._schema_source = "subagent"
    adapter.generate.return_value = GenerateResponse(text="Unvalidated prose, not review JSON.")
    orch._draft_handoffs = {
        "synthetic": {
            "findings": [
                {
                    "chunk_index": 1,
                    "draft": "The first module fails to release its resources.",
                    "sources": [{"path": "first.py"}],
                },
                {
                    "chunk_index": 2,
                    "draft": "The second module retries requests without any bound.",
                    "sources": [{"path": "second.py"}],
                },
            ]
        }
    }
    handoffs = json.dumps(orch._draft_handoffs)
    result, attempts = await orch._run_synthesis(
        {"synthetic": "The unrelated handler fails to check the input."}, ""
    )
    assert not result.ok
    assert result.data is None
    assert result.errors == ["Failed to parse JSON."]
    assert attempts == orch._config.max_retries
    assert json.dumps(orch._draft_handoffs) == handoffs


@pytest.mark.asyncio
async def test_cancelled_parallel_drafts_keep_completed_artifact(tmp_path):
    orch, adapter = runner()
    finished = asyncio.Event()

    async def draft(name, _adapter):
        if name == "fast":
            finished.set()
            return name, "completed draft"
        await asyncio.Event().wait()

    def prepare(subagent):
        orch._subagent_config = {}
        orch._providers = {"fast": adapter, "slow": adapter}
        orch._provider_names = ["fast", "slow"]
        orch._execution_plan = {"selected_providers": ["fast", "slow"]}

    orch._prepare_run = prepare
    orch._generate_draft = draft
    orch._run_parallel_drafts = Orchestrator._run_parallel_drafts.__get__(orch)
    orch._artifact_store = ArtifactStore(
        artifact_dir=tmp_path / "artifacts", db_path=tmp_path / "ledger.db"
    )
    captured = []

    async def call():
        try:
            return await orch.run("synthetic", "synthetic")
        except asyncio.CancelledError as exc:
            captured.append(exc.result)
            raise

    task = asyncio.create_task(call())
    await finished.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    result = captured[0]
    assert result.drafts == {"fast": "completed draft"}
    refs = [r for r in result.execution_plan["artifact_occurrences"] if r["phase"] == "draft"]
    assert len(refs) == 1 and refs[0]["provider"] == "fast"
    assert refs[0]["artifact_id"] in {
        a.artifact_id for a in orch._artifact_store.get_run_artifacts(result.run_id)
    }


@pytest.mark.asyncio
async def test_identical_drafts_preserve_provider_occurrences(tmp_path):
    orch, _ = runner()
    orch._artifact_store = ArtifactStore(
        artifact_dir=tmp_path / "artifacts", db_path=tmp_path / "ledger.db"
    )
    orch._run_parallel_drafts.return_value = {"first": "same", "second": "same"}
    result = await orch.run("synthetic", "synthetic")
    refs = [r for r in result.execution_plan["artifact_occurrences"] if r["phase"] == "draft"]
    assert {r["provider"] for r in refs} == {"first", "second"}
    assert len({r["artifact_id"] for r in refs}) == 2


@pytest.mark.asyncio
async def test_ledger_creation_failure_keeps_usable_output(tmp_path):
    orch, _ = runner()
    orch._artifact_store = ArtifactStore(
        artifact_dir=tmp_path / "artifacts", db_path=tmp_path / "ledger.db"
    )
    with patch.object(orch._artifact_store, "create_run", side_effect=OSError("private")):
        result = await orch.run("synthetic", "synthetic")
    assert result.output == {"ok": True}
    assert result.run_id is None
    assert result.execution_status == "degraded"
    assert result.execution_plan["persistence"]["errors"][0]["operation"] == "create_run"


@pytest.mark.asyncio
async def test_disabled_persistence_is_not_a_failure():
    orch, _ = runner()
    result = await orch.run("synthetic", "synthetic")
    assert result.execution_status == "completed"
    assert not result.execution_plan.get("persistence", {}).get("errors")


def test_intermediate_draft_format_is_not_a_source_finding():
    orch, _ = runner()
    orch._schema = {"type": "object"}
    assert "aligns with the JSON schema" not in orch._format_draft_prompt("Review")
    critique = orch._format_critique_prompt("Review", {"synthetic": "analysis"})
    assert "not whether intermediate drafts match" in critique
    assert "pipeline limitations" in critique
    synthesis = orch._format_synthesis_prompt(
        "Review", {"synthetic": "analysis"}, "critique", {}, []
    )
    assert "not source defects" in synthesis


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_attempt_identity_is_safe_and_preserved_on_failure(fails):
    orch, adapter = runner()
    orch._execution_plan = {}
    identity = {
        "cli_path": "/synthetic/codex",
        "cli_realpath": "/synthetic/bin/codex",
        "cli_version": "codex-cli 0.149.1",
        "secret": "DO_NOT_COPY",
    }
    if fails:
        error = ValueError("unsupported request")
        error.native_identity = identity
        adapter.generate.side_effect = error
    else:
        adapter.generate.return_value = GenerateResponse(
            text="review",
            model="reported",
            raw={"native_identity": identity, "secret": "DO_NOT_COPY"},
        )
    request = GenerateRequest(model="requested", messages=[Message(role="user", content="task")])
    if fails:
        with pytest.raises(ValueError):
            await orch._call_provider("codex", adapter, request, phase="draft")
    else:
        await orch._call_provider("codex", adapter, request, phase="draft")
    attempt = orch._execution_plan["provider_attempts"][0]
    assert attempt["requested_model"] == "requested"
    assert attempt["native_identity"]["cli_version"] == "codex-cli 0.149.1"
    assert attempt["status"] == ("failed" if fails else "completed")
    assert "DO_NOT_COPY" not in str(attempt)
    if not fails:
        assert attempt["adapter_reported_model"] == "reported"


@pytest.mark.asyncio
async def test_cancel_before_retry_is_not_reported_as_executed_retry():
    orch, adapter = runner()
    orch._execution_plan = {}
    orch._providers = {"bad": adapter, "peer": adapter}
    orch._degradation_policy = DegradationPolicy(max_retries=2)
    orch._degradation_policy.BASE_RETRY_DELAY_MS = 100
    adapter.generate.side_effect = RuntimeError("temporary failure")
    request = GenerateRequest(messages=[Message(role="user", content="task")])
    with (
        patch("asyncio.sleep", side_effect=asyncio.CancelledError()),
        pytest.raises(asyncio.CancelledError),
    ):
        await orch._call_provider("bad", adapter, request, phase="draft")
    assert adapter.generate.await_count == 1
    assert orch._get_degradation_report()["total_retries"] == 0
    assert orch._get_degradation_report()["planned_retries"] == 1


@pytest.mark.asyncio
async def test_unsent_synthesis_candidates_do_not_inflate_attempts():
    orch, adapter = runner()
    peer = MagicMock()
    peer.supports = AsyncMock(return_value=False)
    peer.generate = AsyncMock(return_value=GenerateResponse(text='{"ok":true}'))
    orch._candidate_providers_for_phase = AsyncMock(
        return_value=[("queued", adapter), ("peer", peer)]
    )

    @asynccontextmanager
    async def slot(name, **kwargs):
        if name == "queued":
            raise TimeoutError("queue deadline")
        yield 0

    with patch("llm_council.engine.orchestrator.provider_call_slot", slot):
        result = await orch.run("synthetic", "synthetic")
    assert adapter.generate.await_count == 0
    assert peer.generate.await_count == 1
    assert result.synthesis_attempts == 1


def test_unsent_compaction_does_not_degrade_delivered_coverage():
    orch, _ = runner()
    orch._execution_plan = {"required_phases": []}
    orch._select_prompt_profile(
        provider_name="openai",
        phase="synthesis",
        system_prompt="synthesis",
        prompt_builder=lambda profile: orch._compact_text(
            "evidence " * 200, profile.get("draft_limit")
        ),
    )
    result = CouncilResult(success=True, output={}, execution_plan=orch._execution_plan)
    assert result.execution_status == "completed"
    assert not (result.degradation_report or {}).get("context_warnings")


@pytest.mark.asyncio
async def test_runtime_and_occurrence_manifest_survives_reopen(tmp_path):
    orch, _ = runner()
    directory, ledger = tmp_path / "artifacts", tmp_path / "ledger.db"
    orch._artifact_store = ArtifactStore(artifact_dir=directory, db_path=ledger)
    orch._run_parallel_drafts.return_value = {"first": "same", "second": "same"}
    result = await orch.run("synthetic", "synthetic")
    reopened = ArtifactStore(artifact_dir=directory, db_path=ledger)
    manifests = [
        json.loads(reopened.get_artifact_content(a.artifact_id))
        for a in reopened.get_run_artifacts(result.run_id)
        if a.artifact_type == "tool_log"
    ]
    final = next(m for m in manifests if m.get("event") == "settled")
    assert final["runtime_identity"]["council_version"]
    refs = [r for r in final["artifact_occurrences"] if r["phase"] == "draft"]
    assert {r["provider"] for r in refs} == {"first", "second"}
    assert len({r["artifact_id"] for r in refs}) == 2


def test_execution_manifest_filters_nested_metadata(tmp_path):
    orch, _ = runner()
    store = ArtifactStore(artifact_dir=tmp_path / "artifacts", db_path=tmp_path / "ledger.db")
    orch._artifact_store = store
    orch._run_id = store.create_run("synthetic", "synthetic").run_id
    orch._execution_plan = {
        "provider_attempts": [
            {
                "provider": "p",
                "secret": "DO_NOT_COPY",
                "native_identity": {"cli_path": "/synthetic", "token": "DO_NOT_COPY"},
            }
        ],
        "artifact_occurrences": [{"artifact_id": "id", "secret": "DO_NOT_COPY"}],
        "context_preparation": {"coverage": [{"path": "synthetic", "secret": "DO_NOT_COPY"}]},
        "persistence": {"errors": [{"operation": "write", "secret": "DO_NOT_COPY"}]},
    }
    orch._store_execution_manifest("settled")
    data = store.get_artifact_content(store.get_run_artifacts(orch._run_id)[0].artifact_id)
    assert "DO_NOT_COPY" not in data
    assert json.loads(data)["provider_attempts"][0]["native_identity"]["cli_path"] == "/synthetic"


@pytest.mark.asyncio
async def test_start_manifest_precedes_generation_and_cancel_snapshot_is_persisted(tmp_path):
    orch, adapter = runner()
    store = ArtifactStore(artifact_dir=tmp_path / "artifacts", db_path=tmp_path / "ledger.db")
    orch._artifact_store = store

    async def generate(request):
        snapshots = [
            json.loads(store.get_artifact_content(a.artifact_id))
            for a in store.get_run_artifacts(orch._run_id)
            if a.artifact_type == "tool_log"
        ]
        assert snapshots[0]["event"] == "started"
        assert snapshots[0]["provider_attempts"] == []
        raise asyncio.CancelledError()

    adapter.generate.side_effect = generate
    with pytest.raises(asyncio.CancelledError):
        await orch.run("synthetic", "synthetic")
    snapshots = [
        json.loads(store.get_artifact_content(a.artifact_id))
        for a in store.get_run_artifacts(orch._run_id)
        if a.artifact_type == "tool_log"
    ]
    assert snapshots[-1]["event"] == "settled"
    assert snapshots[-1]["execution_status"] == "cancelled"
    assert snapshots[-1]["provider_attempts"][0]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_manifest_write_failure_degrades_result_and_ledger(tmp_path):
    orch, _ = runner()
    store = ArtifactStore(artifact_dir=tmp_path / "artifacts", db_path=tmp_path / "ledger.db")
    orch._artifact_store = store
    write = store.store_artifact

    def fail_settled(**kwargs):
        if (
            kwargs["artifact_type"].value == "tool_log"
            and json.loads(kwargs["content"])["event"] == "settled"
        ):
            raise OSError("private")
        return write(**kwargs)

    with patch.object(store, "store_artifact", side_effect=fail_settled):
        result = await orch.run("synthetic", "synthetic")
    assert result.output == {"ok": True}
    assert result.execution_status == "degraded"
    with store._get_conn() as connection:
        assert (
            connection.execute(
                "SELECT status FROM runs WHERE run_id=?", (result.run_id,)
            ).fetchone()[0]
            == "degraded"
        )
