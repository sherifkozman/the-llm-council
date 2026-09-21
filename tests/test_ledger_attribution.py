"""LG-*: the ledger must record WHICH provider produced WHAT, in WHICH phase.

Background. On 2026-08-08 the live ledger held 821 runs and 3561 artifacts and
could not answer the questions it existed to answer -- "which seat is slow",
"which provider failed", "where did the wall-clock go" -- because the artifacts
table had no provider or phase column at all. Diagnosing a slow council required
probing providers live, which only measures the present and can say nothing about
the recorded past.

These tests pin that gap closed at BOTH ends:

  TestLedgerAttributionStorage -- the store can record and return attribution.
  TestLedgerAttributionWiring  -- the orchestrator actually passes it.

Both halves are required. A store that merely *accepts* provider/phase proves
nothing if every call site keeps dropping the value on the floor, which is exactly
what the old code did: `for _provider_name, draft_text in drafts.items()` had the
provider in hand and discarded it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from test_orchestrator import CaptureProvider

from llm_council.engine.orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    ValidationResult,
)
from llm_council.providers.base import GenerateRequest
from llm_council.storage import ArtifactStore, ArtifactType, Phase

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}


def _store(tmp_path) -> ArtifactStore:
    return ArtifactStore(
        artifact_dir=tmp_path / "artifacts",
        db_path=tmp_path / "ledger.db",
    )


class TestLedgerAttributionStorage:
    """The storage layer records attribution and never fabricates it."""

    def test_provider_and_phase_round_trip(self, tmp_path):
        """LG-01: read it back from the DB, not from the returned object."""
        store = _store(tmp_path)
        run = store.create_run(subagent="critic", task="t")

        store.store_artifact(
            run_id=run.run_id,
            content="claude's draft",
            artifact_type=ArtifactType.DRAFT,
            provider="claude",
            phase=Phase.DRAFT,
        )

        reloaded = store.get_run_artifacts(run.run_id)
        assert len(reloaded) == 1
        assert reloaded[0].provider == "claude"
        assert reloaded[0].phase == "draft"

    def test_phase_enum_and_plain_string_are_equivalent(self, tmp_path):
        """LG-02: callers may pass either form; the stored value is the string."""
        store = _store(tmp_path)
        run = store.create_run(subagent="critic", task="t")

        store.store_artifact(
            run.run_id, "a", ArtifactType.CRITIQUE, provider="codex", phase=Phase.CRITIQUE
        )
        store.store_artifact(
            run.run_id, "b", ArtifactType.CRITIQUE, provider="codex", phase="critique"
        )

        assert {a.phase for a in store.get_run_artifacts(run.run_id)} == {"critique"}

    def test_identical_output_from_two_providers_stays_two_rows(self, tmp_path):
        """LG-03: dedup must not collapse two seats into one.

        Providers really do emit byte-identical short outputs (a one-word verdict,
        a small JSON object). Deduping on content alone would merge them and make
        the new columns lie about who produced what -- worse than having no
        columns at all, because the data would then look authoritative.
        """
        store = _store(tmp_path)
        run = store.create_run(subagent="router", task="t")
        identical = '{"ok": true}'

        a = store.store_artifact(
            run.run_id, identical, ArtifactType.DRAFT, provider="claude", phase=Phase.DRAFT
        )
        b = store.store_artifact(
            run.run_id, identical, ArtifactType.DRAFT, provider="codex", phase=Phase.DRAFT
        )

        assert a.artifact_id != b.artifact_id
        assert {x.provider for x in store.get_run_artifacts(run.run_id)} == {"claude", "codex"}

    def test_same_provider_same_phase_still_deduplicates(self, tmp_path):
        """LG-04: LG-03 must not disable deduplication wholesale."""
        store = _store(tmp_path)
        run = store.create_run(subagent="router", task="t")

        a = store.store_artifact(
            run.run_id, "same", ArtifactType.DRAFT, provider="claude", phase=Phase.DRAFT
        )
        b = store.store_artifact(
            run.run_id, "same", ArtifactType.DRAFT, provider="claude", phase=Phase.DRAFT
        )

        assert a.artifact_id == b.artifact_id

    def test_existing_ledger_gains_columns_without_losing_rows(self, tmp_path):
        """LG-05: migrating a ledger that already has history is non-destructive.

        The live ledger holds three months of runs. `CREATE TABLE IF NOT EXISTS`
        is a no-op there, so without an explicit ALTER the columns would only ever
        appear on brand-new installs -- the machine that needed them most would
        never get them.
        """
        db_path = tmp_path / "ledger.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY, wave_id TEXT, subagent TEXT NOT NULL,
                task_hash TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'running',
                budget_output_tokens INTEGER DEFAULT 4000,
                actual_output_tokens INTEGER DEFAULT 0,
                created_at TEXT NOT NULL, completed_at TEXT);
            CREATE TABLE artifacts (
                artifact_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL, content_hash TEXT NOT NULL,
                byte_size INTEGER NOT NULL, token_estimate INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                processing_state TEXT NOT NULL DEFAULT 'unseen',
                created_at TEXT NOT NULL, summary TEXT,
                summary_tokens INTEGER DEFAULT 0);
            """
        )
        conn.execute(
            "INSERT INTO artifacts VALUES ('old-1','run-1','draft','h',10,3,"
            "'/tmp/x.txt','unseen','2026-05-07T00:00:00+00:00',NULL,0)"
        )
        conn.commit()
        conn.close()

        # Opening the store must migrate the schema in place.
        ArtifactStore(artifact_dir=tmp_path / "artifacts", db_path=db_path)

        conn = sqlite3.connect(db_path)
        try:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(artifacts)")}
            rows = conn.execute(
                "SELECT artifact_id, token_estimate, provider, phase FROM artifacts"
            ).fetchall()
        finally:
            conn.close()

        assert {"provider", "phase"} <= columns
        # The pre-existing row survives untouched and its attribution is NULL:
        # "not recorded", never a backfilled guess.
        assert rows == [("old-1", 3, None, None)]

    def test_two_first_opens_of_a_legacy_ledger_both_succeed(self, tmp_path, monkeypatch):
        """LG-05b: the migration is serialized, so a concurrent first open cannot lose.

        Two processes starting at once on an old ledger could both read the schema
        before either altered it; the second ALTER then raised "duplicate column
        name" out of the ArtifactStore constructor and killed that run. This pins the
        interleaving deterministically: opener A is held right before its first
        ALTER (after its schema read) while opener B does a complete first open.
        """
        db_path = tmp_path / "ledger.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY, wave_id TEXT, subagent TEXT NOT NULL,
                task_hash TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'running',
                budget_output_tokens INTEGER DEFAULT 4000,
                actual_output_tokens INTEGER DEFAULT 0,
                created_at TEXT NOT NULL, completed_at TEXT);
            CREATE TABLE artifacts (
                artifact_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL, content_hash TEXT NOT NULL,
                byte_size INTEGER NOT NULL, token_estimate INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                processing_state TEXT NOT NULL DEFAULT 'unseen',
                created_at TEXT NOT NULL, summary TEXT,
                summary_tokens INTEGER DEFAULT 0);
            """
        )
        conn.close()

        a_has_read_schema = threading.Event()
        b_is_done = threading.Event()

        def hold_a_before_its_first_alter(sql: str) -> None:
            if (
                threading.current_thread().name == "opener-a"
                and sql.lstrip().upper().startswith("ALTER TABLE")
                and not a_has_read_schema.is_set()
            ):
                a_has_read_schema.set()
                # Unserialized, B migrates in this window and A resumes on a stale
                # view. Serialized, B is blocked on A's lock and this wait times out.
                b_is_done.wait(timeout=1.0)

        class _Cursor:
            def __init__(self, real: sqlite3.Cursor) -> None:
                self._real = real

            def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
                hold_a_before_its_first_alter(sql)
                return self._real.execute(sql, *args)

            def __getattr__(self, name: str) -> object:
                return getattr(self._real, name)

        class _Connection:
            def __init__(self, real: sqlite3.Connection) -> None:
                self._real = real

            def cursor(self) -> _Cursor:
                return _Cursor(self._real.cursor())

            def __getattr__(self, name: str) -> object:
                return getattr(self._real, name)

        real_connect = sqlite3.connect
        monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: _Connection(real_connect(*a, **k)))

        errors: dict[str, BaseException] = {}

        def open_store(name: str) -> None:
            try:
                ArtifactStore(artifact_dir=tmp_path / f"artifacts-{name}", db_path=db_path)
            except BaseException as exc:  # recorded, asserted below
                errors[name] = exc
            finally:
                if name == "b":
                    b_is_done.set()

        opener_a = threading.Thread(target=open_store, args=("a",), name="opener-a")
        opener_a.start()
        assert a_has_read_schema.wait(timeout=10), "opener A never reached its ALTER"
        opener_b = threading.Thread(target=open_store, args=("b",), name="opener-b")
        opener_b.start()
        opener_b.join(timeout=15)
        opener_a.join(timeout=15)
        monkeypatch.undo()

        # A thread still blocked here would make `errors == {}` a false green.
        assert not opener_a.is_alive() and not opener_b.is_alive()
        assert errors == {}
        conn = sqlite3.connect(db_path)
        try:
            columns = [row[1] for row in conn.execute("PRAGMA table_info(artifacts)")]
        finally:
            conn.close()
        assert columns[-2:] == ["provider", "phase"]
        assert columns.count("provider") == columns.count("phase") == 1

    def test_unrecorded_attribution_stays_null(self, tmp_path):
        """LG-06: omitting provider/phase stores NULL, not a placeholder string."""
        store = _store(tmp_path)
        run = store.create_run(subagent="critic", task="t")
        store.store_artifact(run.run_id, "no attribution", ArtifactType.TOOL_LOG)

        artifact = store.get_run_artifacts(run.run_id)[0]
        assert artifact.provider is None
        assert artifact.phase is None

    def test_the_previously_impossible_query_now_answers(self, tmp_path):
        """LG-07: the exact analysis that could not be run before.

        This is the regression that matters. Per-provider cost and per-phase
        breakdown previously required live probing because the column did not
        exist. It must now be a plain GROUP BY over stored data.
        """
        store = _store(tmp_path)
        run = store.create_run(subagent="critic", task="t")

        store.store_artifact(
            run.run_id, "evidence blob", ArtifactType.TOOL_LOG, provider=None, phase=Phase.EVIDENCE
        )
        for seat, text in (("claude", "c" * 400), ("codex", "x" * 800), ("sakana", "s" * 200)):
            store.store_artifact(
                run.run_id, text, ArtifactType.DRAFT, provider=seat, phase=Phase.DRAFT
            )
        store.store_artifact(
            run.run_id,
            "the critique",
            ArtifactType.CRITIQUE,
            provider="claude",
            phase=Phase.CRITIQUE,
        )
        store.store_artifact(
            run.run_id,
            "the synthesis",
            ArtifactType.SYNTHESIS,
            provider="codex",
            phase=Phase.SYNTHESIS,
        )

        conn = sqlite3.connect(store.db_path)
        try:
            breakdown = dict(
                conn.execute(
                    """
                    SELECT provider || '/' || phase, SUM(token_estimate)
                      FROM artifacts WHERE run_id = ? AND provider IS NOT NULL
                     GROUP BY provider, phase
                    """,
                    (run.run_id,),
                ).fetchall()
            )
            phases = {
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT phase FROM artifacts WHERE run_id = ?", (run.run_id,)
                )
            }
        finally:
            conn.close()

        # Each seat is individually accountable for its share of the tokens.
        assert breakdown["claude/draft"] == 100
        assert breakdown["codex/draft"] == 200
        assert breakdown["sakana/draft"] == 50
        assert breakdown["claude/critique"] > 0
        assert breakdown["codex/synthesis"] > 0
        # And all four pipeline phases are distinguishable.
        assert phases == {"evidence", "draft", "critique", "synthesis"}


class TestLedgerAttributionWiring:
    """The orchestrator actually passes attribution on a real run().

    Without this class the feature could be fully "implemented" and still record
    nothing: the storage tests above would stay green while every artifact landed
    with provider=NULL.
    """

    @staticmethod
    def _orchestrator(tmp_path, store: ArtifactStore) -> Orchestrator:
        config = OrchestratorConfig(enable_artifacts=True, output_schema=SCHEMA)
        with (
            patch("llm_council.engine.orchestrator.get_registry") as mock_reg,
            patch("llm_council.engine.orchestrator.get_store", return_value=store),
        ):
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            return Orchestrator(providers=["claude", "codex", "sakana"], config=config)

    @pytest.mark.asyncio
    async def test_run_attributes_every_artifact_it_writes(self, tmp_path):
        """LG-08: after a full run, the ledger says who did what, where."""
        store = _store(tmp_path)
        orch = self._orchestrator(tmp_path, store)

        drafts = {
            "claude": '{"ok": true}',
            "codex": '{"ok": false}',
            "sakana": '{"ok": true, "note": "third"}',
        }

        async def _critique(_drafts):
            # Mirror the real path: the critique phase records which seat served it.
            orch._record_phase_provider_used("critique", "claude")
            return "the critique text"

        async def _synthesis(_drafts, _critique):
            orch._record_phase_provider_used("synthesis", "codex")
            return (ValidationResult(ok=True, data={"ok": True}, raw='{"ok": true}'), 1)

        with (
            patch.object(orch, "_run_parallel_drafts", AsyncMock(return_value=drafts)),
            patch.object(orch, "_run_critique", AsyncMock(side_effect=_critique)),
            patch.object(orch, "_run_synthesis", AsyncMock(side_effect=_synthesis)),
        ):
            result = await orch.run("Review this change", "critic")

        assert result.success is True
        assert orch._run_id is not None

        stored = store.get_run_artifacts(orch._run_id)
        by_type = {a.artifact_type: a for a in stored if a.artifact_type != "draft"}
        draft_rows = [a for a in stored if a.artifact_type == "draft"]

        # Every draft is attributed to the seat that produced it -- this is the
        # value the old `for _provider_name, ...` loop threw away.
        assert {a.provider for a in draft_rows} == {"claude", "codex", "sakana"}
        assert {a.phase for a in draft_rows} == {"draft"}

        assert by_type["critique"].provider == "claude"
        assert by_type["critique"].phase == "critique"
        assert by_type["synthesis"].provider == "codex"
        assert by_type["synthesis"].phase == "synthesis"

        # Nothing the run wrote is left unattributed except deliberately
        # provider-less rows (locally collected evidence).
        assert all(a.phase is not None for a in stored)

    @pytest.mark.asyncio
    async def test_per_provider_breakdown_is_queryable_after_a_real_run(self, tmp_path):
        """LG-09: the investigation question, answered from a run's own ledger rows.

        'Which seat produced how much, in which phase' had to be answered with a
        live latency probe on 2026-08-08. After a real run it must be SQL.
        """
        store = _store(tmp_path)
        orch = self._orchestrator(tmp_path, store)

        drafts = {"claude": "a" * 400, "codex": "b" * 1200}

        async def _synthesis(_drafts, _critique):
            orch._record_phase_provider_used("synthesis", "claude")
            return (ValidationResult(ok=True, data={"ok": True}, raw='{"ok": true}'), 1)

        with (
            patch.object(orch, "_run_parallel_drafts", AsyncMock(return_value=drafts)),
            patch.object(orch, "_run_critique", AsyncMock(return_value="")),
            patch.object(orch, "_run_synthesis", AsyncMock(side_effect=_synthesis)),
        ):
            await orch.run("Review this change", "critic")

        conn = sqlite3.connect(store.db_path)
        try:
            rows = dict(
                conn.execute(
                    """
                    SELECT provider, SUM(token_estimate) FROM artifacts
                     WHERE run_id = ? AND phase = 'draft' GROUP BY provider
                    """,
                    (orch._run_id,),
                ).fetchall()
            )
        finally:
            conn.close()

        # codex generated 3x claude's tokens, and the ledger can now say so.
        assert rows == {"claude": 100, "codex": 300}

    @pytest.mark.asyncio
    async def test_attribution_survives_two_providers_emitting_identical_drafts(self, tmp_path):
        """LG-10: the dedup hazard, exercised through a real run rather than the store.

        Two seats returning the same short JSON is the realistic case; if dedup
        collapsed them, a three-seat council would silently look like a two-seat
        one in every later analysis.
        """
        store = _store(tmp_path)
        orch = self._orchestrator(tmp_path, store)

        identical = '{"ok": true}'
        drafts = {"claude": identical, "codex": identical, "sakana": identical}

        with (
            patch.object(orch, "_run_parallel_drafts", AsyncMock(return_value=drafts)),
            patch.object(orch, "_run_critique", AsyncMock(return_value="")),
            patch.object(
                orch,
                "_run_synthesis",
                AsyncMock(
                    return_value=(
                        ValidationResult(ok=True, data={"ok": True}, raw=identical),
                        1,
                    )
                ),
            ),
        ):
            await orch.run("Review this change", "critic")

        draft_rows = [
            a for a in store.get_run_artifacts(orch._run_id) if a.artifact_type == "draft"
        ]
        assert len(draft_rows) == 3
        assert {a.provider for a in draft_rows} == {"claude", "codex", "sakana"}


class TestErrorAttribution:
    """LG-11..15: a failure must be attributable too, not just a success.

    Failures produce no draft/critique/synthesis, so they used to leave no trace
    in the ledger whatsoever. A run could burn ten minutes, lose two seats and be
    abandoned, and the stored record would show a lone evidence tool_log. That is
    why "which provider is failing" had to be answered with a live probe.
    """

    @staticmethod
    def _orchestrator_with_store(store: ArtifactStore, providers=("claude",)) -> Orchestrator:
        config = OrchestratorConfig(
            enable_artifacts=True,
            enable_graceful_degradation=True,
            output_schema=SCHEMA,
        )
        with (
            patch("llm_council.engine.orchestrator.get_registry") as mock_reg,
            patch("llm_council.engine.orchestrator.get_store", return_value=store),
        ):
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            orch = Orchestrator(providers=list(providers), config=config)
        run = store.create_run(subagent="critic", task="t")
        orch._run_id = run.run_id
        return orch

    @staticmethod
    def _dead_adapter(message: str = "upstream request timed out"):
        adapter = MagicMock()
        adapter.generate = AsyncMock(side_effect=TimeoutError(message))
        return adapter

    @staticmethod
    def _error_rows(store: ArtifactStore, run_id: str):
        return [a for a in store.get_run_artifacts(run_id) if a.artifact_type == "error_report"]

    @staticmethod
    def _payload(artifact):
        return json.loads(Path(artifact.file_path).read_text(encoding="utf-8"))

    @pytest.mark.asyncio
    async def test_failure_is_written_with_provider_and_phase(self, tmp_path):
        """LG-11: the seat and the phase that failed are both recorded."""
        store = _store(tmp_path)
        orch = self._orchestrator_with_store(store)

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises((RuntimeError, TimeoutError)),
        ):
            await orch._call_provider(
                "claude",
                self._dead_adapter(),
                GenerateRequest(prompt="hi"),
                phase="critique",
                remaining_providers=0,
            )

        errors = self._error_rows(store, orch._run_id)
        assert errors, "a provider failure left no trace in the ledger"
        assert {a.provider for a in errors} == {"claude"}
        assert {a.phase for a in errors} == {"critique"}

        payload = self._payload(errors[0])
        assert payload["provider"] == "claude"
        assert payload["phase"] == "critique"
        assert payload["error_type"] == "timeout"
        assert "timed out" in payload["error"]

    @pytest.mark.asyncio
    async def test_each_retry_is_its_own_row(self, tmp_path):
        """LG-12: retries must not dedup into a single row.

        The error text is identical on every attempt, so without the attempt
        number in the payload three attempts would collapse to one and the retry
        burn -- the thing that makes a slow failure expensive -- would be
        invisible.
        """
        store = _store(tmp_path)
        orch = self._orchestrator_with_store(store)

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises((RuntimeError, TimeoutError)),
        ):
            await orch._call_provider(
                "claude",
                self._dead_adapter(),
                GenerateRequest(prompt="hi"),
                phase="draft",
                remaining_providers=0,
            )

        errors = self._error_rows(store, orch._run_id)
        attempts = sorted(self._payload(a)["attempt"] for a in errors)
        assert len(errors) >= 2, "retries collapsed into one ledger row"
        assert attempts == sorted(set(attempts)), "attempt numbers repeat"

    @pytest.mark.asyncio
    async def test_error_reports_do_not_inflate_output_tokens(self, tmp_path):
        """LG-13: error text is not model output and must not be counted as such.

        actual_output_tokens is the denominator of every throughput metric. If
        error reports counted, a run that generated nothing would report hundreds
        of output tokens and failure analysis would read backwards.
        """
        store = _store(tmp_path)
        orch = self._orchestrator_with_store(store)

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises((RuntimeError, TimeoutError)),
        ):
            await orch._call_provider(
                "claude",
                self._dead_adapter(),
                GenerateRequest(prompt="hi"),
                phase="draft",
                remaining_providers=0,
            )

        conn = sqlite3.connect(store.db_path)
        try:
            total = conn.execute(
                "SELECT actual_output_tokens FROM runs WHERE run_id = ?", (orch._run_id,)
            ).fetchone()[0]
            stored = conn.execute(
                "SELECT COUNT(*) FROM artifacts WHERE run_id = ? "
                "AND artifact_type = 'error_report'",
                (orch._run_id,),
            ).fetchone()[0]
        finally:
            conn.close()

        assert stored >= 2, "precondition: error rows were written"
        assert total == 0, "error reports inflated the run's output-token count"

    @pytest.mark.asyncio
    async def test_failures_survive_a_run_that_never_completes(self, tmp_path):
        """LG-14: the record exists even though the call ends by raising.

        40 rows in the live ledger are stuck in 'running' -- killed, abandoned or
        hung -- and never reach any end-of-run bookkeeping. Persisting at the
        moment of failure is what makes those runs explainable at all.
        """
        store = _store(tmp_path)
        orch = self._orchestrator_with_store(store)

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises((RuntimeError, TimeoutError)),
        ):
            await orch._call_provider(
                "claude",
                self._dead_adapter(),
                GenerateRequest(prompt="hi"),
                phase="synthesis",
                remaining_providers=0,
            )

        # The run is deliberately left in 'running': nothing completed it.
        conn = sqlite3.connect(store.db_path)
        try:
            status = conn.execute(
                "SELECT status FROM runs WHERE run_id = ?", (orch._run_id,)
            ).fetchone()[0]
            rows = conn.execute(
                "SELECT COUNT(*) FROM artifacts WHERE run_id = ? AND phase = 'synthesis'",
                (orch._run_id,),
            ).fetchone()[0]
        finally:
            conn.close()

        assert status == "running"
        assert rows >= 1, "an abandoned run still explains nothing"

    @pytest.mark.asyncio
    async def test_which_provider_failed_where_is_now_one_query(self, tmp_path):
        """LG-15: the investigation question, answered from stored data.

        Two seats failing in two different phases is the exact shape that
        previously required a live latency probe to untangle.
        """
        store = _store(tmp_path)
        orch = self._orchestrator_with_store(store, providers=("claude", "codex"))

        for seat, phase in (("claude", "draft"), ("codex", "critique")):
            with (
                patch("asyncio.sleep", new_callable=AsyncMock),
                pytest.raises((RuntimeError, TimeoutError)),
            ):
                await orch._call_provider(
                    seat,
                    self._dead_adapter(),
                    GenerateRequest(prompt="hi"),
                    phase=phase,
                    remaining_providers=0,
                )

        conn = sqlite3.connect(store.db_path)
        try:
            failures = dict(
                conn.execute(
                    """
                    SELECT provider || '/' || phase, COUNT(*) FROM artifacts
                     WHERE run_id = ? AND artifact_type = 'error_report'
                     GROUP BY provider, phase
                    """,
                    (orch._run_id,),
                ).fetchall()
            )
        finally:
            conn.close()

        assert set(failures) == {"claude/draft", "codex/critique"}
        assert all(count >= 2 for count in failures.values())


class TestFailoverAttribution:
    """LG-16/17: after a failover, credit the provider that actually worked.

    Caught by a live run, not by the tests above. The console reported
    `claude -> anthropic/claude-opus-5` while the ledger recorded
    `provider=claude` for the resulting draft, critique and synthesis: the dead
    seat was credited for work its fallback did. Callers key their results by the
    seat they configured, and `_call_provider` swaps the adapter internally, so
    the swap has to be published for attribution to stay true.

    An attribution that looks authoritative and is wrong is worse than none.
    """

    @staticmethod
    def _orchestrator(store: ArtifactStore) -> Orchestrator:
        config = OrchestratorConfig(
            enable_artifacts=True,
            enable_graceful_degradation=True,
            fallback_providers={"claude": "openrouter/backup"},
            output_schema=SCHEMA,
        )
        with (
            patch("llm_council.engine.orchestrator.get_registry") as mock_reg,
            patch("llm_council.engine.orchestrator.get_store", return_value=store),
        ):
            mock_reg.return_value = MagicMock()
            mock_reg.return_value.get_provider.return_value = MagicMock()
            return Orchestrator(providers=["claude"], config=config)

    @pytest.mark.asyncio
    async def test_draft_is_credited_to_the_fallback_not_the_dead_seat(self, tmp_path):
        """LG-16: the draft artifact names the provider that produced it."""
        store = _store(tmp_path)
        orch = self._orchestrator(store)

        dead = MagicMock()
        dead.generate = AsyncMock(side_effect=TimeoutError("upstream request timed out"))
        backup = CaptureProvider()

        async def _drafts():
            # Faithful to the real draft phase: the dict is keyed by the
            # CONFIGURED seat even when the call failed over underneath.
            response = await orch._call_provider(
                "claude",
                dead,
                GenerateRequest(prompt="x"),
                phase="draft",
                remaining_providers=0,
            )
            return {"claude": response.text}

        async def _synthesis(_drafts, _critique):
            return (ValidationResult(ok=True, data={"ok": True}, raw='{"ok": true}'), 1)

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch.object(
                orch,
                "_instantiate_providers",
                return_value=({"openrouter/backup": backup}, {}),
            ),
            patch.object(orch, "_run_parallel_drafts", AsyncMock(side_effect=_drafts)),
            patch.object(orch, "_run_critique", AsyncMock(return_value="")),
            patch.object(orch, "_run_synthesis", AsyncMock(side_effect=_synthesis)),
        ):
            result = await orch.run("Review this change", "critic")

        assert result.success is True

        artifacts = store.get_run_artifacts(orch._run_id)
        drafts = [a for a in artifacts if a.artifact_type == "draft"]
        errors = [a for a in artifacts if a.artifact_type == "error_report"]

        # The work was done by the fallback, so the fallback is credited.
        assert [a.provider for a in drafts] == ["openrouter/backup"]
        # ...while the failures stay pinned to the seat that actually failed.
        assert errors, "the failover was preceded by real failures"
        assert {a.provider for a in errors} == {"claude"}
        assert {a.phase for a in errors} == {"draft"}

    @pytest.mark.asyncio
    async def test_without_a_failover_the_configured_seat_is_kept(self, tmp_path):
        """LG-17: the resolution must be a no-op on the normal path."""
        store = _store(tmp_path)
        orch = self._orchestrator(store)

        async def _synthesis(_drafts, _critique):
            orch._record_phase_provider_used("synthesis", "claude")
            return (ValidationResult(ok=True, data={"ok": True}, raw='{"ok": true}'), 1)

        with (
            patch.object(
                orch, "_run_parallel_drafts", AsyncMock(return_value={"claude": "a draft"})
            ),
            patch.object(orch, "_run_critique", AsyncMock(return_value="")),
            patch.object(orch, "_run_synthesis", AsyncMock(side_effect=_synthesis)),
        ):
            await orch.run("Review this change", "critic")

        artifacts = store.get_run_artifacts(orch._run_id)
        by_type = {a.artifact_type: a for a in artifacts}
        assert by_type["draft"].provider == "claude"
        assert by_type["synthesis"].provider == "claude"
