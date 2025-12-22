"""Tests for artifact storage and summarization."""

from pathlib import Path

import pytest

from llm_council.protocol.types import SummaryTier
from llm_council.storage import (
    TIER_TOKEN_LIMITS,
    ArtifactStore,
    ArtifactType,
    ProcessingState,
    Summarizer,
    summarize_for_context,
)


class TestArtifactStore:
    """Tests for ArtifactStore."""

    def test_create_run(self, tmp_path):
        """Test creating a run."""
        store = ArtifactStore(
            artifact_dir=tmp_path / "artifacts",
            db_path=tmp_path / "ledger.db",
        )
        run = store.create_run(subagent="implementer", task="Test task")

        assert run.run_id is not None
        assert run.subagent == "implementer"
        assert run.status == "running"

    def test_store_artifact(self, tmp_path):
        """Test storing an artifact."""
        store = ArtifactStore(
            artifact_dir=tmp_path / "artifacts",
            db_path=tmp_path / "ledger.db",
        )
        run = store.create_run(subagent="test", task="test")

        content = "This is test content for the artifact."
        artifact = store.store_artifact(
            run_id=run.run_id,
            content=content,
            artifact_type=ArtifactType.DRAFT,
        )

        assert artifact.artifact_id is not None
        assert artifact.byte_size == len(content.encode())
        assert artifact.token_estimate == len(content) // 4
        assert Path(artifact.file_path).exists()

    def test_artifact_deduplication(self, tmp_path):
        """Test that identical content is deduplicated."""
        store = ArtifactStore(
            artifact_dir=tmp_path / "artifacts",
            db_path=tmp_path / "ledger.db",
        )
        run = store.create_run(subagent="test", task="test")

        content = "Duplicate content"
        artifact1 = store.store_artifact(run.run_id, content, ArtifactType.DRAFT)
        artifact2 = store.store_artifact(run.run_id, content, ArtifactType.DRAFT)

        # Same content should return same artifact
        assert artifact1.artifact_id == artifact2.artifact_id

    def test_get_artifact_content(self, tmp_path):
        """Test retrieving artifact content."""
        store = ArtifactStore(
            artifact_dir=tmp_path / "artifacts",
            db_path=tmp_path / "ledger.db",
        )
        run = store.create_run(subagent="test", task="test")

        content = "Content to retrieve"
        artifact = store.store_artifact(run.run_id, content, ArtifactType.DRAFT)

        retrieved = store.get_artifact_content(artifact.artifact_id)
        assert retrieved == content

    def test_update_artifact_summary(self, tmp_path):
        """Test updating artifact summary."""
        store = ArtifactStore(
            artifact_dir=tmp_path / "artifacts",
            db_path=tmp_path / "ledger.db",
        )
        run = store.create_run(subagent="test", task="test")

        artifact = store.store_artifact(run.run_id, "content", ArtifactType.DRAFT)
        store.update_artifact_summary(artifact.artifact_id, "Summary text")

        artifacts = store.get_run_artifacts(run.run_id)
        assert len(artifacts) == 1
        assert artifacts[0].summary == "Summary text"
        assert artifacts[0].processing_state == ProcessingState.SUMMARIZED.value

    def test_create_capsule(self, tmp_path):
        """Test creating a result capsule."""
        store = ArtifactStore(
            artifact_dir=tmp_path / "artifacts",
            db_path=tmp_path / "ledger.db",
        )
        run = store.create_run(subagent="test", task="test")
        store.store_artifact(run.run_id, "content", ArtifactType.DRAFT)

        capsule = store.create_capsule(
            run_id=run.run_id,
            status="success",
            summary="Test summary",
            key_findings=["Finding 1", "Finding 2"],
            next_actions=["Action 1"],
        )

        assert capsule.run_id == run.run_id
        assert capsule.status == "success"
        assert len(capsule.key_findings) == 2
        assert capsule.token_estimate > 0

    def test_rehydrate(self, tmp_path):
        """Test rehydrating artifact content."""
        store = ArtifactStore(
            artifact_dir=tmp_path / "artifacts",
            db_path=tmp_path / "ledger.db",
        )
        run = store.create_run(subagent="test", task="test")

        content = "A" * 1000
        artifact = store.store_artifact(run.run_id, content, ArtifactType.DRAFT)

        # Full content
        full = store.rehydrate(artifact.artifact_id, max_chars=2000)
        assert full == content

        # Truncated content
        truncated = store.rehydrate(artifact.artifact_id, max_chars=100)
        assert len(truncated) > 100  # Includes truncation message
        assert "truncated" in truncated

    def test_path_traversal_protection(self, tmp_path):
        """Test that path traversal is prevented."""
        store = ArtifactStore(
            artifact_dir=tmp_path / "artifacts",
            db_path=tmp_path / "ledger.db",
        )

        # Try to escape artifact directory
        with pytest.raises(ValueError, match="escapes artifact directory"):
            store._ensure_path_containment(tmp_path / "artifacts" / ".." / "secret.txt")

    def test_disabled_store(self):
        """Test that disabled store operations are no-ops."""
        store = ArtifactStore(enabled=False)
        run = store.create_run(subagent="test", task="test")

        # Should return artifact but not persist
        artifact = store.store_artifact(run.run_id, "content", ArtifactType.DRAFT)
        assert artifact.artifact_id is not None
        assert artifact.file_path == ""

        # Should return empty list
        artifacts = store.get_run_artifacts(run.run_id)
        assert artifacts == []

    def test_cleanup_stale_runs(self, tmp_path):
        """Test that stale runs are marked as timed_out."""
        import sqlite3
        from datetime import datetime, timedelta, timezone

        store = ArtifactStore(
            artifact_dir=tmp_path / "artifacts",
            db_path=tmp_path / "ledger.db",
        )

        # Create a run and manually backdate it
        run = store.create_run(subagent="test", task="test")
        old_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()

        with sqlite3.connect(store.db_path) as conn:
            conn.execute(
                "UPDATE runs SET created_at = ? WHERE run_id = ?",
                (old_time, run.run_id),
            )
            conn.commit()

        # Cleanup with 1 hour threshold
        cleaned = store.cleanup_stale_runs(age_hours=1.0)
        assert cleaned == 1

        # Verify status changed
        with sqlite3.connect(store.db_path) as conn:
            cursor = conn.execute("SELECT status FROM runs WHERE run_id = ?", (run.run_id,))
            status = cursor.fetchone()[0]
        assert status == "timed_out"

    def test_cleanup_stale_runs_ignores_completed(self, tmp_path):
        """Test that completed runs are not affected by cleanup."""
        import sqlite3
        from datetime import datetime, timedelta, timezone

        store = ArtifactStore(
            artifact_dir=tmp_path / "artifacts",
            db_path=tmp_path / "ledger.db",
        )

        # Create a run and complete it
        run = store.create_run(subagent="test", task="test")
        store.complete_run(run.run_id, status="completed")

        # Backdate it
        old_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        with sqlite3.connect(store.db_path) as conn:
            conn.execute(
                "UPDATE runs SET created_at = ? WHERE run_id = ?",
                (old_time, run.run_id),
            )
            conn.commit()

        # Cleanup should not affect completed runs
        cleaned = store.cleanup_stale_runs(age_hours=1.0)
        assert cleaned == 0

        # Verify status unchanged
        with sqlite3.connect(store.db_path) as conn:
            cursor = conn.execute("SELECT status FROM runs WHERE run_id = ?", (run.run_id,))
            status = cursor.fetchone()[0]
        assert status == "completed"

    def test_cleanup_stale_runs_respects_threshold(self, tmp_path):
        """Test that runs younger than threshold are not cleaned up."""
        store = ArtifactStore(
            artifact_dir=tmp_path / "artifacts",
            db_path=tmp_path / "ledger.db",
        )

        # Create a fresh run
        run = store.create_run(subagent="test", task="test")

        # Cleanup with 1 hour threshold - run is fresh
        cleaned = store.cleanup_stale_runs(age_hours=1.0)
        assert cleaned == 0

        # Verify status unchanged
        import sqlite3

        with sqlite3.connect(store.db_path) as conn:
            cursor = conn.execute("SELECT status FROM runs WHERE run_id = ?", (run.run_id,))
            status = cursor.fetchone()[0]
        assert status == "running"


class TestSummarizer:
    """Tests for Summarizer."""

    def test_should_summarize(self, tmp_path):
        """Test summarization threshold detection."""
        store = ArtifactStore(
            artifact_dir=tmp_path / "artifacts",
            db_path=tmp_path / "ledger.db",
        )
        summarizer = Summarizer(artifact_store=store, threshold_tokens=100)

        # Below threshold
        short_content = "Short content"
        assert not summarizer.should_summarize(short_content)

        # Above threshold
        long_content = "A" * 1000  # 250 tokens
        assert summarizer.should_summarize(long_content)

    def test_summarize_small_content(self, tmp_path):
        """Test that small content is not summarized."""
        store = ArtifactStore(
            artifact_dir=tmp_path / "artifacts",
            db_path=tmp_path / "ledger.db",
        )
        summarizer = Summarizer(artifact_store=store)

        content = "Small content"
        result = summarizer.summarize(content, SummaryTier.ACTIONS, store_full=False)

        assert result.summary == content
        assert result.tokens_saved == 0
        assert not result.truncated

    def test_summarize_large_content(self, tmp_path):
        """Test that large content is summarized."""
        store = ArtifactStore(
            artifact_dir=tmp_path / "artifacts",
            db_path=tmp_path / "ledger.db",
        )
        summarizer = Summarizer(artifact_store=store, threshold_tokens=50)

        content = "This is a large piece of content.\n" * 100
        run = store.create_run(subagent="test", task="test")

        result = summarizer.summarize(
            content,
            SummaryTier.GIST,
            run_id=run.run_id,
            store_full=True,
        )

        assert result.tokens_saved > 0
        assert result.artifact_ref is not None
        assert len(result.summary) <= TIER_TOKEN_LIMITS[SummaryTier.GIST] * 4

    def test_summarize_drafts(self, tmp_path):
        """Test summarizing multiple drafts."""
        store = ArtifactStore(
            artifact_dir=tmp_path / "artifacts",
            db_path=tmp_path / "ledger.db",
        )
        # Use a very low threshold so content will be summarized
        summarizer = Summarizer(artifact_store=store, threshold_tokens=10)
        run = store.create_run(subagent="test", task="test")

        # Make content large enough to exceed threshold
        drafts = {
            "provider1": "Draft content line 1\n" * 100,  # ~500 tokens
            "provider2": "Draft content line 2\n" * 100,  # ~500 tokens
        }

        results = summarizer.summarize_drafts(drafts, SummaryTier.GIST, run_id=run.run_id)

        assert len(results) == 2
        assert "provider1" in results
        assert "provider2" in results

        # Verify summarization happened for large content
        for _provider, result in results.items():
            assert result.original_tokens > 0
            # With GIST tier and large content, should have tokens saved
            if result.original_tokens > TIER_TOKEN_LIMITS[SummaryTier.GIST]:
                assert result.tokens_saved > 0


class TestSummarizeForContext:
    """Tests for the convenience function."""

    def test_summarize_for_context_small(self):
        """Test that small content passes through."""
        content = "Small content"
        result = summarize_for_context(content, threshold_tokens=100)
        assert result == content

    def test_summarize_for_context_large(self):
        """Test that large content is summarized."""
        content = "Large content line\n" * 1000
        result = summarize_for_context(content, SummaryTier.GIST, threshold_tokens=50)
        assert len(result) < len(content)
