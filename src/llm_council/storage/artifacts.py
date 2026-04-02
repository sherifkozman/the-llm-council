"""
Artifact Store for LLM Council.

Implements the Result Capsule + Output Ledger pattern for context management.
Stores verbose outputs outside the transcript and tracks processing state.

SECURITY NOTE: Path traversal protection via directory containment checks.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import ClassVar


class ProcessingState(str, Enum):
    """State of artifact processing."""

    UNSEEN = "unseen"
    SUMMARIZED = "summarized"
    INGESTED = "ingested"
    ARCHIVED = "archived"


class ArtifactType(str, Enum):
    """Types of artifacts stored."""

    DRAFT = "draft"
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"
    TOOL_LOG = "tool_log"
    CODE_DIFF = "code_diff"
    STRUCTURED_JSON = "structured_json"
    ERROR_REPORT = "error_report"


@dataclass
class Artifact:
    """A stored artifact with metadata."""

    artifact_id: str
    run_id: str
    artifact_type: str
    content_hash: str
    byte_size: int
    token_estimate: int
    file_path: str
    processing_state: str = ProcessingState.UNSEEN.value
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    summary: str | None = None
    summary_tokens: int = 0


@dataclass
class Run:
    """A council run with budget tracking."""

    run_id: str
    wave_id: str | None
    subagent: str
    task_hash: str
    status: str = "running"
    budget_output_tokens: int = 4000
    actual_output_tokens: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None


@dataclass
class StoragePaths:
    """Resolved council storage paths and current selection."""

    active_artifact_dir: Path
    active_db_path: Path
    default_artifact_dir: Path
    default_db_path: Path
    legacy_artifact_dir: Path
    legacy_db_path: Path
    using_legacy: bool


@dataclass
class StorageMigrationResult:
    """Outcome of an explicit legacy-to-neutral storage migration."""

    source_artifact_dir: Path
    source_db_path: Path
    target_artifact_dir: Path
    target_db_path: Path
    dry_run: bool
    migrated: bool
    copied_files: int
    copied_db: bool
    message: str


@dataclass
class ResultCapsule:
    """Bounded summary for main context ingestion."""

    run_id: str
    status: str  # success, failed, partial
    summary: str  # Max 500 chars
    key_findings: list[str]  # Max 5 items, 200 chars each
    blockers: list[str]  # Critical issues
    next_actions: list[str]  # Recommended steps
    artifact_refs: list[str]  # Pointers to full artifacts
    token_estimate: int

    def to_context_string(self) -> str:
        """Format for ingestion into main context."""
        lines = [
            f"[Run {self.run_id[:8]}] Status: {self.status}",
            f"Summary: {self.summary}",
        ]
        if self.key_findings:
            lines.append("Key Findings:")
            for f in self.key_findings[:5]:
                lines.append(f"  - {f[:200]}")
        if self.blockers:
            lines.append("Blockers:")
            for b in self.blockers:
                lines.append(f"  ! {b}")
        if self.next_actions:
            lines.append("Next Actions:")
            for a in self.next_actions[:3]:
                lines.append(f"  > {a}")
        if self.artifact_refs:
            lines.append(f"Full details: {len(self.artifact_refs)} artifact(s) available")
        return "\n".join(lines)


class ArtifactStore:
    """
    Local artifact store with SQLite ledger.

    Implements:
    - Artifact persistence outside transcript
    - Content-addressed deduplication
    - Processing state tracking
    - Bounded capsule generation
    - Path traversal protection
    """

    # Default paths
    DEFAULT_HOME_DIR: ClassVar[str] = ".council"
    DEFAULT_ARTIFACT_SUBDIR: ClassVar[str] = "artifacts"
    DEFAULT_DB_NAME: ClassVar[str] = "ledger.db"
    LEGACY_HOME_DIR: ClassVar[str] = ".claude"
    LEGACY_ARTIFACT_SUBDIR: ClassVar[str] = "council-artifacts"
    LEGACY_DB_NAME: ClassVar[str] = "council-ledger.db"
    HOME_ENV_VAR: ClassVar[str] = "COUNCIL_HOME"
    ARTIFACT_DIR_ENV_VAR: ClassVar[str] = "COUNCIL_ARTIFACT_DIR"
    DB_PATH_ENV_VAR: ClassVar[str] = "COUNCIL_DB_PATH"

    # Token budgets (chars / 4 ≈ tokens)
    MAX_CAPSULE_TOKENS: ClassVar[int] = 500
    MAX_SUMMARY_CHARS: ClassVar[int] = 2000
    MAX_FINDING_CHARS: ClassVar[int] = 200
    MAX_FINDINGS: ClassVar[int] = 5

    def __init__(
        self,
        artifact_dir: Path | None = None,
        db_path: Path | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialize the artifact store.

        Args:
            artifact_dir: Directory for artifact files. Defaults to ~/.council/artifacts
            db_path: Path to SQLite ledger. Defaults to ~/.council/ledger.db
            enabled: If False, store operations become no-ops (for testing/disable mode)
        """
        default_artifact_dir, default_db_path = self._resolve_default_paths()
        self.artifact_dir = artifact_dir or default_artifact_dir
        self.db_path = db_path or default_db_path
        self.enabled = enabled

        if self.enabled:
            # Ensure directories exist
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    @classmethod
    def _resolve_default_paths(cls) -> tuple[Path, Path]:
        """Resolve default storage paths with neutral defaults and legacy fallback."""

        home_override = os.environ.get(cls.HOME_ENV_VAR)
        artifact_override = os.environ.get(cls.ARTIFACT_DIR_ENV_VAR)
        db_override = os.environ.get(cls.DB_PATH_ENV_VAR)

        if artifact_override or db_override:
            base_home = Path(home_override).expanduser() if home_override else None
            artifact_dir = (
                Path(artifact_override).expanduser()
                if artifact_override
                else (base_home / cls.DEFAULT_ARTIFACT_SUBDIR if base_home else cls._default_artifact_dir())
            )
            db_path = (
                Path(db_override).expanduser()
                if db_override
                else (base_home / cls.DEFAULT_DB_NAME if base_home else cls._default_db_path())
            )
            return artifact_dir, db_path

        if home_override:
            base_home = Path(home_override).expanduser()
            return base_home / cls.DEFAULT_ARTIFACT_SUBDIR, base_home / cls.DEFAULT_DB_NAME

        default_artifact_dir = cls._default_artifact_dir()
        default_db_path = cls._default_db_path()
        legacy_artifact_dir = cls._legacy_artifact_dir()
        legacy_db_path = cls._legacy_db_path()

        if not default_artifact_dir.exists() and not default_db_path.exists():
            if legacy_artifact_dir.exists() or legacy_db_path.exists():
                return legacy_artifact_dir, legacy_db_path

        return default_artifact_dir, default_db_path

    @classmethod
    def inspect_storage_paths(cls) -> StoragePaths:
        """Inspect active, default, and legacy storage locations."""

        active_artifact_dir, active_db_path = cls._resolve_default_paths()
        legacy_artifact_dir = cls._legacy_artifact_dir()
        legacy_db_path = cls._legacy_db_path()
        return StoragePaths(
            active_artifact_dir=active_artifact_dir,
            active_db_path=active_db_path,
            default_artifact_dir=cls._default_artifact_dir(),
            default_db_path=cls._default_db_path(),
            legacy_artifact_dir=legacy_artifact_dir,
            legacy_db_path=legacy_db_path,
            using_legacy=active_artifact_dir == legacy_artifact_dir and active_db_path == legacy_db_path,
        )

    @classmethod
    def _default_home_dir(cls) -> Path:
        return Path.home() / cls.DEFAULT_HOME_DIR

    @classmethod
    def _default_artifact_dir(cls) -> Path:
        return cls._default_home_dir() / cls.DEFAULT_ARTIFACT_SUBDIR

    @classmethod
    def _default_db_path(cls) -> Path:
        return cls._default_home_dir() / cls.DEFAULT_DB_NAME

    @classmethod
    def _legacy_home_dir(cls) -> Path:
        return Path.home() / cls.LEGACY_HOME_DIR

    @classmethod
    def _legacy_artifact_dir(cls) -> Path:
        return cls._legacy_home_dir() / cls.LEGACY_ARTIFACT_SUBDIR

    @classmethod
    def _legacy_db_path(cls) -> Path:
        return cls._legacy_home_dir() / cls.LEGACY_DB_NAME

    def _ensure_path_containment(self, path: Path) -> None:
        """Ensure path stays within artifact directory (prevent traversal)."""
        resolved = path.resolve()
        base_resolved = self.artifact_dir.resolve()
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            raise ValueError(f"Path escapes artifact directory: {path}") from None

    def _init_db(self) -> None:
        """Initialize SQLite ledger with schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                wave_id TEXT,
                subagent TEXT NOT NULL,
                task_hash TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                budget_output_tokens INTEGER DEFAULT 4000,
                actual_output_tokens INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
        """)

        # Artifacts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                byte_size INTEGER NOT NULL,
                token_estimate INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                processing_state TEXT NOT NULL DEFAULT 'unseen',
                created_at TEXT NOT NULL,
                summary TEXT,
                summary_tokens INTEGER DEFAULT 0,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)

        # Capsules table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS capsules (
                capsule_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                content TEXT NOT NULL,
                token_estimate INTEGER NOT NULL,
                ingested_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_hash ON artifacts(content_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_wave ON runs(wave_id)")

        conn.commit()
        conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection.

        Note: Use as context manager for proper resource cleanup:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                ...
        """
        return sqlite3.connect(self.db_path)

    def _content_hash(self, content: str) -> str:
        """Generate content hash for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (4 chars ≈ 1 token)."""
        return len(text) // 4

    def create_run(
        self,
        subagent: str,
        task: str,
        wave_id: str | None = None,
        budget_tokens: int = 4000,
    ) -> Run:
        """Create a new run record."""
        run = Run(
            run_id=str(uuid.uuid4()),
            wave_id=wave_id,
            subagent=subagent,
            task_hash=self._content_hash(task),
            budget_output_tokens=budget_tokens,
        )

        if not self.enabled:
            return run

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO runs (run_id, wave_id, subagent, task_hash, status,
                                budget_output_tokens, actual_output_tokens, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run.run_id,
                    run.wave_id,
                    run.subagent,
                    run.task_hash,
                    run.status,
                    run.budget_output_tokens,
                    run.actual_output_tokens,
                    run.created_at,
                ),
            )
            conn.commit()

        return run

    def store_artifact(
        self,
        run_id: str,
        content: str,
        artifact_type: ArtifactType,
        force_new: bool = False,
    ) -> Artifact:
        """
        Store an artifact, returning existing if content matches (dedup).

        Args:
            run_id: The run this artifact belongs to
            content: Full artifact content
            artifact_type: Type of artifact
            force_new: Skip deduplication check

        Returns:
            Artifact record (new or existing)
        """
        content_hash = self._content_hash(content)

        if not self.enabled:
            return Artifact(
                artifact_id=str(uuid.uuid4()),
                run_id=run_id,
                artifact_type=artifact_type.value,
                content_hash=content_hash,
                byte_size=len(content.encode()),
                token_estimate=self._estimate_tokens(content),
                file_path="",
            )

        # Check for existing artifact with same hash (dedup)
        if not force_new:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM artifacts WHERE content_hash = ? AND run_id = ?",
                    (content_hash, run_id),
                )
                row = cursor.fetchone()

            if row:
                return Artifact(
                    artifact_id=row[0],
                    run_id=row[1],
                    artifact_type=row[2],
                    content_hash=row[3],
                    byte_size=row[4],
                    token_estimate=row[5],
                    file_path=row[6],
                    processing_state=row[7],
                    created_at=row[8],
                    summary=row[9],
                    summary_tokens=row[10],
                )

        # Create new artifact with safe path
        artifact_id = str(uuid.uuid4())
        file_path = self.artifact_dir / f"{artifact_id}.txt"

        # Security: Ensure path is contained
        self._ensure_path_containment(file_path)

        # Write content atomically
        temp_path = file_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(content)
            temp_path.rename(file_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        artifact = Artifact(
            artifact_id=artifact_id,
            run_id=run_id,
            artifact_type=artifact_type.value,
            content_hash=content_hash,
            byte_size=len(content.encode()),
            token_estimate=self._estimate_tokens(content),
            file_path=str(file_path),
        )

        # Store in ledger
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO artifacts (artifact_id, run_id, artifact_type, content_hash,
                                      byte_size, token_estimate, file_path, processing_state,
                                      created_at, summary, summary_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    artifact.artifact_id,
                    artifact.run_id,
                    artifact.artifact_type,
                    artifact.content_hash,
                    artifact.byte_size,
                    artifact.token_estimate,
                    artifact.file_path,
                    artifact.processing_state,
                    artifact.created_at,
                    artifact.summary,
                    artifact.summary_tokens,
                ),
            )

            # Update run token count
            cursor.execute(
                """
                UPDATE runs SET actual_output_tokens = actual_output_tokens + ?
                WHERE run_id = ?
            """,
                (artifact.token_estimate, run_id),
            )

            conn.commit()

        return artifact

    def get_artifact_content(self, artifact_id: str) -> str | None:
        """Retrieve full artifact content from disk."""
        if not self.enabled:
            return None

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM artifacts WHERE artifact_id = ?", (artifact_id,))
            row = cursor.fetchone()

        if not row:
            return None

        file_path = Path(row[0])

        # Security: Verify path containment before reading
        try:
            self._ensure_path_containment(file_path)
        except ValueError:
            return None

        if not file_path.exists():
            return None

        with open(file_path, encoding="utf-8") as f:
            return f.read()

    def update_artifact_summary(
        self,
        artifact_id: str,
        summary: str,
        mark_processed: bool = True,
    ) -> None:
        """Update artifact with summary and optionally mark as processed."""
        if not self.enabled:
            return

        summary_tokens = self._estimate_tokens(summary)
        state = ProcessingState.SUMMARIZED.value if mark_processed else ProcessingState.UNSEEN.value

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE artifacts
                SET summary = ?, summary_tokens = ?, processing_state = ?
                WHERE artifact_id = ?
            """,
                (summary, summary_tokens, state, artifact_id),
            )
            conn.commit()

    def get_run_artifacts(self, run_id: str) -> list[Artifact]:
        """Get all artifacts for a run."""
        if not self.enabled:
            return []

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM artifacts WHERE run_id = ?", (run_id,))
            rows = cursor.fetchall()

        return [
            Artifact(
                artifact_id=row[0],
                run_id=row[1],
                artifact_type=row[2],
                content_hash=row[3],
                byte_size=row[4],
                token_estimate=row[5],
                file_path=row[6],
                processing_state=row[7],
                created_at=row[8],
                summary=row[9],
                summary_tokens=row[10],
            )
            for row in rows
        ]

    def complete_run(self, run_id: str, status: str = "completed") -> None:
        """Mark a run as completed."""
        if not self.enabled:
            return

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE runs SET status = ?, completed_at = ?
                WHERE run_id = ?
            """,
                (status, datetime.now(timezone.utc).isoformat(), run_id),
            )
            conn.commit()

    def create_capsule(
        self,
        run_id: str,
        status: str,
        summary: str,
        key_findings: list[str],
        blockers: list[str] | None = None,
        next_actions: list[str] | None = None,
    ) -> ResultCapsule:
        """
        Create a bounded result capsule for context ingestion.

        Enforces size limits to prevent context bloat.
        """
        # Get artifact refs for this run
        artifacts = self.get_run_artifacts(run_id)
        artifact_refs = [a.artifact_id for a in artifacts]

        # Enforce limits
        bounded_summary = summary[: self.MAX_SUMMARY_CHARS]
        bounded_findings = [f[: self.MAX_FINDING_CHARS] for f in key_findings[: self.MAX_FINDINGS]]
        bounded_blockers = (blockers or [])[:3]
        bounded_actions = (next_actions or [])[:3]

        capsule = ResultCapsule(
            run_id=run_id,
            status=status,
            summary=bounded_summary,
            key_findings=bounded_findings,
            blockers=bounded_blockers,
            next_actions=bounded_actions,
            artifact_refs=artifact_refs,
            token_estimate=0,
        )

        # Calculate token estimate
        context_string = capsule.to_context_string()
        capsule.token_estimate = self._estimate_tokens(context_string)

        if not self.enabled:
            return capsule

        # Store capsule in ledger
        capsule_id = str(uuid.uuid4())
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO capsules (capsule_id, run_id, content, token_estimate, ingested_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    capsule_id,
                    run_id,
                    context_string,
                    capsule.token_estimate,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

        return capsule

    def rehydrate(
        self,
        artifact_id: str,
        max_chars: int = 4000,
        filter_type: str | None = None,
    ) -> str:
        """
        Retrieve bounded excerpt from artifact for context injection.

        Args:
            artifact_id: The artifact to rehydrate
            max_chars: Maximum characters to return
            filter_type: Optional filter (e.g., "errors_only", "code_only")

        Returns:
            Bounded excerpt with truncation notice if needed
        """
        content = self.get_artifact_content(artifact_id)
        if not content:
            return f"[Artifact {artifact_id} not found]"

        # Apply filters if specified
        if filter_type == "errors_only":
            lines = content.split("\n")
            error_lines = [
                line for line in lines if "error" in line.lower() or "exception" in line.lower()
            ]
            content = "\n".join(error_lines) if error_lines else "[No errors found]"
        elif filter_type == "code_only":
            import re

            code_blocks = re.findall(r"```[\s\S]*?```", content)
            content = "\n\n".join(code_blocks) if code_blocks else "[No code blocks found]"

        # Apply size limit
        if len(content) <= max_chars:
            return content

        return (
            content[:max_chars]
            + f"\n\n[... truncated, {len(content) - max_chars} chars remaining, "
            f"artifact_id={artifact_id}]"
        )

    def cleanup_old_artifacts(self, days_old: int = 7, max_files: int | None = None) -> int:
        """Remove artifacts older than specified days or exceeding max count.

        Args:
            days_old: Remove artifacts older than this many days
            max_files: Maximum number of artifact files to keep (oldest removed first)

        Returns:
            Number of artifacts removed
        """
        if not self.enabled:
            return 0

        removed = 0
        cutoff = datetime.now(timezone.utc).timestamp() - (days_old * 86400)
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()

        with self._get_conn() as conn:
            cursor = conn.cursor()

            # Get old archived artifacts
            cursor.execute(
                """
                SELECT artifact_id, file_path FROM artifacts
                WHERE created_at < ? AND processing_state = 'archived'
            """,
                (cutoff_iso,),
            )
            rows = cursor.fetchall()

            # Delete files and records (in transaction)
            files_to_delete = []
            for artifact_id, file_path in rows:
                cursor.execute("DELETE FROM artifacts WHERE artifact_id = ?", (artifact_id,))
                files_to_delete.append(file_path)
                removed += 1

            # Optionally enforce max file count
            if max_files is not None:
                cursor.execute("SELECT COUNT(*) FROM artifacts")
                total = cursor.fetchone()[0]
                if total > max_files:
                    excess = total - max_files
                    cursor.execute(
                        """
                        SELECT artifact_id, file_path FROM artifacts
                        ORDER BY created_at ASC LIMIT ?
                    """,
                        (excess,),
                    )
                    for artifact_id, file_path in cursor.fetchall():
                        cursor.execute(
                            "DELETE FROM artifacts WHERE artifact_id = ?", (artifact_id,)
                        )
                        files_to_delete.append(file_path)
                        removed += 1

            conn.commit()

        # Delete files after successful commit (outside transaction)
        for file_path in files_to_delete:
            try:
                fp = Path(file_path)
                self._ensure_path_containment(fp)
                fp.unlink(missing_ok=True)
            except Exception:
                pass

        return removed

    def cleanup_stale_runs(self, age_hours: float = 1.0) -> int:
        """Mark stale runs (still 'running' after age_hours) as 'timed_out'.

        Cleans up orphaned runs that were never properly completed due to
        crashes, timeouts, or other failures.

        Args:
            age_hours: Age threshold in hours (default: 1.0)

        Returns:
            Number of runs cleaned up
        """
        if not self.enabled:
            return 0

        cutoff = datetime.now(timezone.utc).timestamp() - (age_hours * 3600)
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE runs
                SET status = 'timed_out', completed_at = ?
                WHERE status = 'running' AND created_at < ?
            """,
                (datetime.now(timezone.utc).isoformat(), cutoff_iso),
            )
            cleaned = cursor.rowcount
            conn.commit()

        return cleaned

    @classmethod
    def migrate_legacy_storage(
        cls,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> StorageMigrationResult:
        """Copy legacy ~/.claude council storage into the neutral target path."""

        paths = cls.inspect_storage_paths()
        source_artifact_dir = paths.legacy_artifact_dir
        source_db_path = paths.legacy_db_path
        target_artifact_dir = paths.default_artifact_dir
        target_db_path = paths.default_db_path

        legacy_exists = source_artifact_dir.exists() or source_db_path.exists()
        if not legacy_exists:
            return StorageMigrationResult(
                source_artifact_dir=source_artifact_dir,
                source_db_path=source_db_path,
                target_artifact_dir=target_artifact_dir,
                target_db_path=target_db_path,
                dry_run=dry_run,
                migrated=False,
                copied_files=0,
                copied_db=False,
                message="No legacy council storage found under ~/.claude.",
            )

        target_exists = target_artifact_dir.exists() or target_db_path.exists()
        if target_exists and not force:
            raise RuntimeError(
                "Neutral council storage already exists. Refusing to merge automatically; "
                "rerun with --force if you want to overwrite the target."
            )

        copied_files = 0
        copied_db = False
        if dry_run:
            if source_artifact_dir.exists():
                copied_files = sum(1 for path in source_artifact_dir.rglob("*") if path.is_file())
            copied_db = source_db_path.exists()
            return StorageMigrationResult(
                source_artifact_dir=source_artifact_dir,
                source_db_path=source_db_path,
                target_artifact_dir=target_artifact_dir,
                target_db_path=target_db_path,
                dry_run=True,
                migrated=False,
                copied_files=copied_files,
                copied_db=copied_db,
                message="Dry run only; no files were copied.",
            )

        if force:
            shutil.rmtree(target_artifact_dir, ignore_errors=True)
            target_db_path.unlink(missing_ok=True)

        target_artifact_dir.mkdir(parents=True, exist_ok=True)
        target_db_path.parent.mkdir(parents=True, exist_ok=True)

        if source_artifact_dir.exists():
            for source_path in source_artifact_dir.rglob("*"):
                if not source_path.is_file():
                    continue
                relative_path = source_path.relative_to(source_artifact_dir)
                destination_path = target_artifact_dir / relative_path
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination_path)
                copied_files += 1

        if source_db_path.exists():
            shutil.copy2(source_db_path, target_db_path)
            copied_db = True
            if source_artifact_dir.exists():
                with sqlite3.connect(target_db_path) as conn:
                    conn.execute(
                        """
                        UPDATE artifacts
                        SET file_path = replace(file_path, ?, ?)
                        WHERE file_path LIKE ?
                        """,
                        (
                            str(source_artifact_dir),
                            str(target_artifact_dir),
                            f"{source_artifact_dir}%",
                        ),
                    )
                    conn.commit()

        if source_artifact_dir.exists():
            source_files = sorted(
                str(path.relative_to(source_artifact_dir))
                for path in source_artifact_dir.rglob("*")
                if path.is_file()
            )
            target_files = sorted(
                str(path.relative_to(target_artifact_dir))
                for path in target_artifact_dir.rglob("*")
                if path.is_file()
            )
            if source_files != target_files:
                raise RuntimeError("Artifact migration verification failed: copied file set mismatch.")

        if source_db_path.exists():
            source_conn = sqlite3.connect(source_db_path)
            target_conn = sqlite3.connect(target_db_path)
            try:
                source_ok = source_conn.execute("PRAGMA integrity_check").fetchone()
                target_ok = target_conn.execute("PRAGMA integrity_check").fetchone()
            finally:
                source_conn.close()
                target_conn.close()
            if source_ok != ("ok",) or target_ok != ("ok",):
                raise RuntimeError("Ledger migration verification failed: SQLite integrity check failed.")

        return StorageMigrationResult(
            source_artifact_dir=source_artifact_dir,
            source_db_path=source_db_path,
            target_artifact_dir=target_artifact_dir,
            target_db_path=target_db_path,
            dry_run=False,
            migrated=True,
            copied_files=copied_files,
            copied_db=copied_db,
            message="Legacy council storage copied to neutral ~/.council home.",
        )


# Module-level default store singleton
_default_store: ArtifactStore | None = None


def get_store(enabled: bool = True) -> ArtifactStore:
    """Get or create the default artifact store."""
    global _default_store
    if _default_store is None:
        _default_store = ArtifactStore(enabled=enabled)
    return _default_store


def reset_store() -> None:
    """Reset the default store singleton (for testing)."""
    global _default_store
    _default_store = None


__all__ = [
    "ArtifactStore",
    "Artifact",
    "ArtifactType",
    "ProcessingState",
    "ResultCapsule",
    "Run",
    "StorageMigrationResult",
    "StoragePaths",
    "get_store",
    "reset_store",
]
