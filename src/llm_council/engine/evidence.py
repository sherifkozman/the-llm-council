"""Local evidence collection for capability-backed council runs."""

from __future__ import annotations

import asyncio
import html
import re
import subprocess
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from llm_council.engine.capabilities import CapabilityPlan

TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".txt",
    ".tsx",
    ".ts",
    ".js",
    ".jsx",
    ".sh",
}
SEARCH_DIRS = ("src", "tests", "config", "docs")
STOPWORDS = {
    "this",
    "that",
    "with",
    "from",
    "into",
    "using",
    "their",
    "there",
    "should",
    "could",
    "would",
    "about",
    "after",
    "before",
    "review",
    "build",
    "plan",
    "security",
    "system",
    "mode",
    "task",
}
SECURITY_PATTERNS = (
    "auth",
    "token",
    "secret",
    "password",
    "session",
    "cookie",
    "oauth",
    "subprocess",
    "shell",
    "exec",
    "eval",
    "encrypt",
    "hash",
    "csrf",
    "jwt",
)
ATTACK_SURFACE_PATTERNS = (
    "app.get",
    "app.post",
    "router.get",
    "router.post",
    "apirouter",
    "@app.",
    "@router.",
    "webhook",
    "callback",
    "oauth",
    "login",
    "session",
    "admin",
    "/api/",
    "graphql",
    "upload",
    "public",
)
DOCS_TARGETS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("react", "reactjs"), "https://react.dev/reference/react"),
    (("next.js", "nextjs"), "https://nextjs.org/docs"),
    (("fastapi",), "https://fastapi.tiangolo.com/"),
    (("pydantic",), "https://docs.pydantic.dev/latest/"),
    (("pytest",), "https://docs.pytest.org/en/stable/"),
    (("typescript",), "https://www.typescriptlang.org/docs/"),
    (("node.js", "nodejs"), "https://nodejs.org/api/documentation.html"),
    (("docker",), "https://docs.docker.com/"),
    (("kubernetes", "k8s"), "https://kubernetes.io/docs/home/"),
    (("redis",), "https://redis.io/docs/latest/"),
    (("postgres", "postgresql"), "https://www.postgresql.org/docs/current/index.html"),
    (("stripe",), "https://docs.stripe.com/api"),
    (("oauth", "oidc", "openid connect", "sso"), "https://datatracker.ietf.org/doc/html/rfc6749"),
    (("rate limit", "rate limiting"), "https://datatracker.ietf.org/doc/html/rfc6585"),
    (("openapi",), "https://spec.openapis.org/oas/latest.html"),
)
DOCS_FETCH_LIMIT = 2
DOCS_TIMEOUT_SECONDS = 5.0
DOCS_USER_AGENT = "llm-council/0.6 docs-research"
DIFF_FETCH_TIMEOUT_SECONDS = 5.0
DIFF_HUNK_LIMIT = 8
DIFF_FILE_LIMIT = 12


class EvidenceItem(BaseModel):
    """One collected evidence item."""

    model_config = ConfigDict(extra="forbid")

    capability: str
    title: str
    summary: str
    details: list[str] = Field(default_factory=list)


class EvidenceBundle(BaseModel):
    """Collected evidence for a council run."""

    model_config = ConfigDict(extra="forbid")

    executed_capabilities: list[str] = Field(default_factory=list)
    pending_capabilities: list[str] = Field(default_factory=list)
    items: list[EvidenceItem] = Field(default_factory=list)

    def to_prompt_block(self) -> str:
        """Render collected evidence as a bounded prompt block."""

        if not self.items and not self.pending_capabilities:
            return ""

        lines = ["## Collected Evidence"]
        if self.executed_capabilities:
            lines.append("Executed capability packs: " + ", ".join(self.executed_capabilities))
        if self.pending_capabilities:
            lines.append(
                "Capability packs not executed in this runtime: "
                + ", ".join(self.pending_capabilities)
            )
        for item in self.items:
            lines.append(f"- [{item.capability}] {item.title}: {item.summary}")
            for detail in item.details[:6]:
                lines.append(f"  - {detail}")
        return "\n".join(lines) + "\n"


async def collect_capability_evidence(
    task: str,
    subagent: str,
    mode: str | None,
    capability_plan: CapabilityPlan,
    *,
    repo_root: Path | None = None,
) -> EvidenceBundle:
    """Collect bounded local evidence for supported capability packs."""

    return await asyncio.to_thread(
        _collect_capability_evidence_sync,
        task,
        subagent,
        mode,
        capability_plan,
        repo_root or Path.cwd(),
    )


def _collect_capability_evidence_sync(
    task: str,
    subagent: str,
    mode: str | None,
    capability_plan: CapabilityPlan,
    repo_root: Path,
) -> EvidenceBundle:
    bundle = EvidenceBundle()
    files = _candidate_files(repo_root)

    for capability in capability_plan.required_capabilities:
        collector = _CAPABILITY_COLLECTORS.get(capability)
        if collector is None:
            bundle.pending_capabilities.append(capability)
            continue

        item = collector(task, subagent, mode, repo_root, files)
        if item is None:
            bundle.pending_capabilities.append(capability)
            continue
        bundle.executed_capabilities.append(capability)
        bundle.items.append(item)

    return bundle


def _candidate_files(repo_root: Path) -> list[Path]:
    """Return bounded candidate files for evidence scanning."""

    files: list[Path] = []
    for dirname in SEARCH_DIRS:
        base = repo_root / dirname
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            files.append(path)
    return files


def _collect_repo_analysis(
    task: str, _subagent: str, _mode: str | None, repo_root: Path, files: list[Path]
) -> EvidenceItem:
    """Collect a bounded repository footprint plus task-keyword hits."""

    sample_files = [str(path.relative_to(repo_root)) for path in files[:8]]
    keywords = _extract_keywords(task)
    hits = _search_files(files, keywords, repo_root, limit=10) if keywords else []

    summary = (
        f"Scanned {len(files)} text files under {', '.join(SEARCH_DIRS)}. "
        f"Task-keyword hits found: {len(hits)}."
    )
    details = []
    if sample_files:
        details.append("Sample files: " + ", ".join(sample_files))
    details.extend(hits)

    return EvidenceItem(
        capability="repo-analysis",
        title="Repository footprint and task-relevant matches",
        summary=summary,
        details=details,
    )


def _collect_planning_assess(
    task: str, _subagent: str, mode: str | None, repo_root: Path, _files: list[Path]
) -> EvidenceItem:
    """Collect planning artifacts and a mode-specific checklist scaffold."""

    planning_artifacts: list[str] = []
    artifact_candidates = [
        repo_root / "README.md",
        repo_root / "ROADMAP.md",
        repo_root / "docs" / "index.md",
    ]
    artifact_candidates.extend(sorted((repo_root / "docs" / "architecture").glob("*.md")))

    for path in artifact_candidates:
        if path.exists() and path.is_file():
            planning_artifacts.append(str(path.relative_to(repo_root)))

    if mode == "assess":
        checklist = [
            "Define the decision question and success criteria",
            "Score alternatives against explicit criteria",
            "Record reversibility, dependencies, and risks",
            "State conditions or next steps before recommending proceed/reject",
        ]
    else:
        checklist = [
            "Define objective and measurable success criteria",
            "Sequence work into phases with dependencies",
            "Identify blockers, risks, and fallback paths",
            "Assign deliverables and exit criteria per phase",
        ]

    summary = (
        f"Found {len(planning_artifacts)} planning artifact(s) and generated a "
        f"{mode or 'plan'} checklist scaffold for '{task[:40]}'."
    )
    details = []
    if planning_artifacts:
        details.append("Planning artifacts: " + ", ".join(planning_artifacts[:6]))
    details.extend(checklist)

    return EvidenceItem(
        capability="planning-assess",
        title="Planning artifacts and checklist scaffold",
        summary=summary,
        details=details,
    )


def _collect_security_code_audit(
    _task: str, _subagent: str, _mode: str | None, repo_root: Path, files: list[Path]
) -> EvidenceItem:
    """Collect security-sensitive code hits for code-audit style analysis."""

    hits = _search_files(files, SECURITY_PATTERNS, repo_root, limit=14)
    summary = (
        f"Scanned {len(files)} text files for security-sensitive patterns. "
        f"High-signal hits found: {len(hits)}."
    )

    details = hits or ["No high-signal security keyword matches found in scanned files."]
    return EvidenceItem(
        capability="security-code-audit",
        title="Security-sensitive code-path matches",
        summary=summary,
        details=details,
    )


def _collect_security_audit(
    _task: str, _subagent: str, _mode: str | None, repo_root: Path, files: list[Path]
) -> EvidenceItem:
    """Compatibility collector for the legacy security-audit capability."""

    item = _collect_security_code_audit(_task, _subagent, _mode, repo_root, files)
    item.capability = "security-audit"
    item.title = "Security-sensitive repository matches"
    return item


def _collect_red_team_recon(
    _task: str, _subagent: str, _mode: str | None, repo_root: Path, files: list[Path]
) -> EvidenceItem:
    """Collect attack-surface indicators for red-team style analysis."""

    hits = _search_files(files, ATTACK_SURFACE_PATTERNS, repo_root, limit=14)
    summary = (
        f"Scanned {len(files)} text files for attack-surface indicators. "
        f"High-signal hits found: {len(hits)}."
    )
    details = hits or ["No obvious attack-surface indicators found in scanned files."]
    return EvidenceItem(
        capability="red-team-recon",
        title="Attack-surface and exposure indicators",
        summary=summary,
        details=details,
    )


def _collect_diff_review(
    _task: str, _subagent: str, _mode: str | None, repo_root: Path, _files: list[Path]
) -> EvidenceItem | None:
    """Collect bounded evidence from the current git diff."""

    diff_text = _load_git_diff(repo_root)
    if not diff_text:
        return None

    parsed = _parse_unified_diff(diff_text)
    if not parsed["files"]:
        return None

    changed_files = parsed["files"][:DIFF_FILE_LIMIT]
    hunk_summaries = parsed["hunks"][:DIFF_HUNK_LIMIT]
    test_hints = _infer_test_impact(repo_root, changed_files)

    details: list[str] = []
    details.append("Changed files: " + ", ".join(changed_files))
    details.extend(hunk_summaries)
    details.extend(test_hints)

    summary = (
        f"Captured {len(changed_files)} changed file(s) and {len(hunk_summaries)} diff hunk summary line(s) "
        "from the current git diff."
    )
    return EvidenceItem(
        capability="diff-review",
        title="Changed files and diff-hunk anchors",
        summary=summary,
        details=details,
    )


def _collect_docs_research(
    task: str, _subagent: str, _mode: str | None, _repo_root: Path, _files: list[Path]
) -> EvidenceItem | None:
    """Fetch bounded official documentation references inferred from the task."""

    targets = _infer_docs_targets(task)
    if not targets:
        return None

    fetched: list[tuple[str, str, str]] = []
    for url in targets[:DOCS_FETCH_LIMIT]:
        page = _fetch_docs_page(url)
        if page is None:
            continue
        title, summary = page
        fetched.append((url, title, summary))

    if not fetched:
        return None

    details = [f"{title} ({url}): {summary}" for url, title, summary in fetched]
    summary = (
        f"Fetched {len(fetched)} official documentation source(s) inferred from the task. "
        f"Unverified targets skipped: {max(0, len(targets[:DOCS_FETCH_LIMIT]) - len(fetched))}."
    )
    return EvidenceItem(
        capability="docs-research",
        title="Official documentation references",
        summary=summary,
        details=details,
    )


def _extract_keywords(task: str, limit: int = 6) -> list[str]:
    """Extract a small keyword set from the task."""

    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", task.lower())
    keywords: list[str] = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def _search_files(
    files: list[Path],
    patterns: list[str] | tuple[str, ...],
    repo_root: Path,
    limit: int,
) -> list[str]:
    """Search candidate files for keyword hits and return bounded detail lines."""

    if not patterns:
        return []

    regex = re.compile("|".join(re.escape(pattern) for pattern in patterns), re.IGNORECASE)
    hits: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            if not regex.search(line):
                continue
            snippet = line.strip()
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            try:
                display_path = str(path.relative_to(repo_root))
            except ValueError:
                display_path = str(path)
            hits.append(f"{display_path}:{line_number}: {snippet}")
            if len(hits) >= limit:
                return hits

    return hits


def _infer_docs_targets(task: str) -> list[str]:
    """Infer official documentation URLs from task keywords and explicit URLs."""

    lower_task = task.lower()
    targets: list[str] = []

    for match in re.findall(r"https?://[^\s)>\"']+", task):
        if match.startswith("https://") and match not in targets:
            targets.append(match.rstrip(".,);"))

    for keywords, url in DOCS_TARGETS:
        if any(keyword in lower_task for keyword in keywords) and url not in targets:
            targets.append(url)

    return targets


def _fetch_docs_page(url: str) -> tuple[str, str] | None:
    """Fetch a documentation page and extract a small grounded summary."""

    try:
        with httpx.Client(
            timeout=DOCS_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers={"User-Agent": DOCS_USER_AGENT},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPError:
        return None

    content_type = response.headers.get("content-type", "").lower()
    if not any(token in content_type for token in ("text/html", "text/plain", "application/xhtml+xml")):
        return None

    return _extract_page_summary(response.text, url)


def _load_git_diff(repo_root: Path) -> str:
    """Load a bounded unified diff for the current repository state."""

    commands = [
        ["git", "diff", "--no-ext-diff", "--unified=3", "--", "."],
        ["git", "diff", "--no-ext-diff", "--cached", "--unified=3", "--", "."],
    ]
    diff_parts: list[str] = []

    for command in commands:
        try:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=DIFF_FETCH_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            return ""

        if completed.returncode not in (0, 1):
            continue
        stdout = completed.stdout.strip()
        if stdout:
            diff_parts.append(stdout)

    if not diff_parts:
        return ""
    return "\n".join(diff_parts)


def _parse_unified_diff(diff_text: str) -> dict[str, list[str]]:
    """Parse a minimal summary from unified diff text."""

    files: list[str] = []
    hunks: list[str] = []
    current_file: str | None = None

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                current_file = parts[3][2:]
                if current_file not in files:
                    files.append(current_file)
            continue

        if line.startswith("@@"):
            summary = line.strip()
            if current_file:
                hunks.append(f"{current_file}: {summary}")
            else:
                hunks.append(summary)
            continue

    return {"files": files, "hunks": hunks}


def _infer_test_impact(repo_root: Path, changed_files: list[str]) -> list[str]:
    """Infer likely test impact from changed files using simple path conventions."""

    hints: list[str] = []
    for rel_path in changed_files[:DIFF_FILE_LIMIT]:
        path = Path(rel_path)
        if path.parts and path.parts[0] == "tests":
            hints.append(f"Test file changed directly: {rel_path}")
            continue

        candidate_tests = _candidate_test_paths(path)
        existing = [candidate for candidate in candidate_tests if (repo_root / candidate).exists()]
        if existing:
            hints.append(f"Likely impacted tests for {rel_path}: {', '.join(existing[:2])}")

    return hints[:6]


def _candidate_test_paths(path: Path) -> list[str]:
    """Generate likely test file paths for a changed source file."""

    candidates: list[str] = []
    if not path.suffix:
        return candidates

    stem = path.stem
    suffix = path.suffix
    file_name = path.name

    candidates.append(str(Path("tests") / f"test_{stem}{suffix}"))
    candidates.append(str(Path("tests") / file_name))

    if path.parts and path.parts[0] == "src":
        relative = Path(*path.parts[1:])
        candidates.append(str(Path("tests") / relative.parent / f"test_{relative.stem}{suffix}"))
        candidates.append(str(Path("tests") / relative.parent / relative.name))

    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _extract_page_summary(content: str, fallback_url: str) -> tuple[str, str]:
    """Extract a short title and summary from HTML or plain-text docs content."""

    title_match = re.search(r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = _clean_text(title_match.group(1))
    else:
        heading_match = re.search(r"^\s*#\s+(.+)$", content, re.MULTILINE)
        title = _clean_text(heading_match.group(1)) if heading_match else fallback_url

    meta_match = re.search(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
        content,
        re.IGNORECASE | re.DOTALL,
    )
    if meta_match:
        summary = _clean_text(meta_match.group(1))
    else:
        text_content = _clean_text(re.sub(r"<[^>]+>", " ", content))
        summary = text_content[:200].rsplit(" ", 1)[0] if len(text_content) > 200 else text_content

    return title or fallback_url, summary or "Documentation page fetched with no extractable summary."


def _clean_text(value: str) -> str:
    """Normalize HTML/plain-text content into a single compact line."""

    normalized = html.unescape(value)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


_CAPABILITY_COLLECTORS: dict[str, Any] = {
    "repo-analysis": _collect_repo_analysis,
    "planning-assess": _collect_planning_assess,
    "security-audit": _collect_security_audit,
    "security-code-audit": _collect_security_code_audit,
    "red-team-recon": _collect_red_team_recon,
    "docs-research": _collect_docs_research,
    "diff-review": _collect_diff_review,
}


__all__ = ["EvidenceBundle", "EvidenceItem", "collect_capability_evidence"]
