"""Local-only import helpers for external review benchmark material."""

from __future__ import annotations

import html
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_PRIVATE_IMPORT_ROOT = Path(".council-private/imports/github")
GREPTILE_AUTHOR_MARKERS = ("greptile",)
GREPTILE_COMMENT_BLOCK_RE = re.compile(
    r"<!--\s*greptile_comment\s*-->.*?<!--\s*/greptile_comment\s*-->",
    re.DOTALL | re.IGNORECASE,
)


class ImportedReviewComment(BaseModel):
    """Normalized inline review comment."""

    model_config = ConfigDict(extra="forbid")

    comment_id: int
    author_login: str
    file_path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    severity: str | None = None
    body_markdown: str
    body_text: str
    suggested_code: str | None = None
    addressed: bool | None = None


class ImportedPullRequest(BaseModel):
    """Locally stored PR import record."""

    model_config = ConfigDict(extra="forbid")

    repo: str
    pr_number: int
    title: str
    state: str
    is_draft: bool
    base_ref: str
    head_ref: str
    additions: int
    deletions: int
    changed_files: int
    url: str
    import_root: str
    diff_path: str
    review_input_path: str
    metadata_path: str
    raw_comments_path: str
    greptile_labels_path: str
    imported_comment_count: int


def import_github_pr_review(
    repo: str,
    pr_number: int,
    *,
    output_root: str | Path | None = None,
    max_diff_lines: int | None = None,
) -> ImportedPullRequest:
    """Import one GitHub PR into a local-only review benchmark bundle."""

    root = Path(output_root) if output_root else DEFAULT_PRIVATE_IMPORT_ROOT
    import_dir = root / _repo_slug(repo) / f"pr-{pr_number}"
    import_dir.mkdir(parents=True, exist_ok=True)

    pr_data = _gh_json(
        [
            "pr",
            "view",
            str(pr_number),
            "--repo",
            repo,
            "--json",
            "number,title,body,state,isDraft,baseRefName,headRefName,additions,deletions,changedFiles,url",
        ]
    )
    comments_data = _gh_json(
        [
            "api",
            f"repos/{repo}/pulls/{pr_number}/comments?per_page=100",
        ]
    )
    diff_text = _gh_text(
        [
            "pr",
            "diff",
            str(pr_number),
            "--repo",
            repo,
            "--patch",
            "--color=never",
        ]
    )
    if max_diff_lines is not None and max_diff_lines > 0:
        diff_text = _truncate_diff(diff_text, max_diff_lines=max_diff_lines)

    greptile_comments = [
        _normalize_review_comment(item)
        for item in comments_data
        if _is_greptile_author(item.get("user", {}).get("login"))
    ]

    metadata_path = import_dir / "pr.json"
    raw_comments_path = import_dir / "review_comments.raw.json"
    labels_path = import_dir / "greptile_labels.json"
    diff_path = import_dir / "diff.patch"
    review_input_path = import_dir / "review_input.md"

    metadata_path.write_text(json.dumps(pr_data, indent=2), encoding="utf-8")
    raw_comments_path.write_text(json.dumps(comments_data, indent=2), encoding="utf-8")
    labels_path.write_text(
        json.dumps([item.model_dump() for item in greptile_comments], indent=2),
        encoding="utf-8",
    )
    diff_path.write_text(diff_text, encoding="utf-8")
    review_input_path.write_text(
        _build_review_input_markdown(pr_data, repo=repo, diff_text=diff_text),
        encoding="utf-8",
    )

    return ImportedPullRequest(
        repo=repo,
        pr_number=pr_number,
        title=pr_data["title"],
        state=pr_data["state"],
        is_draft=bool(pr_data["isDraft"]),
        base_ref=pr_data["baseRefName"],
        head_ref=pr_data["headRefName"],
        additions=int(pr_data["additions"]),
        deletions=int(pr_data["deletions"]),
        changed_files=int(pr_data["changedFiles"]),
        url=pr_data["url"],
        import_root=str(import_dir),
        diff_path=str(diff_path),
        review_input_path=str(review_input_path),
        metadata_path=str(metadata_path),
        raw_comments_path=str(raw_comments_path),
        greptile_labels_path=str(labels_path),
        imported_comment_count=len(greptile_comments),
    )


def _gh_json(args: list[str]) -> Any:
    output = _gh_text(args)
    return json.loads(output)


def _gh_text(args: list[str]) -> str:
    command = ["gh", *args]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _repo_slug(repo: str) -> str:
    return repo.replace("/", "__")


def _is_greptile_author(login: str | None) -> bool:
    normalized = (login or "").lower()
    return any(marker in normalized for marker in GREPTILE_AUTHOR_MARKERS)


def _normalize_review_comment(payload: dict[str, Any]) -> ImportedReviewComment:
    body_markdown = payload.get("body") or ""
    suggestion = _extract_suggested_code(body_markdown)
    return ImportedReviewComment(
        comment_id=int(payload["id"]),
        author_login=payload.get("user", {}).get("login") or "",
        file_path=payload.get("path"),
        line_start=payload.get("line") or payload.get("start_line"),
        line_end=payload.get("line") or payload.get("start_line"),
        severity=_extract_severity(body_markdown),
        body_markdown=body_markdown,
        body_text=_normalize_comment_text(body_markdown),
        suggested_code=suggestion,
        addressed=None,
    )


def _extract_severity(body: str) -> str | None:
    match = re.search(r'alt="(P[0-3])"', body, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b(P[0-3])\b", body)
    if match:
        return match.group(1).upper()
    return None


def _extract_suggested_code(body: str) -> str | None:
    match = re.search(r"```suggestion\n(.*?)```", body, re.DOTALL)
    if not match:
        return None
    return match.group(1).rstrip()


def _normalize_comment_text(body: str) -> str:
    stripped = re.sub(r"<[^>]+>", "", body)
    stripped = html.unescape(stripped)
    stripped = re.sub(r"```suggestion.*?```", "", stripped, flags=re.DOTALL)
    stripped = re.sub(r"`([^`]+)`", r"\1", stripped)
    stripped = re.sub(r"\s+\n", "\n", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


def _build_review_input_markdown(pr_data: dict[str, Any], *, repo: str, diff_text: str) -> str:
    body = _strip_greptile_block(pr_data.get("body") or "")
    lines = [
        "# Pull Request Review Input",
        "",
        f"- Repo: `{repo}`",
        f"- PR: `{pr_data['number']}`",
        f"- Title: {pr_data['title']}",
        f"- State: {pr_data['state']}",
        f"- Draft: {'yes' if pr_data['isDraft'] else 'no'}",
        f"- Base: `{pr_data['baseRefName']}`",
        f"- Head: `{pr_data['headRefName']}`",
        f"- Changed files: {pr_data['changedFiles']}",
        f"- Additions: {pr_data['additions']}",
        f"- Deletions: {pr_data['deletions']}",
        "",
        "## PR Description",
        body.strip() or "_No description provided._",
        "",
        "## Patch",
        "```diff",
        diff_text.rstrip(),
        "```",
        "",
    ]
    return "\n".join(lines)


def _strip_greptile_block(body: str) -> str:
    return GREPTILE_COMMENT_BLOCK_RE.sub("", body).strip()


def _truncate_diff(diff_text: str, *, max_diff_lines: int) -> str:
    lines = diff_text.splitlines()
    if len(lines) <= max_diff_lines:
        return diff_text
    head = lines[:max_diff_lines]
    head.append(f"... [truncated diff at {max_diff_lines} lines]")
    return "\n".join(head) + "\n"


__all__ = [
    "DEFAULT_PRIVATE_IMPORT_ROOT",
    "ImportedPullRequest",
    "ImportedReviewComment",
    "import_github_pr_review",
]
