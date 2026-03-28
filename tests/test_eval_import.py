"""Tests for local-only PR review import helpers."""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from llm_council.eval_import import import_github_pr_review


class TestEvalImport:
    """Tests for GitHub PR import into local-only review bundles."""

    def test_import_github_pr_review_writes_bundle(self, tmp_path):
        """Importer should write diff, metadata, labels, and review input files."""
        pr_json = {
            "number": 42,
            "title": "Fix auth edge case",
            "body": "## Summary\nFixes a bug.\n<!-- greptile_comment -->hidden<!-- /greptile_comment -->",
            "state": "OPEN",
            "isDraft": False,
            "baseRefName": "main",
            "headRefName": "fix/auth",
            "additions": 10,
            "deletions": 2,
            "changedFiles": 1,
            "url": "https://github.com/acme/app/pull/42",
        }
        comments_json = [
            {
                "id": 1001,
                "body": (
                    '<a href="#"><img alt="P1" src="badge"></a> **Leaky internal error**\n\n'
                    "Use an opaque 500 response.\n\n"
                    '```suggestion\nhttp.Error(w, "internal server error", http.StatusInternalServerError)\n```'
                ),
                "path": "backend/handler.go",
                "line": 77,
                "start_line": 77,
                "user": {"login": "greptile-apps[bot]"},
            },
            {
                "id": 1002,
                "body": "human comment",
                "path": "backend/handler.go",
                "line": 80,
                "start_line": 80,
                "user": {"login": "octocat"},
            },
        ]
        diff_text = "diff --git a/a.go b/a.go\n@@ -1 +1 @@\n-old\n+new\n"

        def fake_run(command, check, capture_output, text):
            joined = " ".join(command)
            if "pr view" in joined:
                return CompletedProcess(command, 0, stdout=json.dumps(pr_json), stderr="")
            if "pulls/owner/repo/pulls/42/comments" in joined:
                raise AssertionError("unexpected endpoint")
            if "repos/owner/repo/pulls/42/comments?per_page=100" in joined:
                return CompletedProcess(command, 0, stdout=json.dumps(comments_json), stderr="")
            if "pr diff" in joined:
                return CompletedProcess(command, 0, stdout=diff_text, stderr="")
            raise AssertionError(f"unexpected command: {joined}")

        with patch("llm_council.eval_import.subprocess.run", side_effect=fake_run):
            imported = import_github_pr_review("owner/repo", 42, output_root=tmp_path)

        import_root = Path(imported.import_root)
        assert imported.imported_comment_count == 1
        assert import_root.exists()
        assert Path(imported.diff_path).read_text() == diff_text
        labels = json.loads(Path(imported.greptile_labels_path).read_text())
        assert labels[0]["severity"] == "P1"
        assert labels[0]["file_path"] == "backend/handler.go"
        assert "internal server error" in labels[0]["suggested_code"]
        review_input = Path(imported.review_input_path).read_text()
        assert "Fix auth edge case" in review_input
        assert "hidden" not in review_input
