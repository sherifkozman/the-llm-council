"""Tests for local capability evidence collection."""

from __future__ import annotations

import pytest

from llm_council.engine.capabilities import CapabilityPlan
from llm_council.engine.evidence import collect_capability_evidence


class TestEvidenceCollection:
    """Tests for bounded local evidence collection."""

    @pytest.mark.asyncio
    async def test_collects_planning_and_repo_evidence(self, tmp_path):
        """Planning runs should gather checklist and repo-analysis evidence."""
        (tmp_path / "src").mkdir()
        (tmp_path / "docs" / "architecture").mkdir(parents=True)
        (tmp_path / "src" / "planner.py").write_text("def plan_migration():\n    return True\n")
        (tmp_path / "README.md").write_text("# Project\n")
        (tmp_path / "ROADMAP.md").write_text("# Roadmap\n")
        (tmp_path / "docs" / "index.md").write_text("# Docs\n")

        plan = CapabilityPlan(
            execution_profile="light_tools",
            budget_class="normal",
            required_capabilities=["planning-assess", "repo-analysis"],
        )

        bundle = await collect_capability_evidence(
            "Plan the migration for planner rollout",
            "planner",
            "plan",
            plan,
            repo_root=tmp_path,
        )

        assert bundle.executed_capabilities == ["planning-assess", "repo-analysis"]
        assert bundle.pending_capabilities == []
        assert len(bundle.items) == 2
        prompt_block = bundle.to_prompt_block()
        assert "Collected Evidence" in prompt_block
        assert "Planning artifacts" in prompt_block

    @pytest.mark.asyncio
    async def test_collects_security_matches_and_leaves_unsupported_pending(self, tmp_path):
        """Legacy security-audit alias should still work and leave unsupported packs pending."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "auth.py").write_text(
            "TOKEN='dev'\ndef run_auth(password, session):\n    return password, session, TOKEN\n"
        )

        plan = CapabilityPlan(
            execution_profile="deep_analysis",
            budget_class="premium",
            required_capabilities=["security-audit", "docs-research"],
        )

        bundle = await collect_capability_evidence(
            "Assess the authentication attack surface",
            "critic",
            "security",
            plan,
            repo_root=tmp_path,
        )

        assert bundle.executed_capabilities == ["security-audit"]
        assert bundle.pending_capabilities == ["docs-research"]
        assert any(item.capability == "security-audit" for item in bundle.items)
        assert "Capability packs not executed in this runtime" in bundle.to_prompt_block()

    @pytest.mark.asyncio
    async def test_collects_split_security_evidence(self, tmp_path):
        """Split security packs should gather attack-surface and code-audit evidence."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "auth.py").write_text(
            "TOKEN='dev'\n"
            "router.post('/login')\n"
            "def run_auth(password, session):\n"
            "    return password, session, TOKEN\n"
        )

        plan = CapabilityPlan(
            execution_profile="deep_analysis",
            budget_class="premium",
            required_capabilities=["red-team-recon", "security-code-audit"],
        )

        bundle = await collect_capability_evidence(
            "Assess the authentication attack surface",
            "critic",
            "security",
            plan,
            repo_root=tmp_path,
        )

        assert bundle.executed_capabilities == ["red-team-recon", "security-code-audit"]
        assert bundle.pending_capabilities == []
        assert any(item.capability == "red-team-recon" for item in bundle.items)
        assert any(item.capability == "security-code-audit" for item in bundle.items)

    @pytest.mark.asyncio
    async def test_collects_docs_research_for_known_targets(self, tmp_path, monkeypatch):
        """Docs research should execute when the task maps to supported official docs."""
        fetched_urls: list[str] = []

        def fake_fetch(url: str) -> tuple[str, str] | None:
            fetched_urls.append(url)
            return ("React Docs", "Server and client components guidance.")

        monkeypatch.setattr("llm_council.engine.evidence._fetch_docs_page", fake_fetch)

        plan = CapabilityPlan(
            execution_profile="grounded",
            budget_class="normal",
            required_capabilities=["docs-research"],
        )

        bundle = await collect_capability_evidence(
            "Research React server components best practices",
            "researcher",
            None,
            plan,
            repo_root=tmp_path,
        )

        assert bundle.executed_capabilities == ["docs-research"]
        assert bundle.pending_capabilities == []
        assert fetched_urls == ["https://react.dev/reference/react"]
        assert bundle.items[0].capability == "docs-research"
        assert "React Docs" in bundle.items[0].details[0]

    @pytest.mark.asyncio
    async def test_collects_diff_review_from_git_diff(self, tmp_path, monkeypatch):
        """Review runs should gather changed-file and hunk evidence from a diff."""
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / "src" / "service.py").write_text("def old():\n    return False\n")
        (tmp_path / "tests" / "test_service.py").write_text("def test_old():\n    assert True\n")

        monkeypatch.setattr(
            "llm_council.engine.evidence._load_git_diff",
            lambda repo_root: (
                "diff --git a/src/service.py b/src/service.py\n"
                "--- a/src/service.py\n"
                "+++ b/src/service.py\n"
                "@@ -1,2 +1,2 @@\n"
                "-def old():\n"
                "+def old(flag=True):\n"
            ),
        )

        plan = CapabilityPlan(
            execution_profile="light_tools",
            budget_class="normal",
            required_capabilities=["diff-review"],
        )

        bundle = await collect_capability_evidence(
            "Review the latest implementation changes",
            "critic",
            "review",
            plan,
            repo_root=tmp_path,
        )

        assert bundle.executed_capabilities == ["diff-review"]
        assert bundle.pending_capabilities == []
        assert bundle.items[0].capability == "diff-review"
        assert "Changed files: src/service.py" in bundle.items[0].details[0]
        assert any("@@ -1,2 +1,2 @@" in detail for detail in bundle.items[0].details)
        assert any("Likely impacted tests" in detail for detail in bundle.items[0].details)

    @pytest.mark.asyncio
    async def test_diff_review_stays_pending_without_diff(self, tmp_path, monkeypatch):
        """Without diff context, diff-review should remain pending."""
        monkeypatch.setattr("llm_council.engine.evidence._load_git_diff", lambda repo_root: "")

        plan = CapabilityPlan(
            execution_profile="light_tools",
            budget_class="normal",
            required_capabilities=["diff-review"],
        )

        bundle = await collect_capability_evidence(
            "Review the pending changes",
            "critic",
            "review",
            plan,
            repo_root=tmp_path,
        )

        assert bundle.executed_capabilities == []
        assert bundle.pending_capabilities == ["diff-review"]
