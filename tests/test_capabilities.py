"""Tests for capability selection and capability-pack registry loading."""

from __future__ import annotations

from pathlib import Path

from llm_council.engine.capabilities import select_capability_plan
from llm_council.registry.tool_registry import ToolRegistry

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "tool_registry.yaml"


class TestCapabilitySelection:
    """Tests for runtime capability planning."""

    def test_select_plan_for_planner_assess(self):
        """Assess mode should use grounded planning and research capabilities."""
        registry = ToolRegistry()
        registry.load_from_yaml(CONFIG_PATH)

        plan = select_capability_plan("planner", "assess", tool_registry=registry)

        assert plan.execution_profile == "grounded"
        assert plan.budget_class == "premium"
        assert plan.required_capabilities == ["planning-assess", "docs-research"]
        assert "create_checklist" in plan.tool_names
        assert "web_search" in plan.tool_names
        assert "context7_lookup" in plan.tool_names

    def test_select_plan_for_security(self):
        """Security mode should resolve deep-analysis security capability packs."""
        registry = ToolRegistry()
        registry.load_from_yaml(CONFIG_PATH)

        plan = select_capability_plan("critic", "security", tool_registry=registry)

        assert plan.execution_profile == "deep_analysis"
        assert plan.required_capabilities == [
            "red-team-recon",
            "security-code-audit",
            "repo-analysis",
        ]
        assert "code_analysis" in plan.tool_names
        assert "grep_search" in plan.tool_names
        assert "web_search" in plan.tool_names
        assert any("attack paths" in requirement for requirement in plan.evidence_requirements)

    def test_default_prompt_only_fallback(self):
        """Unknown subagent/mode combinations should stay prompt-only."""
        registry = ToolRegistry()
        registry.load_from_yaml(CONFIG_PATH)

        plan = select_capability_plan("unknown-role", None, tool_registry=registry)

        assert plan.execution_profile == "prompt_only"
        assert plan.required_capabilities == []
        assert plan.tool_names == []

    def test_runtime_overrides_strengthen_plan_without_reducing_it(self):
        """Runtime capability overrides should only strengthen the base plan."""
        registry = ToolRegistry()
        registry.load_from_yaml(CONFIG_PATH)

        plan = select_capability_plan(
            "planner",
            "plan",
            tool_registry=registry,
            requested_execution_profile="grounded",
            requested_budget_class="premium",
            requested_capabilities=["docs-research"],
        )

        assert plan.execution_profile == "grounded"
        assert plan.budget_class == "premium"
        assert plan.required_capabilities == [
            "planning-assess",
            "repo-analysis",
            "docs-research",
        ]
        assert "web_search" in plan.tool_names
        assert "context7_lookup" in plan.tool_names
        assert any("Prefer authoritative" in requirement for requirement in plan.evidence_requirements)

    def test_runtime_overrides_do_not_weaken_plan(self):
        """Lower execution-profile or budget hints should not downgrade the base plan."""
        registry = ToolRegistry()
        registry.load_from_yaml(CONFIG_PATH)

        plan = select_capability_plan(
            "critic",
            "security",
            tool_registry=registry,
            requested_execution_profile="light_tools",
            requested_budget_class="cheap",
            requested_capabilities=["planning-assess"],
        )

        assert plan.execution_profile == "deep_analysis"
        assert plan.budget_class == "premium"
        assert plan.required_capabilities == [
            "red-team-recon",
            "security-code-audit",
            "repo-analysis",
        ]


class TestToolRegistryCapabilityPacks:
    """Tests for capability pack loading from the registry config."""

    def test_loads_capability_packs(self):
        """Capability packs should be available after loading the YAML config."""
        registry = ToolRegistry()
        registry.load_from_yaml(CONFIG_PATH)

        packs = registry.list_capability_packs()

        assert "repo-analysis" in packs
        assert "planning-assess" in packs
        assert "security-audit" in packs
        assert "security-code-audit" in packs
        assert "red-team-recon" in packs

    def test_resolve_capability_tools_filters_by_role(self):
        """Pack resolution should return tools valid for the requested role."""
        registry = ToolRegistry()
        registry.load_from_yaml(CONFIG_PATH)

        tools = registry.resolve_capability_tools(["planning-assess"], role="planner")

        assert [tool.name for tool in tools] == ["create_checklist", "read_file", "grep_search"]
