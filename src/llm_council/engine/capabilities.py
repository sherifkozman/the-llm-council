"""Capability selection policy for mode-aware council execution."""

from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from llm_council.registry.tool_registry import ToolRegistry, get_tool_registry

ExecutionProfile = Literal["prompt_only", "light_tools", "grounded", "deep_analysis"]
BudgetClass = Literal["cheap", "normal", "premium"]


class CapabilityPlan(BaseModel):
    """Resolved capability policy for a council run."""

    model_config = ConfigDict(extra="forbid")

    execution_profile: ExecutionProfile = "prompt_only"
    budget_class: BudgetClass = "normal"
    required_capabilities: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)


_EXECUTION_PROFILE_ORDER: dict[ExecutionProfile, int] = {
    "prompt_only": 0,
    "light_tools": 1,
    "grounded": 2,
    "deep_analysis": 3,
}
_BUDGET_CLASS_ORDER: dict[BudgetClass, int] = {
    "cheap": 0,
    "normal": 1,
    "premium": 2,
}


_CAPABILITY_POLICY: dict[tuple[str, str | None], CapabilityPlan] = {
    (
        "planner",
        "plan",
    ): CapabilityPlan(
        execution_profile="light_tools",
        budget_class="normal",
        required_capabilities=["planning-assess", "repo-analysis"],
        evidence_requirements=[
            "Identify concrete dependencies, blockers, and sequencing assumptions.",
            "Call out risks and mitigations explicitly instead of implying them.",
            "If repository context is missing, state which planning assumptions remain unverified.",
        ],
    ),
    (
        "planner",
        "assess",
    ): CapabilityPlan(
        execution_profile="grounded",
        budget_class="premium",
        required_capabilities=["planning-assess", "docs-research"],
        evidence_requirements=[
            "Define explicit evaluation criteria and score tradeoffs against them.",
            "Name alternatives considered and why they were rejected.",
            "State reversibility and unresolved external dependencies if evidence is incomplete.",
        ],
    ),
    (
        "critic",
        "security",
    ): CapabilityPlan(
        execution_profile="deep_analysis",
        budget_class="premium",
        required_capabilities=["red-team-recon", "security-code-audit", "repo-analysis"],
        evidence_requirements=[
            "Ground findings in realistic attack paths, prerequisites, and affected components.",
            "Distinguish verified weaknesses from speculative concerns.",
            "If attack-surface evidence is missing, say so explicitly instead of fabricating exploitability.",
        ],
    ),
    (
        "critic",
        "review",
    ): CapabilityPlan(
        execution_profile="light_tools",
        budget_class="normal",
        required_capabilities=["diff-review", "repo-analysis"],
        evidence_requirements=[
            "Tie findings to concrete code locations or behaviors.",
            "Separate correctness issues from style or preference commentary.",
        ],
    ),
    (
        "researcher",
        None,
    ): CapabilityPlan(
        execution_profile="grounded",
        budget_class="normal",
        required_capabilities=["docs-research"],
        evidence_requirements=[
            "Use authoritative sources where possible and state freshness limits.",
            "Separate verified facts from inference or opinion.",
        ],
    ),
    (
        "drafter",
        "impl",
    ): CapabilityPlan(
        execution_profile="light_tools",
        budget_class="normal",
        required_capabilities=["repo-analysis"],
        evidence_requirements=[
            "Follow existing repository patterns where evidence exists.",
            "Call out any implementation assumptions that could not be verified locally.",
        ],
    ),
    (
        "drafter",
        "arch",
    ): CapabilityPlan(
        execution_profile="light_tools",
        budget_class="normal",
        required_capabilities=["repo-analysis"],
        evidence_requirements=[
            "Identify affected boundaries, interfaces, and dependencies explicitly.",
        ],
    ),
    (
        "drafter",
        "test",
    ): CapabilityPlan(
        execution_profile="light_tools",
        budget_class="normal",
        required_capabilities=["repo-analysis"],
        evidence_requirements=[
            "Tie proposed tests to concrete code paths, failure cases, and risk areas.",
        ],
    ),
    ("router", None): CapabilityPlan(execution_profile="prompt_only", budget_class="cheap"),
    ("synthesizer", None): CapabilityPlan(execution_profile="prompt_only", budget_class="cheap"),
}


def select_capability_plan(
    subagent: str,
    mode: str | None = None,
    *,
    subagent_config: dict[str, object] | None = None,
    tool_registry: ToolRegistry | None = None,
    requested_execution_profile: str | None = None,
    requested_budget_class: str | None = None,
    requested_capabilities: list[str] | None = None,
) -> CapabilityPlan:
    """Resolve the capability plan for a subagent and mode."""

    template = _CAPABILITY_POLICY.get((subagent, mode)) or _CAPABILITY_POLICY.get((subagent, None))
    plan = template.model_copy(deep=True) if template else CapabilityPlan()

    registry = tool_registry or get_tool_registry()
    registry.ensure_loaded()

    plan.tool_names = [
        tool.name
        for tool in registry.resolve_capability_tools(plan.required_capabilities, role=subagent)
    ]

    if isinstance(subagent_config, dict):
        suggested_tools = subagent_config.get("suggested_tools")
        for suggested_tool in suggested_tools if isinstance(suggested_tools, list) else []:
            if isinstance(suggested_tool, str) and suggested_tool not in plan.tool_names:
                plan.tool_names.append(suggested_tool)

    _apply_capability_overrides(
        plan,
        subagent=subagent,
        registry=registry,
        requested_execution_profile=requested_execution_profile,
        requested_budget_class=requested_budget_class,
        requested_capabilities=requested_capabilities or [],
    )

    return plan


def _apply_capability_overrides(
    plan: CapabilityPlan,
    *,
    subagent: str,
    registry: ToolRegistry,
    requested_execution_profile: str | None,
    requested_budget_class: str | None,
    requested_capabilities: list[str],
) -> None:
    """Safely merge runtime capability overrides into a plan.

    Runtime overrides may strengthen a run, but they should not weaken the
    default policy for a subagent/mode combination.
    """

    normalized_execution_profile = (
        cast(ExecutionProfile, requested_execution_profile)
        if requested_execution_profile in _EXECUTION_PROFILE_ORDER
        else None
    )
    if normalized_execution_profile is not None:
        if (
            _EXECUTION_PROFILE_ORDER[normalized_execution_profile]
            > _EXECUTION_PROFILE_ORDER[plan.execution_profile]
        ):
            plan.execution_profile = normalized_execution_profile

    normalized_budget_class = (
        cast(BudgetClass, requested_budget_class)
        if requested_budget_class in _BUDGET_CLASS_ORDER
        else None
    )
    if normalized_budget_class is not None:
        if _BUDGET_CLASS_ORDER[normalized_budget_class] > _BUDGET_CLASS_ORDER[plan.budget_class]:
            plan.budget_class = normalized_budget_class

    resolved_capability_names: list[str] = []
    for capability_name in requested_capabilities:
        pack = registry.get_capability_pack(capability_name)
        if pack is None:
            continue
        if pack.roles and subagent not in pack.roles:
            continue
        if capability_name not in plan.required_capabilities:
            plan.required_capabilities.append(capability_name)
            resolved_capability_names.append(capability_name)

    if resolved_capability_names:
        for requirement in _collect_evidence_requirements(
            registry,
            resolved_capability_names,
            subagent=subagent,
        ):
            if requirement not in plan.evidence_requirements:
                plan.evidence_requirements.append(requirement)

        for tool in registry.resolve_capability_tools(resolved_capability_names, role=subagent):
            if tool.name not in plan.tool_names:
                plan.tool_names.append(tool.name)


def _collect_evidence_requirements(
    registry: ToolRegistry, capability_names: list[str], *, subagent: str
) -> list[str]:
    """Collect evidence requirements from capability packs valid for the role."""

    requirements: list[str] = []
    for capability_name in capability_names:
        pack = registry.get_capability_pack(capability_name)
        if pack is None:
            continue
        if pack.roles and subagent not in pack.roles:
            continue
        for requirement in pack.evidence_requirements:
            if requirement not in requirements:
                requirements.append(requirement)
    return requirements


__all__ = ["CapabilityPlan", "select_capability_plan"]
