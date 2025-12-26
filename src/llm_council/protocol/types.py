"""
Protocol types for LLM Council.

Defines Pydantic models for council requests, responses, and configuration.
These types enable JSON-lines stdin/stdout communication for IDE integration.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SummaryTier(str, Enum):
    """Summarization detail levels."""

    GIST = "gist"  # ~50 tokens, one-liner
    FINDINGS = "findings"  # ~150 tokens, key points
    ACTIONS = "actions"  # ~300 tokens, actionable items
    RATIONALE = "rationale"  # ~500 tokens, reasoning included
    AUDIT = "audit"  # Full detail for audit trail


class CouncilConfig(BaseModel):
    """Configuration for a council run."""

    model_config = ConfigDict(extra="allow")

    providers: list[str] = Field(
        default=["openrouter"],
        description="List of provider names to use for drafts",
    )
    models: list[str] | None = Field(
        default=None,
        description=(
            "List of OpenRouter model IDs for multi-model council. "
            "When set with providers=['openrouter'], creates virtual providers for each model. "
            "Example: ['anthropic/claude-3.5-sonnet', 'openai/gpt-4o', 'google/gemini-pro']"
        ),
    )
    timeout: int = Field(
        default=120,
        ge=10,
        le=900,
        description="Timeout per provider call in seconds (max 15 min)",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum validation retries",
    )
    summary_tier: SummaryTier = Field(
        default=SummaryTier.ACTIONS,
        description="Summarization detail level",
    )
    max_draft_tokens: int = Field(
        default=4000,
        description="Max tokens per draft before summarization",
    )
    enable_artifact_store: bool = Field(
        default=True,
        description="Store verbose outputs in artifact store",
    )
    enable_health_check: bool = Field(
        default=False,
        description="Run preflight health check before council run",
    )
    enable_graceful_degradation: bool = Field(
        default=True,
        description="Enable graceful degradation on provider failures",
    )
    mode: str | None = Field(
        default=None,
        description="Agent mode for consolidated agents (e.g., 'impl', 'arch', 'review')",
    )


class CouncilRequest(BaseModel):
    """Request to run a council task."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(default="council_request", description="Message type")
    task: str = Field(..., description="The task to process")
    subagent: str = Field(
        default="router",
        description="Subagent type (drafter, critic, planner, etc.)",
    )
    mode: str | None = Field(
        default=None,
        description="Agent mode for consolidated agents",
    )
    config: CouncilConfig | None = Field(
        default=None,
        description="Optional configuration override",
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="Additional context for the task",
    )


class CostEstimate(BaseModel):
    """Estimated cost for a council run."""

    model_config = ConfigDict(frozen=True)

    provider_calls: dict[str, int] = Field(
        default_factory=dict,
        description="Call count per provider",
    )
    total_input_tokens: int = Field(default=0, description="Total input tokens")
    total_output_tokens: int = Field(default=0, description="Total output tokens")
    estimated_cost_usd: float = Field(default=0.0, description="Estimated cost in USD")


class PhaseTiming(BaseModel):
    """Timing information for a council phase."""

    model_config = ConfigDict(frozen=True)

    phase: str = Field(..., description="Phase name")
    duration_ms: int = Field(..., description="Duration in milliseconds")


class CouncilResponse(BaseModel):
    """Response from a council run."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(default="council_response", description="Message type")
    success: bool = Field(..., description="Whether the council succeeded")
    output: dict[str, Any] | None = Field(
        default=None,
        description="Validated JSON output",
    )
    drafts: dict[str, str] | None = Field(
        default=None,
        description="Draft outputs by provider",
    )
    critique: str | None = Field(
        default=None,
        description="Adversarial critique",
    )
    synthesis_attempts: int = Field(
        default=1,
        description="Number of synthesis attempts",
    )
    duration_ms: int = Field(
        default=0,
        description="Total duration in milliseconds",
    )
    phase_timings: list[PhaseTiming] | None = Field(
        default=None,
        description="Timing per phase",
    )
    validation_errors: list[str] | None = Field(
        default=None,
        description="Validation errors if failed",
    )
    cost_estimate: CostEstimate | None = Field(
        default=None,
        description="Cost estimation",
    )
    run_id: str | None = Field(
        default=None,
        description="Artifact store run ID",
    )


class ErrorResponse(BaseModel):
    """Error response for failed requests."""

    model_config = ConfigDict(frozen=True)

    type: str = Field(default="error", description="Message type")
    error: str = Field(..., description="Error message")
    code: str | None = Field(default=None, description="Error code")
    details: dict[str, Any] | None = Field(
        default=None,
        description="Additional error details",
    )
