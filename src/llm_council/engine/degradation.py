"""
Graceful Degradation for LLM Council.

Implements runtime failure handling and degradation policies to keep
council runs alive when individual providers fail.

Uses error classification from providers.base to determine appropriate
actions for different failure types.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from llm_council.providers.base import (
    NON_RETRYABLE_ERRORS,
    ErrorType,
    classify_error,
    get_billing_help_url,
)

logger = logging.getLogger(__name__)


class DegradationAction(str, Enum):
    """Actions to take when a provider fails."""

    CONTINUE = "continue"  # Proceed without this provider
    RETRY = "retry"  # Retry with exponential backoff
    FALLBACK = "fallback"  # Use fallback provider
    ABORT = "abort"  # Abort the entire run
    SKIP = "skip"  # Skip this provider for current phase only


@dataclass
class FailureEvent:
    """Record of a provider failure."""

    provider: str
    phase: str  # drafts, critique, synthesis
    error_type: ErrorType
    error_message: str
    action_taken: DegradationAction
    retry_count: int = 0
    fallback_provider: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "phase": self.phase,
            "error_type": self.error_type.value,
            "error_message": self.error_message[:200],
            "action_taken": self.action_taken.value,
            "retry_count": self.retry_count,
            "fallback_provider": self.fallback_provider,
            "timestamp": self.timestamp,
        }


@dataclass
class DegradationDecision:
    """Decision made by the degradation policy."""

    action: DegradationAction
    reason: str
    retry_delay_ms: int = 0
    fallback_provider: str | None = None
    billing_url: str | None = None
    should_log: bool = True


@dataclass
class DegradationReport:
    """Summary of degradation events during a council run."""

    failures: list[FailureEvent] = field(default_factory=list)
    total_retries: int = 0
    providers_skipped: list[str] = field(default_factory=list)
    fallbacks_used: list[str] = field(default_factory=list)
    aborted: bool = False

    def add_failure(self, event: FailureEvent) -> None:
        """Record a failure event."""
        self.failures.append(event)
        if event.action_taken == DegradationAction.RETRY:
            self.total_retries += 1
        elif event.action_taken == DegradationAction.SKIP:
            if event.provider not in self.providers_skipped:
                self.providers_skipped.append(event.provider)
        elif event.action_taken == DegradationAction.FALLBACK:
            if event.fallback_provider and event.fallback_provider not in self.fallbacks_used:
                self.fallbacks_used.append(event.fallback_provider)
        elif event.action_taken == DegradationAction.ABORT:
            self.aborted = True

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        if not self.failures:
            return "No degradation events"

        lines = [f"Degradation: {len(self.failures)} failure(s)"]
        if self.providers_skipped:
            lines.append(f"  Skipped: {', '.join(self.providers_skipped)}")
        if self.fallbacks_used:
            lines.append(f"  Fallbacks: {', '.join(self.fallbacks_used)}")
        if self.total_retries:
            lines.append(f"  Retries: {self.total_retries}")
        if self.aborted:
            lines.append("  Status: ABORTED")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "failures": [f.to_dict() for f in self.failures],
            "total_retries": self.total_retries,
            "providers_skipped": self.providers_skipped,
            "fallbacks_used": self.fallbacks_used,
            "aborted": self.aborted,
        }


class DegradationPolicy:
    """Policy engine for handling provider failures.

    Determines appropriate actions based on error type, retry count,
    and available fallback options.
    """

    # Default retry limits
    MAX_RETRIES = 2
    BASE_RETRY_DELAY_MS = 1000  # 1 second
    MAX_RETRY_DELAY_MS = 10000  # 10 seconds

    def __init__(
        self,
        max_retries: int = MAX_RETRIES,
        fallback_providers: dict[str, str] | None = None,
        min_providers_required: int = 1,
        abort_on_all_failures: bool = True,
    ) -> None:
        """Initialize the degradation policy.

        Args:
            max_retries: Maximum retries per provider
            fallback_providers: Dict of provider -> fallback_provider
            min_providers_required: Minimum providers needed to continue
            abort_on_all_failures: Abort if all providers fail
        """
        self._max_retries = max_retries
        self._fallbacks = fallback_providers or {}
        self._min_providers = min_providers_required
        self._abort_on_all_failures = abort_on_all_failures
        self._retry_counts: dict[str, int] = {}
        self._report = DegradationReport()

    def reset(self) -> None:
        """Reset state for a new run."""
        self._retry_counts.clear()
        self._report = DegradationReport()

    def get_report(self) -> DegradationReport:
        """Get the current degradation report."""
        return self._report

    def decide(
        self,
        provider: str,
        error: Exception | str,
        phase: str,
        remaining_providers: int,
    ) -> DegradationDecision:
        """Decide how to handle a provider failure.

        Args:
            provider: Name of the failed provider
            error: The exception or error message
            phase: Current phase (drafts, critique, synthesis)
            remaining_providers: Number of other providers still available

        Returns:
            DegradationDecision with action and details
        """
        error_text = str(error)
        error_type = classify_error(error_text, -1)

        # Track retry count
        retry_key = f"{provider}:{phase}"
        current_retries = self._retry_counts.get(retry_key, 0)

        # Determine action based on error type and context
        decision = self._determine_action(
            provider=provider,
            error_type=error_type,
            error_text=error_text,
            phase=phase,
            current_retries=current_retries,
            remaining_providers=remaining_providers,
        )

        # Record the failure event
        event = FailureEvent(
            provider=provider,
            phase=phase,
            error_type=error_type,
            error_message=error_text,
            action_taken=decision.action,
            retry_count=current_retries,
            fallback_provider=decision.fallback_provider,
        )
        self._report.add_failure(event)

        # Update retry count if retrying
        if decision.action == DegradationAction.RETRY:
            self._retry_counts[retry_key] = current_retries + 1

        if decision.should_log:
            logger.warning(
                "Provider %s failed in %s: %s (action=%s)",
                provider,
                phase,
                error_type.value,
                decision.action.value,
            )

        return decision

    def _determine_action(
        self,
        provider: str,
        error_type: ErrorType,
        error_text: str,
        phase: str,
        current_retries: int,
        remaining_providers: int,
    ) -> DegradationDecision:
        """Determine the appropriate action for a failure."""

        # Non-retryable errors - skip or abort immediately
        if error_type in NON_RETRYABLE_ERRORS:
            billing_url = None
            if error_type == ErrorType.BILLING:
                billing_url = get_billing_help_url(provider)
                reason = f"Billing error: check {billing_url}"
            elif error_type == ErrorType.AUTH:
                reason = f"Authentication error: check API key for {provider}"
            else:
                reason = f"Non-retryable error: {error_type.value}"

            # Check for fallback
            if provider in self._fallbacks:
                return DegradationDecision(
                    action=DegradationAction.FALLBACK,
                    reason=reason,
                    fallback_provider=self._fallbacks[provider],
                    billing_url=billing_url,
                )

            # If no remaining providers and this is critical phase, abort
            if remaining_providers == 0 and phase in ("critique", "synthesis"):
                return DegradationDecision(
                    action=DegradationAction.ABORT,
                    reason=f"Critical failure in {phase}: {reason}",
                    billing_url=billing_url,
                )

            # Otherwise skip this provider
            return DegradationDecision(
                action=DegradationAction.SKIP,
                reason=reason,
                billing_url=billing_url,
            )

        # Retryable errors - check retry limit
        if error_type in (ErrorType.RATE_LIMIT, ErrorType.TIMEOUT, ErrorType.NETWORK):
            if current_retries < self._max_retries:
                # Calculate exponential backoff delay
                delay = min(
                    self.BASE_RETRY_DELAY_MS * (2**current_retries),
                    self.MAX_RETRY_DELAY_MS,
                )
                return DegradationDecision(
                    action=DegradationAction.RETRY,
                    reason=f"Retryable error ({error_type.value}), attempt {current_retries + 1}",
                    retry_delay_ms=delay,
                )

        # Model unavailable - try fallback or skip
        if error_type == ErrorType.MODEL_UNAVAILABLE:
            if provider in self._fallbacks:
                return DegradationDecision(
                    action=DegradationAction.FALLBACK,
                    reason="Model unavailable, using fallback",
                    fallback_provider=self._fallbacks[provider],
                )
            return DegradationDecision(
                action=DegradationAction.SKIP,
                reason="Model unavailable, no fallback configured",
            )

        # Max retries exceeded
        if current_retries >= self._max_retries:
            if provider in self._fallbacks:
                return DegradationDecision(
                    action=DegradationAction.FALLBACK,
                    reason=f"Max retries ({self._max_retries}) exceeded, using fallback",
                    fallback_provider=self._fallbacks[provider],
                )

            if remaining_providers == 0:
                if phase in ("critique", "synthesis"):
                    return DegradationDecision(
                        action=DegradationAction.ABORT,
                        reason=f"All providers exhausted in {phase}",
                    )
                elif self._abort_on_all_failures:
                    return DegradationDecision(
                        action=DegradationAction.ABORT,
                        reason="All providers exhausted",
                    )

            return DegradationDecision(
                action=DegradationAction.SKIP,
                reason=f"Max retries exceeded for {provider}",
            )

        # Default: continue without this provider
        if remaining_providers >= self._min_providers:
            return DegradationDecision(
                action=DegradationAction.CONTINUE,
                reason=f"Continuing with {remaining_providers} remaining provider(s)",
            )

        # Not enough providers remaining
        if self._abort_on_all_failures:
            return DegradationDecision(
                action=DegradationAction.ABORT,
                reason=f"Below minimum required providers ({self._min_providers})",
            )

        return DegradationDecision(
            action=DegradationAction.CONTINUE,
            reason="Continuing with degraded capacity",
        )


def create_default_policy(
    max_retries: int = 2,
    fallback_providers: dict[str, str] | None = None,
) -> DegradationPolicy:
    """Create a degradation policy with sensible defaults.

    Args:
        max_retries: Maximum retry attempts per provider
        fallback_providers: Optional fallback mapping

    Returns:
        Configured DegradationPolicy
    """
    return DegradationPolicy(
        max_retries=max_retries,
        fallback_providers=fallback_providers or {},
        min_providers_required=1,
        abort_on_all_failures=True,
    )


__all__ = [
    "DegradationAction",
    "DegradationDecision",
    "DegradationPolicy",
    "DegradationReport",
    "FailureEvent",
    "create_default_policy",
]
