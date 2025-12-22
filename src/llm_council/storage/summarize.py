"""
Tiered Summarization for LLM Council.

Provides tiered output summarization to manage context size while
preserving audit trails via artifact references.

Tiers:
- GIST: ~50 tokens, one-liner summary
- FINDINGS: ~150 tokens, key points
- ACTIONS: ~300 tokens, actionable items
- RATIONALE: ~500 tokens, reasoning included
- AUDIT: Full detail with artifact references
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from llm_council.protocol.types import SummaryTier
from llm_council.storage.artifacts import ArtifactStore, ArtifactType, get_store

# Token budgets per tier (approximate)
TIER_TOKEN_LIMITS = {
    SummaryTier.GIST: 50,
    SummaryTier.FINDINGS: 150,
    SummaryTier.ACTIONS: 300,
    SummaryTier.RATIONALE: 500,
    SummaryTier.AUDIT: 10000,  # Effectively unlimited for audit
}

# Char limits (tokens * 4)
TIER_CHAR_LIMITS = {tier: limit * 4 for tier, limit in TIER_TOKEN_LIMITS.items()}


@dataclass
class SummarizationResult:
    """Result of tiered summarization."""

    tier: SummaryTier
    summary: str
    token_estimate: int
    original_tokens: int
    tokens_saved: int
    artifact_ref: str | None = None  # Reference to full content if stored
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tier": self.tier.value,
            "summary": self.summary,
            "token_estimate": self.token_estimate,
            "original_tokens": self.original_tokens,
            "tokens_saved": self.tokens_saved,
            "artifact_ref": self.artifact_ref,
            "truncated": self.truncated,
        }


@dataclass
class TieredSummary:
    """Multi-tier summary with all levels available."""

    gist: str = ""
    findings: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    rationale: str = ""
    artifact_ref: str | None = None

    def get_tier(self, tier: SummaryTier) -> str:
        """Get summary content for a specific tier."""
        if tier == SummaryTier.GIST:
            return self.gist
        elif tier == SummaryTier.FINDINGS:
            return "Key findings:\n" + "\n".join(f"- {f}" for f in self.findings)
        elif tier == SummaryTier.ACTIONS:
            findings = "Key findings:\n" + "\n".join(f"- {f}" for f in self.findings)
            actions = "\nActions:\n" + "\n".join(f"- {a}" for a in self.actions)
            return findings + actions
        elif tier == SummaryTier.RATIONALE:
            findings = "Key findings:\n" + "\n".join(f"- {f}" for f in self.findings)
            actions = "\nActions:\n" + "\n".join(f"- {a}" for a in self.actions)
            rationale = f"\nRationale:\n{self.rationale}"
            return findings + actions + rationale
        else:  # AUDIT
            ref_note = (
                f"\n[Full details: artifact {self.artifact_ref}]" if self.artifact_ref else ""
            )
            return self.get_tier(SummaryTier.RATIONALE) + ref_note

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gist": self.gist,
            "findings": self.findings,
            "actions": self.actions,
            "rationale": self.rationale,
            "artifact_ref": self.artifact_ref,
        }


class Summarizer:
    """Tiered output summarization with artifact integration."""

    # Threshold above which summarization is triggered (tokens)
    DEFAULT_THRESHOLD = 500

    def __init__(
        self,
        artifact_store: ArtifactStore | None = None,
        threshold_tokens: int = DEFAULT_THRESHOLD,
    ) -> None:
        """Initialize the summarizer.

        Args:
            artifact_store: Store for full content (uses default if None)
            threshold_tokens: Token count above which to summarize
        """
        self._store = artifact_store or get_store()
        self._threshold = threshold_tokens

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (4 chars ≈ 1 token)."""
        return len(text) // 4

    def should_summarize(self, content: str) -> bool:
        """Check if content should be summarized based on size."""
        return self._estimate_tokens(content) > self._threshold

    def summarize(
        self,
        content: str,
        tier: SummaryTier,
        run_id: str | None = None,
        artifact_type: ArtifactType = ArtifactType.DRAFT,
        store_full: bool = True,
    ) -> SummarizationResult:
        """Summarize content to the specified tier level.

        Args:
            content: The full content to summarize
            tier: Target summarization tier
            run_id: Run ID for artifact storage
            artifact_type: Type of artifact if storing
            store_full: Whether to store full content as artifact

        Returns:
            SummarizationResult with summary and metadata
        """
        original_tokens = self._estimate_tokens(content)
        char_limit = TIER_CHAR_LIMITS[tier]

        # If content is small enough, no summarization needed
        if original_tokens <= TIER_TOKEN_LIMITS[tier]:
            return SummarizationResult(
                tier=tier,
                summary=content,
                token_estimate=original_tokens,
                original_tokens=original_tokens,
                tokens_saved=0,
                artifact_ref=None,
                truncated=False,
            )

        # Store full content as artifact if requested
        artifact_ref = None
        if store_full and run_id:
            artifact = self._store.store_artifact(
                run_id=run_id,
                content=content,
                artifact_type=artifact_type,
            )
            artifact_ref = artifact.artifact_id

        # Generate summary based on tier
        summary = self._generate_summary(content, tier, char_limit)
        summary_tokens = self._estimate_tokens(summary)

        # Add artifact reference for AUDIT tier
        if tier == SummaryTier.AUDIT and artifact_ref:
            summary += f"\n\n[Full details: artifact {artifact_ref}]"
            summary_tokens = self._estimate_tokens(summary)

        return SummarizationResult(
            tier=tier,
            summary=summary,
            token_estimate=summary_tokens,
            original_tokens=original_tokens,
            tokens_saved=original_tokens - summary_tokens,
            artifact_ref=artifact_ref,
            truncated=len(content) > char_limit,
        )

    def _generate_summary(self, content: str, tier: SummaryTier, char_limit: int) -> str:
        """Generate summary based on tier.

        This uses heuristic extraction. For LLM-based summarization,
        override this method or use the summarize_with_llm method.
        """
        if tier == SummaryTier.GIST:
            return self._extract_gist(content, char_limit)
        elif tier == SummaryTier.FINDINGS:
            return self._extract_findings(content, char_limit)
        elif tier == SummaryTier.ACTIONS:
            return self._extract_actions(content, char_limit)
        elif tier == SummaryTier.RATIONALE:
            return self._extract_rationale(content, char_limit)
        else:  # AUDIT
            return content[:char_limit] if len(content) > char_limit else content

    def _extract_gist(self, content: str, char_limit: int) -> str:
        """Extract a one-liner gist from content."""
        # Try to find a summary line
        for pattern in [
            r"(?i)summary[:\s]+(.+?)[\n\.]",
            r"(?i)in summary[,:\s]+(.+?)[\n\.]",
            r"(?i)conclusion[:\s]+(.+?)[\n\.]",
            r"(?i)^(?:the\s+)?(?:main\s+)?(?:key\s+)?(?:point|takeaway|finding)[:\s]+(.+?)[\n\.]",
        ]:
            match = re.search(pattern, content)
            if match:
                gist = match.group(1).strip()
                if len(gist) <= char_limit:
                    return gist

        # Fall back to first meaningful line
        lines = [
            line.strip() for line in content.split("\n") if line.strip() and len(line.strip()) > 10
        ]
        if lines:
            first_line = lines[0]
            if len(first_line) <= char_limit:
                return first_line
            return first_line[: char_limit - 3] + "..."

        return content[: char_limit - 3] + "..."

    def _extract_findings(self, content: str, char_limit: int) -> str:
        """Extract key findings from content."""
        findings = []

        # Look for bullet points
        bullet_pattern = r"^[\s]*[-*•]\s*(.+)$"
        for match in re.finditer(bullet_pattern, content, re.MULTILINE):
            finding = match.group(1).strip()
            if len(finding) > 10:  # Skip trivial items
                findings.append(f"- {finding[:100]}")

        # Look for numbered items
        number_pattern = r"^[\s]*\d+[\.)]\s*(.+)$"
        for match in re.finditer(number_pattern, content, re.MULTILINE):
            finding = match.group(1).strip()
            if len(finding) > 10:
                findings.append(f"- {finding[:100]}")

        if findings:
            result = "Key findings:\n" + "\n".join(findings[:5])
            return result[:char_limit] if len(result) > char_limit else result

        # Fall back to first paragraph
        paragraphs = content.split("\n\n")
        if paragraphs:
            return paragraphs[0][:char_limit]

        return content[:char_limit]

    def _extract_actions(self, content: str, char_limit: int) -> str:
        """Extract actionable items from content."""
        findings = self._extract_findings(content, char_limit // 2)
        actions = []

        # Look for action patterns
        action_patterns = [
            r"(?i)(?:should|must|need to|recommend|suggest)\s+(.+?)[\n\.]",
            r"(?i)(?:action|step|task)[:\s]+(.+?)[\n\.]",
            r"(?i)(?:next|todo|to-do)[:\s]+(.+?)[\n\.]",
        ]

        for pattern in action_patterns:
            for match in re.finditer(pattern, content):
                action = match.group(1).strip()
                if len(action) > 10:
                    actions.append(f"- {action[:80]}")

        actions_text = ""
        if actions:
            actions_text = "\n\nActions:\n" + "\n".join(actions[:3])

        result = findings + actions_text
        return result[:char_limit] if len(result) > char_limit else result

    def _extract_rationale(self, content: str, char_limit: int) -> str:
        """Extract findings, actions, and rationale from content."""
        actions = self._extract_actions(content, char_limit // 2)

        # Look for reasoning patterns
        rationale = ""
        rationale_patterns = [
            r"(?i)because\s+(.+?)[\n\.]",
            r"(?i)reason[:\s]+(.+?)[\n\.]",
            r"(?i)(?:this is because|the reason is)\s+(.+?)[\n\.]",
        ]

        reasons = []
        for pattern in rationale_patterns:
            for match in re.finditer(pattern, content):
                reason = match.group(1).strip()
                if len(reason) > 20:
                    reasons.append(reason[:150])

        if reasons:
            rationale = "\n\nRationale:\n" + "\n".join(f"- {r}" for r in reasons[:3])

        result = actions + rationale
        return result[:char_limit] if len(result) > char_limit else result

    def summarize_drafts(
        self,
        drafts: dict[str, str],
        tier: SummaryTier,
        run_id: str | None = None,
    ) -> dict[str, SummarizationResult]:
        """Summarize multiple drafts.

        Args:
            drafts: Dict of provider_name -> draft_content
            tier: Target summarization tier
            run_id: Run ID for artifact storage

        Returns:
            Dict of provider_name -> SummarizationResult
        """
        results = {}
        for provider, content in drafts.items():
            results[provider] = self.summarize(
                content=content,
                tier=tier,
                run_id=run_id,
                artifact_type=ArtifactType.DRAFT,
            )
        return results

    def get_total_tokens_saved(self, results: dict[str, SummarizationResult]) -> int:
        """Calculate total tokens saved from summarization results."""
        return sum(r.tokens_saved for r in results.values())


def summarize_for_context(
    content: str,
    tier: SummaryTier = SummaryTier.ACTIONS,
    threshold_tokens: int = 500,
) -> str:
    """Convenience function for quick summarization.

    Args:
        content: Content to summarize
        tier: Target tier (default ACTIONS)
        threshold_tokens: Only summarize above this threshold

    Returns:
        Summarized content or original if below threshold
    """
    summarizer = Summarizer(threshold_tokens=threshold_tokens)
    if not summarizer.should_summarize(content):
        return content

    result = summarizer.summarize(content, tier, store_full=False)
    return result.summary


__all__ = [
    "Summarizer",
    "SummarizationResult",
    "TieredSummary",
    "summarize_for_context",
    "TIER_TOKEN_LIMITS",
    "TIER_CHAR_LIMITS",
]
