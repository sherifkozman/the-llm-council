"""Adaptive protocol runner for task-based council orchestration.

This module provides the AdaptiveProtocolRunner that automatically selects
the most appropriate governance protocol based on task classification.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llm_council.engine.task_classifier import (
    GovernanceProtocol,
    TaskClass,
    TASK_MODEL_PACK,
    get_protocol_for_task,
)

if TYPE_CHECKING:
    from llm_council.engine.orchestrator import Orchestrator
    from llm_council.protocol.types import CouncilResult

logger = logging.getLogger(__name__)


class AdaptiveProtocolRunner:
    """Runs council with automatically selected protocol based on task type."""

    def __init__(self, orchestrator: Orchestrator) -> None:
        """Initialize the adaptive protocol runner.

        Args:
            orchestrator: The base orchestrator instance to delegate execution to.
        """
        self.orchestrator = orchestrator

    async def run(
        self,
        task: str,
        subagent: str,
        **kwargs,
    ) -> CouncilResult:
        """Run the council with automatic protocol selection.

        Args:
            task: The task description to classify and execute.
            subagent: The subagent identifier.
            **kwargs: Additional arguments to pass to the orchestrator.

        Returns:
            CouncilResult with additional task_class and protocol fields.
        """
        # Classify the task and select protocol
        task_class, protocol = get_protocol_for_task(task)
        
        logger.info(
            f"Task classified as '{task_class.value}', "
            f"using protocol '{protocol.value}'"
        )

        # Get model pack for this task class
        model_pack = TASK_MODEL_PACK.get(task_class, {})
        
        # Execute using the selected protocol
        result = await self._execute_protocol(
            protocol=protocol,
            task=task,
            subagent=subagent,
            model_pack=model_pack,
            **kwargs,
        )

        # Attach classification metadata to result
        result.task_class = task_class.value
        result.protocol = protocol.value

        return result

    async def _execute_protocol(
        self,
        protocol: GovernanceProtocol,
        task: str,
        subagent: str,
        model_pack: dict[str, str],
        **kwargs,
    ) -> CouncilResult:
        """Execute the task using the specified protocol.

        Args:
            protocol: The governance protocol to use.
            task: The task description.
            subagent: The subagent identifier.
            model_pack: Model configuration for this task type.
            **kwargs: Additional arguments.

        Returns:
            CouncilResult from the orchestrator.
        """
        if protocol == GovernanceProtocol.MAJORITY_VOTE:
            return await self._run_majority_vote(task, subagent, model_pack, **kwargs)
        elif protocol == GovernanceProtocol.VOTE_AND_DELIBERATE:
            return await self._run_vote_and_deliberate(task, subagent, model_pack, **kwargs)
        elif protocol == GovernanceProtocol.PEER_REVIEW_CHAIRMAN:
            return await self._run_peer_review_chairman(task, subagent, model_pack, **kwargs)
        elif protocol == GovernanceProtocol.HIERARCHICAL:
            return await self._run_hierarchical(task, subagent, model_pack, **kwargs)
        else:
            # Fallback to default peer review + chairman
            logger.warning(f"Unknown protocol '{protocol}', falling back to PEER_REVIEW_CHAIRMAN")
            return await self._run_peer_review_chairman(task, subagent, model_pack, **kwargs)

    async def _run_majority_vote(
        self, task: str, subagent: str, model_pack: dict[str, str], **kwargs
    ) -> CouncilResult:
        """Fast parallel drafts with majority vote, no synthesis."""
        # TODO: Implement majority vote protocol
        # For now, delegate to default orchestrator
        return await self.orchestrator.run(task=task, subagent=subagent, **kwargs)

    async def _run_vote_and_deliberate(
        self, task: str, subagent: str, model_pack: dict[str, str], **kwargs
    ) -> CouncilResult:
        """Vote + cross-model deliberation + chairman synthesis."""
        # TODO: Implement vote and deliberate protocol
        # For now, delegate to default orchestrator
        return await self.orchestrator.run(task=task, subagent=subagent, **kwargs)

    async def _run_peer_review_chairman(
        self, task: str, subagent: str, model_pack: dict[str, str], **kwargs
    ) -> CouncilResult:
        """Default: parallel drafts + critique + chairman synthesis."""
        # Use the existing orchestrator's default flow
        return await self.orchestrator.run(task=task, subagent=subagent, **kwargs)

    async def _run_hierarchical(
        self, task: str, subagent: str, model_pack: dict[str, str], **kwargs
    ) -> CouncilResult:
        """Parallel sub-councils with meta-synthesizer."""
        # TODO: Implement hierarchical protocol
        # For now, delegate to default orchestrator
        return await self.orchestrator.run(task=task, subagent=subagent, **kwargs)


__all__ = ["AdaptiveProtocolRunner"]
