"""Task classification for adaptive protocol selection.

Classifies incoming tasks into a TaskClass, which determines which
GovernanceProtocol and model pack the council will use for the run.
"""
from __future__ import annotations

import re
from enum import Enum


class TaskClass(str, Enum):
    """Classification of a council task."""

    REASONING = "reasoning"   # deterministic / verifiable: math, logic, algorithms
    CODE = "code"             # open-ended code generation, refactoring, implementation
    SECURITY = "security"     # security audits, red-teaming, adversarial analysis
    RESEARCH = "research"     # summarisation, surveys, grounded Q&A
    STRATEGY = "strategy"     # architecture ADRs, high-stakes business decisions
    GENERAL = "general"       # catch-all for ambiguous tasks


class GovernanceProtocol(str, Enum):
    """Decision protocol variants the council can execute."""

    # Fastest: parallel drafts, majority vote wins — no synthesis LLM call
    MAJORITY_VOTE = "majority_vote"
    # Vote + one cross-model deliberation round, then chairman synthesis
    VOTE_AND_DELIBERATE = "vote_and_deliberate"
    # Anonymous peer-review + chairman synthesis (original flow)
    PEER_REVIEW_CHAIRMAN = "peer_review_chairman"
    # Parallel sub-councils (security/perf/design) + meta-synthesizer
    HIERARCHICAL = "hierarchical"


# ---------------------------------------------------------------------------
# Classification patterns
# ---------------------------------------------------------------------------

_REASONING_RE = re.compile(
    r"\b(calculat|comput|solv|math|equation|proof|logic|puzzle|algorithm|"
    r"complexity|big.?o|optimis|sort|search|tree|graph)\b",
    re.IGNORECASE,
)
_CODE_RE = re.compile(
    r"\b(implement|build|write|generat|code|function|class|module|api|"
    r"endpoint|refactor|migrat|scaffold|test|pytest|unittest|integration)\b",
    re.IGNORECASE,
)
_SECURITY_RE = re.compile(
    r"\b(security|vulnerabilit|inject|xss|csrf|auth|permission|token|secret|"
    r"exploit|attack|pentest|red.?team|owasp|cve|threat|audit)\b",
    re.IGNORECASE,
)
_RESEARCH_RE = re.compile(
    r"\b(research|summarise|summarize|compar|survey|review|explain|"
    r"what is|how does|overview|landscape|analysis|benchmark)\b",
    re.IGNORECASE,
)
_STRATEGY_RE = re.compile(
    r"\b(architect|design|decid|strategy|adr|tradeoff|trade-off|"
    r"recommend|evaluat|assess|should we|which option|decision)\b",
    re.IGNORECASE,
)


def classify_task(task: str) -> TaskClass:
    """Classify a task string into a TaskClass using regex pattern matching.

    Precedence (highest to lowest):
        SECURITY > STRATEGY > CODE > REASONING > RESEARCH > GENERAL
    """
    if _SECURITY_RE.search(task):
        return TaskClass.SECURITY
    if _STRATEGY_RE.search(task):
        return TaskClass.STRATEGY
    if _CODE_RE.search(task):
        return TaskClass.CODE
    if _REASONING_RE.search(task):
        return TaskClass.REASONING
    if _RESEARCH_RE.search(task):
        return TaskClass.RESEARCH
    return TaskClass.GENERAL


# ---------------------------------------------------------------------------
# Protocol + model-pack mapping
# ---------------------------------------------------------------------------

#: Maps TaskClass -> GovernanceProtocol
TASK_PROTOCOL_MAP: dict[TaskClass, GovernanceProtocol] = {
    TaskClass.REASONING:  GovernanceProtocol.MAJORITY_VOTE,
    TaskClass.CODE:       GovernanceProtocol.PEER_REVIEW_CHAIRMAN,
    TaskClass.SECURITY:   GovernanceProtocol.PEER_REVIEW_CHAIRMAN,
    TaskClass.RESEARCH:   GovernanceProtocol.VOTE_AND_DELIBERATE,
    TaskClass.STRATEGY:   GovernanceProtocol.HIERARCHICAL,
    TaskClass.GENERAL:    GovernanceProtocol.PEER_REVIEW_CHAIRMAN,
}

#: Model-pack env-var keys per task class (mirrors README env vars)
TASK_MODEL_PACK: dict[TaskClass, dict[str, str]] = {
    TaskClass.REASONING: {
        "draft":     "COUNCIL_MODEL_REASONING",
        "critique":  "COUNCIL_MODEL_FAST",
        "synthesis": "COUNCIL_MODEL_REASONING",
    },
    TaskClass.CODE: {
        "draft":     "COUNCIL_MODEL_CODE",
        "critique":  "COUNCIL_MODEL_CRITIC",
        "synthesis": "COUNCIL_MODEL_CODE_COMPLEX",
    },
    TaskClass.SECURITY: {
        "draft":     "COUNCIL_MODEL_REASONING",
        "critique":  "COUNCIL_MODEL_CRITIC",
        "synthesis": "COUNCIL_MODEL_REASONING",
    },
    TaskClass.RESEARCH: {
        "draft":     "COUNCIL_MODEL_GROUNDED",
        "critique":  "COUNCIL_MODEL_FAST",
        "synthesis": "COUNCIL_MODEL_GROUNDED",
    },
    TaskClass.STRATEGY: {
        "draft":     "COUNCIL_MODEL_REASONING",
        "critique":  "COUNCIL_MODEL_CRITIC",
        "synthesis": "COUNCIL_MODEL_REASONING",
    },
    TaskClass.GENERAL: {
        "draft":     "COUNCIL_MODEL_CODE",
        "critique":  "COUNCIL_MODEL_CRITIC",
        "synthesis": "COUNCIL_MODEL_CODE_COMPLEX",
    },
}


def get_protocol_for_task(task: str) -> tuple[TaskClass, GovernanceProtocol]:
    """Return (TaskClass, GovernanceProtocol) for a raw task string."""
    task_class = classify_task(task)
    protocol = TASK_PROTOCOL_MAP[task_class]
    return task_class, protocol


__all__ = [
    "TaskClass",
    "GovernanceProtocol",
    "classify_task",
    "get_protocol_for_task",
    "TASK_PROTOCOL_MAP",
    "TASK_MODEL_PACK",
]
