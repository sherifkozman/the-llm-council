# Task-Adaptive Council Protocol

## Overview

The Task-Adaptive Council Protocol extends the standard LLM Council with automatic **task classification** and **governance protocol selection**. Instead of running every task through the same peer-review + chairman flow, the council first classifies the task and then selects the most cost-effective and accurate protocol for that specific task type.

This reduces unnecessary API calls for simpler tasks while preserving rigorous multi-stage review for high-stakes ones.

---

## Architecture

```
council run <subagent> "<task>"
          |
          v
   TaskClassifier
   (regex patterns)
          |
          v
   GovernanceProtocol
   selector
          |
    ______v______
   |      |      |
 VOTE  DELIBER  PEER_REVIEW  HIERARCHICAL
  |      |        |              |
  v      v        v              v
Majority Cross-   Original   Sub-councils
Vote    model     flow       + meta-synth
        debate
```

---

## Task Classes

| TaskClass | Triggers (keywords) | Default Protocol |
|-----------|--------------------|-----------------|
| `SECURITY` | security, vulnerabilit, inject, xss, audit, pentest | `PEER_REVIEW_CHAIRMAN` |
| `STRATEGY` | architect, design, adr, tradeoff, should we, decision | `HIERARCHICAL` |
| `CODE` | implement, build, write, refactor, api, endpoint | `PEER_REVIEW_CHAIRMAN` |
| `REASONING` | calculate, math, algorithm, sort, complexity | `MAJORITY_VOTE` |
| `RESEARCH` | research, summarize, compare, overview, benchmark | `VOTE_AND_DELIBERATE` |
| `GENERAL` | *(catch-all)* | `PEER_REVIEW_CHAIRMAN` |

Classification precedence (highest to lowest): `SECURITY > STRATEGY > CODE > REASONING > RESEARCH > GENERAL`

---

## Governance Protocols

### `MAJORITY_VOTE` (fastest)
Best for deterministic tasks with verifiable answers (math, algorithms, logic).

1. All providers draft independently in parallel
2. Most common answer wins; longest draft wins on tie
3. **No synthesis LLM call** — lowest cost and latency

### `VOTE_AND_DELIBERATE`
Best for nuanced research tasks where models benefit from seeing each other's perspective.

1. Round 1: parallel drafts
2. Deliberation: each provider sees anonymised summaries of other drafts and refines
3. Chairman synthesises the refined drafts

### `PEER_REVIEW_CHAIRMAN` (original flow)
Best for code generation, security review, and general tasks.

1. Parallel drafts
2. Adversarial critic identifies weaknesses
3. Chairman synthesises a validated final output

### `HIERARCHICAL`
Best for architecture/strategy decisions where multiple dimensions matter equally.

1. Parallel sub-councils run on `security`, `performance`, and `design` aspects
2. Each sub-council runs `PEER_REVIEW_CHAIRMAN` internally
3. Meta-synthesizer merges all three sub-council outputs

---

## Model Pack Selection

Each task class maps to a recommended model pack via environment variables. Set these to override defaults:

```bash
# Fast tasks (reasoning / majority vote)
export COUNCIL_MODEL_REASONING="anthropic/claude-opus-4-6"
export COUNCIL_MODEL_FAST="anthropic/claude-haiku-4-5"

# Code tasks
export COUNCIL_MODEL_CODE="openai/gpt-5.4"
export COUNCIL_MODEL_CODE_COMPLEX="anthropic/claude-opus-4-6"
export COUNCIL_MODEL_CRITIC="anthropic/claude-sonnet-4-6"

# Research tasks
export COUNCIL_MODEL_GROUNDED="google/gemini-3.1-pro-preview"
```

| Task Class | Draft model env var | Critique env var | Synthesis env var |
|------------|--------------------|-----------------|-----------------|
| REASONING | `COUNCIL_MODEL_REASONING` | `COUNCIL_MODEL_FAST` | `COUNCIL_MODEL_REASONING` |
| CODE | `COUNCIL_MODEL_CODE` | `COUNCIL_MODEL_CRITIC` | `COUNCIL_MODEL_CODE_COMPLEX` |
| SECURITY | `COUNCIL_MODEL_REASONING` | `COUNCIL_MODEL_CRITIC` | `COUNCIL_MODEL_REASONING` |
| RESEARCH | `COUNCIL_MODEL_GROUNDED` | `COUNCIL_MODEL_FAST` | `COUNCIL_MODEL_GROUNDED` |
| STRATEGY | `COUNCIL_MODEL_REASONING` | `COUNCIL_MODEL_CRITIC` | `COUNCIL_MODEL_REASONING` |
| GENERAL | `COUNCIL_MODEL_CODE` | `COUNCIL_MODEL_CRITIC` | `COUNCIL_MODEL_CODE_COMPLEX` |

---

## Usage

Adaptive protocol is opt-in via the Python API:

```python
from llm_council import Council
from llm_council.protocol.types import CouncilConfig
from llm_council.engine.adaptive_protocol import AdaptiveProtocolRunner

config = CouncilConfig(providers=["openrouter"])
council = Council(config=config)
orchestrator = council._orchestrator  # access internal orchestrator

runner = AdaptiveProtocolRunner(orchestrator)
result = await runner.run(
    task="Calculate the time complexity of quicksort",
    subagent="planner"
)

print(result.task_class)   # "reasoning"
print(result.protocol)     # "majority_vote"
print(result.output)
```

The result includes two new fields on `CouncilResult`:
- `task_class` — the detected `TaskClass` value (e.g. `"code"`, `"security"`)
- `protocol` — the `GovernanceProtocol` that was executed

---

## New Files

| File | Purpose |
|------|---------|
| `src/llm_council/engine/task_classifier.py` | `TaskClass`, `GovernanceProtocol` enums, `classify_task()`, `get_protocol_for_task()` |
| `src/llm_council/engine/adaptive_protocol.py` | `AdaptiveProtocolRunner` — dispatches to correct protocol |
| `tests/test_task_classifier.py` | Unit tests for classification and protocol mapping |
| `docs/adaptive-protocol.md` | This file |

---

## Cost & Latency Impact

| Protocol | Relative LLM calls | Best for |
|----------|--------------------|----------|
| MAJORITY_VOTE | N drafts (no critique/synth) | ~40% cheaper on reasoning tasks |
| VOTE_AND_DELIBERATE | N*2 + 1 synthesis | Balanced research tasks |
| PEER_REVIEW_CHAIRMAN | N + 1 critique + 1 synth | Code, security, general |
| HIERARCHICAL | 3*(N+2) + 1 meta-synth | Strategy — worth the cost |

---

## Extending

To add a new task class:
1. Add a member to `TaskClass` in `task_classifier.py`
2. Add a regex pattern constant and update `classify_task()`
3. Map it in `TASK_PROTOCOL_MAP` and `TASK_MODEL_PACK`
4. Add test cases in `tests/test_task_classifier.py`

To add a new governance protocol:
1. Add a member to `GovernanceProtocol`
2. Implement `_run_<protocol_name>()` in `AdaptiveProtocolRunner`
3. Wire it up in `AdaptiveProtocolRunner.run()`
