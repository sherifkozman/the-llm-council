# Capability-Augmented Council

**Status:** In progress
**Last updated:** 2026-03-27

## Summary

LLM Council should evolve from a prompt-centric multi-model pipeline into a
mode-aware execution system that loads evidence-gathering capabilities only
when the task requires them.

The goal is not to turn council into a generic agent runtime. The goal is to
make each supported mode materially better at its own job while preserving
predictable cost and latency.

This spec includes planning and security modes as first-class targets, not
just implementation and review.

## Problem

The repo already defines multiple modes and roles:

- `drafter --mode impl|arch|test`
- `critic --mode review|security`
- `planner --mode plan|assess`
- `researcher`, `router`, `synthesizer`

The current runtime is still dominated by a generic three-phase flow:

1. parallel drafts
2. critique
3. synthesis

That is useful, but it leaves quality on the table because prompts alone are
not enough for many tasks:

- planning needs dependency and risk evidence
- assessment needs explicit decision criteria and alternatives
- security needs attack-surface and exploit-path grounding
- implementation needs codebase context
- research needs current sources and citations

The repo already points toward a better architecture:

- router output can include `mode`, `model_pack`, and `suggested_tools`
- the tool registry supports role-based capabilities
- the skill package positions council as a reusable agent skill

The missing piece is runtime execution that actually consumes those signals.

## Grounded Observations

This document started as a kickoff spec. The current release line now has a partial
implementation of the design, so these observations describe the remaining gap,
not a blank-slate system.

- `CouncilConfig` now propagates `mode`, `temperature`, `max_tokens`,
  `runtime_profile`, `reasoning_profile`, and `output_schema` into runtime
  execution.
- Consolidated subagent YAML files declare mode-specific schema, prompts,
  reasoning, model packs, and in some cases provider/model preferences.
- Router output can now drive routed follow-up runs, model-pack selection, and
  capability planning metadata.
- `ToolRegistry` and `config/tool_registry.yaml` now participate in capability
  pack selection, but the evidence/execution depth is still incomplete for some
  modes.

This means the package has moved beyond prompt-only wiring, but mode-native
quality and benchmark maturity are still incomplete.

## Goals

- Make mode-specific configuration executable rather than descriptive.
- Keep prompt-only execution as the default for low-risk, low-context tasks.
- Add lazy-loaded capability packs for tasks that need evidence.
- Measure quality and cost per mode instead of using one blended metric.
- Prioritize planning and security alongside implementation and review.

## Non-Goals

- Building a general plugin marketplace in the first phase.
- Loading every tool on every run.
- Replacing the debate/critique/synthesis model with a fully autonomous agent loop.
- Claiming quality gains without mode-specific evaluation.

## Design Principles

### 1. Mode First

The execution engine should optimize for the current mode, not for a generic
"best prompt".

### 2. Evidence Before Confidence

Modes that make claims about code, risks, plans, or external facts should
gather evidence before producing high-confidence output.

### 3. Lazy Capabilities

Prompt-only should remain available and cheap. Tooling and skills should be
loaded only when routing or validation indicates they are needed.

### 4. Measured Tradeoffs

Every added capability must justify itself in quality, precision, acceptance,
or speed. No decorative complexity.

## Target Architecture

### Execution Profiles

Add an explicit execution profile selected by routing and/or runtime policy:

- `prompt_only`
- `light_tools`
- `grounded`
- `deep_analysis`

The router should recommend a profile. The orchestrator may escalate if the
task is high-risk or the first pass lacks required evidence.

### Capability Packs

Capability packs are internal bundles of tools, prompts, and evidence rules.
They are not user-facing plugins in phase 1.

Initial packs:

- `repo-analysis`
- `docs-research`
- `planning-assess`
- `security-code-audit`
- `red-team-recon`
- `diff-review`

### Staged Execution

Replace the flat "always draft, critique, synthesize" mental model with a
staged model:

1. route the task
2. select mode + execution profile + capability pack
3. gather evidence if required
4. run drafts against the evidence-backed context
5. critique the drafts
6. synthesize the final output
7. validate schema and evidence requirements

## Mode-to-Capability Map

| Subagent / Mode | Primary capability packs | Why |
|---|---|---|
| `planner --mode plan` | `planning-assess`, `repo-analysis` | plans need dependencies, risks, owners, and deliverables grounded in repo reality |
| `planner --mode assess` | `planning-assess`, `docs-research` | decisions need criteria, alternatives, reversibility, and external/vendor context |
| `critic --mode security` | `red-team-recon`, `security-code-audit`, `repo-analysis` | security needs distinct attack-surface recon and code-level weakness evidence before synthesizing a red-team assessment |
| `critic --mode review` | `diff-review`, `repo-analysis` | review quality improves with changed-hunk context, symbol context, and test impact |
| `researcher` | `docs-research` | research needs citations, freshness, and source ranking |
| `drafter --mode impl` | `repo-analysis`, optional `docs-research` | implementation should follow local patterns and current APIs |
| `drafter --mode arch` | `repo-analysis` | architecture needs interfaces, boundaries, and dependency awareness |
| `drafter --mode test` | `repo-analysis` | test design needs code-under-test discovery and gap analysis |
| `synthesizer` | no primary pack; consumes evidence | synthesis should merge grounded outputs rather than gather new evidence by default |

## Capability Pack Scope

### `repo-analysis`

Suggested initial tools:

- `read_file`
- `grep_search`
- code inventory / symbol summary
- dependency map
- changed-files summarizer

Target consumers:

- `drafter`
- `critic`
- `planner`

### `docs-research`

Suggested initial tools:

- `web_search`
- `context7_lookup`
- source ranking
- citation extraction

Target consumers:

- `researcher`
- `planner --mode assess`
- `drafter --mode impl` when external APIs are involved

### `planning-assess`

Suggested initial tools:

- checklist generation
- dependency/risk matrix generation
- alternative comparison helper
- effort/reversibility template builder

Target consumers:

- `planner --mode plan`
- `planner --mode assess`

### `security-code-audit`

Suggested initial tools:

- auth/dataflow grep patterns
- secret/config misuse checks
- dependency advisory lookup
- code-path and sink/source evidence

Target consumers:

- `critic --mode security`

### `red-team-recon`

Suggested initial tools:

- attack-surface inventory
- exposed-route and callback discovery
- trust-boundary and entry-point hints
- attacker-prerequisite framing

Target consumers:

- `critic --mode security`

### `diff-review`

Suggested initial tools:

- git diff parser
- hunk-to-file mapper
- nearby context fetcher
- test impact summarizer

Target consumers:

- `critic --mode review`

## Phased Rollout

### P0: Runtime Truthfulness

Objective: make declared mode behavior real.

Deliverables:

- propagate `mode` through `Council` and `Orchestrator`
- apply mode-specific schema selection
- apply mode-specific prompt extensions
- apply mode-specific model pack and model selection
- apply mode-specific provider preferences where supported
- surface the effective execution plan in `--dry-run` and `--verbose`
- add tests proving mode-specific behavior is honored

Primary files:

- `src/llm_council/council.py`
- `src/llm_council/engine/orchestrator.py`
- `src/llm_council/subagents/__init__.py`
- `src/llm_council/cli/main.py`

Success metrics:

- 100% mode/schema selection correctness in integration tests
- 100% mode/prompt selection correctness in integration tests
- zero CLI flags accepted but silently ignored for supported mode/runtime settings

### P1: Capability Selection

Objective: enable lazy-loaded evidence gathering.

Deliverables:

- define execution profiles
- extend router output consumption in runtime
- load role-appropriate capability packs from tool registry
- support prompt-only fallback for low-risk tasks

Primary files:

- `src/llm_council/engine/orchestrator.py`
- `src/llm_council/registry/tool_registry.py`
- `config/tool_registry.yaml`
- `src/llm_council/subagents/router.yaml`
- `src/llm_council/schemas/router.json`

Success metrics:

- prompt-only remains default for low-risk tasks
- capability packs activate only when requested or justified
- tool activation telemetry available per run

### P1.5: Planning and Security Hardening

Objective: improve the two high-value, under-grounded modes first.

Deliverables:

- `planning-assess` capability pack
- `security-code-audit` and `red-team-recon` capability packs
- evidence requirements for planning and security outputs
- mode-specific validation rules for unsupported claims

Success metrics:

- lower unsupported-claim rate in planning outputs
- higher confirmed-finding rate in security outputs
- lower recommendation reversal rate in assess mode

### P2: Mode-Specific Evaluation

Objective: measure value honestly.

Deliverables:

- benchmark sets for `plan`, `assess`, `security`, `review`, `impl`, `arch`, `test`, `research`
- per-mode scorecards
- cost/latency tracking by mode and execution profile

Success metrics:

- every mode has at least one benchmark set and a documented scorecard
- regressions are caught before release

### P3: Feedback and Iteration

Objective: close the loop after first adoption.

Deliverables:

- acceptance / dismissal capture for findings and recommendations
- repeated false-positive tracking
- capability-pack tuning based on observed value

Success metrics:

- measurable drop in repeated low-value outputs
- measurable increase in accepted outcomes per cost dollar

## Per-Mode Scorecards

### Planning (`planner --mode plan`)

- plan acceptance rate
- missed dependency rate
- missed risk rate
- number of follow-up clarifications needed

### Assessment (`planner --mode assess`)

- recommendation reversal rate
- alternatives coverage
- reversibility coverage
- confidence calibration against later outcomes

### Security (`critic --mode security`)

- confirmed finding rate
- invalid exploit-path rate
- severity calibration accuracy
- remediation usefulness score

### Review (`critic --mode review`)

- accepted finding rate
- invalid location/snippet rate
- false-positive rate
- cost per accepted finding

### Research (`researcher`)

- citation validity rate
- freshness success rate
- unsupported claim rate

### Implementation (`drafter --mode impl`)

- first-pass build/test success
- human rework after generation
- local-pattern adherence

### Architecture (`drafter --mode arch`)

- constraint miss rate
- interface completeness
- major issue discovery before implementation

### Test (`drafter --mode test`)

- defect-detection rate
- coverage delta
- maintenance burden of generated tests

## Kickoff Task Breakdown

1. Fix mode plumbing end to end.
2. Extend router output schema with execution profile and required capabilities.
3. Add runtime execution plan object and expose it in CLI output.
4. Implement `planning-assess` capability pack.
5. Implement `security-code-audit` and `red-team-recon` capability packs.
6. Implement `docs-research` capability pack.
7. Implement `repo-analysis` capability pack.
8. Add per-mode eval harness and baseline datasets.

## Recommended Order

- Start with `planner --mode plan`, `planner --mode assess`, and `critic --mode security`.
- Then lift `researcher`.
- Then lift `drafter --mode impl|arch|test`.
- Then refine `critic --mode review`.

This order is intentional:

- planning and assessment improve strategic value quickly
- security is high-risk and benefits strongly from evidence
- research unlocks other modes
- implementation and review benefit from shared repo-analysis after that

## Deferred Work

- broad external plugin ecosystem
- fully autonomous fixing loops
- persistent cross-run memory for all modes
- dynamic learning from arbitrary user chats

These may be useful later, but they are not needed to prove the value of
capability-augmented council.
