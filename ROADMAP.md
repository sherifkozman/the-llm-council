# Roadmap

> Multi-model AI council for higher-quality, more trustworthy outputs.

This document outlines the development direction for The LLM Council. We welcome community input on priorities and feature design.

## Current Release: v0.7.0

The current release foundation includes:

- **Multi-model council** - Parallel drafts from Claude, GPT-5.4, and Gemini via OpenRouter or direct APIs
- **Adversarial critique** - Models challenge each other's outputs
- **Validated synthesis** - JSON schema validation with retry logic
- **Mode-aware subagents** - primary runtime surface is `router`, `planner`, `researcher`, `drafter`, `critic`, and `synthesizer`, with legacy aliases still supported
- **Graceful degradation** - Continues if some providers fail
- **Provider flexibility** - OpenRouter, direct APIs (Anthropic, OpenAI, Google), Vertex AI (Gemini + Claude)
- **CLI tooling** - `council run`, `council doctor`, `council eval`, `council eval-compare`, `council eval-import-pr`, `council config`
- **Extended reasoning** - Support for thinking/reasoning modes (Anthropic, Google, OpenAI)
- **Runtime controls** - `runtime_profile`, `reasoning_profile`, routed handoff, and deeper provider readiness checks
- **Artifact store** - Content-addressed storage with tiered summarization

---

## Planned Features

Features are prioritized by value-to-complexity ratio. Status labels:

| Label | Meaning |
|-------|---------|
| `not started` | Open for contribution |
| `in progress` | Actively being worked on |
| `needs design` | Requires RFC or design discussion |
| `help wanted` | Community contributions welcome |

---

### Policy Engine + Human-in-the-Loop Approvals

**Priority: High** · `not started` · `help wanted`

Gate risky actions (file writes, shell commands, network calls) with approval prompts.

- Default-safe policies out of the box
- "Remember this decision" to reduce friction
- Per-project policy overrides
- Addresses security concerns with autonomous agents

**Why**: Autonomous agents need guardrails. This makes council safe for production use.

---

### Adaptive Cost/Quality Routing

**Priority: High** · `not started`

Route simpler tasks to cheaper models, escalate to stronger models on validation failure.

- Automatic task classification (summarization vs. implementation)
- Validator-driven escalation
- Target: 20-50% cost reduction without quality loss

**Why**: Not every task needs the most expensive reasoning models. Smart
routing reduces costs significantly.

---

### Mode-Native Capability Packs + Staged Execution

**Priority: High** · `in progress`

Move council from prompt-only execution toward mode-aware, lazy-loaded
capabilities selected at runtime.

- Keep `prompt_only` as the default for low-risk tasks
- Add execution profiles such as `light_tools`, `grounded`, and `deep_analysis`
- Load capability packs only when the mode and risk justify them
- Prioritize `planner --mode plan|assess` and `critic --mode security` in the first rollout
- Measure value per mode instead of relying on one blended quality metric

Implemented baseline in the current release line:

- mode-aware runtime selection is wired through `Council` and `Orchestrator`
- routed handoff can follow the router-selected subagent/mode
- capability planning is exposed in execution plans
- deterministic eval tooling is public
- private PR-import benchmarking remains local-only under `.council-private/`

Reference design:
- `docs/architecture/capability-augmented-council.md`

**Why**: Prompts alone are not enough for planning, security, research, and
code-aware tasks. Mode-native evidence gathering should improve output quality
without forcing every run into an expensive agent loop.

---

### Deterministic Caching + Replay

**Priority: Medium** · `not started` · `help wanted`

Content-addressed cache to skip redundant LLM calls and replay runs for comparison.

- Hash key: normalized prompt + model + temperature + tools
- Local-only storage by default
- Configurable TTL and optional encryption
- Target: 30-70% latency reduction on cache hits

**Why**: Repeated queries during development waste time and money.

---

### Local Observability + Run Viewer

**Priority: Medium** · `needs design`

Structured traces for debugging multi-agent workflows.

- JSON trace store with phase boundaries
- TUI viewer (Rich/Textual) for terminal inspection
- Optional OpenTelemetry export for external tools
- Automatic redaction of sensitive content

**Prerequisites** (must be implemented first):
- [ ] Event bus in orchestrator
- [ ] Redaction layer for prompts/responses

**Why**: Debugging multi-agent systems is hard without visibility into each phase.

---

### Context Management

**Priority: Medium** · `not started`

Smarter context selection under token constraints.

- Diff-aware context (prioritize changed files)
- Token budgets per phase
- Artifact prioritization based on relevance

**Why**: Large codebases exceed context limits. Smart selection prevents incorrect pruning.

---

### Persistent Project Memory

**Priority: Medium** · `not started` · `needs design`

Retain learned preferences and past decisions across sessions.

- Local-only, privacy-first storage
- Per-project scopes
- CLI/UI to inspect, edit, delete memories
- Configurable retention policies

**Why**: Agents shouldn't forget project conventions between sessions.

---

### Streaming + Live TUI Dashboard

**Priority: Low** · `not started`

Real-time output streaming with live progress display.

- Rich/Textual TUI with phase indicators
- Token/cost counters
- Partial output display during generation

**Why**: Long-running council sessions benefit from progress visibility.

---

### Sandboxed Execution

**Priority: Low** · `not started` · `needs design`

Safe code execution for implementer and test-designer phases.

- Docker-based isolation
- CPU/memory/time limits
- Filesystem and network controls
- Staged rollout: dry-run → approval → auto

**Why**: Running generated code safely is essential for automated testing workflows.

---

## Contributing

We welcome contributions! Here's how to get involved:

1. **Pick a feature** - Look for `help wanted` labels above
2. **Open an issue** - Discuss your approach before starting
3. **Submit a PR** - Reference the roadmap item

For major features (`needs design`), please open a discussion first to align on the approach.

---

## Suggesting Features

Have an idea not listed here? [Open an issue](https://github.com/sherifkozman/the-llm-council/issues/new) with:

- **Problem**: What pain point does this solve?
- **Proposal**: How would it work?
- **Alternatives**: What other approaches did you consider?

---

*Last updated: 2026-03-27*
