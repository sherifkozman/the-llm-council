# Roadmap

> Multi-model AI council for higher-quality, more trustworthy outputs.

This document outlines the development direction for The LLM Council. We welcome community input on priorities and feature design.

## Current Release: v0.2.0

The foundation is complete:

- **Multi-model council** - Parallel drafts from Claude, GPT-4, Gemini via OpenRouter or direct APIs
- **Adversarial critique** - Models challenge each other's outputs
- **Validated synthesis** - JSON schema validation with retry logic
- **10 specialized subagents** - router, planner, assessor, researcher, architect, implementer, reviewer, test-designer, shipper, red-team
- **Graceful degradation** - Continues if some providers fail
- **Provider flexibility** - OpenRouter (single key) or direct APIs (Anthropic, OpenAI, Google)
- **CLI tooling** - `council run`, `council doctor`, `council config`

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

**Why**: Not every task needs GPT-4. Smart routing reduces costs significantly.

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

*Last updated: 2025-12-20*
