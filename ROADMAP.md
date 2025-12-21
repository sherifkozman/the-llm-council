# The LLM Council - Roadmap

## Completed (v0.2.0)

- [x] Multi-model council via OpenRouter (single API key, multiple models)
- [x] Parallel drafts from Claude, GPT-4, Gemini
- [x] Adversarial critique phase
- [x] Synthesis with JSON schema validation and retry logic
- [x] 10 specialized subagents (router, planner, assessor, researcher, architect, implementer, reviewer, test-designer, shipper, red-team)
- [x] Graceful degradation (continues if some providers fail)
- [x] Health checks and provider registry
- [x] Artifact storage with tiered summarization
- [x] CLI (council run, council doctor, council config)
- [x] Direct API support (Anthropic, OpenAI, Google)
- [x] Trusted publishing to PyPI and TestPyPI
- [x] Multi-model configuration via environment variables

---

## Proposed Features

### Priority 1: Policy Engine + Human-in-the-Loop Approvals
**Value: 8/10 | Complexity: 3/10 | Ratio: 2.67**

Gate risky actions (file writes, commands, network) with approval prompts.
- Default-safe policies
- "Remember decision" mode to reduce friction
- Per-project policy presets
- Addresses "vibe coding is insecure" criticism

**Status**: Not started

---

### Priority 2: Adaptive Cost/Quality Routing
**Value: 8/10 | Complexity: 4/10 | Ratio: 2.00**

Route cheaper models to low-risk tasks, escalate to stronger models on validation failure.
- Task classification (summarization, review, implementation)
- Validator-driven escalation
- Replay-based regression tests
- Target: 20-50% cost reduction

**Status**: Not started

---

### Priority 3: Deterministic Caching + Replay
**Value: 7/10 | Complexity: 4/10 | Ratio: 1.75**

Content-addressed cache to skip redundant calls, replay runs for comparison.
- Hash key: normalized prompt + model + tools + policy
- Local-only by default
- Configurable retention and encryption
- Target: 30-70% latency reduction on cache hits

**Status**: Not started

---

### Priority 4: Local Observability/Tracing + Run Viewer
**Value: 8/10 | Complexity: 5/10 | Ratio: 1.60**

Structured traces for debugging multi-agent systems.
- JSON trace store
- Optional OpenTelemetry export
- Sensitive content redaction by default
- TUI or web-based run viewer

**Status**: Research in progress

---

### Priority 5: Context Management
**Value: 8/10 | Complexity: 6/10 | Ratio: 1.33**

Smarter context selection under token constraints.
- Diff-aware context selection
- Token budgets per phase
- Artifact prioritization
- Prevents incorrect pruning

**Status**: Not started

---

### Priority 6: Persistent Project Memory
**Value: 9/10 | Complexity: 7/10 | Ratio: 1.29**

Retain preferences and past decisions across sessions.
- Local-only, privacy-first
- Per-project scopes
- UI to inspect/edit/delete
- Strict retention defaults

**Status**: Not started

---

### Priority 7: Streaming + Live TUI Dashboard
**Value: 7/10 | Complexity: 6/10 | Ratio: 1.17**

Real-time output streaming with live progress display.
- Rich/Textual TUI rendering
- Phase progress indicators
- Partial output labeling
- Final validated synthesis indicator

**Status**: Not started

---

### Priority 8: Sandboxed Execution
**Value: 9/10 | Complexity: 8/10 | Ratio: 1.13**

Safe code execution for implementer/test phases.
- Docker-based isolation
- Resource limits
- Filesystem/network controls
- Staged rollout: dry-run → HITL → auto

**Status**: Not started (future consideration)

---

## Rejected/Deferred Ideas

| Idea | Reason |
|------|--------|
| aisuite integration | Redundant - OpenRouter already provides unified interface; aisuite lacks streaming |

---

## Research Completed

### Monitoring Terminal/Dashboard (Research Complete)

**Question**: Should we add a TUI or web dashboard to visualize LLM processes, prompts, and responses?

**Rubric Scoring**:

| Option | Feasibility | Value | Effort | Recommendation |
|--------|-------------|-------|--------|----------------|
| Build TUI (Rich/Textual) | 8/10 | 8/10 | 5/10 | **Primary** |
| Integrate OTel + Jaeger/Phoenix | 7/10 | 8/10 | 4/10 | Secondary |
| Build Web Dashboard | 6/10 | 7/10 | 8/10 | Defer |
| Don't Build | 10/10 | 2/10 | 1/10 | Not recommended |

**Key Findings**:
1. Orchestrator already has clean phase boundaries (drafts → critique → synthesis) and metrics
2. No event stream/bus exists - must be added first
3. Redaction not implemented (only placeholder) - privacy risk for prompt/response visualization
4. CLI providers (codex-cli) don't support streaming - monitoring will be coarse-grained

**Recommendation**: Build TUI (opt-in, TTY-aware)
1. Add structured event bus first (prerequisite for all monitoring)
2. Start with Rich `Live` progress + phase/latency/cost display
3. Add Textual panes for full prompt/response inspection later
4. OpenTelemetry as optional extra for web-based tracing

**Prerequisites**:
- [ ] Implement event bus in orchestrator
- [ ] Implement redaction layer (currently placeholder)
- [ ] Add `--monitor tui` flag (opt-in, TTY-only)

---

## Research Queue

- [ ] Claude Code CLI event integration for real-time streaming
- [ ] MCP tool integration patterns

---

*Last updated: 2025-12-20*
