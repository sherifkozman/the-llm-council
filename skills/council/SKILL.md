---
name: council
description: Run multi-LLM council for adversarial debate and cross-validation. Use it for implementation, architecture, review, security, research, and planning tasks with the canonical llm-council subagents and modes.
---

# LLM Council Skill (v0.8.2)

Multi-model council: parallel drafts, adversarial critique, validated synthesis.

> This skill requires the `the-llm-council` package to be installed. The skill
> provides the agent-side interface; actual runs happen through the installed
> `council` CLI.

## Agent Invocation Contract

For tool-driven runs, read [the portable invocation contract](references/invocation-contract.md)
before execution. It requires Council 0.8.2 or later and applies to Codex, Claude
Code, Hermes and generic subprocess callers. Use an explicit executable/CWD,
task-file or stdin input, ordered providers/models, a unique result file, and
wait for completion. Inspect `execution_status` and full diagnostics; exit 0 or
`success: true` alone is not a complete review. Never retry a still-running run.

## Setup

Install the package:

```bash
pip install 'the-llm-council>=0.8.2'
```

Optional extras:

```bash
pip install 'the-llm-council[anthropic,openai,gemini]>=0.8.2'
pip install 'the-llm-council[vertex]>=0.8.2'
```

Configure at least one provider key:

| Provider | Environment Variable |
|----------|----------------------|
| OpenRouter | `OPENROUTER_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini API | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| Vertex AI | `GOOGLE_CLOUD_PROJECT` or `ANTHROPIC_VERTEX_PROJECT_ID` + ADC |

Verify what is usable in the current shell:

```bash
council doctor
council doctor --deep --provider claude --provider gemini-cli --provider codex
```

## Canonical Surface

```bash
council run <subagent> [--mode <mode>] "<task>" [options]
```

Primary subagents:

| Subagent | Modes | Use for |
|----------|-------|---------|
| `drafter` | `impl`, `arch`, `test` | implementation, architecture, tests |
| `critic` | `review`, `security` | code review and security analysis |
| `planner` | `plan`, `assess` | execution plans and decision assessments |
| `researcher` | — | research with sources and evidence |
| `router` | — | task classification and routed handoff |
| `synthesizer` | — | final merged output |

Legacy aliases such as `implementer`, `architect`, `reviewer`, `red-team`,
`assessor`, `test-designer`, and `shipper` still work, but they are no longer
the preferred interface.

## Common Commands

```bash
# Implementation
council run drafter --mode impl "Add pagination to users API"

# Architecture
council run drafter --mode arch "Design a caching layer"

# Tests
council run drafter --mode test "Design tests for cursor pagination"

# Review
council run critic --mode review "Review auth changes"

# Security
council run critic --mode security "Analyze auth system vulnerabilities"

# Planning
council run planner --mode plan "Plan MongoDB to PostgreSQL migration"

# Assessment
council run planner --mode assess "Redis vs Memcached for sessions"

# Research
council run researcher "Research WebSocket libraries for Node.js"

# Router handoff
council run router "Should we buy or build auth?" --route
```

## Useful Options

| Option | Purpose |
|--------|---------|
| `--mode <mode>` | Select a runtime mode for `drafter`, `critic`, or `planner` |
| `--json` | Return structured JSON |
| `--verbose` | Show resolved execution details and council phases |
| `--providers` | Explicit provider list. Omit to use config defaults |
| `--models` | Explicit model list |
| `--runtime-profile bounded` | Lower latency and token budgets |
| `--timeout` | Per-attempt seconds, 10-3600 (default 120); use the default runtime profile for long deadlines |
| `--reasoning-profile off\|light` | Request reduced reasoning; inspect provider compilation metadata for unsupported or uncontrolled native effort |
| `--route` | Follow a router decision into the chosen subagent/mode |
| `--files` | Add file context, capped at 50,000 characters/file and 200,000 retained characters total |
| `--dry-run` | Show the resolved plan without executing |
| `--schema` | Use a custom output schema |

These file limits count characters, not KB. Per-file overflow truncates input;
total retained overflow fails before calls. Inspect
`execution_plan.context_preparation.files` and
`degradation_report.context_warnings`; never present truncated input as a full
review. Context warnings remain visible with graceful degradation disabled.
The caller's outer deadline must cover the whole workflow, not just one attempt.
The `bounded` profile retains shorter phase caps regardless of a larger timeout.
Successful deep doctor probes prove reachability, not large-review completion or
quality. Codex base doctor success reports login status only, not generation readiness.

## Provider Names

Canonical provider names:

- `openrouter`
- `openai`
- `anthropic`
- `gemini`
- `gemini-cli`
- `vertex-ai`
- `claude`
- `codex`

User-selected providers and models should be respected. Health checks and deep
doctor probes are for diagnostics, not for silently overriding explicit
configuration.

For native CLI providers, consult the invocation contract's exact version list:
Codex 0.149.1/0.153.3 and Claude Code 2.1.288/2.1.289/2.1.290/2.1.291/2.1.292.
Unknown versions are unsupported, not automatically admitted by a patch range
or help-flag check. Report the chosen path/version from doctor failure diagnostics;
never edit package constants to bypass the gate. Synthetic contract checks are
not live-provider or future auto-update compatibility proof.

## When To Use It

Use council for:

- non-trivial implementation work
- architecture and system design
- code review and security analysis
- planning and build-vs-buy style decisions
- research that benefits from multiple models critiquing each other

Skip it for:

- trivial one-line edits
- simple lookups
- tasks where a single direct model call is clearly enough
