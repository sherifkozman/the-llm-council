---
name: council
description: Run multi-LLM council for adversarial debate and cross-validation. Use it for implementation, architecture, review, security, research, and planning tasks with the canonical llm-council subagents and modes.
---

# LLM Council Skill (v0.7.7)

Multi-model council: parallel drafts, adversarial critique, validated synthesis.

> This skill requires the `the-llm-council` package to be installed. The skill
> provides the agent-side interface; actual runs happen through the installed
> `council` CLI.

## Setup

Install the package:

```bash
pip install 'the-llm-council>=0.7.5'
```

Optional extras:

```bash
pip install 'the-llm-council[anthropic,openai,gemini]>=0.7.5'
pip install 'the-llm-council[vertex]>=0.7.5'
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
council doctor --deep --provider claude --provider gemini --provider codex
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
| `--reasoning-profile off|light` | Reduce reasoning overhead |
| `--route` | Follow a router decision into the chosen subagent/mode |
| `--files` | Add file context to the task |
| `--dry-run` | Show the resolved plan without executing |
| `--schema` | Use a custom output schema |

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
