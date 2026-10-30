# LLM Council Documentation

LLM Council is a multi-LLM orchestration framework for adversarial debate,
cross-validation, and structured decision-making across multiple model
providers.

For agent and CI integrations, start with the
[portable invocation contract](../skills/council/references/invocation-contract.md).
Council 0.8.1 adds execution status and exact exit semantics. Wait for the process,
read the full JSON result even after a nonzero exit, and inspect degradation.
The shipped skill and invocation examples require 0.8.5 or later for the safety
fixes. An installed version string does not establish live provider verification.

## Automatic Evidence Boundary

Automatic file-content evidence is confined to the selected repository's canonical
root. In-root file links keep their original source anchors; external targets,
broken links and non-regular files are not read. Descriptor-relative no-follow
opens reject symlinks substituted after validation. The root's ancestors are
trusted; this is not protection against hard links, mounts, hostile relocation of
open directories or concurrent content changes.

Platforms without safe descriptor-relative opens (including Windows) omit these
automatic content searches and emit a warning. Do not interpret absent matches
as a clean security review. Explicit `--files` remains a deliberate user selection;
verify those paths before including sensitive material. See the [0.8.5 containment record](releases/0.8.5.md) and [0.8.6 provider notes](releases/0.8.6.md).

The public package supports:

- the core three-phase council flow
- a mode-aware execution path with lightweight capability
  planning, routed handoff, and evaluation tooling
- provider-specific prompt-cache controls and cache telemetry where supported

## Input And Deadline Limits

`--files` retains at most 50,000 characters per file and 200,000 retained
characters in total. These are character limits, not KB or UTF-8 byte limits.
Per-file truncation is visible in `execution_plan.context_preparation.files` and
`degradation_report.context_warnings` (`kind: "file_truncated"`, path, original
and retained character counts), even when graceful degradation is disabled.
A successful truncated run is `degraded`, not a full-file review. Exceeding the
retained total fails before provider calls rather than silently skipping files.

`--timeout` accepts 10-3600 seconds per provider attempt, with the default still
120. Retries and multiple phases can exceed that duration in total; set a
separate outer workflow deadline and allow cancellation cleanup. Long-deadline
runs need `--runtime-profile default`: `bounded` keeps its existing shorter
provider/phase caps. Inspect `execution_plan.provider_request_timeouts_by_provider`
for the effective budgets.

## Overview

LLM Council centers on a three-phase workflow:

1. **Parallel Drafts** - Multiple providers generate independent solutions concurrently
2. **Adversarial Critique** - A critic model identifies weaknesses, contradictions, and blind spots
3. **Synthesis** - The best elements are merged, critiques addressed, and output validated against JSON schemas

This approach is intended to produce more robust, better-audited outputs than a
single-model pass. The newer capability-aware execution path should be treated
as an evolving runtime surface rather than a claim of fully benchmarked review
or security quality.

## Key Features

### Multi-Model Orchestration
Run parallel drafts from Claude (Anthropic), GPT-5.4 (OpenAI), Gemini
(Google/Vertex), or any supported provider. The framework coordinates multiple
provider calls and merges results into a structured output.

### Mode-Aware Runtime
Core subagents such as `drafter`, `critic`, and `planner` now honor explicit
`--mode` selection at runtime instead of treating mode definitions as static
YAML only. Execution profiles such as `prompt_only`,
`light_tools`, `grounded`, and `deep_analysis` are exposed in the runtime and
evaluation surfaces.

### Adversarial Critique
Built-in critique phase systematically identifies:
- Logical inconsistencies between drafts
- Missing edge cases or error handling
- Security vulnerabilities or blind spots
- Contradictory recommendations

### Schema Validation
JSON Schema validation with automatic retry ensures structured outputs conform to expected formats. Failed validations trigger re-synthesis with validation error details.

### Provider Agnostic
Swap between providers seamlessly:
- **OpenRouter** - Single API key for 100+ models (recommended)
- **Direct APIs** - Anthropic, OpenAI, and Gemini API native SDKs
- **CLI Providers** - `codex`, `gemini-cli`, and `claude` via local CLIs
- **Custom Providers** - Plugin architecture via Python entry points

### Prompt Cache Support
Prompt caching is provider-specific. Anthropic and eligible OpenRouter routes
receive explicit cache controls, OpenAI exposes cache telemetry when returned by
the API, and Gemini API / Vertex Gemini support cached-content create, TTL
refresh, expiry handling, and best-effort cleanup.

### Deep Doctor
`council doctor --deep` distinguishes “installed/configured” from “can answer a
trivial non-interactive prompt right now.” This is useful for diagnosing local
CLI providers and flaky auth or SDK setup.
It proves reachability only, not successful large reviews or complete evidence
coverage. Those claims require the actual run result and retained-input checks.
For Codex, base doctor checks record admission and login status for the default
request context. It does not prove generation readiness or supply path/version
evidence unless those details are actually present in the diagnostic.

Claude CLI admission checks required option definitions in the selected binary's
help on every call; its version is diagnostic, not an allowlist gate. Required
controls depend on the selected auth route and request. Help is captured through
a bounded terminal stream, or a bounded pipe if a terminal is unavailable. The
permission controls must advertise their required values. Missing or incomplete
controls stop execution before task text is sent, rather than weakening isolation.
Base doctor reports `cli_compatibility: required_flags_advertised` and leaves
authentication and generation unverified. This is syntax advertisement, not proof
of future isolation semantics or successful generation; use deep probes and
inspect actual results. Managed-policy checks and terminal validation remain.
When no model is selected in Council, Codex uses the model in the
selected Codex configuration; if neither selects one, the user must choose one.
The Codex provider requires a current compatibility record matching
the selected executable, resources, platform and request context. It runs version
and catalog checks from a verified private copy and rejects unknown or changed
builds. The published record currently covers macOS arm64 with ChatGPT
authentication and selected request contexts; other previously usable contexts
fail closed. Record updates can add qualified builds without changing Council;
they do not certify arbitrary future versions or provider behavior.

Codex runs with an isolated temporary `HOME`, the selected authentication file,
and a restricted environment that does not inherit proxy, CA or loader overrides.
An explicitly qualified API-key custom endpoint can use the user's selected URL;
ChatGPT custom routing is not currently qualified. Base doctor checks only the
default model, default reasoning, text output and login status. It does not prove
credential validity or generation readiness.

### Evaluation Tooling
`council eval`, `council eval-compare`, and `council eval-import-pr` provide a
public deterministic evaluation harness plus local-only PR-import support for
private benchmark creation.

### IDE Integration
JSON-lines protocol enables seamless integration with Claude Code and other IDEs. Results stream as structured JSON for programmatic consumption.

### Secret-Safe Logging
Built-in redaction pipeline prevents API keys and credentials from leaking into logs or artifacts.

## Installation

Basic installation:

```bash
pip install the-llm-council
```

With specific providers:

```bash
# OpenRouter only (recommended)
pip install the-llm-council

# Direct APIs
pip install the-llm-council[anthropic,openai,gemini]

# All providers
pip install the-llm-council[all]
```

## Quick Start

### CLI Usage

```bash
# Set API key
export OPENROUTER_API_KEY="your-key"

# Run a council task
council run drafter --mode impl "Build a REST API for user authentication"

# Check provider status
council doctor

# Verify actual non-interactive generation readiness
# May incur API/CLI usage.
council doctor --deep --provider claude --provider gemini-cli --provider codex

# Router handoff: classify, then run the chosen subagent/mode
council run router "Assess whether we should adopt a hosted vector store" --route

# Bound cost and latency for a run
council run critic --mode review "Review these auth changes" \
  --runtime-profile bounded \
  --reasoning-profile off

# Get JSON output
council run planner "Add dark mode support" --json
```

### Python API

```python
from llm_council import Council

# Initialize council
council = Council(providers=["openrouter"])

# Run a task
result = await council.run(
    task="Build a login page with OAuth",
    subagent="drafter"
)

# Access structured output
if result.success:
    print(result.output)  # Dict validated against JSON schema
    print(f"Duration: {result.duration_ms}ms")
else:
    print(result.validation_errors)
```

## Subagents

LLM Council currently exposes six primary subagents, with several backwards
compatible aliases still supported:

| Subagent | Purpose | Output Schema |
|----------|---------|---------------|
| `router` | Classify and route tasks to appropriate subagents | Task classification with routing recommendation |
| `planner` | Create execution roadmaps and assessment outputs | Plans, phases, risks, decisions |
| `researcher` | Research with citations and evidence | Research findings and references |
| `drafter` | Implementation, architecture, and test design | Code/design/test outputs by mode |
| `critic` | Code review and security analysis | Review or red-team findings by mode |
| `synthesizer` | Final merge and structured output production | Final schema-conforming output |

Primary runtime modes:

- `drafter --mode impl|arch|test`
- `critic --mode review|security`
- `planner --mode plan|assess`

Legacy aliases such as `implementer`, `architect`, `reviewer`, `red-team`,
`assessor`, and `test-designer` still work for backwards compatibility.

## How It Works

```
USER TASK
    |
    v
[ROUTE / MODE RESOLUTION]
    |
    +-- Optional router handoff
    +-- Resolve mode, schema, model pack, providers
    |
    v
[EVIDENCE SELECTION]
    |
    +-- prompt_only (default)
    +-- or capability-backed execution
    |
    v
[PARALLEL DRAFTS]
    |
    +-- Provider A --> Draft A
    +-- Provider B --> Draft B
    +-- Provider C --> Draft C
    |
    v
[ADVERSARIAL CRITIQUE]
    |
    +-- Challenge contradictions and weak assumptions
    |
    v
[SYNTHESIS]
    |
    +-- Merge best elements
    +-- Validate against JSON Schema
    +-- Retry on validation failure
    |
    v
[STRUCTURED OUTPUT]
```

## Configuration

Configuration can be provided via:

1. **Environment variables** (highest priority)
2. **Config file** at `~/.config/llm-council/config.yaml`
3. **Programmatic overrides** via `CouncilConfig`

Example config file:

```yaml
# ~/.config/llm-council/config.yaml

providers:
  - name: openrouter
    api_key: ${OPENROUTER_API_KEY}
    default_model: anthropic/claude-sonnet-4-6

  - name: anthropic
    api_key: ${ANTHROPIC_API_KEY}
    default_model: claude-sonnet-4-6

defaults:
  timeout: 120              # Timeout per API call (seconds)
  max_retries: 3            # Max synthesis retry attempts
  summary_tier: actions     # Artifact summarization depth
  enable_schema_validation: true
```

Config values support standard shell-style environment expansion such as `${OPENROUTER_API_KEY}` at load time. That expansion only affects values declared in `config.yaml`; if a provider config omits a field, the adapter still falls back to its normal provider-specific environment variables. For ad hoc provider selection, you can still override the model at runtime with `--models`.

Initialize config file:

```bash
council config --init
```

View current config:

```bash
council config --show
```

## Architecture

```
llm_council/
├── council.py           # Main Council facade
├── engine/
│   └── orchestrator.py  # Three-phase orchestration logic
├── providers/
│   ├── base.py          # ProviderAdapter abstract base class
│   ├── registry.py      # Provider discovery and registration
│   ├── openrouter.py    # OpenRouter adapter
│   ├── anthropic.py     # Anthropic adapter
│   ├── openai.py        # OpenAI adapter
│   ├── gemini.py       # Gemini API adapter
│   └── cli/             # CLI-based providers
├── subagents/           # Subagent prompt + schema definitions
├── schemas/             # JSON Schema definitions
├── validation/          # Schema validation utilities
└── cli/                 # Typer-based CLI
```

## Next Steps

- [Capability-Augmented Council](architecture/capability-augmented-council.md) - Kickoff spec for mode-native capabilities, staged execution, and per-mode evaluation
- [Eval Datasets](../evals/README.md) - Baseline datasets and usage for `council eval`
- [OpenRouter Quickstart](quickstart/openrouter.md) - Recommended setup with single API key
- [Direct APIs Quickstart](quickstart/direct-apis.md) - Using Anthropic, OpenAI, and Gemini API directly
- [Creating Custom Providers](providers/creating-providers.md) - Build your own provider adapters

## Support

- **GitHub**: [sherifkozman/the-llm-council](https://github.com/sherifkozman/the-llm-council)
- **Issues**: [Report bugs or request features](https://github.com/sherifkozman/the-llm-council/issues)
- **License**: MIT
