# LLM Council Documentation

LLM Council is a multi-LLM orchestration framework for adversarial debate,
cross-validation, and structured decision-making across multiple model
providers.

The public package now supports both:

- the core three-phase council flow
- a mode-aware execution path with lightweight capability
  planning, routed handoff, and evaluation tooling

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

### Deep Doctor
`council doctor --deep` distinguishes “installed/configured” from “can answer a
trivial non-interactive prompt right now.” This is useful for diagnosing local
CLI providers and flaky auth or SDK setup.

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
