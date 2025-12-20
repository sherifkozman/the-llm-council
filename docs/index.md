# LLM Council Documentation

LLM Council is a Multi-LLM orchestration framework that enables adversarial debate, cross-validation, and structured decision-making across multiple language model providers.

## Overview

LLM Council orchestrates multiple LLM backends in a three-phase workflow:

1. **Parallel Drafts** - Multiple providers generate independent solutions concurrently
2. **Adversarial Critique** - A critic model identifies weaknesses, contradictions, and blind spots
3. **Synthesis** - The best elements are merged, critiques addressed, and output validated against JSON schemas

This approach produces more robust, well-considered outputs than single-model generation.

## Key Features

### Multi-Model Orchestration
Run parallel drafts from Claude (Anthropic), GPT (OpenAI), Gemini (Google), or any supported provider. The framework automatically coordinates multiple API calls and merges results.

### Adversarial Critique
Built-in critique phase systematically identifies:
- Logical inconsistencies between drafts
- Missing edge cases or error handling
- Security vulnerabilities or blind spots
- Contradictory recommendations

### Schema Validation
JSON Schema validation with automatic retry ensures structured outputs conform to expected formats. Failed validations trigger re-synthesis with error feedback.

### Provider Agnostic
Swap between providers seamlessly:
- **OpenRouter** - Single API key for 100+ models (recommended)
- **Direct APIs** - Anthropic, OpenAI, Google native SDKs
- **CLI Providers** - Codex CLI, Gemini CLI for local/offline use
- **Custom Providers** - Plugin architecture via Python entry points

### IDE Integration
JSON-lines protocol enables seamless integration with Claude Code and other IDEs. Results stream as structured JSON for programmatic consumption.

### Secret-Safe Logging
Built-in redaction pipeline prevents API keys and credentials from leaking into logs or artifacts.

## Installation

Basic installation:

```bash
pip install llm-council
```

With specific providers:

```bash
# OpenRouter only (recommended)
pip install llm-council

# Direct APIs
pip install llm-council[anthropic,openai,google]

# All providers
pip install llm-council[all]
```

## Quick Start

### CLI Usage

```bash
# Set API key
export OPENROUTER_API_KEY="your-key"

# Run a council task
council run implementer "Build a REST API for user authentication"

# Check provider status
council doctor

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
    subagent="implementer"
)

# Access structured output
if result.success:
    print(result.output)  # Dict validated against JSON schema
    print(f"Duration: {result.duration_ms}ms")
else:
    print(result.validation_errors)
```

## Subagents

LLM Council includes 10 specialized subagents, each with tailored prompts and JSON schemas:

| Subagent | Purpose | Output Schema |
|----------|---------|---------------|
| `router` | Classify and route tasks to appropriate subagents | Task classification with routing recommendation |
| `planner` | Create execution roadmaps with phases and dependencies | Phased plan with milestones and risks |
| `assessor` | Make build/buy/no-build decisions with weighted criteria | Decision matrix with scores and rationale |
| `researcher` | Deep market and technical research with citations | Research findings with formal references |
| `architect` | Design system architecture, APIs, data models | Architecture diagrams and specifications |
| `implementer` | Generate production-ready code with tests | Code files with test coverage |
| `reviewer` | Review code for bugs, security, style issues | Findings with CWE IDs and severity |
| `test-designer` | Design comprehensive test suites | Test plan with coverage goals |
| `shipper` | Generate release notes and deployment guides | Release documentation |
| `red-team` | Security threat modeling and attack vectors | Threat model with mitigations |

Each subagent enforces structured output via JSON Schema, ensuring consistent, parseable results.

## How It Works

```
USER TASK
    |
    v
[PARALLEL DRAFTS]
    |
    +-- Provider A (e.g., Claude Opus)      --> Draft A
    +-- Provider B (e.g., GPT-4)            --> Draft B
    +-- Provider C (e.g., Gemini Pro)       --> Draft C
    |
    v
[ADVERSARIAL CRITIQUE]
    |
    +-- Critic Model analyzes all drafts
    +-- Identifies weaknesses, contradictions, blind spots
    |
    v
[SYNTHESIS]
    |
    +-- Merge best elements from drafts
    +-- Address critique points
    +-- Validate against JSON Schema
    +-- Retry on validation failure (max 3 attempts)
    |
    v
[STRUCTURED OUTPUT]
    |
    +-- Success: JSON matching schema
    +-- Failure: Validation errors + partial output
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
    default_model: anthropic/claude-3.5-sonnet

  - name: anthropic
    api_key: ${ANTHROPIC_API_KEY}
    default_model: claude-3-opus-20240229

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
│   ├── google.py        # Google Gemini adapter
│   └── cli/             # CLI-based providers
├── subagents/           # Subagent prompt + schema definitions
├── schemas/             # JSON Schema definitions
├── validation/          # Schema validation utilities
└── cli/                 # Typer-based CLI
```

## Next Steps

- [OpenRouter Quickstart](quickstart/openrouter.md) - Recommended setup with single API key
- [Direct APIs Quickstart](quickstart/direct-apis.md) - Using Anthropic, OpenAI, Google directly
- [Creating Custom Providers](providers/creating-providers.md) - Build your own provider adapters

## Support

- **GitHub**: [sherifkozman/llm-council](https://github.com/sherifkozman/llm-council)
- **Issues**: [Report bugs or request features](https://github.com/sherifkozman/llm-council/issues)
- **License**: MIT
