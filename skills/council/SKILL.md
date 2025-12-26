---
name: council
description: Run multi-LLM council for adversarial debate and cross-validation. Orchestrates Claude, GPT-4, and Gemini for production-grade implementation, code review, architecture design, research, and security analysis.
---

# LLM Council Skill

Multi-model council: parallel drafts → adversarial critique → validated synthesis.

> **Prerequisite:** This skill requires the `the-llm-council` Python package to be installed. The skill provides IDE integration but the actual council runs via the installed CLI. If you see `command not found: council`, run `pip install the-llm-council` first.

## Setup

### 1. Install
```bash
pip install the-llm-council>=0.5.0

# With specific provider SDKs
pip install the-llm-council[anthropic,openai,google]
```

### 2. Configure API Keys

| Provider | Environment Variable | Notes |
|----------|---------------------|-------|
| OpenRouter | `OPENROUTER_API_KEY` | **Recommended** - single key for all models |
| OpenAI | `OPENAI_API_KEY` | Direct GPT access |
| Anthropic | `ANTHROPIC_API_KEY` | Direct Claude access |
| Google | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Direct Gemini access |

```bash
# Minimum setup (OpenRouter)
export OPENROUTER_API_KEY="your-key"
```

### 3. Verify
```bash
council doctor
```

## Usage

```bash
council run <subagent> "<task>" [options]
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--mode` | Agent mode (e.g., `impl`/`arch`/`test` for drafter) |
| `--json` | Output structured JSON |
| `--verbose, -v` | Verbose output |
| `--models, -m` | Comma-separated model IDs |
| `--providers, -p` | Comma-separated provider list |
| `--no-artifacts` | Disable artifact storage (faster) |

### Other Commands
```bash
council doctor    # Check provider health
council config    # Show current configuration
```

## Subagents (v0.5.0)

### Core Agents

| Subagent | Modes | Use For | Details |
|----------|-------|---------|---------|
| `drafter` | `impl`, `arch`, `test` | Code, architecture, tests | See below |
| `critic` | `review`, `security` | Code review, security audit | See below |
| `synthesizer` | - | Merge and finalize outputs | See `subagents/synthesizer.md` |
| `researcher` | - | Technical research | See `subagents/researcher.md` |
| `planner` | `plan`, `assess` | Roadmaps, decisions | See `subagents/planner.md` |
| `router` | - | Task classification | See `subagents/router.md` |

### Agent Modes

**drafter modes:**
- `--mode impl` - Feature implementation, bug fixes (default)
- `--mode arch` - System design, API schemas
- `--mode test` - Test suite design

**critic modes:**
- `--mode review` - Code review with CWE IDs (default)
- `--mode security` - Security threat analysis

**planner modes:**
- `--mode plan` - Execution roadmaps (default)
- `--mode assess` - Build vs buy decisions

### Deprecated Aliases (Backwards Compatible)

The following legacy agent names still work but will be removed in v1.0:

| Old Name | Use Instead | Removed In |
|----------|-------------|------------|
| `implementer` | `drafter --mode impl` | v1.0 |
| `architect` | `drafter --mode arch` | v1.0 |
| `test-designer` | `drafter --mode test` | v1.0 |
| `reviewer` | `critic --mode review` | v1.0 |
| `red-team` | `critic --mode security` | v1.0 |
| `assessor` | `planner --mode assess` | v1.0 |
| `shipper` | `synthesizer` | v1.0 |

## Multi-Model Configuration

Run multiple models in parallel for adversarial debate:

```bash
# Via CLI flag
council run drafter --mode arch "Design caching layer" \
  --models "anthropic/claude-3.5-sonnet,openai/gpt-4o,google/gemini-pro"

# Via environment variable
export COUNCIL_MODELS="anthropic/claude-3.5-sonnet,openai/gpt-4o,google/gemini-pro"
```

### Model Pack Overrides

Fine-tune which models handle specific task types:

```bash
export COUNCIL_MODEL_FAST="anthropic/claude-3-haiku"      # Quick tasks
export COUNCIL_MODEL_REASONING="anthropic/claude-3-opus"  # Deep analysis
export COUNCIL_MODEL_CODE="openai/gpt-4o"                 # Code generation
export COUNCIL_MODEL_CRITIC="anthropic/claude-3.5-sonnet" # Adversarial critique
```

## Config File

Optional YAML configuration:

```yaml
# ~/.config/llm-council/config.yaml
providers:
  - name: openrouter
    api_key: ${OPENROUTER_API_KEY}
    default_model: anthropic/claude-3-opus

defaults:
  providers:
    - openrouter
  timeout: 120
  max_retries: 3
  summary_tier: actions
```

## Python API

```python
from llm_council import Council
from llm_council.protocol.types import CouncilConfig

config = CouncilConfig(
    providers=["openrouter"],
    mode="impl"  # Optional: set agent mode
)
council = Council(config=config)
result = await council.run(
    task="Build a login page with OAuth",
    subagent="drafter"
)
print(result.output)
```

## When to Use

**Use council for:**
- Feature implementation requiring production quality
- Code review with security analysis (CWE IDs)
- Architecture design decisions
- Technical research informing decisions
- Build vs buy assessments
- Security threat modeling

**Skip council for:**
- Quick file lookups
- Single-line fixes
- Simple questions

## Examples

```bash
# Feature implementation (new v0.5.0 syntax)
council run drafter --mode impl "Add pagination to users API" --json

# Code review
council run critic --mode review "Review the authentication changes" --json

# Multi-model architecture design
council run drafter --mode arch "Design caching layer" \
  --models "anthropic/claude-3.5-sonnet,openai/gpt-4o" --json

# Security threat model
council run critic --mode security "Analyze auth system vulnerabilities" --json

# Build vs buy decision
council run planner --mode assess "Should we build or buy a payment system?" --json

# Legacy syntax (still works, shows deprecation warning)
council run implementer "Add pagination" --json
council run reviewer "Review changes" --json
```

## Security Notes

- **API Keys**: Never embed secrets in task descriptions or skill files. Use environment variables.
- **Data Sensitivity**: Avoid sending files containing secrets (`.env`, credentials) to the council. Context is sent to external LLM providers.
- **Skill Integrity**: Treat `SKILL.md` and `subagents/*.md` as configuration code. Keep under version control.

## Troubleshooting

```bash
# Check all providers
council doctor

# Verbose output for debugging
council run drafter --mode impl "task" --verbose

# Faster runs (skip artifact storage)
council run drafter "task" --no-artifacts
```
