# The LLM Council - Development Guide

## Project Overview

Multi-LLM Council Framework (v0.8.3 release target) that orchestrates multiple LLM backends for adversarial debate, cross-validation, structured decision-making, and provider-specific prompt-cache handling. Package name: `the-llm-council` on PyPI. See [0.8.3 release notes](docs/releases/0.8.3.md) for pending verification and publication status.

- **Python**: >=3.10 (tested 3.10, 3.11, 3.12)
- **Build**: hatchling
- **CLI**: typer + rich
- **Source**: `src/llm_council/`
- **Tests**: `tests/` (pytest + pytest-asyncio)
- **Linting**: ruff, mypy (strict)

## Quick Reference

```bash
council run <subagent> "<task>"       # Run a council task
council run drafter --mode impl "..." # With specific mode
council doctor                        # Check provider health
council config --show                 # View configuration
council version                       # Show version
```

## Architecture

Agent and CI callers must follow the
[portable invocation contract](skills/council/references/invocation-contract.md).
Use an explicit installed executable and CWD, wait for the existing process,
then read its full result file even on nonzero exit. Inspect `execution_status`,
provider errors, degradation and actual request-compilation metadata. Native
CLI compatibility and effort support are versioned; see the release notes.
Exact Codex 0.160.1 is admitted, not future versions or every model. Its tested
catalog omits the unchanged `gpt-5.4` default; explicitly select a supported model
through `--models` or provider configuration. Version-specific tool controls
preserve authentication, selected model and reasoning effort.
No caller should source host secrets to bypass its credential policy.

`success: true` and exit 0 can describe a degraded result, not a complete review.
Ingestion counts in `execution_plan.context_preparation.files` do not prove
delivery: inspect `context_preparation.coverage` and `phase_prompt_compaction`
under the execution plan, including profile 0. `evidence_compacted` excludes
schema-only omission. `submitted` distinguishes unsent candidates from adapter
dispatch; only submitted evidence cuts count toward compaction-based degradation.
Council's pipeline cuts are not defects in the reviewed source.
The added prepared-coverage prompt boundary applies only to the built-in reviewer,
not security/custom schemas. Schema-validation exhaustion fails while preserving
drafts, critique and errors; arbitrary prose no longer manufactures reviewer
findings. Invalid raw synthesis is artifact-only if storage succeeds. A synthesis
exception can still fall back to a schema-valid JSON draft as degraded output;
schema validity does not establish semantic truth. Public result schemas are unchanged;
the optional, non-abstract `resolve_model` adapter hook is additive.

Read executed `degradation_report.total_retries` separately from `planned_retries`,
and inspect `synthesis_attempts` (adapter starts, not queued requests), completed
cancellation evidence,
`execution_plan.artifact_occurrences` and `execution_plan.persistence`. Hard-killed
`running` ledger rows do not prove liveness; never automatically repair them or
restart paid runs. Runtime identity and `execution_plan.provider_attempts` must
distinguish selected CLI path/realpath/version, `requested_model`,
`adapter_resolved_model` and `adapter_reported_model`. Resolution can use the
instantiated adapter default; it is not explicit selection or provider confirmation.
Defaults resolve before capability policy without an environment re-read or
default change. OpenAI `gpt-5.4` schema forwarding is restored, but Council's
unchanged policy forwards reasoning controls only for o-series, not `gpt-5.4`.
The shared engine identity helper covers library callers too. `TOOL_LOG`
`council_execution` manifests retain allowlisted `started`/`settled` metadata;
the settled snapshot says `ledger_finalization: not_yet_attempted`, so read the
terminal ledger separately. A hard kill may preserve only the start snapshot;
learned native identity is not durable there until settlement is persisted.
No heartbeat or automatic repair is added.
Reject `provider:model` mappings while preserving official slash/colon model IDs.
Use the portable contract for approved exact-version uv tool reinstalls; isolated
test installs do not update the real executable or pin. Do not delete native
installations or change PATH to hide selection mismatches.

```
Council (facade) → Orchestrator → Provider Adapters
                        ↓
                  3-phase flow:
                  1. Parallel drafts (asyncio.gather)
                  2. Adversarial critique
                  3. Schema-validated synthesis (with retry)
```

### Source Layout

```
src/llm_council/
├── cli/main.py          # CLI entry point (typer app)
├── council.py           # Council facade class
├── config/models.py     # Model packs, env var config
├── engine/
│   ├── orchestrator.py  # Core 3-phase orchestration
│   ├── degradation.py   # Graceful degradation policy
│   └── health.py        # Provider health checks
├── protocol/types.py    # Pydantic types (CouncilConfig, etc.)
├── providers/
│   ├── base.py          # ProviderAdapter interface
│   ├── registry.py      # Provider entry point registry
│   ├── openrouter.py    # OpenRouter (recommended)
│   ├── anthropic.py     # Direct Anthropic API
│   ├── openai.py        # Direct OpenAI API
│   ├── google.py        # Direct Google API
│   ├── vertex.py        # Vertex AI (Gemini + Claude)
│   └── cli/             # CLI-based providers
│       ├── codex.py     # OpenAI Codex CLI
│       └── gemini.py    # Gemini CLI
├── schemas/             # JSON schemas per subagent
├── storage/
│   ├── artifacts.py     # Artifact store (aiosqlite)
│   └── summarize.py     # Tiered summarization
├── subagents/           # YAML configs per subagent
├── registry/            # Base agent, tool registry
└── validation/          # Output validation
```

## Subagents

### Core Agents (v0.5.0+)

| Subagent | Use For | Modes |
|----------|---------|-------|
| `drafter` | Generate code, designs, tests | `impl`, `arch`, `test` |
| `critic` | Code review, security analysis | `review`, `security` |
| `synthesizer` | Merge and finalize outputs | - |
| `researcher` | Technical/market research | - |
| `planner` | Roadmaps, build-vs-buy | `plan`, `assess` |
| `router` | Classify and route tasks | - |

### Deprecated Aliases (Backwards Compatible)

| Old Name | Maps To | Removed In |
|----------|---------|------------|
| `implementer` | `drafter --mode impl` | v1.0 |
| `architect` | `drafter --mode arch` | v1.0 |
| `test-designer` | `drafter --mode test` | v1.0 |
| `reviewer` | `critic --mode review` | v1.0 |
| `red-team` | `critic --mode security` | v1.0 |
| `assessor` | `planner --mode assess` | v1.0 |
| `shipper` | `synthesizer` | v1.0 |

### Subagent Configs

YAML files in `src/llm_council/subagents/` define per-agent:
- Mode-specific model packs and schemas
- Reasoning budgets (OpenAI effort, Anthropic budget_tokens, Gemini thinking_level)
- Provider preferences (preferred, fallback, exclude)
- Model overrides per provider
- System and mode-specific prompts

## CLI Reference

### `council run`

```bash
council run <subagent> "<task>" [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--mode` | Agent mode (impl/arch/test, review/security, plan/assess) |
| `--providers, -p` | Comma-separated provider list |
| `--models, -m` | Literal model IDs in provider order, or multiple OpenRouter IDs; not `provider:model` mappings |
| `--timeout, -t` | Request timeout in seconds |
| `--temperature` | Model temperature (0.0-2.0) |
| `--max-tokens` | Max output tokens |
| `--files, -f` | File paths as context (repeatable or comma-separated; 50,000 characters/file, 200,000 retained characters total) |
| `--input, -i` | Read task from file (use `-` for stdin) |
| `--output, -o` | Write output to file |
| `--context, --system` | Additional system context/instructions |
| `--schema` | Custom output schema JSON file |
| `--no-artifacts` | Disable artifact storage |
| `--format` | Output format: `json` or `markdown` (alias `md`). Renders to clean human-readable Markdown on stdout and to `--output` files. Precedence: `--format` > `--json` > config `output_format`; default keeps the rich panel |
| `--json` | Output structured JSON (shorthand for `--format json`) |
| `--verbose, -v` | Show detailed output with metrics |
| `--dry-run` | Show what would run without executing |

### `council doctor`

```bash
council doctor [--json] [--provider NAME]
```

### `council config`

```bash
council config --show          # Show current configuration
council config --init          # Create default config at ~/.config/llm-council/config.yaml
council config --path          # Show config file path
council config --validate      # Validate config file
council config --get KEY       # Get value by dot notation
council config --set KEY VALUE # Set value
council config --edit          # Open in $EDITOR
```

### Global Options

```bash
council --version/-V    # Show version
council --quiet/-q      # Suppress non-essential output
council --debug         # Enable debug logging
council --config/-c     # Custom config file path
council --no-color      # Disable colored output
```

## When to Use Council

**Use council for:**
- Feature implementation (`council run drafter --mode impl "..."`)
- Code review (`council run critic --mode review "..."`)
- Architecture design (`council run drafter --mode arch "..."`)
- Security analysis (`council run critic --mode security "..."`)
- Build vs buy decisions (`council run planner --mode assess "..."`)
- Task classification (`council run router "..."`)
- Multi-model debate (`council run drafter --models "model1,model2,model3" "..."`)

**Skip council for:**
- Quick file lookups or single-line fixes
- Simple questions with obvious answers
- Tasks that only need one model's opinion

## Provider Setup

### Registered Providers (7)

| Provider | Entry Point | Env Variable |
|----------|-------------|--------------|
| `openrouter` | HTTP API | `OPENROUTER_API_KEY` |
| `openai` | Native SDK | `OPENAI_API_KEY` |
| `anthropic` | Native SDK | `ANTHROPIC_API_KEY` |
| `google` | Native SDK | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| `vertex-ai` | Native SDK | `GOOGLE_CLOUD_PROJECT` / `ANTHROPIC_VERTEX_PROJECT_ID` |
| `claude-code` | CLI subprocess | Claude Code CLI (`claude`) |
| `codex-cli` | CLI subprocess | Codex CLI (`codex`) |
| `gemini-cli` | CLI subprocess | Gemini CLI (`gemini`) |

### OpenRouter (Recommended)

```bash
export OPENROUTER_API_KEY="your-key"
council doctor
```

### Multi-Model Council

Run multiple models in parallel via OpenRouter:

```bash
# Via CLI flag
council run drafter --models "anthropic/claude-opus-4-6,openai/gpt-5.4,google/gemini-3.1-pro-preview" "task"

# Via environment variable
export COUNCIL_MODELS="anthropic/claude-opus-4-6,openai/gpt-5.4,google/gemini-3.1-pro-preview"
council run drafter "task"
```

### Model Pack Overrides

```bash
export COUNCIL_MODEL_FAST="anthropic/claude-haiku-4-5"        # Quick tasks (router, synthesizer)
export COUNCIL_MODEL_REASONING="anthropic/claude-opus-4-6"    # Deep analysis (architect, planner)
export COUNCIL_MODEL_CODE="openai/gpt-5.4"                   # Code generation (implementer)
export COUNCIL_MODEL_CRITIC="anthropic/claude-sonnet-4-6"     # Adversarial critique
export COUNCIL_MODEL_GROUNDED="google/gemini-3.1-pro-preview" # Research tasks
export COUNCIL_MODEL_CODE_COMPLEX="anthropic/claude-opus-4-6" # Complex refactoring
```

### Vertex AI (Enterprise GCP)

**Gemini models:**
```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
council run drafter --providers vertex-ai "task"
```

**Claude models:**
```bash
gcloud auth application-default login
export ANTHROPIC_VERTEX_PROJECT_ID="your-project-id"
export CLOUD_ML_REGION="global"
export ANTHROPIC_MODEL="claude-opus-4-6@20260301"
council run drafter --providers vertex-ai "task"
```

## Python API

```python
from llm_council import Council
from llm_council.protocol.types import CouncilConfig

config = CouncilConfig(
    providers=["openrouter"],
    models=["anthropic/claude-opus-4-6", "openai/gpt-5.4"],
    mode="impl",
    timeout=120,
    max_retries=3,
    temperature=0.7,
    max_tokens=4000,
    system_context="Additional instructions...",
    enable_artifact_store=True,
    enable_health_check=False,
    enable_graceful_degradation=True,
)
council = Council(config=config)
result = await council.run(task="Build a login page", subagent="drafter")
print(result.output)      # Validated JSON output
print(result.execution_status) # Inspect diagnostics, not success alone
print(result.success)     # Legacy bool; also true for degraded output
print(result.duration_ms) # Execution time
print(result.cost_estimate.estimated_cost_usd)
```

## Config File

```yaml
# ~/.config/llm-council/config.yaml
providers:
  - name: openrouter
    default_model: anthropic/claude-opus-4-6

defaults:
  providers:
    - openrouter
  timeout: 120
  max_retries: 3
  summary_tier: actions
  enable_degradation: true
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Linting
ruff check src tests
ruff format --check src tests
vulture src tests
mypy src/llm_council

# Install all providers
pip install -e ".[all]"
```

## Known Issues

Check `gh issue list` (or the GitHub repo) for current issues. CI enforces strict
mypy, Ruff lint/format and tests on Python 3.10, 3.11 and 3.12. Release verification
must state the tested native CLI versions and actual caller/provider coverage;
unit tests and doctor reachability alone do not establish that coverage.

### Resolved (historical, v0.6.x)
- ~~Config wiring: `providers[].default_model` not passed to provider constructors~~ — Fixed in v0.6.0
- ~~Model defaults: hardcoded Gemini model name outdated~~ — Fixed in v0.6.0
- ~~File injection: `--files` via wrapper not reaching models~~ — Native `--files` flag added in v0.6.2
- ~~Config defaults: no `output_format` config option~~ — Fixed in v0.6.0
- ~~`council version` shows stale version~~ — Fixed in v0.6.2

### Recent additions (v0.7.x)
- Human-readable Markdown output via `council run --format markdown` (and the
  `output_format` config key); honored for both stdout and `--output` writes.

## Troubleshooting

```bash
council doctor                              # Check all providers
council doctor --provider openrouter        # Check specific provider
council run drafter "task" --verbose         # Verbose output with metrics
council run drafter "task" --dry-run         # See what would execute
council run drafter "task" --no-artifacts    # Skip artifact storage (faster)
council --debug run drafter "task"           # Debug logging to stderr
```
