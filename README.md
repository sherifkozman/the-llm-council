# The LLM Council

```
$ council run drafter --mode arch "Design a mass hallucination prevention system"

                    ╔══════════════════════════════════════════════════════════╗
                    ║             ⚖️  THE LLM COUNCIL CONVENES  ⚖️              ║
                    ╚══════════════════════════════════════════════════════════╝

      ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
      │  ┌───────────┐  │      │  ┌───────────┐  │      │  ┌───────────┐  │
      │  │ ╭───────╮ │  │      │  │ ╭───────╮ │  │      │  │ ╭───────╮ │  │
      │  │ │GPT5.4│ │  │      │  │ │ CLAUDE│ │  │      │  │ │GEMINI │ │  │
      │  │ ╰───────╯ │  │      │  │ ╰───────╯ │  │      │  │ ╰───────╯ │  │
      │  │   ◉ ◉     │  │      │  │   ◉ ◉     │  │      │  │   ◉ ◉     │  │
      │  │    ⌣      │  │      │  │    ▽      │  │      │  │    ○      │  │
      │  └───────────┘  │      │  └───────────┘  │      │  └───────────┘  │
      │    JUDGE #1     │      │    JUDGE #2     │      │    JUDGE #3     │
      └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
               │                        │                        │
               │ "I propose we use      │ "Actually, I must      │ "Interesting, but
               │  a vector database..." │  respectfully disagree" │  what about...?"
               │                        │                        │
               └────────────────────────┼────────────────────────┘
                                        ▼
                         ┌──────────────────────────────┐
                         │     🔥 ADVERSARIAL DEBATE 🔥   │
                         │                              │
                         │  GPT5.4: "Your approach has  │
                         │          a cold start issue" │
                         │                              │
                         │  CLAUDE: "Fair, but yours    │
                         │          doesn't scale"      │
                         │                              │
                         │  GEMINI: "Both valid. What   │
                         │          if we combine..."   │
                         └──────────────┬───────────────┘
                                        ▼
                         ┌──────────────────────────────┐
                         │      ✅ VERDICT REACHED ✅     │
                         │                              │
                         │   Synthesized best ideas     │
                         │   Schema-validated output    │
                         │   Confidence: 94%            │
                         └──────────────────────────────┘

[Council] Task completed in 45.2s | 3 judges | 2 debate rounds | Cost: $0.12
```

<p align="center">
  <img src="assets/council-hero.png" alt="The LLM Council - Multiple AI models debating as judges" width="800">
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OS: Cross-platform](https://img.shields.io/badge/OS-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](https://github.com/sherifkozman/the-llm-council)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

A multi-model orchestration package that runs adversarial council workflows across OpenAI, Anthropic, Google, Vertex, OpenRouter, and local CLIs.

This is not a Claude-only framework. Claude Code is one supported client and one supported provider path among several.

For tool-driven calls from Codex, Claude Code, Hermes or CI, use the
[portable invocation contract](skills/council/references/invocation-contract.md).
It covers installed-binary identity, timeouts, process completion, credential
boundaries, complete result files and execution-status handling. The shipped
skill and caller examples require Council 0.8.3 or later for the reporting fixes.
The [0.8.3 release notes](docs/releases/0.8.3.md) are a candidate record with
verification pending, not a publication or live-provider compatibility claim.

The native CLI allowlist covers Codex **0.149.1, 0.153.3 and 0.160.1**, and Claude Code
**2.1.288, 2.1.289, 2.1.290, 2.1.291 and 2.1.292**. Unknown versions fail
explicitly; this is not a future auto-update compatibility guarantee. Report the
chosen executable path and observed version from doctor failure diagnostics,
not an assumed shell binary. Do not edit package constants to bypass the check.
See the invocation contract and release notes for native versus synthetic coverage;
version acceptance alone does not prove live reachability or review quality.
The Codex default remains `gpt-5.4`, which is absent from the tested 0.160.1
catalog. That version requires an explicit supported model through `--models`
or provider configuration; contained native checks used `gpt-6.1-sol`. Exact
0.160.1 tool controls preserve authentication, selected model and reasoning effort.
This is not support for every model or future CLI release.

This release also includes a mode-aware execution path with runtime profiles,
routed handoff, capability planning, and deterministic eval tooling. Those
capabilities materially extend the package runtime and make the public surface
more explicit for planning, review, security, and research workflows.

The `0.8.0` release fixes Windows large-prompt handling in the CLI provider
adapters (stdin instead of argv), surfaces Codex `turn.failed` server
rejections instead of empty successes, restores Claude Code CLI auth on
subscription and Vertex machines, and isolates CLI subprocess working
directories from the caller's repo.

The `0.7.18` release adds provider-specific prompt-cache support. Cache request
controls are adapter-owned, cache telemetry is normalized when providers return
it, and Gemini/Vertex cached-content resources can be created, refreshed, and
cleaned up with explicit lifecycle metadata.

## Why Use a Council?

Single-model outputs have blind spots. By running multiple models in parallel and having them critique each other, the council:

- **Catches errors** that any single model might miss
- **Reduces hallucination** through cross-validation
- **Produces higher-quality outputs** via adversarial refinement
- **Validates structure** with JSON schema enforcement and retry logic

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Council** | Run Claude, GPT-5.4, and Gemini in parallel via OpenRouter or direct APIs |
| **Mode-Aware Runtime** | `drafter`, `critic`, and `planner` honor runtime modes and execution profiles |
| **Adversarial Critique** | Built-in critique phase identifies weaknesses and blind spots |
| **Schema Validation** | JSON schema validation with automatic retry for structured outputs |
| **Provider Agnostic** | Swap between OpenRouter, direct APIs, or CLI-based providers |
| **Prompt Cache Support** | Anthropic/OpenRouter controls, OpenAI telemetry, and Gemini/Vertex cached-content lifecycle handling |
| **Deep Doctor** | `council doctor --deep` checks real non-interactive generation readiness |
| **Graceful Degradation** | Automatic retry, fallback, and skip strategies for failures |
| **Artifact Store** | Persistent storage of drafts with tiered summarization |
| **Eval Tooling** | `eval`, `eval-compare`, and local-only `eval-import-pr` support reproducible checks |
| **Secret-Safe Logging** | Redaction pipeline prevents credential leakage |

## Requirements

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10, 3.11, or 3.12 |
| **OS** | macOS, Linux, Windows (native or WSL) |
| **Credentials** | At least one provider credential or authenticated local CLI (see below) |

### Supported Providers

| Provider | Environment Variable / Auth | Notes |
|----------|-----------------------------|-------|
| OpenRouter | `OPENROUTER_API_KEY` | **Recommended** - single key for all models |
| OpenAI | `OPENAI_API_KEY` | Direct OpenAI API access (GPT models) |
| Anthropic | `ANTHROPIC_API_KEY` | Direct Anthropic API access (Claude models) |
| Gemini API | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Direct Gemini API access |
| Vertex AI | `GOOGLE_CLOUD_PROJECT` or `ANTHROPIC_VERTEX_PROJECT_ID` + ADC | Enterprise GCP - Gemini + Claude |
| Claude Code | Local `claude` CLI login | CLI subprocess provider |
| Codex CLI | Local `codex` CLI login | CLI subprocess provider |
| Gemini CLI | Local `gemini` CLI login | CLI subprocess provider |

## Installation

```bash
pip install the-llm-council
```

With specific providers:

```bash
# OpenRouter (recommended - single API key for all models)
pip install the-llm-council

# Direct APIs
pip install the-llm-council[anthropic,openai,gemini]

# Vertex AI (Enterprise GCP)
pip install the-llm-council[vertex]

# All providers
pip install the-llm-council[all]

# Development
pip install the-llm-council[dev]
```

## Agent Skills and Plugins

The LLM Council is available as an **Agent Skill** following the open [Agent Skills](https://agentskills.io) standard. It works across OpenAI Codex, Claude Code, Cursor, VS Code, and other skill-compatible agents.

### OpenAI Codex

```bash
# Copy skills directory to Codex skills location
cp -r skills/council ~/.codex/skills/
```

### Claude Code

```bash
# Step 1: Add the repo as a marketplace
/plugin marketplace add sherifkozman/the-llm-council

# Step 2: Install the plugin
/plugin install llm-council@the-llm-council
```

Once installed, the `council` skill is auto-invoked when relevant, or use the `/council` command:

```
/council drafter --mode impl "Build a login page with OAuth"
```

### Other Agents (Cursor, VS Code, GitHub, etc.)

Copy the `skills/council/` directory to your agent's skills folder. The skill follows the open Agent Skills spec and works with any compatible agent.

## Quick Start

### CLI Usage

```bash
# Set your API key
export OPENROUTER_API_KEY="your-key"

# Run a council task (v0.7.x syntax with modes)
council run drafter --mode impl "Build a login page with OAuth"

# Multi-model council (Claude + GPT-5 + Gemini debating)
council run drafter --mode arch "Design a caching layer" \
  --models "anthropic/claude-opus-4-6,openai/gpt-5.4,google/gemini-3.1-pro-preview"

# Or set via environment variable
export COUNCIL_MODELS="anthropic/claude-opus-4-6,openai/gpt-5.4,google/gemini-3.1-pro-preview"
council run drafter "Build a login page"

# OpenRouter model IDs keep vendor namespaces like google/... .
# Those model IDs are separate from council provider names such as gemini or gemini-cli.

# Code review with security analysis
council run critic --mode review "Review auth changes" --verbose

# Ask the router to choose the next subagent/mode, then follow through
council run router "Assess whether we should add a hosted vector store" --route

# Bound latency/cost for review-style runs
council run critic --mode review "Review auth changes" \
  --runtime-profile bounded \
  --reasoning-profile off

# Disable artifact storage for faster runs
council run drafter "Quick fix" --no-artifacts

# Get structured JSON output
council run planner "Add user authentication" --json

# Legacy aliases still work, but prefer canonical subagents and modes
council run drafter --mode impl "Build a login page"
```

### Python API

```python
from llm_council import Council
from llm_council.protocol.types import CouncilConfig

# With mode configuration
config = CouncilConfig(providers=["openrouter"], mode="impl")
council = Council(config=config)
result = await council.run(
    task="Build a login page with OAuth",
    subagent="drafter"
)
print(result.output)
```

### Check Provider Health

```bash
council doctor

# Verify actual non-interactive generation readiness
# May incur API/CLI usage.
council doctor --deep --provider claude --provider gemini-cli --provider codex
```

The Codex CLI adapter runs nested `codex exec` calls under an isolated temporary
`HOME`, copying only the Codex auth files required for login. That keeps council
subprocesses from inheriting the parent Codex agent's MCP tools, plugins, or
skills while preserving your local Codex authentication. It also preserves the
ambient Codex runtime environment, so `council doctor` reflects real login
status instead of a false healthy result caused by over-stripped subprocess env.
For Codex, a base doctor success reports login status only, not generation
readiness. Require `probe_ok: true` from a deep probe for reachability; neither
result proves large-review success. Do not infer path/version details absent
from the diagnostic.

### Run Deterministic Evals

```bash
# Public deterministic runtime checks
council eval evals/runtime-baseline.yaml --providers openrouter

# Compare named variants on the same dataset
council eval-compare evals/runtime-baseline.yaml variants.yaml --providers openai
```

For private benchmark creation, import external PR review material into
`.council-private/` only:

```bash
council eval-import-pr owner/repo 123
```

That path is gitignored and intended for local-only evaluation inputs. Do not
commit imported diffs, copied code, or review fixtures into tracked repo paths.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           LLM Council                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │    CLI      │───▶│  Council    │───▶│     Orchestrator        │ │
│  │  (typer)    │    │   (API)     │    │                         │ │
│  └─────────────┘    └─────────────┘    │  ┌───────────────────┐  │ │
│                                        │  │  Health Checker   │  │ │
│  ┌─────────────────────────────────┐   │  ├───────────────────┤  │ │
│  │        Provider Registry        │◀──│  │ Degradation Policy│  │ │
│  │  ┌─────────┐ ┌─────────┐       │   │  ├───────────────────┤  │ │
│  │  │OpenRouter│ │Anthropic│ ...  │   │  │  Artifact Store   │  │ │
│  │  └─────────┘ └─────────┘       │   │  └───────────────────┘  │ │
│  └─────────────────────────────────┘   └─────────────────────────┘ │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Subagent Configs                          │   │
│  │  router | planner | researcher | drafter | critic | ...     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     JSON Schemas                             │   │
│  │  Validation & retry logic for structured outputs             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

```
0. HEALTH CHECK (optional)
   └── Preflight check of all providers, skip unhealthy ones

1. PARALLEL DRAFTS
   ├── Provider A generates draft
   ├── Provider B generates draft
   └── Provider C generates draft
   └── (Graceful degradation on failures)

2. ADVERSARIAL CRITIQUE
   └── Critic identifies weaknesses, contradictions, blind spots

3. SYNTHESIS
   └── Merge best elements, address critique, validate schema

4. VALIDATION
   └── JSON schema check with retry on failure

5. ARTIFACT STORAGE (optional)
   └── Store drafts and outputs for context management
```

Artifact storage defaults to `~/.council/` (`artifacts/` plus `ledger.db`).
You can override the root with `COUNCIL_HOME`, or set `COUNCIL_ARTIFACT_DIR`
and `COUNCIL_DB_PATH` explicitly. Existing legacy `~/.claude/` council stores
are still picked up automatically during migration.

Inspect or migrate storage explicitly:

```bash
council storage status
council storage migrate --dry-run
council storage migrate
```

## Subagents (v0.7.x)

### Core Agents

| Subagent | Modes | Purpose | Example |
|----------|-------|---------|---------|
| `drafter` | `impl`, `arch`, `test` | Generate code, designs, tests | "Build the login page" |
| `critic` | `review`, `security` | Review and analyze | "Review this PR for security" |
| `synthesizer` | - | Merge and finalize | "Generate changelog for v1.2" |
| `researcher` | - | Technical research | "Research OAuth providers" |
| `planner` | `plan`, `assess` | Roadmaps and decisions | "Plan the auth implementation" |
| `router` | - | Classify and route tasks | "Is this a bug or feature?" |

### Agent Modes

```bash
# drafter modes
council run drafter --mode impl "Build login page"     # Implementation (default)
council run drafter --mode arch "Design caching layer" # Architecture
council run drafter --mode test "Design test suite"    # Test design

# critic modes
council run critic --mode review "Review PR"           # Code review (default)
council run critic --mode security "Analyze auth"      # Security analysis

# planner modes
council run planner --mode plan "Plan implementation"  # Planning (default)
council run planner --mode assess "Redis vs Memcached" # Build vs buy
```

### Legacy Aliases

Legacy names such as `implementer`, `architect`, `reviewer`, `red-team`,
`assessor`, `test-designer`, and `shipper` still work for backwards
compatibility, but public docs and examples now use the canonical subagents and
modes.

## Writing a Provider

Providers are pluggable via Python entry points. See the full [Provider Development Guide](docs/providers/creating-providers.md) for detailed instructions.

### Quick Example

```python
from llm_council.providers.base import ProviderAdapter, GenerateRequest, GenerateResponse

class MyProvider(ProviderAdapter):
    name = "myprovider"

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        # Your implementation
        return GenerateResponse(text="...", content="...")

    async def doctor(self) -> DoctorResult:
        return DoctorResult(ok=True, message="Healthy")
```

Register via `pyproject.toml`:

```toml
[project.entry-points."llm_council.providers"]
myprovider = "my_package.providers:MyProvider"
```

### Reference Implementations

| Provider | Type | File |
|----------|------|------|
| OpenRouter | HTTP API | `src/llm_council/providers/openrouter.py` |
| Anthropic | Native SDK | `src/llm_council/providers/anthropic.py` |
| OpenAI | Native SDK | `src/llm_council/providers/openai.py` |
| Gemini API | Native SDK | `src/llm_council/providers/gemini.py` |
| Vertex AI | Native SDK | `src/llm_council/providers/vertex.py` |
| Claude Code | CLI subprocess | `src/llm_council/providers/cli/claude_code.py` |
| Codex CLI | CLI subprocess | `src/llm_council/providers/cli/codex.py` |
| Gemini CLI | CLI subprocess | `src/llm_council/providers/cli/gemini_cli.py` |

## Configuration

### Environment Variables

```bash
# OpenRouter (recommended - single key for all models)
export OPENROUTER_API_KEY="your-key"

# Direct APIs
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."

# Vertex AI - Gemini (Enterprise GCP)
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="global"  # optional, default for Gemini on Vertex
export VERTEX_AI_MODEL="gemini-3.1-pro-preview"  # optional

# Vertex AI - Claude (Enterprise GCP)
export ANTHROPIC_VERTEX_PROJECT_ID="your-project-id"
export CLOUD_ML_REGION="global"              # Claude uses global region
export ANTHROPIC_MODEL="claude-opus-4-6@20260301"  # model with version

# Auth for Vertex AI: gcloud auth application-default login OR
# export GOOGLE_APPLICATION_CREDENTIALS="/path/to/sa.json"

# Multi-model council: comma-separated OpenRouter model IDs
export COUNCIL_MODELS="anthropic/claude-opus-4-6,openai/gpt-5.4,google/gemini-3.1-pro-preview"

# Per-provider model override (v0.7.0+)
export OPENAI_MODEL="gpt-5.4"               # Override OpenAI default
export ANTHROPIC_MODEL="claude-opus-4-6"     # Override Anthropic default
export GEMINI_MODEL="gemini-3.1-pro-preview" # Override Gemini API default
export OPENROUTER_MODEL="anthropic/claude-opus-4-6"  # Override OpenRouter default

# Model pack overrides for specific task types
export COUNCIL_MODEL_FAST="anthropic/claude-haiku-4-5"        # Quick tasks
export COUNCIL_MODEL_REASONING="anthropic/claude-opus-4-6"    # Deep analysis
export COUNCIL_MODEL_CODE="openai/gpt-5.4"                   # Code generation
export COUNCIL_MODEL_CRITIC="anthropic/claude-sonnet-4-6"     # Adversarial critique
export COUNCIL_MODEL_GROUNDED="google/gemini-3.1-pro-preview" # Research tasks
export COUNCIL_MODEL_CODE_COMPLEX="anthropic/claude-opus-4-6" # Complex refactoring
```

### Per-Subagent Reasoning Configuration (v0.3.0+)

Subagents can be configured with provider preferences, model overrides, and extended reasoning/thinking budgets in their YAML configs:

```yaml
# src/llm_council/subagents/critic.yaml (security mode)
name: critic
model_pack: harsh_critic

# Provider preferences
providers:
  preferred: [anthropic, openai]
  fallback: [openrouter]
  exclude: [gemini]

# Model overrides per provider
models:
  anthropic: claude-opus-4-6
  openai: o3-mini
  gemini: gemini-3-pro

# Extended reasoning/thinking configuration
reasoning:
  enabled: true
  effort: high           # OpenAI o-series: low/medium/high
  budget_tokens: 32768   # Anthropic: 1024-128000
  thinking_level: high   # Google Gemini 3.x: minimal/low/medium/high
```

| Provider | Parameter | Values | Description |
|----------|-----------|--------|-------------|
| OpenAI | `effort` | low/medium/high | Reasoning effort for o-series models |
| Anthropic | `budget_tokens` | 1024-128000 | Extended thinking token budget |
| Gemini API | `thinking_level` | minimal/low/medium/high | Gemini 3.x thinking level |

#### Default Reasoning Tiers (v0.4.0+)

All subagents have pre-configured reasoning defaults based on task complexity:

| Tier | Subagents | Config | Use Case |
|------|-----------|--------|----------|
| **High** | drafter (arch), critic, planner | `effort: high`, `budget_tokens: 16384` | Deep analysis, critical decisions |
| **Medium** | drafter (impl), researcher | `effort: medium`, `budget_tokens: 8192` | Balanced code/research tasks |
| **Disabled** | router, synthesizer, drafter (test) | `enabled: false` | Fast tasks, no overhead |

### Config File

```yaml
# ~/.config/llm-council/config.yaml
providers:
  - name: openrouter
    default_model: anthropic/claude-opus-4-6
  - name: openai
    default_model: gpt-5.4
  - name: gemini
    default_model: gemini-3.1-pro-preview

defaults:
  providers:
    - openrouter
  timeout: 120
  max_retries: 3
  summary_tier: actions
  output_format: json  # or "rich" (default) — useful for non-interactive clients
```

Provider `default_model` is forwarded to each provider's constructor, overriding the hardcoded default. Per-provider env vars (e.g. `OPENAI_MODEL`) take precedence over the config file, and the `--models` CLI flag overrides both.

## CLI Reference

```bash
council run <subagent> "<task>"    # Run a council task
council doctor                      # Check provider health
council config                      # Show configuration
council version                     # Show installed version

# Options
--mode             Agent mode (impl/arch/test for drafter, review/security for critic, etc.)
--providers, -p    Comma-separated provider list
--models, -m       Comma-separated OpenRouter model IDs for multi-model council
--files, -f        File paths as context (repeatable or comma-separated; 50,000 chars/file, 200,000 total)
--context, --system  Additional system context/instructions
--timeout, -t      Per-attempt timeout in seconds (10-3600; default 120)
--temperature      Model temperature (0.0-2.0)
--max-tokens       Max output tokens
--input, -i        Read task from file (use '-' for stdin)
--output, -o       Write output to file
--schema           Custom output schema JSON file
--dry-run          Show what would run without executing
--no-artifacts     Disable artifact storage
--json             Output structured JSON
--verbose, -v      Verbose output
```

### File Context

Pass source files directly to council for review, implementation, or analysis:

```bash
# Comma-separated
council run critic --mode review --files src/auth.py,src/middleware.py "Review these files"

# Repeatable -f (cleaner for many files)
council run critic --mode review \
  -f src/auth.py \
  -f src/middleware.py \
  -f src/handler.py \
  "Review these files"

# Security audit
council run critic --mode security -f src/payment.py "Audit payment handler"
```

Limits are **50,000 characters per file** and **200,000 retained characters in
total**, not KB or UTF-8 bytes. A file above the per-file limit is truncated with
a warning. If the retained total exceeds the total limit, the run fails before
provider calls; it does not silently skip the excess file.

Inspect `execution_plan.context_preparation.files` for `path`, `original_chars`,
`retained_chars`, and `truncated`. Truncation is also reported in
`degradation_report.context_warnings` with `kind: "file_truncated"`, even when
graceful degradation is disabled. A successful run with truncated input has
`execution_status: "degraded"`. Do not describe it as a full-file review. Review
the omitted material separately or narrow the input and state the actual scope.

### Execution And Evidence Reporting

Use `execution_status`, not `success` alone, in JSON, Markdown and console output.
A valid degraded result can have `success: true` and exit 0. A `completed` result
does not prove substantive independent reviews or correct findings. Inspect
`drafts`, `provider_errors`, `degradation_report`, `execution_plan` and `run_id`.

For the built-in reviewer, critique and synthesis receive bounded prepared-coverage
metadata as data, not source evidence or proof of every-phase delivery. The review
prompt separates Council pipeline limitations from source findings; real defects
in pipeline code under review still require source evidence. This added prompt
boundary does not apply to security or custom schemas and adds no task-intent
detector or semantic regex filter. Public result schemas are unchanged; the
adapter model-resolution hook described below is additive.

Schema-validation exhaustion returns `failed`, preserving drafts, critique and
validation errors. Council no longer manufactures reviewer findings from arbitrary
draft/critique prose by inferring severity, category, location or remediation.
Invalid raw synthesis is retained only as an artifact when storage succeeds,
not promoted to valid output. A synthesis exception can still use an existing
schema-valid JSON draft as a degraded fallback. Schema validity is not semantic
truth; inspect the findings and diagnostics rather than treating fallback as approval.

File ingestion is not delivered coverage. The `context_preparation.files`
counts above describe files read, before source selection and downstream prompt
compaction. Inspect `execution_plan.context_preparation.coverage` and
`execution_plan.phase_prompt_compaction` for delivered source and draft evidence.
Even profile index 0 can cut content. `truncated: false` at ingestion does not
establish full delivery to each phase. Council slicing, cut draft text and schema
omission are pipeline limitations, not defects in the source under review or
proof that a provider generated incomplete text.
Compaction metadata includes `original_prompt_chars` and `delivered_prompt_chars`;
`evidence_compacted` excludes schema-only omission. `submitted` distinguishes
unsent candidates from adapter dispatch; only submitted evidence cuts count
toward compaction-based degradation.

`degradation_report.total_retries` counts extra calls actually started;
`planned_retries` separately records decisions, not executed work or retry budgets.
Inspect `synthesis_attempts` even when the final output comes from a draft fallback;
it counts actual adapter starts, not queued requests that never start.
An empty `fallbacks_used` list alone cannot rule out final-output fallback.
Handled cancellation retains completed evidence where available, not unfinished
work. Inspect `execution_plan.artifact_occurrences` for each phase/provider's
artifact reference and `execution_plan.persistence` for storage/finalization
errors. Identical text does not make two phase occurrences the same evidence.
Disabled persistence is not a storage failure; absent artifacts are not proof
that nothing executed.

With persistence enabled, `TOOL_LOG` manifests of type `council_execution` capture
`started` and `settled` snapshots of allowlisted runtime, attempt, occurrence,
coverage and persistence-error metadata. The settled snapshot records
`ledger_finalization: not_yet_attempted`; read the terminal ledger separately.
A hard kill may retain only the start snapshot; learned native identity is not
durable in these manifests until settlement is persisted. No heartbeat or
automatic repair is added.

A hard kill can leave a ledger row marked `running`. That row does not prove
process liveness, and a stopped caller does not prove every descendant exited.
Check the original process handle and caller termination evidence. Do not
automatically repair ledger rows or restart paid runs.

`execution_plan.runtime_identity` identifies the Council entrypoint, imported
package and Python runtime for CLI and library callers through a shared engine
helper. Native diagnostics must distinguish the selected
executable and observed version from the requested model. A requested model is
not a provider-confirmed model identity. Inspect `execution_plan.provider_attempts`
for `cli_path`, `cli_realpath`, `cli_version`, `requested_model`,
`adapter_resolved_model` and `adapter_reported_model` where available. The resolved
model can come from the adapter's default, not an explicit caller choice; neither
resolution nor adapter reporting is independent provider confirmation. Do not infer historical executable
selection from the current shell, or fill in identity fields absent from a report.

Before applying capability policy, the engine resolves an omitted model from
the instantiated adapter's default. The optional, non-abstract `resolve_model`
hook is backward compatible: it does not re-read environment variables, change
defaults or override an explicit model; unknown defaults remain unknown.
This restores schema forwarding for the OpenAI adapter's `gpt-5.4` default.
It does **not** restore or enable `gpt-5.4` reasoning controls: Council's existing
OpenAI capability policy forwards those controls only for o-series models.
That policy and model defaults are unchanged. Do not claim that high reasoning
reached `gpt-5.4`; inspect request-compilation metadata. O-series reasoning
forwarding is covered separately by regression checks.

`--models` takes literal model IDs in provider order, not provider-prefixed
`provider:model` mappings: `codex:gpt-5.4` and `vertex-ai:gemini-3.1-pro-preview`
are invalid mappings. Preserve official IDs such as `openai/gpt-5.4` and
`qwen/qwen3.6-plus:free` for providers that accept them; slash namespaces and
colon suffixes are not a reason to rewrite a model ID.

### Updating A Pinned uv Tool

After publication and approval to update the installed tool, preserve any local
package edits and the uv receipt, then replace the exact version constraint:

```bash
uv tool install --reinstall 'the-llm-council[all]==0.8.3'
```

This targets the uv-managed tool and its exposed executable, and updates the
stored pin. Installing into an isolated test environment does not update the
command used by real callers. `uv tool upgrade` respects the existing constraint;
an exact `==0.8.2` pin must be replaced explicitly. See
[uv tool upgrade rules](https://docs.astral.sh/uv/concepts/tools/#upgrading-tools).
Verify the resolved Council path, version, imported source and receipt from each
actual caller. Report native CLI selection mismatches without deleting native
installations or changing PATH. This instruction does not itself authorize an
installation, prove publication, or establish live-provider compatibility.

### Timeout Budgets

`--timeout` accepts 10 through 3600 seconds; its default remains 120. It limits
each provider attempt, not the whole workflow. Retries, multiple phases, queue
waits, backoff and cleanup can make the workflow longer. Set the caller's outer
deadline separately and wait for the original process to finish before reading
its result or considering another run.

For long-deadline work, use `--runtime-profile default`. The `bounded` profile
keeps its shorter provider/phase caps even with `--timeout 3600`. Inspect
`execution_plan.provider_request_timeouts_by_provider` for effective budgets.
A successful `doctor --deep` probe proves reachability only, not large-review
completion, full input coverage, or review quality.

## Development

```bash
# Clone the repository
git clone https://github.com/sherifkozman/the-llm-council.git
cd the-llm-council

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/
mypy src/llm_council
```

## Contributing

Contributions are welcome! See our [Roadmap](ROADMAP.md) for planned features and [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Start

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/the-llm-council.git
cd the-llm-council

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/ && mypy src/llm_council
```

### Contribution Workflow

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Test** your changes (`pytest`)
5. **Lint** your code (`ruff check src/ && mypy src/llm_council`)
6. **Commit** with a clear message (`git commit -m 'Add amazing feature'`)
7. **Push** to your branch (`git push origin feature/amazing-feature`)
8. **Open** a Pull Request

### What We're Looking For

- **New Providers**: Add support for more LLM backends
- **New Subagents**: Create specialized agents for specific tasks
- **Bug Fixes**: Found a bug? We'd love a fix!
- **Documentation**: Improvements to docs are always welcome
- **Tests**: More test coverage is great

## Security

For security concerns, please see our [Security Policy](SECURITY.md) or email vibecode@sherifkozman.com.

**Key security features:**
- CLI adapters use exec-style subprocess (no shell injection)
- Environment variable allowlisting prevents secret leakage
- Path traversal protection in artifact storage
- Configurable secret redaction in logs

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with:
- [Pydantic](https://docs.pydantic.dev/) - Data validation
- [Typer](https://typer.tiangolo.com/) - CLI framework
- [Rich](https://rich.readthedocs.io/) - Terminal formatting
- [httpx](https://www.python-httpx.org/) - Async HTTP client

---

<p align="center">
  <i>When one model isn't enough, convene a council.</i>
</p>

<p align="center">
  <sub>~ vibe coded by <a href="https://twitter.com/sherifkozman">Sherif Kozman</a> & The LLM Council ~</sub>
</p>
