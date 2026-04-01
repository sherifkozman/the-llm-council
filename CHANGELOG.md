# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.9] - 2026-04-01

### Changed
- Refreshed the packaged README, docs, Claude plugin metadata, and council skill to the `0.7.9` release surface

### Fixed
- Codex CLI doctor now preserves the ambient Codex runtime environment and reports real login status instead of false healthy results from an over-sanitized subprocess env
- Codex CLI generation now streams JSONL events incrementally and can return as soon as a completed answer arrives, instead of waiting only on subprocess shutdown
- Nested Codex runs now fail fast after `turn.started` if no answer ever arrives, which avoids burning the entire request timeout on known stalled subprocesses

## [0.7.8] - 2026-04-01

### Changed
- Refreshed the packaged README, docs, Claude plugin metadata, and council skill to the `0.7.8` release surface

### Fixed
- Codex CLI subprocess runs now execute under an isolated temporary `HOME` with only Codex auth files copied in, so nested council runs do not inherit the parent Codex agent's MCP tools, plugins, or skills
- Codex CLI parsing now consumes `--json` output for agent messages and usage metadata, and it preserves stdout-only JSONL error payloads instead of assuming meaningful failures always land on stderr
- Codex structured-output requests now normalize nested object schemas to the stricter validation contract enforced by the current Codex CLI
- Orchestrator regression tests now match the bounded runtime timeout caps and single-retry budget introduced in `e0e7320`

## [0.7.7] - 2026-03-31

### Fixed
- Increased bounded runtime timeout caps across all providers to prevent degradation from provider lock contention and API latency
- Bounded retry budget raised from 0 to 1 so transient timeouts get one retry instead of immediate skip
- Added `openrouter` to bounded timeout caps (was falling through to the minimal default)
- Added `codex-cli` and `claude-code` aliases to bounded timeout caps
- Fixed pre-existing line-length lint violations in orchestrator

## [0.7.6] - 2026-03-31

### Fixed
- Anthropic provider now forces `temperature=1` when extended thinking is enabled
- Anthropic provider now ensures `max_tokens` exceeds `budget_tokens` for extended thinking

## [0.7.5] - 2026-03-28

### Changed
- tightened provider/model wording in the README and direct-API quickstart so API providers, model families, and CLI providers are named explicitly
- reframed CLI provider docs around local CLI tooling and CLI-managed auth instead of implying those flows are inherently offline
- refreshed the direct-API pricing table to current March 2026 provider pricing and updated the Gemini fast-model reference to `Gemini 3 Flash Preview`

## [0.7.4] - 2026-03-28

### Changed
- renamed the direct Google Gemini API surface to `gemini` and the Gemini CLI surface to `gemini-cli`, while keeping `google` as a compatibility alias
- refreshed packaged docs, skills, and Claude plugin metadata to the current provider naming and `0.7.4` package version
- Vertex Gemini now defaults to the `global` location instead of the stale `us-central1` default
- large bounded review runs now expose preflight timing, prompt metrics, provider-specific timeout caps, and provider queue wait data

### Fixed
- Claude Code CLI parsing now handles list-style JSON envelopes from the CLI instead of assuming a single dict payload
- Codex CLI handling now follows the current headless contract and better preserves real stderr causes when calls fail
- Gemini CLI auth now respects the CLI's configured auth mode and avoids overriding it from council
- same-provider council calls are throttled across concurrent processes to reduce timeout cascades during overlapping doctor and review runs

### Known Issues
- `gemini-cli` deep doctor now passes on the local Vertex-authenticated setup, but review-mode smoke remains unstable and can overrun timeout budgets or return empty non-JSON output under council orchestration

## [0.7.2] - 2026-03-28

### Changed
- Codex CLI provider now defaults to `gpt-5.4` so ChatGPT-authenticated local Codex installs work without extra model config
- bounded runtime gives Codex longer per-phase request caps for planner-style council runs that exceed the generic 15s draft timeout
- Codex quickstart docs now note the compatibility default and when to opt into `*-codex` variants explicitly

### Fixed
- Codex CLI provider now rewrites incompatible `*-codex` model names when local login status shows `Logged in using ChatGPT`
- Codex CLI error reporting now preserves the meaningful trailing `ERROR:` lines instead of truncating the banner and hiding the real cause
- provider error classification no longer treats generic `invalid_request_error` responses as auth failures by default

## [0.7.1] - 2026-03-27

### Changed
- aligned public docs and shipped skills to the canonical `drafter` / `critic` / `planner` surface
- added canonical repo-shipped subagent docs for `drafter`, `critic`, and `synthesizer`
- refreshed quickstart examples to use current provider names and mode-aware commands

## [0.7.0] - 2026-03-27

### Added
- Mode-aware runtime wiring across `Council` and `Orchestrator`
  - runtime `mode`, `runtime_profile`, `reasoning_profile`, `temperature`, `max_tokens`, and output-schema overrides now flow into execution
  - execution plans now expose effective mode, schema, model pack, providers, phase budgets, and degradation details
- Routed handoff support
  - `council run router ... --route` can now continue into the router-selected subagent and mode
  - routed runs can carry model-pack, execution-profile, budget-class, and capability recommendations into the follow-up run
- Capability-aware execution baseline
  - capability planning and execution-profile selection added to the runtime
  - initial packs include `repo-analysis`, `docs-research`, `planning-assess`, `diff-review`, `security-code-audit`, and `red-team-recon`
- Deterministic evaluation tooling
  - new `council eval`, `council eval-compare`, and `council eval-import-pr` commands
  - public runtime baseline dataset under `evals/runtime-baseline.yaml`
- Deep doctor support
  - `council doctor --deep` now distinguishes installed/configured providers from providers that can answer a trivial non-interactive prompt

### Changed
- Canonical CLI provider names are now `codex`, `gemini`, and `claude`, with legacy aliases preserved for compatibility
- Public docs now describe the current runtime surface, local-only eval import boundary, and routed execution behavior
- Bounded runtime request caps were recalibrated to better match real review workloads

### Fixed
- Explicit user-selected providers and configured models now take precedence over subagent defaults
- Auto-fallback now stays limited to the default unhealthy path instead of overriding explicit user provider choices
- CLI provider timeout settlement now kills the full subprocess tree instead of hanging on timed-out children
- Gemini CLI approval-mode handling updated to current CLI behavior
- OpenAI timeout and provider error reporting now preserve the underlying cause chain for easier diagnosis
- Bounded review prompts are slimmer and no longer force schema-heavy draft/critique instructions

## [0.6.4] - 2026-03-26

### Added
- `claude-code` provider — wraps Claude Code CLI for agent-to-agent delegation (#37)
  - Uses `--bare` mode to prevent recursion when council runs inside Claude Code
  - Parses structured JSON output from `--output-format json`
  - `claude` alias resolves to `claude-code` in provider registry
- CLI providers are no longer deprecated — useful for agent delegation workflows

### Fixed
- Import order (E402) in `codex.py` and `gemini.py` CLI providers
- Removed deprecation warnings from `codex-cli` and `gemini-cli` providers

## [0.6.3] - 2026-03-26

### Added
- `--files/-f` is now repeatable: `-f file1.py -f file2.py` works alongside comma-separated `--files a.py,b.py`
- `vertex` accepted as alias for `vertex-ai` in provider registry (#38)

### Changed
- File context limits increased: 50KB per file (was 20KB), 200KB total (was 60KB)

## [0.6.2] - 2026-03-18

### Added
- `--files/-f` flag on `council run` — comma-separated file paths injected as context (20KB/file, 60KB total)
- Documented all v0.6.0+ CLI flags in README (--timeout, --temperature, --files, --context, etc.)

### Fixed
- `council version` now reads from `importlib.metadata` instead of a hardcoded string (#35)
- `council doctor` no longer emits `RuntimeError: Event loop is closed` — health checks run in parallel via `asyncio.gather`
- Remaining stale `gpt-5.2` references updated to `gpt-5.4` in docstrings and codex CLI provider
- ASCII art banner in README updated from GPT5.2 to GPT5.4

## [0.6.1] - 2026-03-17

### Changed
- Updated default models to March 2026 generation:
  - Anthropic: claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5
  - OpenAI: gpt-5.4, gpt-5.4-codex, gpt-5.4-mini
  - Vertex AI: claude-opus-4-6@20260301, claude-sonnet-4-6@20260301
- Updated skills/council/SKILL.md to v0.6.0 subagent names and modes
- Refreshed all documentation examples with current model names

## [0.6.0] - 2026-03-14

### Fixed
- **Config wiring** (#26): `providers[].default_model` in config.yaml is now forwarded to provider constructors via the registry
- **Doctor output** (#28): `council doctor` shows configured models instead of hardcoded defaults
- **Gemini model name** (#30): Updated default from `gemini-3-pro-preview` to `gemini-3.1-pro-preview`
- **Context injection** (#31): `--context`/`--system` content now reaches all three prompt phases (draft, critique, synthesis)

### Added
- **Per-provider env var model override** (#27): `OPENAI_MODEL`, `ANTHROPIC_MODEL`, `GOOGLE_MODEL`, `OPENROUTER_MODEL` — consistent fallback across all providers
- **Config output_format** (#29): Set `output_format: json` in config defaults to avoid passing `--json` on every invocation
- **Provider configs in CouncilConfig**: New `provider_configs` field for per-provider constructor kwargs via Python API

### Changed
- **Prompt quality improvements**: Router examples updated to v0.5.0 agent names; synthesizer prompt generalized beyond release notes; planner deduplicated; drafter/critic gain schema awareness and anti-hallucination instructions
- **Router prompt streamlined**: Removed heavyweight Council Deliberation Protocol from fast classifier

### Security
- **Context injection hardened**: User-provided `--context` content wrapped in `<reference_material>` XML delimiters and framed as reference data, not instructions — mitigates prompt injection

## [0.5.3] - 2026-01-03

### Fixed
- **Provider Token Handling**: Fixed validation error when LLM APIs return `None` for token counts (#22)
  - Applied defensive `or 0` fallback to all providers (google, vertex, anthropic, openai, openrouter)
  - Prevents Pydantic validation errors with `GenerateResponse.usage` field

## [0.5.2] - 2026-01-02

### Added
- **CLI Flags Enhancement**: 21 new CLI flags for improved usability and scripting

  **Global flags:**
  - `--version / -V` - Show version and exit
  - `--quiet / -q` - Suppress non-essential output
  - `--debug` - Enable debug logging
  - `--config / -c` - Custom config file path
  - `--no-color` - Disable colored output

  **`council run` flags:**
  - `--timeout / -t` - Request timeout in seconds
  - `--temperature` - Model temperature (0.0-2.0)
  - `--max-tokens` - Max output tokens
  - `--input / -i` - Read task from file (or `-` for stdin)
  - `--output / -o` - Write output to file
  - `--dry-run` - Show what would run without executing
  - `--context / --system` - System prompt injection
  - `--schema` - Custom output schema file

  **`council doctor` flags:**
  - `--json` - Output as JSON for scripting
  - `--provider` - Check specific provider only

  **`council config` flags:**
  - `--path` - Show config file path
  - `--validate` - Validate configuration file
  - `--get KEY` - Get config value by key (dot notation)
  - `--set KEY VALUE` - Set config value
  - `--edit` - Open config in $EDITOR

- **CouncilConfig fields**: Added `temperature`, `max_tokens`, `system_context`, `output_schema` fields

### Security
- Path validation for `--config` flag (prevents path traversal)
- Task length limit (100KB max) to prevent resource exhaustion
- Editor command validation for `--edit` flag

## [0.5.1] - 2026-01-02

### Changed
- **Gemini model upgrade**: Default model updated to `gemini-3-pro-preview` across all providers (Google, Vertex AI)
- **Router schema updated**: Now returns v0.5.0 agent names (`drafter`, `critic`) with `mode` field

### Added
- **Model pack expansion**: Added `CODE_COMPLEX` and `GROUNDED` packs to `ModelPack` enum
- **YAML-to-enum mapping**: New `YAML_TO_MODEL_PACK` dict and `resolve_model_pack()` function
  - Maps YAML strings (`fast_generator`, `deep_reasoner`, etc.) to `ModelPack` enum values
- **Subagent model pack resolution**: New functions in `subagents/__init__.py`:
  - `get_model_pack(config, mode)` - Resolve ModelPack from subagent config
  - `get_model_for_subagent(config, mode)` - Get OpenRouter model ID for subagent
  - `get_mode_config(config, mode)` - Get mode-specific configuration
- **New tests**: 7 new tests for model pack resolution in `test_multi_model.py`

### Deprecated
- **CLI providers deprecated**: `codex-cli` and `gemini-cli` providers now emit `DeprecationWarning` on instantiation
  - Use `openai` provider instead of `codex-cli`
  - Use `google` provider instead of `gemini-cli`
  - CLI providers will be removed in v1.0

## [0.5.0] - 2025-12-25

### Changed
- **Agent Consolidation: 10 → 6 Core Agents** (Council V2)
  - **drafter**: Unified generation agent with modes (`--mode impl|arch|test`)
    - Replaces: `implementer`, `architect`, `test-designer`
  - **critic**: Unified evaluation agent with modes (`--mode review|security`)
    - Replaces: `reviewer`, `red-team`
  - **planner**: Decision and planning agent with modes (`--mode plan|assess`)
    - Replaces: `assessor` (planner already existed)
  - **synthesizer**: Final output synthesis (replaces `shipper`)
  - **researcher**: Unchanged - technical research
  - **router**: Unchanged - task classification

- **CLI Enhancements**
  - New `--mode` flag for agent mode selection
  - Backwards-compatible alias resolution (old names still work with deprecation warning)
  - Simplified CLI by moving `--timeout`, `--max-retries`, `--no-degradation`, `--health-check` to config file
  - Config-based defaults via `~/.config/llm-council/config.yaml`

- **Enhanced Prompt Protocol**
  - All subagent YAML files now include "Council Deliberation Protocol" instructions
  - Promotes cross-model synthesis awareness and critique responsiveness
  - Improved output quality through multi-perspective reasoning

### Added
- **Registry Module** (`src/llm_council/registry/`)
  - `ToolRegistry`: Declarative tool registration for agent tool use
  - `BaseAgent`: Thin base class for building custom agents
  - Foundation for future MCP integration and tool orchestration

### Deprecated
- **Legacy agent names** (will be removed in v1.0):
  - `implementer` → use `drafter --mode impl`
  - `architect` → use `drafter --mode arch`
  - `test-designer` → use `drafter --mode test`
  - `reviewer` → use `critic --mode review`
  - `red-team` → use `critic --mode security`
  - `assessor` → use `planner --mode assess`
  - `shipper` → use `synthesizer`

## [0.4.13] - 2025-12-24

### Fixed
- **Fixed researcher schema for cross-provider compatibility**
  - `performance_benchmarks` now uses array of objects instead of `additionalProperties`
  - OpenAI structured outputs require explicit `properties` - cannot use empty properties with `additionalProperties`
  - Removed unsupported `"format": "uri"` from URL field (OpenAI only supports: date-time, time, date, duration, email, hostname, ipv4, ipv6, uuid)
  - Schema now works with OpenAI, Gemini, Anthropic, and all other providers
  - Tested against real providers to confirm fix

## [0.4.12] - 2025-12-24

### Fixed
- **CI lint and type check fixes**
  - Fixed ruff import sorting in `vertex.py` (I001)
  - Fixed ruff formatting in `main.py`
  - Fixed mypy type annotation in `_load_config_defaults()`

## [0.4.11] - 2025-12-24

### Changed
- **Increased max timeout from 600s to 900s (15 min)**
  - Complex subagents like `red-team` can now use longer timeouts
  - Fixes validation error when using `--timeout 660` or higher

## [0.4.10] - 2025-12-24

### Fixed
- **CLI now reads `defaults.providers` from config file** (fixes #19)
  - Previously, `council run` always defaulted to `["openrouter"]` when `--providers` flag was not passed
  - Now reads `defaults.providers` from `~/.config/llm-council/config.yaml`
  - Fallback chain: CLI flag → config file → `["openrouter"]`
  - Updated default config template to include `providers` under `defaults` section
  - Example config:
    ```yaml
    defaults:
      providers:
        - openai
        - anthropic
    ```

## [0.4.9] - 2025-12-23

### Added
- **Vertex AI provider** - Enterprise access to Gemini and Claude via Google Cloud
  - **Gemini support**: Uses `google-genai` SDK (region: us-central1)
    - Env vars: `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `VERTEX_AI_MODEL`
  - **Claude support**: Uses `anthropic[vertex]` SDK (region: global)
    - Env vars: `ANTHROPIC_VERTEX_PROJECT_ID`, `CLOUD_ML_REGION`, `ANTHROPIC_MODEL`
  - Install: `pip install the-llm-council[vertex]` (includes both SDKs)
  - Automatic model routing based on model prefix (claude-* vs gemini-*)
  - Two authentication methods (equally supported):
    - **ADC**: `gcloud auth application-default login`
    - **Service Account**: `GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json`
  - Usage: `council run architect "Design a cache" --providers vertex-ai`
  - Live tested: doctor, generation, streaming, structured output for both Gemini and Claude

## [0.4.8] - 2025-12-23

### Added
- **Vertex AI provider (Gemini only)** - Initial Vertex AI support for Gemini models

## [0.4.7] - 2025-12-23

### Fixed
- **Complex subagents failing with "Empty synthesis response"** (fixes #17)
  - **Root cause**: OpenAI structured output returns empty text when JSON is truncated due to token limit
  - Complex schemas (architect, planner, red-team) need 3000-5000+ tokens, but default was 2000
  - When `max_tokens` is too low and `finish_reason: length`, OpenAI returns empty content
  - **Fix**: Increased default token limits:
    - `max_synthesis_tokens`: 2000 → 8000
    - `max_draft_tokens`: 2000 → 4000
    - `max_critique_tokens`: 1200 → 2000
  - All subagents now work with OpenAI and Google providers

## [0.4.6] - 2025-12-23

### Changed
- **Google provider: Migrated to new google-genai SDK**
  - Replaced deprecated `google-generativeai` package with new `google-genai>=1.0`
  - Complete rewrite of `GoogleProvider` using new SDK patterns:
    - `genai.Client(api_key=...)` instead of `genai.configure()`
    - `client.aio.models.generate_content()` for async generation
    - `config={}` parameter instead of `generation_config`
  - Default model updated to `gemini-2.0-flash`
  - Fixed streaming implementation for new SDK async pattern

## [0.4.5] - 2025-12-23

### Fixed
- **Google provider: "title" property incorrectly stripped from schemas** (Issue #18)
  - `_strip_schema_meta_fields()` was stripping `title` everywhere, including when it's a property name
  - Fix: Only strip `title` as a schema meta field, preserve it when inside `properties`
  - Example: `{"properties": {"title": {"type": "string"}}}` now correctly keeps the "title" property
  - Fixes red-team schema failing with "required[1]: property is not defined" error

## [0.4.4] - 2025-12-23

### Changed
- **CI: Bump GitHub Actions to Node 24 versions**
  - `actions/checkout`: v4 → v6
  - `actions/setup-python`: v5 → v6
  - `astral-sh/setup-uv`: v4 → v7
  - Improved CI performance with Node 24 runtime

## [0.4.3] - 2025-12-22

### Fixed
- **OpenAI/OpenRouter: Strict mode requires additionalProperties: false** (fixes #17)
  - OpenAI strict mode requires ALL object schemas to have `additionalProperties: false`
  - Updated `_make_schema_strict_compatible()` in both providers to:
    - Strip existing `additionalProperties` field from input schema
    - Add `additionalProperties: false` to all object types
    - Recursively process nested objects and arrays of objects
  - Prevents "additionalProperties is required" validation errors

### Documentation
- Added **Default Reasoning Tiers** table to README documenting v0.4.0 reasoning defaults per subagent

## [0.4.2] - 2025-12-22

### Fixed
- **Missing logger import in all providers** (fixes CI failure)
  - Added `import logging` and `logger = logging.getLogger(__name__)` to OpenAI, Google, and Anthropic providers
  - Reasoning warnings now log correctly without `NameError`

## [0.4.1] - 2025-12-22

### Fixed
- **Missing logger import in OpenAI and Google providers**
  - Partial fix - Anthropic provider also needed the logger import

## [0.4.0] - 2025-12-22

### Added
- **Extended reasoning config for all subagents** (implements #16)
  - **High reasoning** (architect, assessor, planner, reviewer): `effort: high`, `budget_tokens: 16384`, `thinking_level: high`
  - **Medium reasoning** (implementer, researcher): `effort: medium`, `budget_tokens: 8192`, `thinking_level: medium`
  - **No reasoning** (router, shipper, test-designer): `enabled: false` for fast tasks
  - All configs are optional with sensible defaults for quick installation

## [0.3.1] - 2025-12-22

### Fixed
- **OpenAI provider: GPT-5.x and o-series require max_completion_tokens** (fixes #13)
  - New models return 400 error with `max_tokens` parameter
  - Added `MAX_COMPLETION_TOKENS_PREFIXES` to detect affected models
  - Automatically uses `max_completion_tokens` for gpt-5.x and o-series

- **Google provider: Schema field 'minItems' not supported** (fixes #14)
  - Added `minItems`, `maxItems`, `uniqueItems` to stripped schema fields
  - Fixes failures with complex schemas (e.g., red-team) on Google provider

- **Google provider: Suppress ALTS gRPC warnings** (fixes #12)
  - Set `GRPC_VERBOSITY=ERROR` at module load to suppress noisy warnings
  - Warnings are harmless but cluttered output when running outside GCP

## [0.3.0] - 2025-12-22

### Added
- **Per-subagent provider, model, and reasoning budget configuration** (implements #11)
  - New YAML fields: `providers`, `models`, `reasoning` for fine-grained control
  - `providers.preferred`, `providers.fallback`, `providers.exclude` for provider selection
  - `models.<provider>` for per-provider model overrides
  - `reasoning.enabled`, `reasoning.effort`, `reasoning.budget_tokens`, `reasoning.thinking_level`
  - `ReasoningConfig` class in `providers/base.py` for provider-agnostic reasoning configuration
  - Pydantic validation models: `ProviderPreferences`, `ModelOverrides`, `ReasoningBudget`
  - Helper functions: `get_provider_preferences()`, `get_model_overrides()`, `get_reasoning_budget()`

- **Reasoning/thinking API support for all major providers**
  - OpenAI: `reasoning_effort` parameter for o-series models (o1, o3, o4-mini)
  - Anthropic: `thinking` block with `budget_tokens` for extended thinking
  - Google: `thinking_config` with `thinking_level` (Gemini 3.x) or `thinking_budget` (Gemini 2.5)
  - OpenRouter: Pass-through support for underlying provider reasoning APIs

### Changed
- **Updated default models to December 2025 releases**
  - OpenAI: `gpt-5.1` (was `gpt-4o`)
  - Anthropic: `claude-opus-4-5` (was `claude-3-5-sonnet-20241022`)
  - Google: `gemini-3-flash-preview` (was `gemini-2.0-flash-exp`)
  - OpenRouter model packs updated to match

- **red-team subagent now uses extended reasoning by default**
  - `reasoning.enabled: true` with `effort: high`, `budget_tokens: 32768`, `thinking_level: high`
  - Uses o3-mini for OpenAI, claude-opus-4-5 for Anthropic

### Fixed
- **Token budget validation and warnings** (post-review security fixes)
  - Added maximum limit on `budget_tokens` (1024-128000) to prevent cost explosion
  - Anthropic: Use beta API for extended thinking, log warning when budget clamped
  - Google: Log warning when `thinking_budget` capped at provider maximum (24576)
  - OpenAI: Default `reasoning_effort` to "medium" when enabled but not specified
  - OpenAI: Warn and fallback when "none" effort used with o-series models
  - OpenAI: Warn when reasoning requested for non-reasoning models

## [0.2.3] - 2025-12-22

### Fixed
- **Google provider: Strip all unsupported schema fields** (fixes #10)
  - Google's `protos.Schema` only accepts: `type`, `properties`, `required`, `items`, `enum`, `description`
  - Now strips: `title`, `additionalProperties`, `default`, `examples`, `minLength`, `maxLength`, `minimum`, `maximum`, `pattern`, `format`
  - Recursive stripping for nested schemas

## [0.2.2] - 2025-12-22

### Fixed
- **Schema compatibility issues with OpenAI strict mode and Google** (fixes #9)
  - OpenAI/OpenRouter: Transform schemas to make ALL properties required (strict mode requirement)
  - Google: Strip `$schema` meta field which Google's SDK doesn't accept
  - Anthropic: Strip `$schema` meta field for consistency
  - All transformations are recursive for nested object schemas

## [0.2.1] - 2025-12-22

### Added
- **Agent Skills plugin** for cross-platform IDE integration (Claude Code, OpenAI Codex, Cursor, VS Code)
  - `.claude-plugin/plugin.json` and `marketplace.json` for Claude Code marketplace
  - `skills/council/SKILL.md` with comprehensive configuration reference
  - 10 subagent reference files with JSON output examples (`skills/council/subagents/*.md`)
  - `/council` slash command (`commands/council.md`)
- `cleanup_stale_runs()` method in `ArtifactStore` to mark orphaned runs as timed out
- Stale run cleanup tests

### Security
- Added security notes to skill docs (API key handling, data sensitivity, skill integrity)
- Version pinning requirement (`>=0.2.0`) in skill prerequisites

### Documentation
- **Provider development guide** updated with comprehensive structured output documentation
  - `StructuredOutputConfig` class usage and examples
  - Provider-specific API format transformations (OpenAI, Anthropic, Google)
  - Model capability checking patterns with prefix-based matching
  - Complete implementation checklist for new providers

### Fixed
- **Structured output format for all providers** (fixes #8)
  - OpenAI/OpenRouter: Fixed `response_format` to use correct `json_schema` wrapper format
  - Anthropic: Added `output_format` support with beta header (`structured-outputs-2025-11-13`)
  - Google: Added `response_schema` support in `generation_config`
  - Added `StructuredOutputConfig` for provider-agnostic structured output configuration
  - Model-specific capability checks for each provider family
  - OpenAI: Added GPT-5.x family support (gpt-5.1, gpt-5.2, gpt-5.1-codex, gpt-5.2-codex)
  - Google: Added Gemini 2.x/3.x pro and flash model support, including Gemini 3 preview family

### Changed
- Updated ruff config to ignore pre-existing lint patterns (E402, F401, B904, SIM102, SIM105)
- Configured mypy to ignore optional provider packages (anthropic, openai, google)
- Applied ruff formatting across entire codebase

## [0.2.0] - 2025-12-20

### Added
- Multi-model council via OpenRouter (single API key for 100+ models)
- Direct API support for Anthropic, OpenAI, and Google providers
- 10 specialized subagents: router, planner, assessor, researcher, architect, implementer, reviewer, test-designer, shipper, red-team
- Adversarial critique and synthesis workflow
- Graceful degradation when providers fail
- Artifact storage with SQLite-backed ledger
- Tiered summarization (GIST, FINDINGS, ACTIONS, RATIONALE, AUDIT)
- CLI tooling: `council run`, `council doctor`, `council config`
- Trusted publishing to PyPI and TestPyPI via GitHub Actions

### Security
- Path traversal protection in artifact store
- Content-addressed deduplication

## [0.1.0] - 2025-12-19

### Added
- Initial release with core council functionality
- Basic provider adapters
- JSON schema validation for subagent outputs

[Unreleased]: https://github.com/sherifkozman/the-llm-council/compare/v0.7.9...HEAD
[0.7.9]: https://github.com/sherifkozman/the-llm-council/compare/v0.7.8...v0.7.9
[0.7.8]: https://github.com/sherifkozman/the-llm-council/compare/v0.7.7...v0.7.8
[0.7.2]: https://github.com/sherifkozman/the-llm-council/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/sherifkozman/the-llm-council/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/sherifkozman/the-llm-council/compare/v0.6.4...v0.7.0
[0.6.4]: https://github.com/sherifkozman/the-llm-council/compare/v0.6.3...v0.6.4
[0.6.3]: https://github.com/sherifkozman/the-llm-council/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/sherifkozman/the-llm-council/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/sherifkozman/the-llm-council/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/sherifkozman/the-llm-council/compare/v0.5.3...v0.6.0
[0.5.3]: https://github.com/sherifkozman/the-llm-council/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/sherifkozman/the-llm-council/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/sherifkozman/the-llm-council/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/sherifkozman/the-llm-council/compare/v0.4.13...v0.5.0
[0.4.13]: https://github.com/sherifkozman/the-llm-council/compare/v0.4.12...v0.4.13
[0.4.12]: https://github.com/sherifkozman/the-llm-council/compare/v0.4.11...v0.4.12
[0.4.11]: https://github.com/sherifkozman/the-llm-council/compare/v0.4.10...v0.4.11
[0.4.10]: https://github.com/sherifkozman/the-llm-council/compare/v0.4.9...v0.4.10
[0.4.9]: https://github.com/sherifkozman/the-llm-council/compare/v0.4.8...v0.4.9
[0.4.8]: https://github.com/sherifkozman/the-llm-council/compare/v0.4.7...v0.4.8
[0.4.7]: https://github.com/sherifkozman/the-llm-council/compare/v0.4.6...v0.4.7
[0.4.6]: https://github.com/sherifkozman/the-llm-council/compare/v0.4.5...v0.4.6
[0.4.5]: https://github.com/sherifkozman/the-llm-council/compare/v0.4.4...v0.4.5
[0.4.4]: https://github.com/sherifkozman/the-llm-council/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/sherifkozman/the-llm-council/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/sherifkozman/the-llm-council/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/sherifkozman/the-llm-council/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/sherifkozman/the-llm-council/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/sherifkozman/the-llm-council/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/sherifkozman/the-llm-council/compare/v0.2.3...v0.3.0
[0.2.3]: https://github.com/sherifkozman/the-llm-council/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/sherifkozman/the-llm-council/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/sherifkozman/the-llm-council/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/sherifkozman/the-llm-council/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sherifkozman/the-llm-council/releases/tag/v0.1.0
