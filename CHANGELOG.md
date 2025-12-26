# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/sherifkozman/the-llm-council/compare/v0.5.0...HEAD
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
