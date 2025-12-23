# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/sherifkozman/the-llm-council/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/sherifkozman/the-llm-council/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/sherifkozman/the-llm-council/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sherifkozman/the-llm-council/releases/tag/v0.1.0
