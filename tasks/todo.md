# Cross-Provider Prompt Caching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the proven Anthropic prompt-caching proof slice into a conservative cross-provider prompt-caching feature with provider-specific semantics, deterministic tests, live telemetry proof, and council review gates.

**Architecture:** Keep the public request surface additive and provider-neutral, but keep wire-format behavior adapter-owned. The compiler decides whether a provider can keep, drop, or transform `prompt_cache`; each adapter performs final request serialization and usage parsing. Do not pretend all providers support the same caching model.

**Tech Stack:** Python, Pydantic v2, pytest, pytest-asyncio, Anthropic SDK, OpenAI SDK, Google GenAI SDK, HTTPX/OpenRouter direct API, existing `council` reviewer.

---

## Current State

- [x] Branch is based on `origin/main` at `24e23b8`.
- [x] Anthropic proof slice implemented locally.
- [x] Anthropic proof slice locally verified with `PYTHONPATH=src uv run pytest tests/test_request_compiler.py tests/test_providers.py -q`.
- [x] Current local result: `114 passed`.
- [x] Anthropic live proof succeeded with identical large prompt:
  - first response: `cache_creation_input_tokens=10988`
  - second response: `cache_read_input_tokens=10988`
- [x] Council final focused review approved the Anthropic proof slice with no blocking issues.
- [x] Design spec exists at `docs/providers/2026-06-09-anthropic-prompt-caching-design.md`.

## Implementation Status Update

- [x] Shared `PromptCacheConfig` supports Anthropic-style `auto` and Gemini-style `cached_content` passthrough.
- [x] Compiler keeps or drops `prompt_cache` by provider, route, model family, and cache mode with explicit decisions.
- [x] Anthropic baseline is hardened for direct, beta, stream, write, and cache-hit-only paths.
- [x] OpenAI telemetry-only parsing is implemented without request mutation.
- [x] OpenRouter route-gated passthrough is implemented for `anthropic/` routes only.
- [x] Gemini cached-content lifecycle is implemented for create, TTL refresh, expiry recreation, and best-effort cleanup.
- [x] Vertex split paths are implemented: Claude uses Anthropic-style controls, Gemini uses cached-content lifecycle APIs.
- [x] Execution metadata records normalized cache usage without changing cost token totals.
- [x] Provider support matrix is documented at `docs/providers/2026-06-09-cross-provider-prompt-caching.md`.
- [x] Full local suite passed after the final documentation updates: `418 passed`.
- [x] Final council follow-up review approved with comments and no blocking issues.
- [x] OpenRouter live proof succeeded on `anthropic/claude-haiku-4.5`: second identical call returned `cache_read_tokens=8116`.
- [x] Gemini API live proof succeeded on `gemini-3.1-pro-preview`: cache created, `cache_read_tokens=12603`, cleanup deleted the resource.
- [x] Vertex Gemini live proof succeeded on `gemini-3.1-pro-preview`: cache created, `cache_read_tokens=14002`, cleanup deleted the resource.
- [ ] Vertex Claude still needs separate live proof for the Anthropic Vertex SDK path.

## Non-Negotiable Design Rules

- [ ] Provider-specific request serialization stays inside adapters.
- [ ] Compiler emits decisions; it does not build provider wire payloads.
- [ ] Unsupported providers drop `prompt_cache` with an explicit compilation decision.
- [ ] Tests assert request shape and usage telemetry, not latency or cost.
- [ ] Live tests are opt-in and skipped by default.
- [ ] OpenRouter is route/model gated; it is not advertised as a stable universal cache provider.
- [ ] Gemini cached content is treated as object lifecycle work, not as the same feature as Anthropic `cache_control`.
- [ ] Vertex is split into Claude and Gemini paths.

## Provider Assessment Matrix

| Provider | First supported scope | Request behavior | Usage telemetry | Risk | Go/No-Go rule |
| --- | --- | --- | --- | --- | --- |
| Anthropic | Implemented proof slice | `extra_body={"cache_control": ...}` for locked SDK | `cache_read_input_tokens`, `cache_creation_input_tokens`, `cache_creation` buckets | SDK field drift | Keep as baseline; re-check when upgrading Anthropic SDK |
| OpenAI | Telemetry-only | No request mutation; prompt caching is automatic | `usage.prompt_tokens_details.cached_tokens` | False expectation of control | Ship only usage parsing and docs |
| OpenRouter | Route-gated passthrough | Only for explicitly cache-capable model routes; pass `cache_control` through body | `prompt_tokens_details.cached_tokens`, `cache_write_tokens` when present | Routed provider variability | Ship disabled by default unless route guard passes |
| Gemini | Cached content object lifecycle | Create/reuse/delete `cachedContents`; pass cached content name in config | `usage_metadata` cache token fields when present | Lifecycle/storage billing | Ship after object lifecycle tests exist |
| Vertex Claude | Anthropic-like path | Use Anthropic Vertex SDK path if `extra_body` or equivalent is accepted | Anthropic-style usage if returned | SDK parity uncertainty | Ship after live Vertex Claude proof |
| Vertex Gemini | Gemini-like path | Use Google GenAI Vertex cached content APIs | Gemini-style usage metadata | IAM/region/resource complexity | Ship after direct Gemini lifecycle is stable |
| CLI providers | Telemetry-only | No request mutation in this plan | Preserve existing cached-input parsing | CLI output drift | Do not expand request surface in this phase |

---

## Phase 1: Stabilize Shared Cache Contract

**Files:**
- Modify: `src/llm_council/providers/base.py`
- Modify: `src/llm_council/providers/compiler.py`
- Test: `tests/test_providers.py`
- Test: `tests/test_request_compiler.py`

- [ ] **Step 1: Lock public cache config semantics**

Keep `PromptCacheConfig` intentionally narrow until more providers are implemented:

```python
class PromptCacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = Field(default=True)
    mode: Literal["auto"] = Field(default="auto")
    ttl: Literal["5m", "1h"] = Field(default="5m")
```

Do not add `breakpoints`, `provider_options`, or `cached_content_name` until the provider slice that needs them begins.

- [ ] **Step 2: Test conservative capability semantics**

Add or retain tests proving:

```python
assert ProviderCapabilities().prompt_caching == "none"
assert OpenAIProvider(api_key="test").supports("prompt_caching") is False
assert AnthropicProvider(api_key="test").supports("prompt_caching") is True
```

Expected: non-Anthropic providers do not become truthy just because `prompt_caching` is a string field.

- [ ] **Step 3: Test compiler decision matrix**

Add table-driven tests:

```python
cases = [
    ("anthropic", "claude-opus-4-6", "supported"),
    ("openai", "gpt-5.4", "dropped"),
    ("openrouter", "anthropic/claude-opus-4-6", "dropped"),
    ("gemini", "gemini-3.1-pro-preview", "dropped"),
    ("vertex-ai", "gemini-3.1-pro-preview", "dropped"),
]
```

Expected:

- Anthropic keeps `prompt_cache`.
- All others drop it until their provider slice is implemented.
- Every drop emits a `CompilationDecision(option="prompt_cache", action="dropped", ...)`.

- [ ] **Step 4: Run shared-contract tests**

Run:

```bash
PYTHONPATH=src uv run pytest tests/test_request_compiler.py tests/test_providers.py -q
```

Expected: pass.

---

## Phase 2: Anthropic Baseline Hardening

**Files:**
- Modify: `src/llm_council/providers/anthropic.py`
- Modify: `tests/test_providers.py`
- Modify: `docs/providers/2026-06-09-anthropic-prompt-caching-design.md`

- [ ] **Step 1: Preserve SDK 0.75.0 request-shape workaround**

Keep `cache_control` inside `extra_body`:

```python
kwargs["extra_body"] = {"cache_control": cache_control}
```

Reason: locked `anthropic==0.75.0` rejects `cache_control` as a direct `messages.create()` keyword.

- [ ] **Step 2: Add beta-path cache propagation test**

Add a test combining `prompt_cache` with structured output or reasoning so `client.beta.messages.create(...)` receives:

```python
extra_body={"cache_control": {"type": "ephemeral"}}
```

Expected: no regression when existing beta paths are combined with prompt caching.

- [ ] **Step 3: Add streaming path cache propagation test**

Add a stream-path test that verifies `_generate_stream()` receives kwargs containing:

```python
{"extra_body": {"cache_control": {"type": "ephemeral"}}}
```

Expected: streaming and non-streaming request shaping stay consistent.

- [ ] **Step 4: Add hit and write usage tests**

Keep current cache write test and add cache-hit-only test:

```python
usage=SimpleNamespace(
    input_tokens=3,
    cache_read_input_tokens=10988,
    cache_creation_input_tokens=0,
    output_tokens=16,
)
```

Expected parsed usage:

```python
{
    "prompt_tokens": 10991,
    "completion_tokens": 16,
    "total_tokens": 11007,
    "cache_read_tokens": 10988,
}
```

- [ ] **Step 5: Run Anthropic local tests**

Run:

```bash
PYTHONPATH=src uv run pytest tests/test_providers.py::TestAnthropicPromptCaching -q
```

Expected: pass.

- [ ] **Step 6: Run optional Anthropic live proof**

Guard with `ANTHROPIC_API_KEY`.

Test shape:

- Same model.
- Same system prompt.
- Same user prompt.
- Large stable prefix above cache threshold.
- Two identical calls within 5 minutes.

Expected:

- First call returns `cache_creation_input_tokens > 0`.
- Second call returns `cache_read_input_tokens > 0`.
- Do not assert latency.

---

## Phase 3: OpenAI Telemetry-Only Support

**Files:**
- Modify: `src/llm_council/providers/openai.py`
- Modify: `tests/test_providers.py`
- Modify: `docs/providers/creating-providers.md` or provider support docs

- [ ] **Step 1: Do not send request cache controls**

OpenAI prompt caching is automatic. Do not add any request field to OpenAI SDK kwargs for `prompt_cache`.

- [ ] **Step 2: Parse cached-token telemetry**

In `_parse_response()`, inspect:

```python
response.usage.prompt_tokens_details.cached_tokens
```

Add normalized field when nonzero:

```python
usage["cache_read_tokens"] = cached_tokens
```

Use `cache_read_tokens` rather than `cache_creation_tokens` because OpenAI reports cached prompt tokens, not explicit writes.

- [ ] **Step 3: Happy-path test**

Mock OpenAI usage:

```python
usage = SimpleNamespace(
    prompt_tokens=2000,
    completion_tokens=100,
    total_tokens=2100,
    prompt_tokens_details=SimpleNamespace(cached_tokens=1200),
)
```

Expected:

```python
usage["cache_read_tokens"] == 1200
```

- [ ] **Step 4: Unhappy-path tests**

Cover:

- `prompt_tokens_details` missing.
- `cached_tokens` missing.
- `cached_tokens=0`.
- `prompt_cache` requested and compiler drops it for OpenAI.

Expected: parsing remains backward-compatible and no SDK kwargs include cache controls.

- [ ] **Step 5: Run OpenAI tests**

Run:

```bash
PYTHONPATH=src uv run pytest tests/test_providers.py::TestOpenAIProviderRequestParams tests/test_request_compiler.py -q
```

Expected: pass.

---

## Phase 4: OpenRouter Route-Gated Passthrough

**Files:**
- Modify: `src/llm_council/providers/openrouter.py`
- Modify: `src/llm_council/providers/compiler.py`
- Modify: `tests/test_providers.py`
- Modify: `tests/test_request_compiler.py`
- Modify: `docs/quickstart/openrouter.md`

- [ ] **Step 1: Define route guard**

Only keep `prompt_cache` for OpenRouter when the model route is explicitly cache-capable.

Initial allowlist:

```python
OPENROUTER_CACHE_CONTROL_MODEL_PREFIXES = (
    "anthropic/",
)
```

This is intentionally narrow.

- [ ] **Step 2: Compiler behavior**

For OpenRouter:

- Keep `prompt_cache` only when model starts with an allowed prefix.
- Drop otherwise with decision detail:

```text
OpenRouter prompt caching is route-dependent; model is not in cache-control allowlist
```

- [ ] **Step 3: Request serialization**

In `_build_request_body()`, when allowed:

```python
body["cache_control"] = {"type": "ephemeral"}
```

For `ttl="1h"`:

```python
body["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
```

- [ ] **Step 4: Parse telemetry**

Parse:

```python
usage.prompt_tokens_details.cached_tokens
usage.cache_write_tokens
```

Normalize:

```python
usage["cache_read_tokens"] = cached_tokens
usage["cache_creation_tokens"] = cache_write_tokens
```

- [ ] **Step 5: Happy-path tests**

Cover:

- `anthropic/claude-opus-4-6` keeps cache config.
- request body includes `cache_control`.
- response parsing includes read/write cache tokens.

- [ ] **Step 6: Unhappy-path tests**

Cover:

- `openai/gpt-5.4` route drops cache config.
- no `cache_control` emitted after compiler drop.
- malformed `prompt_tokens_details` does not crash.
- missing usage does not crash.
- virtual provider name like `anthropic/claude-opus-4-6` resolves through OpenRouter path.

- [ ] **Step 7: Run OpenRouter tests**

Run:

```bash
PYTHONPATH=src uv run pytest tests/test_request_compiler.py tests/test_providers.py::TestOpenRouterProviderEnvModel -q
```

Expected: pass.

---

## Phase 5: Gemini Cached Content Lifecycle

**Files:**
- Modify: `src/llm_council/providers/base.py`
- Modify: `src/llm_council/providers/gemini.py`
- Modify: `tests/test_providers.py`
- Create if needed: `tests/test_gemini_prompt_cache.py`
- Modify: provider docs

- [ ] **Step 1: Extend cache config only when Gemini work begins**

Add cached content fields:

```python
class PromptCacheConfig(BaseModel):
    enabled: bool = True
    mode: Literal["auto", "cached_content"] = "auto"
    ttl: Literal["5m", "1h"] = "5m"
    cached_content_name: str | None = None
```

Do not add this before the Gemini slice begins.

- [ ] **Step 2: Decide lifecycle ownership**

Use a conservative first pass:

- `cached_content_name` passthrough only.
- Do not auto-create/delete cached contents in `generate()` yet.

Reason: auto lifecycle introduces storage billing, cleanup, and concurrent reuse concerns.

- [ ] **Step 3: Request serialization**

When `cached_content_name` is provided:

```python
config["cached_content"] = request.prompt_cache.cached_content_name
```

Use the exact Google GenAI SDK field name verified against current installed `google-genai`.

- [ ] **Step 4: Parse telemetry**

Inspect `response.usage_metadata` for cache token fields available in current SDK.

Expected normalized fields:

```python
usage["cache_read_tokens"] = cached_token_count
```

Use raw fallback only when field names differ.

- [ ] **Step 5: Happy-path tests**

Mock `client.aio.models.generate_content()` and assert:

- `config.cached_content` or dict equivalent is present.
- regular generation config fields still pass.
- cache usage telemetry is parsed when present.

- [ ] **Step 6: Unhappy-path tests**

Cover:

- `mode="cached_content"` without `cached_content_name` raises validation error or compiler drop.
- invalid cached content name shape is rejected if SDK requires a resource path.
- missing `usage_metadata` does not crash.
- existing non-cache Gemini requests unchanged.

- [ ] **Step 7: Deferred lifecycle tests**

Before auto-create/delete support, write design notes for:

- create cached content
- reuse cached content
- TTL expiration
- delete cleanup
- IAM/API failures
- storage billing warnings

Do not implement lifecycle until this design is approved.

---

## Phase 6: Vertex Split Paths

**Files:**
- Modify: `src/llm_council/providers/vertex.py`
- Modify: `tests/test_providers.py`
- Modify: `tests/test_request_compiler.py`
- Modify: Vertex provider docs

- [ ] **Step 1: Vertex Claude path**

Mirror Anthropic only after verifying `AsyncAnthropicVertex` accepts:

```python
extra_body={"cache_control": {"type": "ephemeral"}}
```

Happy path:

- Claude Vertex model keeps `prompt_cache`.
- request kwargs include `extra_body.cache_control`.
- usage parser extracts Anthropic-style cache fields.

Unhappy path:

- Vertex Claude SDK rejects `extra_body`.
- missing project/region still returns normal doctor errors.
- prompt cache does not mask auth/model errors.

- [ ] **Step 2: Vertex Gemini path**

Mirror Gemini cached content passthrough only after direct Gemini slice is green.

Happy path:

- `cached_content_name` is passed to Google GenAI Vertex client config.
- usage metadata parsed.

Unhappy path:

- Gemini Vertex model without cached content support drops config.
- invalid resource/project/region does not get swallowed.
- non-cache Vertex Gemini requests unchanged.

- [ ] **Step 3: Compiler split**

For `vertex-ai`:

- Claude model: Anthropic-like behavior only if Vertex Claude support is verified.
- Gemini model: Gemini-like behavior only if cached-content passthrough is implemented.
- Unknown model: drop prompt cache.

---

## Phase 7: Cross-Provider Reporting

**Files:**
- Modify: `src/llm_council/engine/orchestrator.py`
- Modify: tests around execution metadata if present
- Modify: docs

- [ ] **Step 1: Preserve current token accounting**

Do not change existing `prompt_tokens`, `completion_tokens`, and `total_tokens` semantics without a separate compatibility decision.

- [ ] **Step 2: Add optional cache summary metadata**

When provider usage includes cache keys, execution metadata can aggregate:

```python
cache_read_tokens
cache_creation_tokens
cache_creation_5m_tokens
cache_creation_1h_tokens
```

- [ ] **Step 3: Happy-path tests**

Create provider fake results with cache usage and assert summary fields are present.

- [ ] **Step 4: Unhappy-path tests**

Cover:

- usage is `None`.
- cache usage fields absent.
- cache usage fields zero.
- provider returns unexpected non-int values.

Expected: no crash; raw provider response remains available for diagnostics.

---

## Phase 8: Full Test Matrix

Run after each provider slice:

```bash
PYTHONPATH=src uv run pytest tests/test_request_compiler.py tests/test_providers.py -q
```

Run before council review:

```bash
PYTHONPATH=src uv run pytest -q
```

Live tests are opt-in only:

```bash
ANTHROPIC_API_KEY=... PYTHONPATH=src uv run --extra anthropic pytest tests/integration/test_prompt_cache_anthropic.py -q
OPENAI_API_KEY=... PYTHONPATH=src uv run --extra openai pytest tests/integration/test_prompt_cache_openai.py -q
OPENROUTER_API_KEY=... PYTHONPATH=src uv run pytest tests/integration/test_prompt_cache_openrouter.py -q
GOOGLE_API_KEY=... PYTHONPATH=src uv run --extra gemini pytest tests/integration/test_prompt_cache_gemini.py -q
GOOGLE_CLOUD_PROJECT=... PYTHONPATH=src uv run --extra vertex pytest tests/integration/test_prompt_cache_vertex.py -q
```

Live-test assertions:

- assert telemetry fields, not latency
- use large stable prefixes
- run duplicate requests only where provider caching requires it
- skip when credentials are absent
- do not fail CI by default

---

## Phase 9: Council Review Gates

- [x] **After Anthropic hardening**

Run:

```bash
council run reviewer --runtime-profile bounded --reasoning-profile light --files src/llm_council/providers/base.py --files src/llm_council/providers/compiler.py --files src/llm_council/providers/anthropic.py --files tests/test_providers.py --files tests/test_request_compiler.py "Review Anthropic prompt-cache baseline hardening and local/live proof."
```

- [x] **After each provider slice**

Run focused council review with:

- changed provider file
- shared model/compiler files
- tests
- local test result
- live telemetry proof if available

- [x] **Before full rollout**

Run broad council review:

```bash
council run planner --mode assess "Assess cross-provider prompt-caching feature readiness, provider support matrix, implementation risks, and test evidence."
```

Go criteria:

- no blocking council issues
- local tests green
- at least Anthropic live proof retained
- OpenRouter/Gemini/Vertex live proofs either green or explicitly deferred

---

## Known Unhappy Paths To Preserve

- [x] Unsupported provider receives `prompt_cache`: compiler drops with decision.
- [x] Unsupported model route receives `prompt_cache`: compiler drops with decision.
- [x] SDK rejects cache field: test fails locally before merge.
- [x] Cache telemetry missing: parser returns normal usage without cache keys.
- [x] Cache telemetry zero: parser omits optional cache keys or returns zero consistently.
- [x] Provider auth failure: cache logic does not hide auth error.
- [x] Provider model unavailable: cache logic does not hide model error.
- [x] Live cache read not observed: do not infer from latency; report telemetry absence.
- [ ] Gemini cached content expires: lifecycle layer must create a fresh object or return actionable error.
- [ ] Gemini cached content delete fails: do not hide generation result; surface cleanup warning in lifecycle metadata.
- [x] Vertex project/region mismatch: preserve existing doctor/error behavior.

## Completion Criteria

- [x] Provider support matrix is documented.
- [x] Each provider slice has request-shape tests.
- [x] Each provider slice has usage parsing tests.
- [x] Each provider slice has unsupported-provider/model tests.
- [x] Live proof exists for Anthropic and any provider marked fully supported.
- [x] Full local test suite passes.
- [x] Council final review approves.
- [x] Follow-up docs explain which providers are supported, implicit, passthrough, deferred, or telemetry-only.
