# Anthropic Prompt Caching Proof Slice Design

## Goal

Prove prompt-caching support in LLM Council with the smallest provider slice that has explicit request controls and observable usage telemetry: Anthropic automatic prompt caching.

This is not the full multi-provider implementation. It is the proof slice that decides whether the common request, compiler, adapter, and telemetry surfaces are sound enough to expand later.

## Decision

Implement Anthropic first.

Anthropic is the right proof target because it supports top-level `cache_control`, has documented `5m` and `1h` TTL behavior, and returns cache-specific usage fields. OpenAI is automatic and telemetry-only, OpenRouter depends on routed upstream behavior, and Gemini/Vertex Gemini require cached content object lifecycle work. Those are useful follow-ups, but they are weaker first proofs.

## Current Repo Gap

`GenerateRequest` has no cache control field.

`ProviderCapabilities` has no prompt-caching capability.

`compile_request_for_provider()` has no caching decisions.

The Anthropic adapter does not send `cache_control` and does not parse:

- `cache_read_input_tokens`
- `cache_creation_input_tokens`
- `usage.cache_creation.ephemeral_5m_input_tokens`
- `usage.cache_creation.ephemeral_1h_input_tokens`

CLI providers already parse some cached-token usage, but SDK providers do not.

## Provider Research Summary

Anthropic supports two request modes:

- Automatic caching: top-level `cache_control={"type": "ephemeral"}`.
- Explicit breakpoints: `cache_control` on individual content blocks.

Automatic caching is the proof-slice scope. Explicit breakpoints are deferred because the current `Message.content` shape is mostly plain text, while block-level breakpoints need content-block arrays and clearer breakpoint placement rules.

Anthropic TTL support:

- Default: 5 minutes.
- Optional: `ttl="1h"`.

Anthropic usage telemetry:

- `input_tokens`
- `output_tokens`
- `cache_read_input_tokens`
- `cache_creation_input_tokens`
- optional `cache_creation` bucket fields for 5-minute and 1-hour writes.

## Proposed Public Shape

Add a provider-agnostic but conservative cache config:

```python
class PromptCacheConfig(BaseModel):
    enabled: bool = True
    mode: Literal["auto"] = "auto"
    ttl: Literal["5m", "1h"] = "5m"
```

Add to `GenerateRequest`:

```python
prompt_cache: PromptCacheConfig | None = None
```

This is deliberately smaller than a final multi-provider abstraction. It allows Anthropic automatic caching without claiming support for Gemini cached content objects, OpenAI implicit caching controls, or OpenRouter route guarantees.

## Capability Shape

Add a conservative capability field:

```python
prompt_caching: Literal["none", "implicit", "explicit", "context_object", "passthrough"] = "none"
```

For this proof slice:

- Anthropic: `explicit`
- OpenAI: `implicit`
- OpenRouter: `passthrough`
- Gemini: `context_object`
- Vertex AI: `passthrough`
- CLI providers: leave unchanged unless current usage parsing needs capability visibility.

The capability field is descriptive. It does not require implementing every provider in this slice.

## Compiler Behavior

For `provider_identity == "anthropic"`:

- Keep `prompt_cache` when `mode == "auto"`.
- Transform `ttl="5m"` to `{"type": "ephemeral"}` in the adapter.
- Transform `ttl="1h"` to `{"type": "ephemeral", "ttl": "1h"}` in the adapter.
- Record a `supported` decision for Anthropic automatic prompt caching.

For other providers in this proof slice:

- Preserve only descriptive capability metadata.
- Drop `prompt_cache` with a compilation decision unless provider implementation exists.
- OpenAI should not receive a request mutation because OpenAI prompt caching is automatic.
- OpenRouter should not receive `cache_control` until route/model handling is implemented.
- Gemini and Vertex Gemini should not receive `prompt_cache` until cached content lifecycle is implemented.

## Anthropic Adapter Behavior

When `request.prompt_cache` is present:

- Add top-level `cache_control` to the request body. With the currently locked Anthropic SDK, this must be sent through `extra_body={"cache_control": ...}` because `cache_control` is not yet an explicit SDK keyword.
- Use the existing normal messages/system handling.
- Do not add block-level `cache_control`.
- Do not force beta APIs solely for caching.
- Continue using beta only for existing structured-output or reasoning paths.

Usage parsing should keep current token totals stable:

- `prompt_tokens` should include all input-equivalent tokens:
  - `input_tokens`
  - `cache_read_input_tokens`
  - `cache_creation_input_tokens`
- `completion_tokens` remains `output_tokens`.
- `total_tokens` is prompt plus completion.

Add optional normalized cache telemetry keys:

- `cache_read_tokens`
- `cache_creation_tokens`
- `cache_creation_5m_tokens`
- `cache_creation_1h_tokens`

Do not infer a cache hit from latency.

## Proof Criteria

The proof slice is successful when all are true:

- A `GenerateRequest(prompt_cache=PromptCacheConfig(...))` compiles unchanged for Anthropic.
- The Anthropic adapter sends the expected top-level `cache_control`.
- Existing non-cache Anthropic requests do not change.
- Cache usage fields are parsed when present.
- Missing cache usage fields default to zero or omission without breaking current usage consumers.
- Unsupported providers either drop the cache config in compiler decisions or leave behavior unchanged.

## Test Plan

Unit tests:

- `PromptCacheConfig` accepts default auto caching and `ttl="1h"`.
- Invalid modes or TTLs are rejected.
- Anthropic compiler keeps supported automatic caching.
- OpenAI/OpenRouter/Gemini/Vertex compiler drops cache config until implemented.

Provider request tests:

- Anthropic request with default TTL sends `cache_control={"type": "ephemeral"}`.
- Anthropic request with `ttl="1h"` sends `cache_control={"type": "ephemeral", "ttl": "1h"}`.
- Anthropic request without cache sends no `cache_control`.
- Structured output and reasoning still select the same beta/non-beta paths as before.

Provider usage tests:

- Anthropic parser extracts read and write cache tokens.
- Anthropic parser includes cached tokens in `prompt_tokens`.
- Anthropic parser handles missing `cache_creation` buckets.

Optional live integration:

- Marked separately and skipped by default.
- Requires `ANTHROPIC_API_KEY`.
- Sends the same long stable-prefix prompt twice.
- Asserts only usage telemetry fields, not latency or cost.

## Risks

The main risk is designing a cache abstraction that becomes too broad too early. The proof slice avoids this by supporting only automatic Anthropic caching and by treating other provider shapes as future work.

Anthropic explicit block-level breakpoints are intentionally deferred. They need content block support and placement rules that are not part of the current `Message` model.

OpenRouter should not be treated as equivalent to Anthropic yet. It may pass through cache controls for Claude routes, but route/model/provider behavior can vary.

Gemini context caching is a separate lifecycle feature. It should not be squeezed into the Anthropic proof slice.

## Review Gate

After this proof slice is implemented and tested, run a council review with the changed files and test results.

If council accepts the slice and live telemetry proves useful, write the full implementation and testing plan for:

- OpenAI telemetry parsing
- OpenRouter passthrough with route safeguards
- Gemini cached content lifecycle
- Vertex Claude and Vertex Gemini split paths
- normalized cache reporting in council execution metadata
