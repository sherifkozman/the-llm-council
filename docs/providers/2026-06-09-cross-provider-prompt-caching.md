# Cross-Provider Prompt Caching Support

## Summary

Prompt caching is provider-specific in LLM Council. `PromptCacheConfig` is a
shared request surface, but each adapter owns wire serialization and each
provider keeps only the cache mode it can actually support.

## Current Support Matrix

| Provider | Support level | Request behavior | Usage telemetry | Live proof |
| --- | --- | --- | --- | --- |
| Anthropic | Fully supported for automatic cache controls | Sends `extra_body={"cache_control": {"type": "ephemeral"}}`; includes `ttl="1h"` when requested | Parses `cache_read_input_tokens`, `cache_creation_input_tokens`, and 5m/1h creation buckets | Proven locally with first-call creation and second-call read telemetry |
| OpenAI | Telemetry-only | Sends no request cache controls because caching is automatic | Parses `usage.prompt_tokens_details.cached_tokens` into `cache_read_tokens` | Deferred; no request controls to prove |
| OpenRouter | Route-gated passthrough | Sends top-level `cache_control` only for `anthropic/` routes | Parses `prompt_tokens_details.cached_tokens` and `cache_write_tokens` | Proven live on `anthropic/claude-haiku-4.5`: second identical call returned `cache_read_tokens=8116` |
| Gemini API | Cached-content lifecycle supported | Creates cached-content resources from explicit `source_text`, refreshes TTL, passes `config["cached_content"]`, retries once after expiry, and optionally deletes after generation | Parses `usage_metadata.cached_content_token_count` into `cache_read_tokens`; returns lifecycle metadata and cleanup warnings | Proven live on `gemini-3.1-pro-preview`: created `cachedContents/...`, returned `cache_read_tokens=12603`, and deleted successfully |
| Vertex Claude | Anthropic-like split path | Sends `extra_body.cache_control` for Claude models | Parses Anthropic-style cache read/write usage | Deferred until Vertex Claude live credentials verify SDK parity |
| Vertex Gemini | Gemini-like lifecycle supported | Uses the Google GenAI Vertex cache APIs for create, TTL refresh, cached generation, expiry recreation, and cleanup | Parses `usage_metadata.cached_content_token_count`; returns lifecycle metadata and cleanup warnings | Proven live on `gemini-3.1-pro-preview`: created `projects/.../cachedContents/...`, returned `cache_read_tokens=14002`, and deleted successfully |
| CLI providers | Telemetry preservation only | No prompt-cache request controls in this phase | Existing CLI usage parsing remains provider-owned | Deferred |

## Request Contract

`PromptCacheConfig(mode="auto")` is valid for Anthropic-style automatic cache
controls. The compiler keeps it for Anthropic and Vertex Claude, and keeps it
for OpenRouter only when the route model starts with `anthropic/`.

`PromptCacheConfig(mode="cached_content", cached_content_name="cachedContents/...")`
is valid for Gemini-style reuse of an existing cached-content resource.

`PromptCacheConfig(mode="cached_content", create_if_missing=True, source_text="...")`
creates a Gemini-style cached-content resource from explicit stable source text.
The adapter then forwards the returned cache name in generation config.

Optional lifecycle flags:

- `refresh_ttl=True` updates the cached-content TTL before generation.
- `delete_after=True` performs best-effort cleanup after generation.
- Expired or missing cached-content resources are recreated once when
  `create_if_missing=True` and `source_text` is available.
- Streaming expiry is retried only before any chunk has been yielded. If expiry
  is detected after partial streamed output, the provider propagates the error
  after cleanup instead of concatenating partial output with a retried stream.
- Successful streaming responses include a final metadata-only
  `GenerateResponse` chunk with cached-content cleanup status.

Created or refreshed cached-content resources can incur provider storage billing
until TTL expiry or successful cleanup. Cleanup failures are returned as
warnings and do not hide a successful generation response.

Unsupported providers, unsupported routes, disabled configs, and incompatible
modes are dropped or ignored by the compiler with explicit request-compilation
decisions.

## Testing Policy

Unit tests assert request shape and usage telemetry only. They do not assert
latency or cost.

Execution metadata separates cache intent from cache proof:

- `requested`: the caller supplied an enabled `prompt_cache`.
- `forwarded`: the compiler preserved `prompt_cache` for adapter serialization.
- `metrics_available`: the provider returned normalized cache telemetry.
- `observed`: positive normalized cache telemetry was returned by the provider.
- `usage`: the positive cache counters observed in the provider response.

`forwarded=True` is not proof of a cache hit. Cache use is only observed when
the provider response contains positive normalized cache counters.

Live tests must be opt-in and skipped by default. A provider is marked fully
supported only when its request controls and telemetry are proven live. Providers
without live proof are documented as telemetry-only, passthrough, cached-content
passthrough, or deferred.

## Remaining Lifecycle Caveats

Gemini and Vertex Gemini lifecycle handling is intentionally scoped to a single
request. LLM Council can create, reuse by name, refresh TTL, retry once after
expiry, and best-effort delete. It does not yet provide a shared cache registry,
cross-process locking, or automatic reuse discovery.

Live proof has been captured for OpenRouter, Gemini API, and Vertex Gemini on
non-streaming generation paths. Streaming lifecycle behavior is covered by
regression tests and still needs separate live proof before being called
production-proven.
Vertex Claude still needs separate live proof because it uses the Anthropic
Vertex SDK path, not the Google GenAI cached-content lifecycle.
