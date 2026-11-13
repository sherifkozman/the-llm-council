# Eval Datasets

This directory contains baseline evaluation datasets for `council eval`.

Current datasets:

- `runtime-baseline.yaml`: checks mode-aware runtime behavior, capability
  activation, and minimal structured outputs for the most important modes.

Example:

```bash
council eval evals/runtime-baseline.yaml --providers openrouter
```

This harness is intentionally deterministic. It checks execution plans and
required output structure first. It does not try to replace human judgment for
semantic quality.

For internal benchmark creation, keep any raw PR diffs, copied code, or review
fixtures sourced from other repositories under `.council-private/`. That path is
gitignored and intended for local-only evaluation inputs. Do not commit those
materials into `evals/` or other tracked repo paths.

A case expecting `success: true` also requires `execution_status: completed`
by default. Set `execution_status: degraded` explicitly when testing a degraded
response, or `success: false` and `execution_status: failed` for an expected failure. Setting
`success: null` leaves success unconstrained; set an explicit execution status
if completion matters. Results include the actual execution status. A completed
status does not itself establish provider quorum or semantic accuracy.

Comparisons rank variants only when every selected case was evaluated. In
particular, fail-fast runs with incomplete case coverage are reported but not
ranked, and an empty selection has no winner. A non-completed run under an
unconstrained `success: null` case is also unranked unless the expected status
is explicit. The CLI returns a nonzero exit when no cases or no ranked variants
are available.

Rankings are relative comparisons; the best-ranked variant is not necessarily
an acceptable one. Check case results and pass rates before selecting it.
