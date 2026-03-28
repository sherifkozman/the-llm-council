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
