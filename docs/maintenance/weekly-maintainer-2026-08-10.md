# Weekly Maintainer Digest — 2026-08-10

Baseline: `origin/main` @ `1736d69` (v0.8.0). Previous digest: 2026-07-13 (4 weeks ago).

> **Note on location.** Previous digests lived in `tasks/`. Commit `379b8d0`
> ("chore: untrack internal task plans, ignore tasks/") made that directory
> gitignored, so this digest moved to `docs/maintenance/` rather than being
> force-added against that decision.

---

## Executive summary

- **Built** — `claude/feature-cli-reasoning-effort-2026-08-10`: CLI providers now honor reasoning effort (issue #58, RICE 30.0). 464 tests green, ruff + mypy strict clean. Branch only, no PR.
- **Planned** — nothing deferred to a plan this week; the top candidate cleared the gate and shipped.
- **Watch** — 2 PRs idle 16–19d despite green CI, 10 stale `claude/` branches ready for deletion, and a docstring in `logging/__init__.py` promising secret redaction that does not exist.

---

## 1. Repo health — `HEALTH: watch`

**Environment caveat found first.** The local checkout could not `git fetch`
(`Permission denied (publickey)`) and was pinned 16 commits behind, still
reading `0.7.18` while `origin/main` was at `0.8.0`. Fetching over HTTPS with
the `gh` credential helper resolved it for this run without altering the remote
config. Any local-git-only tooling in this workspace will keep reporting stale
state until SSH push/fetch credentials are fixed — **this is worth fixing, as it
silently invalidates local-only health checks.**

| Area | Status |
|---|---|
| Open PRs | 2, both `MERGEABLE` and CI-green, both idle: **#57** (19d), **#59** dependabot (16d) |
| CI | Last run on `main` **success** (2026-07-22). No failing/cancelled runs in the last 50 across any branch |
| Security | Dependabot: 11 alerts, **all `fixed`, 0 open** |
| Deps | Only 1 open bump (#59, `setup-python` 6→7), idle 16d despite green CI |
| Releases | **v0.8.0**, 2026-07-22 (19d ago). 0 commits on `main` since — HEAD == tag. Within normal cadence |
| Branches | 13 live on GitHub; **10 have no commits in >30d** |

### Drift worth a maintainer's eyes

1. **`logging/__init__.py` claims redaction that does not exist.** The docstring
   advertises *"Secret-safe logging… redaction pipeline to prevent credential and
   prompt leakage"*, but `grep -r "redact" src/` matches **only that docstring** —
   no implementation, no tests. For a project that juggles API keys across seven
   providers, this is either dead scaffolding to delete or a real gap to fill.
   Either way the docstring currently overstates what the code does.
2. **`validation/` is an empty stub.** Its docstring claims JSON-schema
   validation with retry logic; that logic actually lives in
   `engine/orchestrator.py` and `schemas/__init__.py`. Lower severity.
3. **CLAUDE.md's "~78 mypy errors" note is stale.** `mypy --strict` reports
   `Success: no issues found in 43 source files` today. Left untouched — outside
   this week's feature surface, but it should be corrected.
4. Weak signal (grep-based, unconfirmed): `base_agent.py` and
   `providers/cli/codex.py` had no name-matched test file. Possibly covered
   indirectly. Flagged as "worth a look", not a proven gap.

---

## 2. Feature candidates (RICE)

| # | Feature | Source | R | I | C | E | **Final** | Flags |
|---|---|---|---|---|---|---|---|---|
| 1 | **Fix CLI provider reasoning-effort passthrough (#58)** | Open issue | 3 | 4 | 5 | 2 | **30.0** | — |
| 2 | Deterministic caching + replay | Roadmap | 4 | 4 | 4 | 4 | 16.0 | multi-run |
| 3 | Adaptive cost/quality routing | Roadmap | 5 | 4 | 3 | 5 | 12.0 | multi-run, speculative |
| 4 | CLI live streaming output | Codebase / Roadmap | 3 | 3 | 3 | 3 | 9.0 | new-deps |
| 5 | Test coverage for CLI provider adapters | Codebase gap | 2 | 3 | 4 | 3 | 8.0 | — |
| 6 | Persistent budget tracking + spend alerts | Competitor (LiteLLM) | 3 | 3 | 3 | 4 | 6.75 | — |
| 7 | Policy engine + human-in-the-loop approvals | Roadmap | 4 | 4 | 2 | 5 | 6.4 | multi-run, speculative |
| 8 | Web side-by-side comparison UI | Competitor | 3 | 3 | 2 | 5 | 3.6 | multi-run, new-deps, speculative |

**TOP: #1 — Final 30.0.** The gap to #2 is genuine, not inflated: #58 is a
documented, already-root-caused bug in a *released* version that silently breaks
a shipped feature for an entire provider class, with the fix localized to two
files. The roadmap items below it are honestly larger and less certain.

Competitor scan covered LiteLLM (cost tracking, budgets, fallback routing),
Karpathy's `llm-council`, and Suprmind (named debate modes + UI).

---

## 3. What was built

**Branch:** `claude/feature-cli-reasoning-effort-2026-08-10`
**Commit:** `bd31204` — *fix(cli-providers): honor reasoning effort in the Codex CLI adapter*
**Pushed:** yes. **No PR opened, nothing merged, issue #58 untouched.**

### The bug (all three gaps verified present on `origin/main`)

`grep -n reasoning src/llm_council/providers/cli/` returned **zero** hits.
CLI-backed providers had no working lever for reasoning effort at all:

1. Codex `_build_command` never emitted `-c model_reasoning_effort=…`, so
   `request.reasoning` was inert.
2. `_load_provider_configs` forwarded only `api_key` and `default_model`,
   silently dropping `default_flags` — which `CodexCliProvider.__init__`
   already accepted. No config-level escape hatch.
3. The isolated `CODEX_HOME` ignored `~/.codex/config.toml`.

Net: none of the three normal levers worked, quietly defeating the per-subagent
reasoning config for every CLI-backed council.

### Fix

- **codex.py** — an enabled `ReasoningConfig` maps to `-c model_reasoning_effort=<effort>`.
- **cli/main.py** — `default_flags` forwarded from `config.yaml` via a new
  `_normalize_provider_flags()`, accepting a string or token list and dropping
  unparseable values with a warning instead of failing inside a constructor mid-run.
- **claude_code.py** — docstring note that the `claude` CLI has no effort flag,
  so `reasoning` is inert there. **No flag was invented.**
- **docs/** + CLAUDE.md — new config surface, precedence rule, and rationale.

**Precedence rule:** `default_flags` **wins over** `request.reasoning`. If the
operator pinned an effort (`-c k=v`, `-ck=v`, or `--config=k=v`, all detected via
`shlex.split`), the adapter emits nothing. Operator config outranks a shipped
subagent default, and it guarantees exactly one `model_reasoning_effort` token
in argv rather than relying on codex's duplicate-flag resolution order.

**Gap 3 deliberately skipped.** `~/.codex/config.toml` can carry `sandbox_mode`,
`approval_policy`, MCP server definitions and `notify` hooks. Copying it into the
run sandbox would let ambient user config silently widen the least-privilege
defaults the adapter sets on purpose. Documented in
`_copy_isolated_runtime_state`'s docstring instead. Gaps 1+2 give two clean
levers, so the user-facing need is met.

### Verification (re-run independently by the orchestrator, not just reported)

```
pytest                        →  464 passed, 1 warning in 9.55s
ruff check src/ tests/        →  All checks passed!
ruff format --check           →  61 files already formatted
mypy --strict src/llm_council →  Success: no issues found in 43 source files
```

Remote state re-checked after the push: `main` still at `1736d69`, open PRs
still only #57 and #59.

**Fail-first honesty.** The builder confirmed the load-bearing test
(`test_generate_forwards_request_reasoning_to_cli`) fails pre-fix on a real
assertion (`assert [] == ['high']`). It also reported, unprompted, that the
`_build_command` tests fail pre-fix via `TypeError` on a new kwarg rather than a
behavioral assertion, and that 3 of the new tests pass both before and after —
those are guards against a naive implementation, not fail-first proof. Recorded
here rather than rounded up to "all tests fail-first verified".

### Builder deviations worth review

1. `enabled=True, effort=None` emits **no** flag — diverges from the OpenAI
   adapter's `effort or "medium"`. For an HTTP call "medium" restates the API
   default; for a CLI it would actively override a real user default. No shipped
   subagent config hits this case.
2. `effort="none"` passes through rather than being downgraded. An older codex
   CLI that rejects `none` would error; no shipped config sets it.
3. `default_flags` accepts a list, normalized via `shlex.join` — slightly beyond
   "forward the value", but YAML users naturally write flag lists.
4. Value emitted unquoted (`model_reasoning_effort=high`); we exec via argv with
   no shell, so the quoted form seen in upstream docs would be wrong here.
5. `default_flags` forwards for **any** provider name, matching existing `api_key`
   behavior — setting it on a non-CLI provider raises `TypeError`. Documented
   rather than adding constructor introspection.

---

## 4. Cleanup list — stale `claude/` branches (>30d, no unique unlanded work)

Each is `ahead 1–3 / behind 15–20` vs `main`, consistent with content already
landed via squash-merge. **Not deleted by this run** — listed for a human.

| Branch | Last commit | Age |
|---|---|---|
| `claude/feature-markdown-output-2026-06-01` | 2026-06-01 | 70d |
| `claude/weekly-maintainer-2026-06-01` | 2026-06-01 | 70d |
| `claude/weekly-maintainer-2026-06-08` | 2026-06-08 | 63d |
| `claude/weekly-maintainer-2026-06-15` | 2026-06-15 | 56d |
| `claude/feature-engine-tests-2026-06-22` | 2026-06-22 | 49d |
| `claude/weekly-maintainer-2026-06-22` | 2026-06-22 | 49d |
| `claude/feature-registry-tests-2026-06-29` | 2026-06-29 | 42d |
| `claude/weekly-maintainer-2026-06-29` | 2026-06-29 | 42d |
| `claude/feature-resilience-provider-tests-2026-07-06` | 2026-07-06 | 35d |
| `claude/weekly-maintainer-2026-07-06` | 2026-07-06 | 35d |

Same pattern, non-`claude/`: `codex/feature/prompt-caching-providers-research`
(2026-06-10, 61d), superseded by merged PR #51.

Local-only, never pushed: `claude/feature-openrouter-schema-strip-2026-07-13`.

---

## 5. Suggested next actions

1. Triage the two idle green PRs (#57, #59) — 16–19d with no review.
2. Review and merge `claude/feature-cli-reasoning-effort-2026-08-10`, then close #58.
3. Decide on `logging/__init__.py`: implement redaction or delete the claim.
4. Delete the 11 stale branches above.
5. Fix SSH fetch/push credentials in this checkout so local health checks stop reading stale state.
6. Correct the stale "~78 mypy errors" line in CLAUDE.md's Known Issues.
