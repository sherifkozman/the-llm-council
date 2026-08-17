# Weekly Maintainer Digest — 2026-08-17

Baseline: `origin/main` @ `1736d69` (v0.8.0). Previous digest: 2026-08-10 (7 days ago).

---

## Executive summary

- **Nothing built this week** — because the top-scored candidate is "merge PR #57", an action this run is forbidden to take, and the top *buildable* item (secret redaction) is a security-surface change whose first decision is a maintainer's to make, not a builder's.
- **Planned** — implementation plans below for the top candidate (#57 review) and the top buildable candidate (#2, redaction).
- **Watch** — the repo is functionally healthy but **entirely blocked on human review**: 2 green PRs now idle 22–26d, issue #58's fix sitting unmerged on a branch since last week, and all 10 stale branches from last week's cleanup list still live.

---

## 1. Repo health — `HEALTH: green (with a review-latency caveat)`

| Area | Status |
|---|---|
| Open PRs | 2, both `MERGEABLE` and CI-green, both idle: **#57** (26d), **#59** dependabot (22d) |
| CI | Last run on `main` (`1736d69`, 2026-07-22) **success**; no failures in the last 15 runs |
| Security | Dependabot: **0 open alerts**. One historical `pyasn1` DoS (GHSA-hm4w-wwcw-mr6r) — state `fixed` |
| Deps | 1 open bump (#59, `setup-python` 6→7), idle 22d despite green CI |
| Releases | **v0.8.0**, tagged 2026-07-22 (26d ago). **0 commits on `main` since** — HEAD == tag |
| Branches | 15 live on GitHub; **10 `claude/` branches with no commits in >30d** |
| Open issues | 1 — **#58** (fix already built last week, unmerged) |

> **Correction to the health scan.** The `repo-health` subagent reported "no
> `claude/` branches older than 30 days — clean." That is wrong. Querying the
> GitHub branch list directly returns **10** such branches, unchanged from last
> week's cleanup list. The table above and §4 use the verified data, not the
> subagent's claim. Recorded here rather than silently overwritten, because a
> health check that reports "clean" when nothing was cleaned is a failure mode
> worth seeing.

### The real signal: nothing is moving

Every health metric is green, and that is slightly misleading. `main` has not
received a commit in 26 days. Meanwhile:

- PR #57 (+340 lines, 5 files, external contributor `beysa`) — green, mergeable, **26 days unreviewed**
- PR #59 (CI action bump) — green, mergeable, **22 days unreviewed**
- Issue #58's fix — built and pushed by last week's run, **7 days unreviewed**
- 10 stale branches — flagged last week, **0 deleted**

The bottleneck is not engineering capacity; it is maintainer review. Automation
adding an eleventh branch this week would make that worse, not better. That
observation drove the gate decision below.

### Drift worth a maintainer's eyes

1. **`logging/__init__.py` is a 5-line docstring-only stub** claiming *"a
   redaction pipeline to prevent credential and prompt leakage."* Verified:
   `grep -rn "redact" src/` matches **that one docstring and nothing else**.
   Carried over unresolved from last week. See plan in §3.
2. **`validation/__init__.py` — same shape**, also 5 lines, claiming JSON-schema
   validation with retry that actually lives in `engine/orchestrator.py` and
   `schemas/__init__.py`. Lower severity, same root cause.
3. **README version markers are stale** — sections still labelled "v0.7.x
   syntax", "(v0.3.0+)", "(v0.4.0+)" against a v0.8.0 package. Cosmetic.
4. **CLAUDE.md's Known Issues is now accurate** (mypy-clean since v0.7.18) —
   last week's stale "~78 mypy errors" note has been corrected. No action.
5. `registry/base_agent.py` still has no direct test file; exercised indirectly
   via `engine/capabilities.py` (`tests/test_capabilities.py`). Weak, not zero.

### Environment caveat (third consecutive week)

The local checkout still cannot `git fetch` over SSH (`Permission denied
(publickey)`) and sat **16 commits behind** `origin/main`, with local `main`
reading `24e23b8` while `origin/main` is `1736d69`. Fetching over HTTPS via the
`gh` credential helper worked around it again without touching remote config.
**Any local-git-only tooling in this workspace silently reports stale state.**
This has now been flagged three runs running and is worth a real fix.

---

## 2. Feature candidates (RICE)

| # | Feature | Source | R | I | C | E | **Final** | Flags |
|---|---|---|---|---|---|---|---|---|
| 1 | **Merge Sakana/Fugu provider PR #57** | Open PR | 2 | 3 | 5 | 1 | **30.0** | not buildable by this run |
| 2 | **Implement the secret-redaction pipeline `logging/` claims** | Codebase gap | 5 | 4 | 4 | 3 | **26.67** | new this run |
| 3 | Deterministic caching + replay | Roadmap | 4 | 4 | 3 | 4 | 12.0 | multi-run (4th) |
| 3 | Adaptive cost/quality routing | Roadmap | 5 | 4 | 3 | 5 | 12.0 | multi-run (4th), speculative |
| 5 | Test coverage for `claude_code.py` / `gemini_cli.py` | Codebase gap | 2 | 3 | 4 | 3 | 8.0 | multi-run (3rd) |
| 6 | Persistent budget tracking + spend alerts | Competitor (LiteLLM) | 3 | 3 | 3 | 4 | 6.75 | new-deps |
| 7 | Policy engine + human-in-the-loop approvals | Roadmap | 4 | 4 | 2 | 5 | 6.4 | multi-run (3rd), speculative |

**TOP: #1 — Final 30.0.**

Scoring honesty, carried through from the scout rather than smoothed over:

- **#1 ranks first only because its effort is ~0**, not because it is a big win.
  Reach is genuinely low (2): Sakana/Fugu is a niche provider nobody uses
  *because* it isn't merged. A near-zero denominator is inflating this score;
  treat the ranking as "cheapest unblock", not "highest value".
- **#2 is new this run** — it appeared as a watch item last week but was never
  RICE-scored. It is the highest-scoring item that is actually *code work*.
- **#3/#4 have now appeared unbuilt in four consecutive digests** (07-06, 07-13,
  08-10, 08-17); **#5 and #7 in three**. Scores were deliberately **not**
  adjusted to manufacture a different outcome. Four identical re-scorings is
  itself the finding: these roadmap items are either genuinely too large for a
  single clean run (both are multi-file — hashing/storage/CLI for caching;
  classifier + escalation + validators for routing) or the roadmap overstates
  their near-term priority. **They need a maintainer decision — kill, re-scope
  smaller, or formally deprioritize — not a fifth identical score.**

Competitor scan this week covered LiteLLM (per-key/team budgets, block-on-
exhaustion, load balancing), Portkey and TrueFoundry (gateway-level cost
governance), LangGraph and the OpenAI Agents SDK (both ship the human-in-the-
loop pattern behind #7).

---

## 3. What was built — nothing, and why

**No feature branch was created this week.** The gate was applied to the single
highest-Final candidate, as specified, and it failed:

**#1 (Final 30.0) is not eligible.** The action it names *is* the merge. This
run is explicitly forbidden to merge, and PR #57 is +340 lines of third-party
provider code from an external contributor (`beysa`) — precisely the class of
change that needs a human reviewer, not an automated one. There is no
"self-contained change to complete cleanly" here; the work is already written
and waiting on judgment.

I did **not** fall through to #2 and build that instead. Two reasons, and the
second is the one that matters:

1. **The gate is written against the top candidate**, not "the top item I happen
   to be able to build." Rewriting the rule mid-run to justify producing output
   would defeat the point of having a threshold.
2. **#2's first question is not an engineering question.** The docstring
   promises redaction the code does not do. There are two legitimate fixes —
   implement the pipeline, or delete the false claim — and choosing between
   them is a product/security call for the maintainer. Implementing a redaction
   pipeline across seven providers' credential paths also lands squarely on the
   security surface, where this repo's own process requires a plan review
   *before* code. Shipping it unreviewed on a branch nobody has time to read
   would add risk, not remove it.

Given four PRs/branches already queued for review, the honest highest-value
output this week is a decision list, not an eleventh branch.

### Implementation plan — #1 (PR #57), for the reviewer

PR #57 adds a `sakana` provider adapter plus two fixes: sanitizing
`reasoning_effort` (which was causing a silent 400 that dropped the provider
from councils) and flooring its per-phase timeout so it survives
`--runtime-profile bounded`. It reports 7 provider tests + 2 orchestrator
timeout tests, with ruff, ruff-format, mypy --strict and the full suite green —
and CI independently confirms green. Review should focus on three things the
green checks cannot answer: whether the new adapter's auth and subprocess/HTTP
isolation match the hardening applied to the other adapters in v0.8.0 (the
release explicitly themed "CLI adapter reliability & security"); whether the
timeout floor interacts correctly with the degradation policy in
`engine/degradation.py` rather than merely making the provider slower to fail;
and whether adding a seventh-plus provider to the registry obliges a
corresponding entry in CLAUDE.md's provider table and `council doctor`. If those
hold, it is a straightforward merge; the 26-day idle time is the actual problem,
not the diff.

### Implementation plan — #2 (redaction), for whoever picks it up

First decide the question the code cannot: does LLM Council promise
secret-safe logging, or not? If **no**, the fix is three lines — delete the
misleading docstrings in `logging/__init__.py` and `validation/__init__.py` and
let the modules be honest stubs, closing the drift permanently. If **yes**, scope
it as: a `redact()` helper with a pattern set covering the provider key formats
actually in play (`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GOOGLE_API_KEY`/`GEMINI_API_KEY`, Vertex ADC material) plus a generic
high-entropy-token rule; a logging filter installed at the framework's log
setup point so it applies uniformly rather than per-call-site; and explicit
tests asserting that a known key value never reaches a log record, including
through the CLI subprocess adapters where keys pass through argv and env. Route
it through the plan-review gate before writing code — it touches credential
handling across every provider — and keep prompt redaction a separate,
opt-in decision from credential redaction, since councils legitimately need
prompt text in artifacts for debugging.

---

## 4. Cleanup list — stale `claude/` branches (>30d)

**Unchanged from last week — none were deleted.** Verified live on the remote
today via the GitHub branch API. Each is ahead 1–3 / behind 15–20 vs `main`,
consistent with content already landed by squash-merge.

| Branch | Last commit | Age |
|---|---|---|
| `claude/feature-markdown-output-2026-06-01` | 2026-06-01 | 77d |
| `claude/weekly-maintainer-2026-06-01` | 2026-06-01 | 77d |
| `claude/weekly-maintainer-2026-06-08` | 2026-06-08 | 70d |
| `claude/weekly-maintainer-2026-06-15` | 2026-06-15 | 63d |
| `claude/feature-engine-tests-2026-06-22` | 2026-06-22 | 56d |
| `claude/weekly-maintainer-2026-06-22` | 2026-06-22 | 56d |
| `claude/feature-registry-tests-2026-06-29` | 2026-06-29 | 49d |
| `claude/weekly-maintainer-2026-06-29` | 2026-06-29 | 49d |
| `claude/feature-resilience-provider-tests-2026-07-06` | 2026-07-06 | 42d |
| `claude/weekly-maintainer-2026-07-06` | 2026-07-06 | 42d |

Same pattern, non-`claude/`: `codex/feature/prompt-caching-providers-research`
(2026-06-10, 68d), superseded by merged PR #51.

**Not** cleanup candidates (<30d): `claude/weekly-maintainer-2026-08-10`,
`claude/feature-cli-reasoning-effort-2026-08-10` (both 7d, the latter awaiting
review), and this week's `claude/weekly-maintainer-2026-08-17`.

Local-only, never pushed: `claude/feature-openrouter-schema-strip-2026-07-13`,
`claude/weekly-maintainer-2026-07-13`.

---

## 5. Suggested next actions

Ordered by what unblocks the most. Every item is a human decision:

1. **Review PR #57** (26d idle) — see plan in §3.
2. **Merge PR #59** (22d idle) — a CI action bump with green checks.
3. **Review `claude/feature-cli-reasoning-effort-2026-08-10`** and close #58.
4. **Decide the redaction question** (implement vs. delete the claim) — §3.
5. **Rule on the recurring roadmap items** (caching, adaptive routing, policy
   engine). Four unbuilt appearances means the backlog is lying about priority.
6. **Delete the 11 stale branches** in §4 — carried over untouched.
7. **Fix SSH fetch/push credentials** in this checkout — third week flagged.
