# Weekly Maintainer Digest — 2026-11-02

Baseline: `origin/main` @ `2bca169` (v0.8.8). Previous digest: 2026-10-26
(branch `claude/weekly-maintainer-2026-10-26`, never merged to `main`).

---

## Executive summary (3 lines)

- **Built:** ROADMAP.md "Current Release" refreshed from v0.7.0 to v0.8.8 on
  branch `claude/feature-roadmap-current-release-2026-11-02` (1 file, +7/−2,
  every claim cited to release notes). No PR opened.
- **Planned:** nothing new. #64's reserved override half still waits on the
  maintainer, and v0.8.6–0.8.8 removed most of its urgency.
- **Why it's small:** the build cleared the gate on the scoring formula rather
  than on real value. A one-effort docs edit scores 20 on the uncalibrated
  `(R×I×C)/E` scale. Threshold recalibration is now overdue by 3+ runs (see §2).

---

## 1. Repo health — `HEALTH: watch`

| Area | Status |
|---|---|
| CI | Latest `CI` on `main` (`2bca169`, 2026-10-30) **success**. Two transient failures during the 0.8.6–0.8.8 sequence, both fixed by later commits |
| Security | Dependabot **0 open alerts** |
| Releases | **v0.8.6, v0.8.7, v0.8.8**, all on 2026-10-30 within ~6.5h (Codex compatibility sequence). `main` == v0.8.8 |
| Open PRs | 3, all `REVIEW_REQUIRED`: **#57** (103d, `CONFLICTING`), **#59** dependabot setup-python 7 (99d, now `UNKNOWN`), **#76** dependabot setup-node 7 (2d, `MERGEABLE`, new) |
| Open issues | 1: **#64** (27d), unchanged |
| Branches | **13 stale `claude/` branches**, the same 13 as last week. None pruned |
| Coverage | **Not measured this run.** Last measured: 79% on v0.8.5, which predates 4 commits of provider/CLI changes |

### What changed since 2026-10-26

- **The maintainer reshaped #64 without closing it.** v0.8.6–0.8.8 moved Codex
  admission to a *separately maintained, fetched compatibility record* covering
  six platform/arch builds. Unknown builds still fail closed (`docs/releases/0.8.8.md`).
  So qualifying a new Codex build no longer needs a package release. That was
  the bulk of #64's reported pain ("every auto-update disables them"). Last
  digest's §3 "safe half" plan is therefore **superseded**: the maintainer
  shipped something stronger. The reserved override/range half is still open.
  Consider updating or narrowing #64 to say so.
- **PR #59 dropped from `MERGEABLE` to `UNKNOWN`**, and a sibling dependabot PR
  (#76) arrived. These two GitHub Actions major bumps are cheap to batch-merge
  once CI is green.
- **`CLAUDE.md` drift moved but didn't close.** Its header now says "v0.8.3
  release target" (actual: 0.8.8), and Known Issues still doesn't mention #64.
  It is **still not auto-fixed**: `CLAUDE.md` is under the config-safety rule,
  and an unattended run can't get confirmation.
- **Local SSH fetch is still broken, fifth consecutive run.** The checkout started
  4 commits behind `origin/main`. Both subagents had to work around it via `gh`,
  and the scout initially reasoned from stale local source.

### Not checked this run

`pip`/`uv` dependency audit, measured coverage, whether the October `codex/*`
branches are merged, and doc drift beyond `CLAUDE.md` and `ROADMAP.md`.

---

## 2. Feature candidates (RICE)

Final = (R×I×C)/E, each dimension 1–5.

| # | Feature | Source | R | I | C | E | **Final** | Flags |
|---|---|---|---|---|---|---|---|---|
| 1 | **Refresh ROADMAP.md current release (v0.7.0 → v0.8.8)** | Roadmap drift (verified on `main`) | 2 | 2 | 5 | 1 | **20.0** | — |
| 2 | Opt-in anonymized, shuffled-label peer critique | Competitor scan | 4 | 3 | 2 | 2 | **12.0** | design work; changes default critique behaviour |
| 3 | Local / OpenAI-compatible provider (Ollama, vLLM, LM Studio) | Competitor scan | 3 | 4 | 3 | 3 | **12.0** | design work; local-endpoint security posture |
| 4 | Tests for `providers/cli/_subprocess.py` (33%) + `storage/summarize.py` (50%) | 2026-10-26 measured coverage | 2 | 2 | 4 | 2 | **8.0** | re-measure first; subset of a multi-run item |
| 5 | #64 reserved half: unverified-version override / range | Issue #64 | 4 | 4 | 2 | 4 | **8.0** | security-policy decision reserved by maintainer |
| 6 | Local observability + run viewer | ROADMAP | 4 | 4 | 2 | 5 | **6.4** | design; new deps; multi-run (2026-10-26) |
| 7 | Deterministic caching + replay | ROADMAP | 3 | 3 | 2 | 4 | **4.5** | design |
| 8 | Triage PR #57 (Sakana/Fugu, conflicting, 5.2k lines) | PR #57 | 2 | 2 | 2 | 4 | **2.0** | maintainer action; multi-run (2026-10-26) |

**Scout corrections, recorded instead of silently overwritten:**
- The scout's flags were honest this week. The only `multi-run` citations point
  to the 2026-10-26 digest, which is correct.
- The scout said #64's "safe half is already shipped". That's partly right. The
  allowlist and doctor output shipped in v0.8.2, but the structural-match idea
  in last week's plan was overtaken by the fetched record in v0.8.6–0.8.8, not
  implemented as described.
- The scout reasoned from a stale local checkout. Candidate #1's drift was
  re-verified against `origin/main` via the GitHub API before the build:
  `ROADMAP.md:7` read `## Current Release: v0.7.0`, and the file was last
  changed 2026-03-28.

**Threshold calibration, now overdue (5+ runs).** The gate compares a 1–10
mental model (`THRESHOLD = 8`) against a `(R×I×C)/E` score that runs from 0.2
to 125 in theory and from 2 to 30 in practice. Effort sits in the denominator,
so any E=1 chore with high confidence outranks real features. This week a
9-line docs edit (20.0) beat two substantive product ideas (12.0). Recommend one
of these:
- Normalise `Final` to 1–10.
- Add a floor such as `I ≥ 3` so that cosmetic work cannot be top-ranked.

---

## 3. Gate decision: built #1

Candidate #1 met all three conditions: Final 20.0 ≥ 8, no flags, and
self-contained.

- **Branch:** `claude/feature-roadmap-current-release-2026-11-02`, commit
  `9effc64`. Parent is `2bca169` = `origin/main`, verified on the remote.
- **Diff:** `ROADMAP.md` only, +7/−2. Changes:
  - Heading set to v0.8.8.
  - Five foundation bullets added: Markdown output, provider prompt caching,
    `execution_status` + invocation contract, native CLI compatibility
    admission, and evidence-read containment.
  - "Last updated" footer bumped.
- **Sources:** each bullet is cited to `docs/releases/0.7.18.md`, `0.8.1.md`,
  `0.8.4.md`, `0.8.5.md`, `0.8.6.md`, `0.8.8.md` or `CHANGELOG.md`. Markdown
  output (0.7.17) is cited to the CHANGELOG only, because no release-notes file
  exists for that version.
- **Unchanged:**
  - No roadmap status labels changed. Nothing marked not-started had fully shipped.
  - Provider prompt caching ≠ "Deterministic Caching + Replay".
- **Checks:** `tests/test_evidence.py` passed (6/6). That suite is the only
  code path that references `ROADMAP.md`. Under the 10-line council-review
  gate, so no `Council-Review:` trailer.
- **Not done:** no PR opened, nothing merged, tagged or released.

Honest value read: this is correct and safe, but it is a docs tidy-up, not a
feature. It shipped because the gate let it through, which is the calibration
problem above.

---

## 4. Stale `claude/` branch cleanup list

13 branches with no commits in >30d, verified against the GitHub branches API.
**This is the same list as last week. Nothing has been pruned in three digests.**

| Branch | Age |
|---|---|
| `claude/feature-markdown-output-2026-06-01` | 154d |
| `claude/weekly-maintainer-2026-06-01` | 154d |
| `claude/weekly-maintainer-2026-06-08` | 147d |
| `claude/weekly-maintainer-2026-06-15` | 140d |
| `claude/feature-engine-tests-2026-06-22` | 133d |
| `claude/weekly-maintainer-2026-06-22` | 133d |
| `claude/feature-registry-tests-2026-06-29` | 126d |
| `claude/weekly-maintainer-2026-06-29` | 126d |
| `claude/feature-resilience-provider-tests-2026-07-06` | 119d |
| `claude/weekly-maintainer-2026-07-06` | 119d |
| `claude/feature-cli-reasoning-effort-2026-08-10` | 84d |
| `claude/weekly-maintainer-2026-08-10` | 84d |
| `claude/weekly-maintainer-2026-08-17` | 77d |

Also old, non-`claude/`:
- `codex/feature/prompt-caching-providers-research` (145d). Prompt caching shipped in 0.7.18, so it is probably superseded.
- `dependabot/github_actions/actions/setup-python-7` (100d), which backs PR #59.

`claude/feature-markdown-output-2026-06-01` is almost certainly superseded:
`--format markdown` shipped in 0.7.17.

---

## 5. Maintainer action list

1. **Review and merge `claude/feature-roadmap-current-release-2026-11-02`**: 9 lines, docs only.
2. **Recalibrate the gate** (§2). Without it, next week will select another E=1 chore.
3. **Update #64** to reflect the fetched compatibility record, and decide
   whether the reserved override half still warrants an open issue.
4. **Batch-merge dependabot #59 + #76.**
5. **Delete the 13 stale branches** in §4. Start with `feature-markdown-output`
   and all `weekly-maintainer-*` digests older than this one.
6. **Approve the `CLAUDE.md` drift fix**: header v0.8.3 → v0.8.8, and mention #64 in Known Issues.
7. **Rebase or close PR #57** (103d, conflicting).
8. **Fix local SSH fetch**: flagged on five runs in a row.
