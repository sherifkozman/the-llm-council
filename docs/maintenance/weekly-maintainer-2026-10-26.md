# Weekly Maintainer Digest — 2026-10-26

Baseline: `origin/main` @ `0839581` (v0.8.5). Previous digest: 2026-08-17 — **70 days ago**.

---

## Executive summary

- **Nothing built this week** — because the top-scored candidate (close issue #64) is the one piece of #64 the maintainer explicitly reserved: a deliberate fail-closed security gate whose own source says version strings "cannot prove tool or instruction isolation."
- **Planned** — implementation plan for #64's remaining scope in §3, written as a decision request rather than a diff.
- **Watch** — the maintainer shipped **5 releases in October unaided** while this automation sat idle for 10 weeks; meanwhile PR #57 rotted from `MERGEABLE` to `CONFLICTING` and the stale-branch pile grew 10 → 13.

---

## 1. Repo health — `HEALTH: watch`

| Area | Status |
|---|---|
| CI | Last run on `main` (`0839581`, 2026-10-23) **success**; last 15 runs all green |
| Security | Dependabot **0 open / 16 fixed**. Secret scanning + push protection enabled |
| Releases | **v0.8.5**, 2026-10-23 (3d ago). 0 commits on `main` since — HEAD == tag. 5 patch releases in October |
| Open PRs | 2, both `REVIEW_REQUIRED`: **#57** (96d, now **`CONFLICTING`**), **#59** dependabot (93d, `MERGEABLE`) |
| Open issues | 1 — **#64**, partially fixed in v0.8.5 via #68, intentionally left open |
| Branches | **13 stale `claude/` branches** (>30d), up from 10 last digest |
| Coverage | 79% overall (measured). Weak: `providers/cli/_subprocess.py` 33%, `storage/summarize.py` 50%, `providers/openai.py` 56%, `cli/main.py` 59%, `providers/vertex.py` 62% |

### The real signal: the automation stopped, the maintainer didn't

The previous digest is dated 2026-08-17. It is now 2026-10-26 — this ritual
missed roughly ten weeks. That gap is not visible in any health metric, because
the maintainer carried the repo manually in the meantime: v0.8.1 through v0.8.5
all shipped in October, including a security release (`6f88c4d`, "contain
automatic evidence reads"). The project got healthier while its automation was
dark.

Two things decayed specifically *because* nothing was watching:

1. **PR #57 degraded from `MERGEABLE` to `CONFLICTING`.** Last digest recorded
   it as green and mergeable at 26 days old. At 96 days it now needs a rebase
   before anyone can even evaluate it. The cost of the review backlog is no
   longer just latency — it is rework.
2. **The stale-branch list grew rather than shrank**: 10 → 13, and every branch
   on last digest's cleanup list is still live. Three of the new entries are
   maintainer digests from runs that happened and were never pruned.

### Corrections to this run's own subagents

Recorded rather than silently overwritten, in the same spirit as last digest's
correction of `repo-health`.

- **`feature-scout` flagged candidate #1 as `(multi-run)`. That is wrong.**
  Issue #64 was opened 2026-10-06; every prior digest (2026-07-06, 2026-08-10,
  2026-08-17) predates it. It cannot have appeared in a previous run. The flag
  is disregarded in the gate below — it did not change the outcome, but a
  fabricated "we've seen this before" flag could easily have blocked real work
  on a future run.
- **`repo-health` reported 14 untested `src/` files from filename matching
  alone.** A real `pytest --cov` run puts the project at 79% with a different
  and shorter list of weak spots (table above). The heuristic overstated the
  gap; the measured numbers are used here.

### Drift worth a maintainer's eyes

1. **`CLAUDE.md` is stale on two counts**: its header says "v0.8.0" (actual:
   0.8.5) and its Known Issues says "No open issues are currently tracked"
   (actual: #64 is open and referenced from the v0.8.2 release notes).
   **Not auto-fixed** — `CLAUDE.md` is a config file under this workspace's
   config-safety rule and needs explicit maintainer confirmation, which a
   scheduled non-interactive run cannot obtain. It is a two-line edit awaiting
   a yes.
2. **Local checkout cannot `git fetch` over SSH — fourth consecutive run.**
   The working tree read `version = "0.8.0"` and a `providers/cli/` layout
   predating v0.8.2 while `origin/main` was 16+ commits ahead. Worked around
   again via the `gh` credential helper. Any local-git-only tooling in this
   workspace silently reports stale state, and this run's first grep of the
   version-gating code returned nothing because of it.

---

## 2. Feature candidates (RICE)

Final = (R×I×C)/E, each dimension 1–5.

| # | Feature | Source | R | I | C | E | **Final** | Flags |
|---|---|---|---|---|---|---|---|---|
| 1 | **CLI provider version-range/override policy (close #64)** | Issue #64 | 4 | 4 | 4 | 4 | **16.0** | safety decision reserved by maintainer |
| 2 | Fix `CLAUDE.md` drift + CI doc-drift check | Health scan | 3 | 2 | 5 | 2 | **15.0** | recurring; blocked on config-safety confirmation |
| 3 | Raise coverage on `_subprocess.py`, `summarize.py`, `openai.py`, `cli/main.py`, `vertex.py` | Measured coverage | 3 | 2 | 4 | 3 | **8.0** | multi-run |
| 4 | Local observability + run viewer | ROADMAP + competitor scan | 4 | 4 | 2 | 5 | **6.4** | needs design; new deps |
| 5 | Adaptive cost/quality routing | ROADMAP | 4 | 4 | 2 | 5 | **6.4** | needs design |
| 6 | Policy engine + human-in-the-loop approvals | ROADMAP | 3 | 3 | 2 | 4 | **4.5** | needs design |
| 7 | Triage stale PR #57 (Sakana/Fugu, now conflicting) | PR #57 | 2 | 2 | 2 | 4 | **2.0** | not buildable by this run |

**Threshold calibration — overdue.** The task template sets `THRESHOLD = 8` with
a note to recalibrate after three runs; four-plus runs have now passed. The
template assumes a 1–10 score, but every run has scored on `(R×I×C)/E`, which
ranges 2–30 (last digest's top item scored 30.0). On that scale 8 is a low bar —
it admits candidate #3, routine test-coverage work, as "high value." Recommend
either normalising Final to 1–10 or raising the threshold to ~12 so the gate
selects for genuinely high-value work rather than merely non-trivial work.

---

## 3. Gate decision: nothing built

Candidate #1 scores 16.0, clears the threshold, and — once the bogus
`(multi-run)` flag is discarded — fails the gate on the third condition: it is
**not a self-contained change completable cleanly in one run.**

The reason is specific, not procedural caution. Issue #64's remaining scope *is*
the safety control:

- `providers/cli/_compatibility.py` gates on an exact-string allowlist and fails
  closed. Its module docstring reads *"Admit only native releases covered by the
  synthetic contract matrix"*, and `check_verified_version`'s own docstring
  states *"Version/help compatibility cannot prove tool or instruction
  isolation."* The strictness is the design, not an oversight.
- The maintainer said so explicitly on the issue on 2026-10-06: *"A compatible
  range or an unverified-version override would need a separate safety decision
  and evidence that native isolation and completion contracts remain intact.
  This release does not claim arbitrary forward compatibility."*

The buildable core of #64 is therefore exactly the part a human reserved. An
autonomous run shipping `allow_unverified_cli_version` would be weakening a
security boundary on the strength of a version string — the precise inference
the code says is invalid.

### Implementation plan (for maintainer decision, not immediate build)

The work splits into a safe half and a reserved half, and only the first half
should move without a decision. **Safe half:** widen `_VERIFIED_VERSIONS` to a
parsed `(name, version)` comparison instead of exact strings so that
`codex-cli 0.160.1` and `2.1.292` are matched structurally, keep failing closed
on anything outside the tested set, and have `council doctor` print the observed
version, the selected executable, and the tested set — turning today's opaque
"unsupported or unverified version" into an actionable message that names what
would need testing. This is mechanical, testable against the existing contract
matrix, and changes no policy. **Reserved half (needs the maintainer's call
first):** an `allow_unverified_cli_version` escape hatch. Before it is built,
three questions need answers only the maintainer has — does an override default
to warn-and-proceed or refuse-unless-set; must it be re-affirmed per CLI version
or set once; and what evidence (which subset of the native isolation and
completion contract tests, run against an untested CLI build) is sufficient to
let an unverified version through. Until those are settled, implementing the
override means inventing a security policy, which is out of scope for this run.
The honest smallest step is the safe half plus a doctor message that tells users
how to request coverage for a new version — that removes most of the reported
pain (opaque hard failure after every auto-update) without claiming forward
compatibility the project cannot back.

---

## 4. Stale `claude/` branch cleanup list

13 branches with no commits in >30d, verified against the GitHub branches API.
All 10 from last digest's list are still live.

| Branch | Age |
|---|---|
| `claude/weekly-maintainer-2026-06-01` | 147d |
| `claude/feature-markdown-output-2026-06-01` | 147d |
| `claude/weekly-maintainer-2026-06-08` | 140d |
| `claude/weekly-maintainer-2026-06-15` | 133d |
| `claude/feature-engine-tests-2026-06-22` | 126d |
| `claude/weekly-maintainer-2026-06-22` | 126d |
| `claude/feature-registry-tests-2026-06-29` | 119d |
| `claude/weekly-maintainer-2026-06-29` | 119d |
| `claude/feature-resilience-provider-tests-2026-07-06` | 112d |
| `claude/weekly-maintainer-2026-07-06` | 112d |
| `claude/feature-cli-reasoning-effort-2026-08-10` | 77d |
| `claude/weekly-maintainer-2026-08-10` | 77d |
| `claude/weekly-maintainer-2026-08-17` | 70d |

Digest branches whose contents are already merged or superseded are safe to
delete. The `feature-*` branches should be checked for unmerged work first —
`claude/feature-cli-reasoning-effort-2026-08-10` in particular may overlap the
reasoning-effort handling that shipped in 0.8.x.

---

## 5. Maintainer action list

1. **Decide #64's reserved half** — the three questions in §3. Unblocks the
   highest-value item in the backlog.
2. **Approve the two-line `CLAUDE.md` drift fix** (version 0.8.0 → 0.8.5,
   drop the false "no open issues" claim).
3. **Rebase or close PR #57** — it is conflicting and 96 days old; its 5k-line
   diff is getting more expensive to review every week.
4. **Merge or close #59** (dependabot, green, 93 days).
5. **Delete the 13 stale branches** in §4.
6. **Fix the local SSH fetch** — flagged four runs running; it silently feeds
   stale state to any local-git-only tooling.
7. **Recalibrate `THRESHOLD`** per §2.

---

## Executive summary (3 lines)

- **Built:** nothing.
- **Planned:** issue #64's CLI version-gating policy — split into a safe
  structural-version-match half and a reserved override half (§3).
- **Nothing this week — because** the top candidate's buildable core is a
  deliberate fail-closed security gate the maintainer explicitly reserved for a
  separate safety decision, and the runner-up is a `CLAUDE.md` edit that needs
  config-safety confirmation this unattended run cannot obtain.
