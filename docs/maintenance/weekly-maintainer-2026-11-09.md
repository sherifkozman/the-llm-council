# Weekly Maintainer Digest — 2026-11-09

Baseline: `origin/main` @ `9923cf9` (v0.8.10 + Codex 0.167.0 Homebrew
qualification). Previous digest: 2026-11-02 (branch
`claude/weekly-maintainer-2026-11-02`, never merged to `main`).

---

## Executive summary (3 lines)

- **Planned, not built:** the top candidate is a configurable `base_url` for the
  OpenAI provider (Ollama, vLLM, LM Studio, LiteLLM), with a RICE score of 18.0. It
  cleared the score bar but failed the "self-contained" condition: it opens a new
  outbound data path and changes provider-identity semantics. Both need maintainer
  decisions (plan in §3).
- **Superseded:** the maintainer's #80 already refreshed ROADMAP.md and CLAUDE.md
  to v0.8.10. Last week's `claude/feature-roadmap-current-release-2026-11-02`
  branch is therefore redundant and can be deleted.
- **Health is green on CI, security and releases.** Branch hygiene has not
  improved: the 13 stale `claude/` branches have gone through four digests without
  being pruned.

---

## 1. Repo health — `HEALTH: attention-needed`

| Area | Status |
|---|---|
| CI | `CI` and `Publish to PyPI` both **succeeded** on `main` (`9923cf9`, 2026-11-08). No failures in the last 6 runs |
| Security | Dependabot **0 open alerts**. Branch `dependabot/uv/urllib3-2.8.0` exists with no PR |
| Releases | **v0.8.9 and v0.8.10** both shipped 2026-11-08. pyproject, PyPI and the latest tag all read 0.8.10. `main` is 1 commit ahead of the tag (#82) |
| Open PRs | 3, unchanged: **#57** Sakana/Fugu (110d, `CONFLICTING`), **#59** setup-python 7 (107d), **#76** setup-node 7 (9d) |
| Open issues | 1: **#64** (CLI version pinning), unchanged |
| Branches | 38 in total. **16 `claude/`**, of which 13 are older than 30 days. 7+ old `codex/*` release branches look merged or superseded (not verified) |
| Coverage / doc drift | **Not measured this run.** The last measurement (79%) was on v0.8.5 |

### What changed since 2026-11-02

- **Two releases shipped:** 0.8.9 (provider compatibility) and 0.8.10. Codex
  0.167.0 Homebrew on macOS ARM64 was also qualified.
- **Drift fixes landed upstream.** I checked `origin/main` directly:
  - `ROADMAP.md:7` now reads `## Current Release: v0.8.10`.
  - The `CLAUDE.md` header now reads v0.8.10.
  - So items 1 and 6 of last week's action list are resolved by the maintainer,
    and `claude/feature-roadmap-current-release-2026-11-02` is superseded.
- **Local SSH fetch is still broken, sixth consecutive run.** The checkout was 5
  commits behind until it was fetched over gh HTTPS.

---

## 2. Feature candidates (RICE)

Final = (R×I×C)/E, each dimension 1–5, so the range is 0.2–125 and the practical
range is about 2–30. This week the scout capped chores at I ≤ 2 and excluded them
from TOP. That was the fix recommended last week, applied in the scout prompt
rather than in the task config.

| # | Feature | Source | R | I | C | E | **Final** | Flags |
|---|---|---|---|---|---|---|---|---|
| 1 | **Configurable `base_url` for the OpenAI provider** (OpenAI-compatible endpoints) | Competitor scan, 11-02 digest #3, gap verified in `providers/openai.py` | 3 | 4 | 3 | 2 | **18.0** | security/data-boundary decision |
| 2 | Opt-in anonymized, shuffled-label peer critique (`--blind-critique`) | karpathy/llm-council, 11-02 digest #2 | 4 | 3 | 3 | 3 | **12.0** | touches the very large `orchestrator.py` |
| 3 | #64 reserved half: unverified-version override plus a doctor warning | Issue #64 | 4 | 4 | 2 | 4 | **8.0** | security-policy decision reserved by the maintainer |
| 4 | Tests for `cli/_subprocess.py` (33%) and `storage/summarize.py` (50%) | 10-26 measured coverage | 2 | 2 | 4 | 2 | **8.0** | figures are stale (v0.8.5) |
| 5 | Local observability + run viewer | ROADMAP | 4 | 4 | 2 | 5 | **6.4** | multi-run (10-26, 11-02), needs design |
| 6 | Deterministic caching + replay | ROADMAP | 3 | 3 | 2 | 4 | **4.5** | needs design |
| — | Chore: prune stale branches, merge dependabot #59/#76 | Health | 2 | 2 | 5 | 1 | (20.0 raw) | chore, excluded; maintainer action |

**Scout verification and corrections:**
- **Flags are honest.** The only `multi-run` flag (#5) cites the 10-26 and 11-02
  digests, and both list it.
- **The gap in #1 is real but partly misdescribed.** `providers/openai.py` builds
  `AsyncOpenAI(api_key=...)` with no `base_url` parameter. However, the OpenAI
  SDK already honours the `OPENAI_BASE_URL` environment variable on its own.
  Users can already point the provider at another endpoint today:
  - The behaviour is undocumented.
  - Council does not report it: the execution plan would still call the backend
    "openai".
  - An API key is still required, even for local servers.
- **The competitor scan is thin.** It covered one project (karpathy/llm-council).

---

## 3. Gate decision: nothing built. Plan for #1

The gate requires Final ≥ 8, no `multi-run` flag, and a self-contained change.

- **Score:** 18.0 clears 8. It also clears the ~12 bar that would be equivalent
  after recalibration.
- **Flag:** none.
- **Self-contained: fails.** This is not a pure provider tweak:
  1. **Data boundary.** The change creates a supported way to send prompts and
     `--files` contents (possibly private code) to an arbitrary host. The repo's
     posture elsewhere is fail-closed: unknown Codex builds are rejected, and
     evidence reads are contained. Whether non-loopback endpoints are allowed is
     a policy call for the maintainer.
  2. **Identity invariant.** `execution_plan.provider_attempts` and the runtime
     identity rules depend on distinguishing requested, resolved and reported
     models. A run served by Ollama but recorded as provider `openai` would
     break that contract unless the endpoint is recorded. That is cross-cutting,
     and getting it wrong silently degrades audit evidence.
  3. Under the owner's rules, a security-surface plan needs a dual review (council
     plus Codex) and user approval before implementation. An unattended run
     can't get that approval.

**Implementation plan (for maintainer approval):** add an optional `base_url` to
`OpenAIProvider`, set via constructor, `providers[].base_url` in config, or the
already-honoured `OPENAI_BASE_URL`, and pass it explicitly to both
`AsyncOpenAI(...)` call sites (`openai.py:260` and the health check near `:498`).

- **Endpoint policy:**
  - Allow `localhost`, `127.0.0.1` and `::1` without further opt-in.
  - Require `allow_remote_base_url: true` for any other host.
- **API key:**
  - Make the key optional only when the endpoint is loopback; send a placeholder
    in that case.
  - Keep the existing missing-key error for api.openai.com.
- **Identity:**
  - Record the effective endpoint origin (scheme+host+port only, never a path or
    query) in runtime identity and `provider_attempts`.
  - Mark the run so it can't be mistaken for an OpenAI-hosted one.
- **Capabilities:**
  - Leave reasoning-effort and schema forwarding unchanged for known OpenAI models.
  - Treat unknown model IDs on a custom endpoint as having no o-series controls.
- **Doctor:** report the endpoint and whether it is loopback.
- **Tests:**
  - Loopback without a key → OK.
  - Remote without opt-in → configuration error.
  - Endpoint origin appears in the execution plan.
  - `OPENAI_BASE_URL` pass-through is explicit.
- **Docs:** README provider table, plus a release note.
- **Size:** about 4 files plus tests, roughly 150 LOC.
- **Review:** dual review before implementation.

Runner-up if the maintainer declines arbitrary endpoints: #2, opt-in blind
critique. It doesn't change the default behaviour and has no data-boundary
impact.

---

## 4. Stale branch cleanup list

**`claude/` branches older than 30 days.** There are 13, the same 13 as the last
three digests, and none has been merged:

| Branch | Age |
|---|---|
| `claude/feature-markdown-output-2026-06-01` | 161d (superseded: `--format markdown` shipped in 0.7.17) |
| `claude/weekly-maintainer-2026-06-01` | 161d |
| `claude/weekly-maintainer-2026-06-08` | 154d |
| `claude/weekly-maintainer-2026-06-15` | 147d |
| `claude/feature-engine-tests-2026-06-22` | 140d (unmerged tests, review before deleting) |
| `claude/weekly-maintainer-2026-06-22` | 140d |
| `claude/feature-registry-tests-2026-06-29` | 133d (unmerged tests, review before deleting) |
| `claude/weekly-maintainer-2026-06-29` | 133d |
| `claude/feature-resilience-provider-tests-2026-07-06` | 126d (unmerged tests, review before deleting) |
| `claude/weekly-maintainer-2026-07-06` | 126d |
| `claude/feature-cli-reasoning-effort-2026-08-10` | 91d |
| `claude/weekly-maintainer-2026-08-10` | 91d |
| `claude/weekly-maintainer-2026-08-17` | 84d |

**Newly redundant:**
- `claude/feature-roadmap-current-release-2026-11-02` (7d), superseded by #80.
- `claude/weekly-maintainer-2026-10-26` (14d), a digest only.

**Old non-`claude/` branches:**
- `codex/feature/prompt-caching-providers-research` (prompt caching shipped in 0.7.18).
- `codex/council-client-repair`, `codex/release-0.8.2-issue-hardening`,
  `codex/release-0.8.3-reporting`, `codex/claude-capability-compat`,
  `codex/security-release-0.8.5`. Merge status was not verified.
- `dependabot/uv/urllib3-2.8.0` (no PR).

Suggestion: digests have accumulated on 16 unmerged branches. Either merge
digests into `docs/maintenance/` on `main`, or keep only the latest digest
branch.

---

## 5. Maintainer action list

1. **Decide the endpoint policy for #1** (§3): loopback-only, opt-in remote, or
   not at all. Then approve the plan or pick #2.
2. **Delete the stale branches** in §4. That is 15 `claude/` branches; check the
   three `*-tests` branches before deleting them.
3. **Batch-merge dependabot #59 and #76.** Look at the orphaned urllib3 branch.
4. **Rebase or close PR #57** (110d, conflicting).
5. **Recalibrate the gate** in the task config (still pending). Either normalise
   Final to 1–10 or add an `I ≥ 3` floor. This week the scout applied the floor
   through its prompt, which is fragile.
6. **Update #64** to reflect the fetched compatibility record (carried over).
7. **Fix local SSH fetch** (six runs in a row).

### Not checked this run
- Measured coverage.
- `pip`/`uv` audit.
- Merge status of the `codex/*` branches.
- Doc drift beyond ROADMAP.md and CLAUDE.md headers.
- Competitor scan beyond one project.
