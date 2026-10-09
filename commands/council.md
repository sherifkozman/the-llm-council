---
name: council
description: Run a multi-LLM council task with adversarial debate
arguments:
  - name: subagent
    description: "Subagent type: drafter, critic, planner, researcher, router, synthesizer"
    required: true
  - name: task
    description: Task description in quotes
    required: true
---

Run the LLM Council with the specified subagent and task.

> **Requires:** Council 0.8.4 or later and usable credentials for each selected provider.

## Subagents
- `drafter --mode impl|arch|test` - Implementation, architecture, tests
- `critic --mode review|security` - Review and security analysis
- `planner --mode plan|assess` - Plans and assessments
- `researcher` - Technical research
- `synthesizer` - Merge findings
- `router` - Task classification

## Execution

Read [the portable invocation contract](../skills/council/references/invocation-contract.md)
from the installed plugin before calling Council. Resolve the intended executable,
verify its version, preserve the user's providers/models, and pass the task through
a UTF-8 input file or stdin. Use an explicit CWD and a unique JSON output file.
Wait on the original process handle, then inspect the complete result and
`execution_status` even on nonzero exit; `success: true` alone is insufficient.
A `request_changes` verdict is not an execution failure; degraded coverage is
not a full review. Never automatically
retry a live or valid degraded run, invoke a personal wrapper, or bypass a
client's credential/approval policy.

The built-in reviewer distinguishes prepared-coverage metadata from source evidence;
the added prompt boundary leaves security/custom schemas unchanged. Exhausted
schema validation fails with drafts, critique and errors preserved: arbitrary
prose is not converted into reviewer findings. Invalid raw synthesis is artifact-only
if storage succeeds. Exception-triggered schema-valid JSON draft fallback remains
degraded and is not evidence of semantic truth or approval.

File limits count characters, not KB: 50,000 per file and 200,000 retained in
total. Inspect `execution_plan.context_preparation.files` and
`degradation_report.context_warnings`; truncated input is not a full-file review.
`--timeout` accepts 10-3600 seconds per attempt (default 120), not per workflow.
Use `--runtime-profile default` for long attempt deadlines and set the caller's
outer deadline separately; `bounded` retains shorter phase caps. Doctor
reachability is not proof that a large review completed.

Ingestion counts are not delivered coverage. Inspect
`execution_plan.context_preparation.coverage` and
`execution_plan.phase_prompt_compaction`,
including profile 0; source selection and draft cuts are pipeline limitations,
not source defects. Only `submitted` evidence cuts count toward compaction-based
degradation. Read `degradation_report.total_retries` separately from
`planned_retries`, the top-level `synthesis_attempts`, completed evidence on
cancellation, `execution_plan.artifact_occurrences` and `execution_plan.persistence`.
Synthesis attempts count adapter starts, not queued requests. The contract covers
CLI/library runtime identity and `started`/`settled` execution manifests; the
settled snapshot precedes ledger finalization, not a terminal-ledger guarantee.
A hard-killed run's `running` ledger row does not prove liveness;
never automatically repair it or restart a paid run.

Preserve official slash/colon model IDs, but reject `provider:model` mappings.
Distinguish selected executable/version from requested, not confirmed, model
identity. Follow the contract for an approved exact-version uv tool reinstall;
an isolated environment does not update the real caller's executable or pin.
Do not delete native installations or change PATH to hide selection mismatches.

Claude CLI admission checks required capabilities on every call, not exact build
numbers. Report missing controls from doctor; never strip isolation flags to
proceed. Help advertises syntax, not authentication or behavioral compatibility.
Codex still uses the exact version allowlist in the invocation contract.
Codex 0.160.1 requires an explicit supported model: the unchanged `gpt-5.4`
default is absent from its tested catalog. Its tool settings are version-specific,
not universal model support or permission to change authentication/model/effort.
Unknown Codex versions remain unsupported. Report doctor failure diagnostics with the
chosen executable path and version; do not edit package constants or assume
future builds are behaviorally verified merely because help flags still exist.
