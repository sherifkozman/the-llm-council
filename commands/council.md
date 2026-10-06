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

> **Requires:** Council 0.8.2 or later and usable credentials for each selected provider.

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
execution_status even on nonzero exit. A request_changes verdict is not an
execution failure; degraded coverage is not a full review. Never automatically
retry a live or valid degraded run, invoke a personal wrapper, or bypass a
client's credential/approval policy.

File limits count characters, not KB: 50,000 per file and 200,000 retained in
total. Inspect `execution_plan.context_preparation.files` and
`degradation_report.context_warnings`; truncated input is not a full-file review.
`--timeout` accepts 10-3600 seconds per attempt (default 120), not per workflow.
Use `--runtime-profile default` for long attempt deadlines and set the caller's
outer deadline separately; `bounded` retains shorter phase caps. Doctor
reachability is not proof that a large review completed.

Native CLI support uses the exact version allowlist in the invocation contract.
Unknown versions remain unsupported. Report doctor failure diagnostics with the
chosen executable path and version; do not edit package constants or assume
future auto-updates are compatible because help flags still exist.
