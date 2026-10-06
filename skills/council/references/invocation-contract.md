# Portable Council Invocation Contract

Requires Council 0.8.2 or later for the safety fixes. This file travels with a copied Council skill;
it does not require a repository checkout or a client-specific wrapper.

## Before Calling

Resolve the intended Council executable in the actual caller environment. Check
its version before a paid call. During candidate testing, use the candidate's
absolute executable path and verify its installed source identity. A version
string alone does not prove that local fixes were installed.

Select providers and their model IDs explicitly. Keep model order aligned with
provider order. Use models verified for those accounts, not a model substituted
after a failure. For substantial review, use a mixed-provider run and inspect
every selected draft; an exit code is not evidence of independent participation.

Use an explicit working directory, UTF-8 task file (`--input`), absolute reference
paths (`--files`), and a new result path for every invocation. Alternatively send
the task to `--input -` on stdin and close stdin. Do not interpolate task or file
contents into shell code. Do not assume a TTY or interactive/login shell startup.

Supply credentials only through the caller's sanctioned configuration or inherited
environment. Never source a host secrets file to bypass a client's credential
policy. Do not print key values, token fragments, auth files, or full environments.
Run deep diagnostics in the same effective caller context, where permitted:

```text
<council-executable> doctor --deep --provider <selected-provider> --probe-timeout 30 --json
```

Repeat for each selected provider. A deep probe proves reachability only, not a
valid review, correct model, full phase coverage, or isolation. A sandbox denial,
missing child credential or expired outer tool timeout is not model-health proof.

Doctor JSON is not a run-result envelope. It contains a `providers` array;
`probe_ok` is inside each provider object, not at the JSON root. For the
one-provider command above, require exactly the selected provider's row and
check `diagnostic["providers"][0]["probe_ok"] is True`. A missing row, missing
field, skipped probe or false value is not a pass. Exit code 0 alone is not a
passing deep probe. Inspect the actual saved JSON rather than inventing a
top-level status field. Do not remove or overwrite earlier diagnostic/result
files to make a retry look like the first attempt; choose a new output directory.
An empty or whitespace-only generation is a failed probe with an `empty_response`
diagnostic. Doctor requests a non-streaming response; an unexpected stream is
rejected, not converted to a success message.
For Codex, base doctor `ok: true` reports login status only, not generation
readiness. Require a successful deep probe for reachability. Do not infer an
executable path or version from a healthy row that does not report those fields.

Claude managed-policy environments are not certified for generation-only use.
The adapter rejects known local/cached policy but does not detect every remote
organization policy. Native `claude --safe-mode doctor` can report the current
auth route's policy status; a failed, pending or skipped eligible fetch is not
proof of absence. Account, policy or native-version changes need fresh evidence.
Do not change credentials, endpoints or administrative controls to bypass policy.

### Native CLI Versions

Council uses an explicit native-version allowlist, not a patch-version range:

| Provider | Accepted native versions |
| --- | --- |
| `codex` | 0.149.1, 0.153.3 |
| `claude` | 2.1.288, 2.1.289, 2.1.290, 2.1.291, 2.1.292 |

These entries are backed by synthetic native-contract checks. They do not prove
live provider access, review quality, or compatibility with future auto-updates.
See the release notes for the recorded checks and final verification scope.
Unknown versions remain explicitly unsupported, even if their help output lists
the same flags. On an unsupported-version error, report the executable path and
observed version in doctor diagnostics from the actual caller environment. Do
not edit package version constants or bypass the check; use a verified version
or wait for a release that verifies the new one. This resolves the known
versions, not generic future native-CLI compatibility.

## Run And Wait

The examples below require these caller-provided environment values: COUNCIL_BIN,
COUNCIL_WORKDIR, COUNCIL_TASK_FILE, COUNCIL_REFERENCE_FILE, COUNCIL_PROVIDERS and
COUNCIL_MODELS. Paths are absolute; providers and models are comma-separated.
These are example inputs, not new Council configuration variables.

`--timeout` accepts 10-3600 seconds; the default remains 120. It is a per-attempt
budget, not an outer workflow deadline. Queue wait, CLI discovery, spawn, generation
and stream consumption share it. Retries receive a new attempt budget; phases,
backoff and bounded cleanup can make a whole run much longer. Set the outer tool
budget accordingly. The Python example uses 900 seconds as an example, not a
package-wide guarantee. On an outer timeout, send SIGTERM and allow cleanup before
forcing termination. A second signal, SIGKILL or host death can leave cleanup
incomplete, especially outside the owned POSIX process groups.

Use `--runtime-profile default` when a long attempt deadline is required.
`bounded` keeps its existing shorter provider/phase caps even with `--timeout 3600`;
inspect `execution_plan.provider_request_timeouts_by_provider`. Raising
the attempt limit neither extends the caller's deadline nor proves completion.

For Claude and Codex CLI providers, `max_tokens` is not a verified whole-generation
limit; inspect its dropped-option entry in request compilation. Native output may
exceed the requested value. Claude's per-request token control can cause internal
continuations, so it is not used as a hard total cap. Keep token limits separate
from deadlines. If a valid high-effort request exceeds its deadline, inspect the
captured diagnostics before choosing a larger explicit timeout and outer budget;
do not lower reasoning, swap models, or repeat the failed run automatically.

### Python Subprocess

```python
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

binary = os.environ["COUNCIL_BIN"]
version = subprocess.run([binary, "--version"], check=True, capture_output=True, text=True)
match = re.fullmatch(r"LLM Council v(\d+)\.(\d+)\.(\d+)", version.stdout.strip())
if not match or tuple(map(int, match.groups())) < (0, 8, 2):
    raise SystemExit("Council 0.8.2 or later is required")
result_file = Path(tempfile.mkdtemp(prefix="council-result-")) / "result.json"
argv = [binary, "run", "critic", "--mode", "review",
        "--providers", os.environ["COUNCIL_PROVIDERS"],
        "--models", os.environ["COUNCIL_MODELS"],
        "--reasoning-profile", "default", "--runtime-profile", "default",
        "--timeout", "120", "--input", os.environ["COUNCIL_TASK_FILE"],
        "--files", os.environ["COUNCIL_REFERENCE_FILE"],
        "--json", "--output", str(result_file)]
proc = subprocess.Popen(argv, cwd=os.environ["COUNCIL_WORKDIR"], stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
try:
    stdout, stderr = proc.communicate(timeout=900)
except subprocess.TimeoutExpired:
    proc.terminate()
    try:
        stdout, stderr = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate(timeout=2)
        raise RuntimeError("Outer deadline expired; cleanup was not confirmed")
if stderr:
    print(stderr, file=sys.stderr, end="")
if stdout or not result_file.is_file():
    raise RuntimeError("Missing result or unexpected stdout; do not reuse an old result")
payload = json.loads(result_file.read_text(encoding="utf-8"))
status = payload.get("execution_status")
expected = {"completed": (True, {0}), "degraded": (True, {0}),
            "failed": (False, {1}), "cancelled": (False, {130, 143})}
if status not in expected:
    raise RuntimeError("Missing or unsupported execution contract")
success, codes = expected[status]
if payload.get("success") is not success or proc.returncode not in codes:
    raise RuntimeError("Result and process outcome disagree")
print(json.dumps({"execution_status": status, "run_id": payload.get("run_id"),
                  "result_file": str(result_file)}))
raise SystemExit(proc.returncode)
```

### Shell With Errexit

This remains safe under `set -e`: it reads a valid failure result before returning
the process code. It does not retry. The surrounding client owns its timeout and
signal handling; use managed background execution when its foreground cap is too
short. The temporary result directory is retained for inspection.

```sh
set -eu
version=$("$COUNCIL_BIN" --version)
python3 -c 'import re,sys; m=re.fullmatch(r"LLM Council v(\d+)\.(\d+)\.(\d+)",sys.argv[1]); sys.exit(0 if m and tuple(map(int,m.groups())) >= (0,8,2) else "Council 0.8.2 or later is required")' "$version"
run_dir=$(mktemp -d)
result_file="$run_dir/result.json"
cd "$COUNCIL_WORKDIR"
code=0
"$COUNCIL_BIN" run critic --mode review \
  --providers "$COUNCIL_PROVIDERS" --models "$COUNCIL_MODELS" \
  --reasoning-profile default --runtime-profile default --timeout 120 \
  --input "$COUNCIL_TASK_FILE" --files "$COUNCIL_REFERENCE_FILE" \
  --json --output "$result_file" >"$run_dir/stdout.log" 2>"$run_dir/stderr.log" || code=$?
if [ -s "$run_dir/stdout.log" ] || [ ! -f "$result_file" ]; then
  printf '%s\n' "Missing result or unexpected stdout; inspect $run_dir" >&2
  exit 1
fi
python3 - "$result_file" "$code" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
status = payload.get("execution_status")
expected = {"completed": (True, {0}), "degraded": (True, {0}),
            "failed": (False, {1}), "cancelled": (False, {130, 143})}
if status not in expected:
    raise SystemExit("Missing or unsupported execution contract")
success, codes = expected[status]
if payload.get("success") is not success or int(sys.argv[2]) not in codes:
    raise SystemExit("Result and process outcome disagree")
print(json.dumps({"execution_status": status, "run_id": payload.get("run_id"),
                  "result_file": str(path)}))
PY
exit "$code"
```

## Interpret The Result

JSON run mode writes one result to stdout, with diagnostics on stderr. With
`--json --output`, stdout is empty and the full result is written atomically in
the destination directory. Read the file only after the process is terminal.
Atomic writing cannot guarantee an artifact after disk-full, permission, kill or
host failures. Use a unique path so an old artifact cannot be mistaken for this run.

| Execution status | Legacy success | CLI exit | Meaning |
| --- | --- | --- | --- |
| completed | true | 0 | Required execution completed without known lost coverage. |
| degraded | true | 0 | Valid output, but selected coverage, a required phase or input was lost, or fallback was used. |
| failed | false | 1 | Execution/input/output failure. Inspect the available result and stderr. |
| cancelled | false | 130 or 143 | Gracefully handled SIGINT or SIGTERM. Inspect cleanup diagnostics. |
| not_executed | not a result | 0 | JSON dry-run, with kind=plan. No provider-health or execution claim. |

Parser usage errors remain stderr plus exit 2 and need not produce JSON. A
`request_changes` review verdict is not an execution failure. Recovered retries
with restored required coverage are not automatically degraded. Missing explicit
reference files and files skipped entirely by the total cap fail before model
calls. Retained-but-truncated files make the execution degraded. The existing
50,000-character per-file and 200,000-retained-character total caps still apply.
They count characters, not KB or UTF-8 bytes. Inspect
`execution_plan.context_preparation.files` for `path`, `original_chars`,
`retained_chars`, and `truncated`. The corresponding
`degradation_report.context_warnings` entry includes `kind: "file_truncated"`,
path and original/retained counts, even with graceful degradation disabled.
Do not claim a full-file review when a file was truncated. State the retained
scope and review omitted material separately instead of treating exit 0 as proof.

New execution plans distinguish `requested_capabilities` (effective caller or
router requirements) from the default local-evidence collection profile.
A requested capability still listed in `pending_capabilities` makes execution
degraded. Default capabilities without collected evidence remain visible in that
list but do not, by themselves, change execution status. This applies to every
mode, including security and research: `completed` does not certify evidence
sufficiency. Plans without the new field retain conservative pending-capability
classification. Inspect the evidence requirements and pending diagnostics before
accepting a conclusion, even when all provider phases completed.

Before accepting a review, inspect the complete `drafts`, `provider_errors`,
`degradation_report`, `execution_plan`, schema-validation result and `run_id`.
Check actual providers/models, required phase participation, evidence retained,
and any context truncation/slicing/chunking. A valid execution does not prove the
model's conclusions. Do not accept fallback or missing drafts as full coverage.
Do not retry a valid degraded run automatically; assess the specific failure first.
For large reviews, synthesis retains compact draft evidence rather than silently
dropping every draft. No usable drafts, or no evidence-bearing synthesis prompt
that fits the budget, must not be accepted as a full review. Inspect the failure
or degraded fallback and `execution_plan.phase_prompt_compaction` before accepting
the result. Empty provider text is a non-retryable `empty_response` failure, not
a successful independent draft.
OpenRouter accepts final-answer content only. Reasoning-only content is not
promoted to a final answer, even if it contains schema-shaped JSON; a missing
final answer remains `empty_response`. Do not rely on the legacy promotion path.
If an OpenRouter chunk reports `empty_response` with `finish_reason=length`,
inspect its output budget; this is not by itself a model-health failure. An
explicit mitigation is `--runtime-profile default --timeout 300 --max-tokens 8000`,
keeping `--reasoning-profile default` and the configured model unchanged. These
are example overrides, not new defaults: the default chunk output budget remains
900 tokens, while an explicit `--max-tokens` override is honored. Larger budgets
can increase cost and do not guarantee success; do not retry automatically.
When critique compaction records `omit_schema: 1`, full output-schema details were
omitted from that critique. Inspect its warning; do not claim equivalent
schema-specific scrutiny. Synthesis's configured schema handling is unchanged.

For `router --route`, the returned child `run_id` is also the final workflow
handle; its ledger status reflects combined router/child coverage. The child's
original status is retained in `execution_plan.routed_execution_status`, and
`routing_execution_plan` retains the separate router ID, status and diagnostics.
Phase artifacts remain scoped to their own subagent. Cancellation retains this
provenance when the engine can settle and attach a result before the CLI deadline.

## Client Differences

- Codex: use the intended executable and explicit working directory. When an
  execution tool returns a live handle, wait on that handle. An observation
  timeout is not process completion. Do not restart the same paid run or weaken
  sandbox/approval settings to conceal a denial. Where active runtime policy
  permits, request required out-of-sandbox execution through the caller's normal
  approval mechanism. Distinguish untested escalation, explicit denial,
  unsupported escalation and missing sanctioned credentials. Do not alter
  policy or source secrets to bypass these limits.
- Claude Code: check which installed command or skill is active. Use this
  packaged contract, not a missing personal wrapper. Record its path/version
  during verification. Updating repository docs does not update a stale local
  slash command. Do not change owned local instructions without authorization.
  In headless `claude -p` sessions, do not issue a final response while Council
  is running: exiting the session can terminate its background Bash tasks.
  Prefer foreground execution (`run_in_background: false`). If the tool yields
  or automatically backgrounds the command, keep the session active with bounded
  foreground waits and completion checks for that same invocation, within the
  original outer deadline. Do not restart Council or finish with "waiting for
  notification". Inspect its terminal exit and complete result before responding.
- Hermes: use terminal plus its managed process wait/poll path and a result file.
  Displayed terminal output can be truncated or redacted. A wait timeout may mean
  still running. Its provider-key denylist is not fixed by generic environment
  passthrough. If sanctioned child credentials are unavailable, report that cell
  as blocked; do not use one-shot approval bypass or source host secrets.
- Generic callers and CI: require only subprocess execution and full-file JSON
  reading. Do not assume a plugin, login shell, TTY, secrets file or mandatory
  wrapper. Reject legacy payloads without execution_status explicitly.

Native CLI flags and authentication capabilities are version-dependent. Consult
the release notes for tested versions and exact live coverage. Unsupported native
effort is not controlled reasoning; omitting an effort flag does not prove off.
Windows cleanup covers the direct child only; POSIX containment covers the owned
group, not descendants that escape it. Never convert those limits into a model
health claim.
