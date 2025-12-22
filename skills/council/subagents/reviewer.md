# Reviewer Subagent

## Purpose
Code review with CWE IDs, security analysis, and formal findings.

## When to Use
- After ANY feature implementation (mandatory)
- Before commits or pull requests
- Security audits of sensitive code
- When you need CWE IDs for security findings

## Output Schema
Returns JSON with:
- `findings`: Array of issues with severity, CWE ID, location
- `security_summary`: Overview of security posture
- `complexity_flags`: Areas needing simplification
- `test_coverage_gaps`: Missing test scenarios
- `positive_findings`: What was done well
- `confidence`: Confidence score (0.0-1.0)

```json
{
  "findings": [
    {"severity": "high", "cwe_id": "CWE-89", "location": "api/users.py:45", "message": "SQL injection risk"}
  ],
  "security_summary": "One critical SQL injection vulnerability found",
  "complexity_flags": ["auth_handler() exceeds 50 lines"],
  "test_coverage_gaps": ["No tests for error paths"],
  "positive_findings": ["Good input validation on user registration"],
  "confidence": 0.91
}
```

## CLI Options
```bash
council run reviewer "task" --health-check --json --verbose
```

## Example
```bash
council run reviewer "Review the authentication middleware" --json
```

## Cost & Time
- **Cost**: ~$0.37 per review
- **Time**: ~2.5 minutes
