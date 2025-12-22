# Shipper Subagent

## Purpose
Release notes, changelogs, and deployment documentation from git history.

## When to Use
- Generating release notes for versions
- Creating changelogs from git history
- Documenting breaking changes
- Preparing deployment runbooks
- Creating migration guides

## Output Schema
Returns JSON with:
- `release_notes`: Formatted notes with features, fixes, breaking changes
- `version`: Release version
- `summary`: High-level overview
- `breaking_changes`: With migration guides
- `deployment_notes`: Prerequisites, steps, rollback plan
- `stakeholder_summary`: Executive/non-technical overview
- `confidence`: Confidence score (0.0-1.0)

## CLI Options
```bash
council run shipper "task" --health-check --json --verbose
```

## Example
```bash
council run shipper "Generate release notes for v2.0.0" --json
```

## Cost & Time
- **Cost**: ~$0.35 per release
- **Time**: ~3 minutes
