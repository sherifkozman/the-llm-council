# Synthesizer

Primary finalization subagent.

Use it for:

- release notes
- changelogs
- final merged summaries
- deployment notes or migration guides

## Examples

```bash
council run synthesizer "Generate release notes for v2.0.0" --json
council run synthesizer "Summarize architecture review findings" --json
council run synthesizer --input findings.md --output release-notes.json
```
