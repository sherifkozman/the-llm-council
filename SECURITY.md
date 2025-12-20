# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Security Features

The LLM Council implements several security measures:

### CLI Adapter Security
- **No shell injection**: All subprocess calls use exec-style argument lists, not shell strings
- **Environment allowlisting**: Only explicitly allowed environment variables are passed to subprocesses
- **Timeout enforcement**: All external calls have configurable timeouts

### Secret Protection
- **Redaction pipeline**: Sensitive values are redacted from logs and error messages
- **Minimal environment exposure**: CLI adapters only pass necessary environment variables
- **No credential logging**: API keys are never logged

### Storage Security
- **Path traversal protection**: Artifact storage validates paths to prevent directory escape
- **Local-only storage**: Artifacts are stored locally, not transmitted externally

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public issue
2. Email security concerns to the maintainers directly
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will:
- Acknowledge receipt within 48 hours
- Provide an initial assessment within 7 days
- Work with you to understand and resolve the issue
- Credit you in the fix (unless you prefer anonymity)

## Security Best Practices for Users

### API Key Management
```bash
# Use environment variables, not command line arguments
export OPENROUTER_API_KEY="your-key"  # Good
council run impl "task" --api-key="key"  # Bad (visible in process list)
```

### Artifact Storage
- Review artifact contents before sharing
- Consider disabling artifacts for sensitive tasks: `--no-artifacts`
- Artifacts are stored in `~/.local/share/llm-council/artifacts/`

### CLI Provider Usage
- CLI providers (codex-cli, gemini-cli) run external tools
- Review the default sandbox/approval modes
- Use read-only sandbox for untrusted inputs

## Known Security Considerations

1. **LLM Output**: Model outputs are not sanitized and may contain harmful content
2. **Provider Trust**: You must trust your configured providers with your prompts
3. **Artifact Persistence**: Drafts and outputs are stored locally by default
