"""
CLI-based provider adapters.

These adapters invoke external CLI tools (claude, codex, gemini-cli) via subprocess.
Useful for agent-to-agent delegation, environments where CLI auth is available
but API keys may not be, or when leveraging CLI-specific features.

For direct API access with full feature support (streaming, structured output),
use the API-based providers: 'openai', 'anthropic', 'gemini', 'vertex-ai', 'openrouter'.
"""
