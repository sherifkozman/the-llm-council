"""
CLI-based provider adapters.

.. deprecated:: 0.5.1
    CLI providers are deprecated and will be removed in v1.0.
    Use direct API providers instead:
    - 'openai' for GPT models
    - 'anthropic' for Claude models
    - 'google' for Gemini models
    - 'vertex-ai' for Google Cloud (Gemini + Claude)
    - 'openrouter' for multi-provider routing

These adapters invoke external CLI tools (codex, gemini) via subprocess.
Use with caution in CI environments - prefer API-based providers for reliability.
"""
