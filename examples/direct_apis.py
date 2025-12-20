#!/usr/bin/env python3
"""
Direct API Provider Examples

This example shows how to use llm-council with direct API providers
instead of OpenRouter. This is useful when you want to:
- Use specific provider features (e.g., Claude's tool use)
- Have existing API credits with providers
- Need guaranteed access to specific models
- Want to optimize costs with specific providers

Supported Direct Providers:
- Anthropic (Claude models)
- OpenAI (GPT models)
- Google (Gemini models)

Prerequisites:
    # Set one or more provider API keys:
    export ANTHROPIC_API_KEY="sk-ant-..."
    export OPENAI_API_KEY="sk-..."
    export GOOGLE_API_KEY="..."

Usage:
    python examples/direct_apis.py
"""

import asyncio
import os

from llm_council import Council, OrchestratorConfig


async def single_provider_example():
    """Using a single direct API provider."""

    print("=" * 70)
    print("Single Direct Provider Example")
    print("=" * 70)

    # Example 1: Anthropic only
    if os.getenv("ANTHROPIC_API_KEY"):
        print("\n[Example 1] Using Anthropic Claude")
        print("-" * 70)

        council = Council(providers=["anthropic"])

        # Check provider health
        health = await council.doctor()
        if health.get("anthropic", {}).get("ok"):
            print("✓ Anthropic API connected")

            result = await council.run(
                task="Create a Python function to parse JSON with error handling",
                subagent="implementer"
            )

            if result.success:
                print(f"✓ Implementation complete: {result.output.get('implementation_title', 'N/A')}")
                print(f"  Duration: {result.duration_ms}ms")
                print(f"  Files: {len(result.output.get('files', []))}")

                # Anthropic-specific: Check which model was used
                if result.cost_estimate:
                    print(f"  Tokens: {result.cost_estimate.tokens}")
            else:
                print(f"✗ Failed: {result.error}")
        else:
            print("✗ Anthropic API not configured")
            print("  Set ANTHROPIC_API_KEY to use Claude models")

    # Example 2: OpenAI only
    if os.getenv("OPENAI_API_KEY"):
        print("\n[Example 2] Using OpenAI GPT")
        print("-" * 70)

        council = Council(providers=["openai"])

        health = await council.doctor()
        if health.get("openai", {}).get("ok"):
            print("✓ OpenAI API connected")

            result = await council.run(
                task="Design a caching strategy for a high-traffic API",
                subagent="architect"
            )

            if result.success:
                print(f"✓ Architecture complete: {result.output.get('design_title', 'N/A')}")
                print(f"  Duration: {result.duration_ms}ms")
            else:
                print(f"✗ Failed: {result.error}")
        else:
            print("✗ OpenAI API not configured")
            print("  Set OPENAI_API_KEY to use GPT models")

    # Example 3: Google Gemini only
    if os.getenv("GOOGLE_API_KEY"):
        print("\n[Example 3] Using Google Gemini")
        print("-" * 70)

        council = Council(providers=["google"])

        health = await council.doctor()
        if health.get("google", {}).get("ok"):
            print("✓ Google API connected")

            result = await council.run(
                task="Research best practices for API rate limiting",
                subagent="researcher"
            )

            if result.success:
                print(f"✓ Research complete: {result.output.get('research_title', 'N/A')}")
                print(f"  Findings: {len(result.output.get('findings', []))}")
            else:
                print(f"✗ Failed: {result.error}")
        else:
            print("✗ Google API not configured")
            print("  Set GOOGLE_API_KEY to use Gemini models")


async def multi_provider_example():
    """Using multiple direct API providers for adversarial council."""

    print("\n\n" + "=" * 70)
    print("Multi-Provider Council (Adversarial Debate)")
    print("=" * 70)

    available_providers = []
    if os.getenv("ANTHROPIC_API_KEY"):
        available_providers.append("anthropic")
    if os.getenv("OPENAI_API_KEY"):
        available_providers.append("openai")
    if os.getenv("GOOGLE_API_KEY"):
        available_providers.append("google")

    if len(available_providers) < 2:
        print("\n✗ Multi-provider council requires at least 2 providers")
        print(f"  Currently configured: {len(available_providers)}")
        print("\nTo enable multi-provider mode, set API keys for 2+ providers:")
        print("  export ANTHROPIC_API_KEY='...'")
        print("  export OPENAI_API_KEY='...'")
        print("  export GOOGLE_API_KEY='...'")
        return

    print(f"\n✓ Using {len(available_providers)} providers: {', '.join(available_providers)}")
    print("\nMulti-provider councils enable:")
    print("  - Adversarial debate between different models")
    print("  - Cross-validation of outputs")
    print("  - More robust decision-making")
    print("  - Diverse perspectives on problems")

    # Create council with multiple providers
    council = Council(providers=available_providers)

    print("\n[Example] Implementing a feature with multi-provider debate")
    print("-" * 70)

    result = await council.run(
        task="Implement a rate limiter with Redis, handling edge cases and errors",
        subagent="implementer"
    )

    if result.success:
        print("✓ Multi-provider council completed!")

        print(f"\nResult: {result.output.get('implementation_title', 'N/A')}")
        print(f"Duration: {result.duration_ms}ms")
        print(f"Synthesis attempts: {result.synthesis_attempts}")

        # Show which providers contributed
        if result.drafts:
            print(f"\nDrafts from {len(result.drafts)} providers:")
            for provider in result.drafts.keys():
                print(f"  - {provider}")

        # Show phase timings
        if result.phase_timings:
            print("\nPhase breakdown:")
            for timing in result.phase_timings:
                print(f"  {timing.phase:12} {timing.duration_ms:6}ms")

        # Show cost estimate
        if result.cost_estimate:
            cost = result.cost_estimate
            print(f"\nCost estimate:")
            print(f"  Provider calls: {cost.provider_calls}")
            print(f"  Total tokens: {cost.tokens:,}")
            print(f"  Estimated cost: ${cost.estimated_cost_usd:.4f}")

    else:
        print(f"✗ Council failed: {result.error}")


async def model_override_example():
    """Customize which models are used for each provider."""

    print("\n\n" + "=" * 70)
    print("Model Override Example")
    print("=" * 70)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n✗ This example requires ANTHROPIC_API_KEY")
        return

    print("\nYou can override the default models for each provider:")

    # Configure specific models
    config = OrchestratorConfig(
        timeout=120,
        max_retries=3,
        model_overrides={
            "anthropic": "claude-3-opus-20240229",  # Use Opus instead of Sonnet
            "openai": "gpt-4-turbo-preview",
            "google": "gemini-pro",
        }
    )

    council = Council(providers=["anthropic"], config=config)

    print("\nModel overrides configured:")
    print(f"  Anthropic: claude-3-opus-20240229 (highest quality)")
    print(f"  OpenAI: gpt-4-turbo-preview")
    print(f"  Google: gemini-pro")

    result = await council.run(
        task="Write a complex authentication system with JWT and refresh tokens",
        subagent="implementer"
    )

    if result.success:
        print(f"\n✓ Implementation with custom model: {result.output.get('implementation_title', 'N/A')}")
        print(f"  Duration: {result.duration_ms}ms")
    else:
        print(f"✗ Failed: {result.error}")


async def cost_optimization_example():
    """Configure custom cost tracking for direct providers."""

    print("\n\n" + "=" * 70)
    print("Cost Optimization Example")
    print("=" * 70)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n✗ This example requires ANTHROPIC_API_KEY")
        return

    print("\nYou can track costs with custom pricing for each provider:")

    # Configure with cost tracking
    # Prices are per 1K tokens (example rates, check actual pricing)
    config = OrchestratorConfig(
        cost_per_1k_input={
            "anthropic": 0.015,  # Claude Opus input
            "openai": 0.01,      # GPT-4 Turbo input
            "google": 0.001,     # Gemini Pro input
        },
        cost_per_1k_output={
            "anthropic": 0.075,  # Claude Opus output
            "openai": 0.03,      # GPT-4 Turbo output
            "google": 0.002,     # Gemini Pro output
        }
    )

    council = Council(providers=["anthropic"], config=config)

    print("\nCost tracking configured:")
    print("  Anthropic: $0.015 / 1K input, $0.075 / 1K output")
    print("  OpenAI:    $0.010 / 1K input, $0.030 / 1K output")
    print("  Google:    $0.001 / 1K input, $0.002 / 1K output")

    result = await council.run(
        task="Create a simple TODO list API with CRUD operations",
        subagent="implementer"
    )

    if result.success and result.cost_estimate:
        cost = result.cost_estimate
        print(f"\n✓ Cost tracking results:")
        print(f"  Input tokens:  {cost.total_input_tokens:,}")
        print(f"  Output tokens: {cost.total_output_tokens:,}")
        print(f"  Total tokens:  {cost.tokens:,}")
        print(f"  Estimated cost: ${cost.estimated_cost_usd:.4f}")
        print(f"\n  Provider breakdown: {cost.provider_calls}")


async def provider_comparison():
    """Compare the same task across different providers."""

    print("\n\n" + "=" * 70)
    print("Provider Comparison")
    print("=" * 70)

    task = "Implement a binary search algorithm in Python with type hints"

    results = {}

    # Test each provider individually
    for provider_name, env_var in [
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("google", "GOOGLE_API_KEY"),
    ]:
        if not os.getenv(env_var):
            continue

        print(f"\n[Testing {provider_name}]")
        print("-" * 70)

        council = Council(providers=[provider_name])
        result = await council.run(task=task, subagent="implementer")

        if result.success:
            results[provider_name] = {
                "success": True,
                "duration": result.duration_ms,
                "tokens": result.cost_estimate.tokens if result.cost_estimate else 0,
                "attempts": result.synthesis_attempts,
            }
            print(f"✓ Success in {result.duration_ms}ms")
        else:
            results[provider_name] = {"success": False, "error": result.error}
            print(f"✗ Failed: {result.error}")

    # Show comparison
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("Comparison Summary")
        print("=" * 70)

        print(f"\n{'Provider':<12} {'Status':<10} {'Duration':<12} {'Tokens':<10}")
        print("-" * 70)

        for provider_name, data in results.items():
            status = "✓ Success" if data.get("success") else "✗ Failed"
            duration = f"{data.get('duration', 0)}ms" if data.get("success") else "N/A"
            tokens = str(data.get("tokens", 0)) if data.get("success") else "N/A"

            print(f"{provider_name:<12} {status:<10} {duration:<12} {tokens:<10}")


async def main():
    """Run all direct API examples."""

    # Check which providers are configured
    configured = []
    if os.getenv("ANTHROPIC_API_KEY"):
        configured.append("Anthropic")
    if os.getenv("OPENAI_API_KEY"):
        configured.append("OpenAI")
    if os.getenv("GOOGLE_API_KEY"):
        configured.append("Google")

    print("=" * 70)
    print("LLM Council - Direct API Providers")
    print("=" * 70)
    print(f"\nConfigured providers: {', '.join(configured) if configured else 'None'}")

    if not configured:
        print("\n✗ No provider API keys found")
        print("\nTo use direct API providers, set one or more:")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  export GOOGLE_API_KEY='...'")
        print("\nOr use OpenRouter for easier setup:")
        print("  export OPENROUTER_API_KEY='sk-or-v1-...'")
        print("  See: examples/openrouter_only.py")
        return

    # Run examples
    await single_provider_example()
    await multi_provider_example()
    await model_override_example()
    await cost_optimization_example()
    await provider_comparison()

    print("\n" + "=" * 70)
    print("Direct API Examples Complete!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  - Single provider: Simple, uses one model")
    print("  - Multi-provider: Adversarial debate, more robust")
    print("  - Model overrides: Control which models are used")
    print("  - Cost tracking: Monitor token usage and costs")
    print("\nWhen to use direct APIs vs OpenRouter:")
    print("  Direct APIs:")
    print("    - You have existing credits with providers")
    print("    - Need specific provider features")
    print("    - Want to optimize costs per provider")
    print("  OpenRouter:")
    print("    - Easiest setup (one API key)")
    print("    - Access to 100+ models")
    print("    - Automatic failover")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
