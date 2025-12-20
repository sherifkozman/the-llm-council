#!/usr/bin/env python3
"""
OpenRouter-Only Quickstart

This is the simplest way to get started with llm-council.
OpenRouter provides access to 100+ models through a single API key,
making it the recommended default for most users.

Why OpenRouter?
- Single API key for all models (Claude, GPT-4, Gemini, etc.)
- Automatic failover and routing
- Pay-as-you-go with competitive pricing
- No need to manage multiple provider accounts

Prerequisites:
    1. Sign up at https://openrouter.ai/
    2. Get your API key at https://openrouter.ai/keys
    3. Set environment variable:
       export OPENROUTER_API_KEY="sk-or-v1-..."

Usage:
    python examples/openrouter_only.py
"""

import asyncio
import os

from llm_council import Council


async def quickstart():
    """The simplest possible council usage."""

    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not set")
        print("\nTo get started:")
        print("  1. Visit https://openrouter.ai/keys")
        print("  2. Create an API key")
        print("  3. Run: export OPENROUTER_API_KEY='your-key-here'")
        return

    print("=" * 70)
    print("LLM Council - OpenRouter Quickstart")
    print("=" * 70)

    # That's it! Just create a council and run tasks
    council = Council(providers=["openrouter"])

    print("\n[Quickstart] Implementing a simple feature")
    print("-" * 70)

    result = await council.run(
        task="Create a Python function to calculate the Fibonacci sequence",
        subagent="implementer"
    )

    if result.success:
        print("✓ Success!")
        print(f"\nTitle: {result.output.get('implementation_title', 'N/A')}")
        print(f"Duration: {result.duration_ms}ms")

        # Show the implementation
        files = result.output.get('files', [])
        if files:
            print(f"\nGenerated {len(files)} file(s):")
            for file in files:
                print(f"  - {file.get('path', 'N/A')}: {file.get('description', 'N/A')}")
    else:
        print(f"✗ Failed: {result.error}")


async def common_tasks():
    """Examples of common tasks you can run with OpenRouter."""

    if not os.getenv("OPENROUTER_API_KEY"):
        return

    print("\n\n" + "=" * 70)
    print("Common Tasks with OpenRouter")
    print("=" * 70)

    council = Council(providers=["openrouter"])

    # Task 1: Code Implementation
    print("\n[Task 1] Code Implementation")
    print("-" * 70)

    result = await council.run(
        task="Build a REST API endpoint for user registration with email validation",
        subagent="implementer"
    )

    if result.success:
        print(f"✓ {result.output.get('implementation_title', 'N/A')}")
        print(f"  Files: {len(result.output.get('files', []))}")
        print(f"  Duration: {result.duration_ms}ms")

    # Task 2: Code Review
    print("\n[Task 2] Code Review")
    print("-" * 70)

    code = '''
def transfer_money(from_account, to_account, amount):
    from_account.balance -= amount
    to_account.balance += amount
    save_accounts(from_account, to_account)
'''

    result = await council.run(
        task=f"Review this code for bugs and security issues:\n\n{code}",
        subagent="reviewer"
    )

    if result.success:
        findings = result.output.get('findings', [])
        print(f"✓ Review complete: {len(findings)} findings")
        for finding in findings[:2]:  # Show first 2
            print(f"  - [{finding.get('severity', 'N/A')}] {finding.get('issue_type', 'N/A')}")

    # Task 3: Planning
    print("\n[Task 3] Feature Planning")
    print("-" * 70)

    result = await council.run(
        task="Plan the implementation of real-time chat with WebSocket",
        subagent="planner"
    )

    if result.success:
        phases = result.output.get('phases', [])
        print(f"✓ Plan created: {len(phases)} phases")
        for i, phase in enumerate(phases[:3], 1):
            print(f"  {i}. {phase.get('phase_name', 'N/A')}")

    # Task 4: Architecture Design
    print("\n[Task 4] Architecture Design")
    print("-" * 70)

    result = await council.run(
        task="Design a microservices architecture for an e-commerce platform",
        subagent="architect"
    )

    if result.success:
        services = result.output.get('services', [])
        print(f"✓ Architecture designed: {len(services)} services")
        for service in services[:3]:
            print(f"  - {service.get('name', 'N/A')}: {service.get('description', 'N/A')[:50]}...")


async def check_setup():
    """Verify OpenRouter setup is working."""

    print("\n\n" + "=" * 70)
    print("Checking OpenRouter Setup")
    print("=" * 70)

    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n✗ OPENROUTER_API_KEY not set")
        print("\nSetup instructions:")
        print("  1. Sign up at: https://openrouter.ai/")
        print("  2. Get API key: https://openrouter.ai/keys")
        print("  3. Set environment variable:")
        print("     export OPENROUTER_API_KEY='sk-or-v1-...'")
        return

    council = Council(providers=["openrouter"])

    print("\nRunning health check...")
    health = await council.doctor()

    for provider_name, status in health.items():
        ok = status.get('ok', False)
        message = status.get('message', 'Unknown')
        latency = status.get('latency_ms', 0)

        if ok:
            print(f"✓ {provider_name}: {message}")
            print(f"  Latency: {latency:.0f}ms")
        else:
            print(f"✗ {provider_name}: {message}")

    if all(s.get('ok', False) for s in health.values()):
        print("\n✓ OpenRouter is configured correctly and ready to use!")
    else:
        print("\n✗ There are configuration issues. Check your API key.")


async def cost_example():
    """Show cost tracking features."""

    if not os.getenv("OPENROUTER_API_KEY"):
        return

    print("\n\n" + "=" * 70)
    print("Cost Tracking with OpenRouter")
    print("=" * 70)

    council = Council(providers=["openrouter"])

    result = await council.run(
        task="Write a simple hello world function in Python",
        subagent="implementer"
    )

    if result.success and result.cost_estimate:
        cost = result.cost_estimate

        print("\nCost Breakdown:")
        print(f"  Provider calls: {cost.provider_calls}")
        print(f"  Input tokens:  {cost.total_input_tokens:,}")
        print(f"  Output tokens: {cost.total_output_tokens:,}")
        print(f"  Total tokens:  {cost.tokens:,}")
        print(f"  Estimated cost: ${cost.estimated_cost_usd:.4f}")

        print("\nNote: Actual costs shown in OpenRouter dashboard")
        print("      https://openrouter.ai/activity")


async def main():
    """Run all OpenRouter examples."""

    # 1. Check setup first
    await check_setup()

    # Only continue if API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        return

    # 2. Run quickstart
    await quickstart()

    # 3. Show common tasks
    await common_tasks()

    # 4. Show cost tracking
    await cost_example()

    print("\n" + "=" * 70)
    print("OpenRouter Quickstart Complete!")
    print("=" * 70)
    print("\nWhat's next?")
    print("  - Explore other subagents: researcher, assessor, red-team, etc.")
    print("  - Try custom configurations with OrchestratorConfig")
    print("  - Check out examples/basic_council.py for more features")
    print("  - Read docs at: docs/")
    print("\nResources:")
    print("  - OpenRouter docs: https://openrouter.ai/docs")
    print("  - Available models: https://openrouter.ai/models")
    print("  - Usage dashboard: https://openrouter.ai/activity")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
