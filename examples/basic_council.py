#!/usr/bin/env python3
"""
Basic Council Usage Example

This example demonstrates the core functionality of llm-council:
- Creating a Council instance with OpenRouter
- Running different subagents
- Accessing structured output
- Checking provider health

Prerequisites:
    export OPENROUTER_API_KEY="your-key-here"

Usage:
    python examples/basic_council.py
"""

import asyncio
import json
import os

from llm_council import Council, OrchestratorConfig


async def main():
    """Run basic council examples."""

    # Ensure API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Get your key at: https://openrouter.ai/keys")
        return

    print("=" * 70)
    print("LLM Council - Basic Usage Example")
    print("=" * 70)

    # Example 1: Simple council with default configuration
    print("\n[Example 1] Simple council run with default settings")
    print("-" * 70)

    council = Council(providers=["openrouter"])
    result = await council.run(
        task="Create a simple Python function that validates email addresses using regex",
        subagent="implementer"
    )

    if result.success:
        print("✓ Council completed successfully!")
        print(f"\nImplementation Summary:")
        print(f"  Title: {result.output.get('implementation_title', 'N/A')}")
        print(f"  Type: {result.output.get('implementation_type', 'N/A')}")
        print(f"  Files: {len(result.output.get('files', []))} files")
        print(f"\nDuration: {result.duration_ms}ms")
        print(f"Synthesis attempts: {result.synthesis_attempts}")

        # Show cost estimate
        if result.cost_estimate:
            cost = result.cost_estimate
            print(f"\nCost Estimate:")
            print(f"  Provider calls: {cost.provider_calls}")
            print(f"  Total tokens: {cost.tokens}")
            print(f"  Estimated cost: ${cost.estimated_cost_usd:.4f}")
    else:
        print(f"✗ Council failed: {result.error}")
        if result.validation_errors:
            print(f"  Validation errors: {result.validation_errors}")

    # Example 2: Using custom configuration
    print("\n\n[Example 2] Council with custom configuration")
    print("-" * 70)

    config = OrchestratorConfig(
        timeout=60,  # Reduce timeout to 60 seconds
        max_retries=5,  # Increase retry attempts
        max_draft_tokens=1500,  # Limit draft size
        draft_temperature=0.8,  # More creative drafts
        synthesis_temperature=0.1,  # More deterministic synthesis
    )

    council = Council(providers=["openrouter"], config=config)
    result = await council.run(
        task="Design a REST API for a task management system",
        subagent="architect"
    )

    if result.success:
        print("✓ Architecture design completed!")
        print(f"\nSystem Design:")
        print(f"  Title: {result.output.get('design_title', 'N/A')}")

        endpoints = result.output.get('api_endpoints', [])
        print(f"  API Endpoints: {len(endpoints)}")
        for endpoint in endpoints[:3]:  # Show first 3
            print(f"    - {endpoint.get('method', 'N/A')} {endpoint.get('path', 'N/A')}")

        print(f"\nPhase Timings:")
        for timing in result.phase_timings or []:
            print(f"  {timing.phase}: {timing.duration_ms}ms")
    else:
        print(f"✗ Architecture design failed: {result.error}")

    # Example 3: Multiple subagent types
    print("\n\n[Example 3] Using different subagent types")
    print("-" * 70)

    # Planning a feature
    council = Council(providers=["openrouter"])
    result = await council.run(
        task="Plan the implementation of user authentication with OAuth2",
        subagent="planner"
    )

    if result.success:
        print("✓ Planning completed!")
        print(f"\nPlan: {result.output.get('plan_title', 'N/A')}")

        phases = result.output.get('phases', [])
        print(f"Number of phases: {len(phases)}")
        for i, phase in enumerate(phases[:3], 1):
            print(f"\n  Phase {i}: {phase.get('phase_name', 'N/A')}")
            tasks = phase.get('tasks', [])
            print(f"    Tasks: {len(tasks)}")
    else:
        print(f"✗ Planning failed: {result.error}")

    # Example 4: Code review
    print("\n\n[Example 4] Code review subagent")
    print("-" * 70)

    code_to_review = '''
def process_payment(user_id, amount):
    # Process payment
    db.execute("UPDATE users SET balance = balance - " + str(amount) + " WHERE id = " + str(user_id))
    return True
'''

    result = await council.run(
        task=f"Review this payment processing code for security issues:\n\n{code_to_review}",
        subagent="reviewer"
    )

    if result.success:
        print("✓ Code review completed!")

        findings = result.output.get('findings', [])
        print(f"\nFindings: {len(findings)} issues")

        for finding in findings[:3]:  # Show first 3
            severity = finding.get('severity', 'N/A')
            issue_type = finding.get('issue_type', 'N/A')
            print(f"\n  [{severity.upper()}] {issue_type}")
            print(f"  {finding.get('description', 'N/A')}")
    else:
        print(f"✗ Code review failed: {result.error}")

    # Example 5: Health check
    print("\n\n[Example 5] Provider health check")
    print("-" * 70)

    council = Council(providers=["openrouter"])
    health = await council.doctor()

    print("Provider Status:")
    for provider_name, status in health.items():
        ok = status.get('ok', False)
        message = status.get('message', 'Unknown')
        latency = status.get('latency_ms', 0)

        status_icon = "✓" if ok else "✗"
        print(f"  {status_icon} {provider_name}: {message}")
        if ok and latency:
            print(f"    Latency: {latency:.0f}ms")

    # Example 6: List available subagents
    print("\n\n[Example 6] Available subagents")
    print("-" * 70)

    subagents = Council.available_subagents()
    print(f"Available subagents ({len(subagents)}):")
    for subagent in subagents:
        print(f"  - {subagent}")

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
