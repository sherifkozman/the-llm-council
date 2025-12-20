#!/usr/bin/env python3
"""
Custom Subagent Example

This example demonstrates how to:
- Create a custom subagent configuration
- Define custom JSON schemas for output validation
- Use custom prompts and settings
- Integrate custom subagents with Council

The example creates a "code-optimizer" subagent that analyzes code
and suggests performance improvements.

Prerequisites:
    export OPENROUTER_API_KEY="your-key-here"

Usage:
    python examples/custom_subagent.py
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path

import yaml

from llm_council import Council, OrchestratorConfig


def create_custom_subagent_config():
    """Create a custom subagent configuration.

    Returns:
        Tuple of (config_dict, schema_dict) for the custom subagent
    """

    # Custom subagent configuration (YAML format)
    subagent_config = {
        "name": "code-optimizer",
        "description": (
            "Analyzes code for performance bottlenecks and suggests optimizations. "
            "Focuses on algorithmic complexity, memory usage, and best practices."
        ),
        "model_pack": "code_specialist_normal",
        "calls": "4-5",  # Drafts + critique + synthesis
        "schema": "code-optimizer",  # Reference to JSON schema
        "prompts": {
            "system": """You are a senior performance engineer specializing in code optimization.

Your job is to analyze code for performance issues and suggest concrete improvements.

Focus on:
1. Algorithmic complexity (Big O analysis)
2. Memory efficiency
3. Database query optimization
4. Caching opportunities
5. Parallelization potential
6. Framework-specific best practices

For each optimization:
- Explain the current performance issue
- Estimate the impact (high/medium/low)
- Provide specific code suggestions
- Include before/after complexity analysis
""",
            "examples": [
                {
                    "task": "Optimize this user search function",
                    "expected": {
                        "optimization_title": "User Search Performance Optimization",
                        "summary": "Reduced search time from O(n) to O(log n) by adding database index",
                        "optimizations": [
                            {
                                "issue": "Linear scan through users table",
                                "impact": "high",
                                "recommendation": "Add composite index on (email, username)",
                            }
                        ],
                    }
                }
            ]
        },
        "classification_rules": {
            "task_signals": ["optimize", "performance", "speed up", "improve", "bottleneck"]
        }
    }

    # Custom JSON schema for output validation
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "CodeOptimizerOutput",
        "type": "object",
        "required": [
            "optimization_title",
            "summary",
            "optimizations"
        ],
        "properties": {
            "optimization_title": {
                "type": "string",
                "description": "Title describing the optimization focus"
            },
            "summary": {
                "type": "string",
                "description": "High-level summary of optimization opportunities"
            },
            "current_complexity": {
                "type": "object",
                "properties": {
                    "time": {"type": "string"},
                    "space": {"type": "string"}
                }
            },
            "optimized_complexity": {
                "type": "object",
                "properties": {
                    "time": {"type": "string"},
                    "space": {"type": "string"}
                }
            },
            "optimizations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["issue", "impact", "recommendation"],
                    "properties": {
                        "issue": {
                            "type": "string",
                            "description": "Description of the performance issue"
                        },
                        "impact": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "Expected performance impact"
                        },
                        "recommendation": {
                            "type": "string",
                            "description": "Specific optimization suggestion"
                        },
                        "code_example": {
                            "type": "string",
                            "description": "Code snippet showing the optimization"
                        },
                        "estimated_improvement": {
                            "type": "string",
                            "description": "Quantified improvement estimate (e.g., '50% faster')"
                        }
                    }
                }
            },
            "profiling_recommendations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Suggestions for profiling and measurement"
            },
            "caching_opportunities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "strategy": {"type": "string"},
                        "cache_key": {"type": "string"}
                    }
                }
            }
        }
    }

    return subagent_config, schema


async def main():
    """Run custom subagent examples."""

    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Get your key at: https://openrouter.ai/keys")
        return

    print("=" * 70)
    print("LLM Council - Custom Subagent Example")
    print("=" * 70)

    # Create temporary directory for custom subagent files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create custom subagent configuration
        config_dict, schema_dict = create_custom_subagent_config()

        # Note: This is a demonstration of the configuration structure.
        # In production, you would:
        # 1. Save these to src/llm_council/subagents/code-optimizer.yaml
        # 2. Save schema to src/llm_council/schemas/code-optimizer.json
        # 3. The council would automatically discover them

        print("\n[Info] Custom Subagent Configuration")
        print("-" * 70)
        print("Configuration structure created:")
        print(f"  Name: {config_dict['name']}")
        print(f"  Description: {config_dict['description'][:60]}...")
        print(f"  Schema: {config_dict['schema']}")
        print("\nTo use this in production:")
        print("  1. Save config to: src/llm_council/subagents/code-optimizer.yaml")
        print("  2. Save schema to: src/llm_council/schemas/code-optimizer.json")
        print("  3. Run: council run code-optimizer 'your task'")

        # For demonstration, we'll use an existing subagent
        # In a real scenario with the files saved, you would use "code-optimizer"

        print("\n\n[Example 1] Using built-in implementer subagent")
        print("-" * 70)

        council = Council(providers=["openrouter"])

        code_to_optimize = """
def find_duplicates(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates

def get_user_posts(user_id):
    user = db.query("SELECT * FROM users WHERE id = ?", user_id)
    posts = db.query("SELECT * FROM posts WHERE user_id = ?", user_id)
    comments = db.query("SELECT * FROM comments WHERE user_id = ?", user_id)
    likes = db.query("SELECT * FROM likes WHERE user_id = ?", user_id)

    for post in posts:
        post['comments'] = db.query("SELECT * FROM comments WHERE post_id = ?", post['id'])
        post['likes'] = db.query("SELECT * FROM likes WHERE post_id = ?", post['id'])

    return {'user': user, 'posts': posts, 'comments': comments, 'likes': likes}
"""

        result = await council.run(
            task=f"""Analyze this code for performance issues and suggest optimizations.
Focus on algorithmic complexity and database query efficiency.

Code to analyze:
{code_to_optimize}

Provide:
1. Current complexity analysis (Big O)
2. Specific performance bottlenecks
3. Concrete optimization suggestions with code examples
4. Expected performance improvements
""",
            subagent="implementer"
        )

        if result.success:
            print("✓ Code analysis completed!")

            print(f"\nAnalysis Summary:")
            print(f"  {result.output.get('summary', 'N/A')}")

            print(f"\nDuration: {result.duration_ms}ms")
            print(f"Synthesis attempts: {result.synthesis_attempts}")

            # Show phase breakdown
            if result.phase_timings:
                print("\nPhase Timings:")
                for timing in result.phase_timings:
                    print(f"  {timing.phase:12} {timing.duration_ms:6}ms")

            # Show cost
            if result.cost_estimate:
                cost = result.cost_estimate
                print(f"\nCost Estimate:")
                print(f"  Tokens: {cost.total_input_tokens} in + {cost.total_output_tokens} out")
                print(f"  Estimated: ${cost.estimated_cost_usd:.4f}")

            # Save full output
            output_file = temp_path / "optimization_result.json"
            with open(output_file, 'w') as f:
                json.dump(result.output, f, indent=2)
            print(f"\nFull output saved to: {output_file}")

        else:
            print(f"✗ Analysis failed: {result.error}")
            if result.validation_errors:
                print(f"\nValidation errors:")
                for error in result.validation_errors:
                    print(f"  - {error}")

        # Example 2: Show how to create reusable custom configurations
        print("\n\n[Example 2] Creating reusable custom subagent templates")
        print("-" * 70)

        print("\nCustom subagent template structure:")
        print("\nYAML Configuration (subagent.yaml):")
        print("-" * 40)
        print(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))

        print("\nJSON Schema (schema.json):")
        print("-" * 40)
        print(json.dumps(schema_dict, indent=2)[:500] + "...")

        # Example 3: Advanced configuration with custom settings
        print("\n\n[Example 3] Advanced orchestrator configuration")
        print("-" * 70)

        advanced_config = OrchestratorConfig(
            timeout=90,
            max_retries=5,
            max_draft_tokens=2500,
            max_critique_tokens=1500,
            max_synthesis_tokens=2500,
            draft_temperature=0.7,  # Balanced creativity
            critique_temperature=0.2,  # Strict critique
            synthesis_temperature=0.1,  # Deterministic synthesis
            enable_schema_validation=True,
            strict_providers=True,
        )

        council = Council(providers=["openrouter"], config=advanced_config)

        print("Advanced configuration:")
        print(f"  Timeout: {advanced_config.timeout}s")
        print(f"  Max retries: {advanced_config.max_retries}")
        print(f"  Draft tokens: {advanced_config.max_draft_tokens}")
        print(f"  Draft temperature: {advanced_config.draft_temperature}")
        print(f"  Critique temperature: {advanced_config.critique_temperature}")
        print(f"  Synthesis temperature: {advanced_config.synthesis_temperature}")
        print(f"  Schema validation: {advanced_config.enable_schema_validation}")

        print("\n" + "=" * 70)
        print("Custom subagent example complete!")
        print("\nNext steps:")
        print("  1. Create your subagent YAML in src/llm_council/subagents/")
        print("  2. Create matching JSON schema in src/llm_council/schemas/")
        print("  3. Use with: council = Council(); council.run(task, 'your-subagent')")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
