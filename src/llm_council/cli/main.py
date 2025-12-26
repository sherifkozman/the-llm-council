"""
CLI entry point for LLM Council.

Commands:
    council run <subagent> <task>  - Run a council task
    council doctor                  - Check provider status
    council config                  - Manage configuration
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="council",
    help="Multi-LLM Council Framework - Orchestrate multiple LLM backends",
    no_args_is_help=True,
)
console = Console()

# Agent aliases for backwards compatibility (deprecated -> new agent, mode)
AGENT_ALIASES: dict[str, tuple[str, str | None]] = {
    "implementer": ("drafter", "impl"),
    "architect": ("drafter", "arch"),
    "test-designer": ("drafter", "test"),
    "reviewer": ("critic", "review"),
    "red-team": ("critic", "security"),
    "assessor": ("planner", "assess"),
    "shipper": ("synthesizer", None),
}


def _resolve_agent_alias(subagent: str, mode: str | None = None) -> tuple[str, str | None, bool]:
    """Resolve deprecated agent names to new consolidated agents.

    Returns:
        Tuple of (agent_name, mode, was_deprecated)
    """
    if subagent in AGENT_ALIASES:
        new_agent, default_mode = AGENT_ALIASES[subagent]
        # Only print warning if not in JSON mode (checked by caller)
        # Use specified mode if provided, otherwise use alias default
        return new_agent, mode or default_mode, True  # True = was_deprecated
    return subagent, mode, False  # False = not deprecated


def _get_config_file() -> Path:
    """Get the config file path. Computed at runtime for test compatibility."""
    return Path.home() / ".config" / "llm-council" / "config.yaml"


def _load_config_defaults() -> dict[str, Any]:
    """Load defaults from config file if it exists.

    Returns:
        Dictionary with default values from config file, or empty dict if not found.
    """
    config_file = _get_config_file()
    if not config_file.exists():
        return {}

    try:
        config = yaml.safe_load(config_file.read_text()) or {}
        defaults: dict[str, Any] = config.get("defaults", {})
        return defaults
    except (yaml.YAMLError, OSError):
        return {}


@app.command()
def run(
    subagent: str = typer.Argument(..., help="Subagent type (drafter, critic, planner, etc.)"),
    task: str = typer.Argument(..., help="Task description"),
    mode: str | None = typer.Option(
        None,
        "--mode",
        help="Agent mode (e.g., impl/arch/test for drafter, review/security for critic)",
    ),
    providers: str | None = typer.Option(
        None,
        "--providers",
        "-p",
        help="Comma-separated provider list (default: from config or openrouter)",
    ),
    models: str | None = typer.Option(
        None,
        "--models",
        "-m",
        help=(
            "Comma-separated OpenRouter model IDs for multi-model council. "
            "Example: 'anthropic/claude-3.5-sonnet,openai/gpt-4o,google/gemini-pro'"
        ),
    ),
    no_artifacts: bool = typer.Option(False, "--no-artifacts", help="Disable artifact storage"),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Run a council task with the specified subagent."""
    # Load defaults from config file
    config_defaults = _load_config_defaults()

    # Resolve agent aliases (deprecated names -> new consolidated agents)
    resolved_agent, resolved_mode, was_deprecated = _resolve_agent_alias(subagent, mode)

    # Show deprecation warning (but not in JSON mode to avoid polluting output)
    if was_deprecated and not output_json:
        console.print(
            f"[yellow]Warning:[/yellow] '{subagent}' is deprecated. "
            f"Use '{resolved_agent}' instead. (Will be removed in v1.0)",
            style="dim",
        )

    # Use CLI args if provided, otherwise fall back to config, then hardcoded defaults
    if providers:
        provider_list = providers.split(",")
    else:
        provider_list = config_defaults.get("providers", ["openrouter"])

    # Get config-based defaults for removed CLI flags
    timeout = config_defaults.get("timeout", 120)
    max_retries = config_defaults.get("max_retries", 3)
    enable_degradation = config_defaults.get("enable_degradation", True)

    model_list = [m.strip() for m in models.split(",") if m.strip()] if models else None

    if not output_json:
        mode_str = f" --mode {resolved_mode}" if resolved_mode else ""
        if model_list and len(model_list) > 1:
            console.print(
                f"[bold blue]Council[/bold blue] Running {resolved_agent}{mode_str} with "
                f"{len(model_list)} models (multi-model council)..."
            )
        else:
            console.print(
                f"[bold blue]Council[/bold blue] Running {resolved_agent}{mode_str} with "
                f"{len(provider_list)} provider(s)..."
            )

    try:
        from llm_council import Council
        from llm_council.protocol.types import CouncilConfig

        config = CouncilConfig(
            providers=provider_list,
            models=model_list,
            timeout=timeout,
            max_retries=max_retries,
            enable_artifact_store=not no_artifacts,
            enable_health_check=False,  # Lazy failure instead of preflight
            enable_graceful_degradation=enable_degradation,
            mode=resolved_mode,  # Pass mode to config
        )

        council = Council(config=config)
        result = asyncio.run(council.run(task=task, subagent=resolved_agent))

        if output_json:
            print(json.dumps(result.model_dump(), indent=2, default=str))
        else:
            if result.success:
                console.print(
                    Panel(
                        json.dumps(result.output, indent=2),
                        title="[green]Council Result: SUCCESS[/green]",
                        border_style="green",
                    )
                )
            else:
                console.print(
                    Panel(
                        "\n".join(result.validation_errors or ["Unknown error"]),
                        title="[red]Council Result: FAILED[/red]",
                        border_style="red",
                    )
                )

            if verbose:
                console.print("\n[bold]Metrics:[/bold]")
                console.print(f"  Duration: {result.duration_ms}ms")
                console.print(f"  Attempts: {result.synthesis_attempts}")
                if result.cost_estimate:
                    console.print(f"  Est. cost: ${result.cost_estimate.estimated_cost_usd:.4f}")

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@app.command()
def doctor() -> None:
    """Check provider availability and configuration."""
    console.print("[bold blue]Council Doctor[/bold blue] Checking providers...\n")

    from llm_council.providers.registry import get_registry

    # Get registered providers
    registry = get_registry()
    provider_names = registry.list_providers()

    if not provider_names:
        console.print("[yellow]No providers registered.[/yellow]")
        console.print("Install provider packages: pip install the-llm-council[all]")
        return

    table = Table(title="Provider Status")
    table.add_column("Provider", style="cyan")
    table.add_column("Status")
    table.add_column("Message")
    table.add_column("Latency")

    for name in provider_names:
        try:
            provider = registry.get_provider(name)
            result = asyncio.run(provider.doctor())

            status = "[green]OK[/green]" if result.ok else "[red]FAIL[/red]"
            message = result.message or "-"
            latency = f"{result.latency_ms:.0f}ms" if result.latency_ms else "-"

            table.add_row(name, status, message, latency)
        except Exception as e:
            table.add_row(name, "[red]ERROR[/red]", str(e), "-")

    console.print(table)


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    init: bool = typer.Option(False, "--init", help="Initialize default configuration"),
) -> None:
    """Manage LLM Council configuration."""
    config_file = _get_config_file()
    config_dir = config_file.parent

    if show:
        if config_file.exists():
            console.print(config_file.read_text())
        else:
            console.print("[yellow]No configuration file found.[/yellow]")
            console.print(f"Run 'council config --init' to create one at {config_file}")
        return

    if init:
        config_dir.mkdir(parents=True, exist_ok=True)
        default_config = """\
# LLM Council Configuration

# Provider configurations (API keys, models, etc.)
providers:
  - name: openrouter
    # api_key: ${OPENROUTER_API_KEY}
    default_model: anthropic/claude-opus-4-5

# Default settings for 'council run' command
defaults:
  # Providers to use when --providers flag is not specified
  providers:
    - openrouter
  timeout: 120
  max_retries: 3
  summary_tier: actions
"""
        config_file.write_text(default_config)
        console.print(f"[green]Created configuration at {config_file}[/green]")
        return

    console.print("Usage: council config [--show | --init]")


@app.command()
def version() -> None:
    """Show version information."""
    from llm_council import __version__

    console.print(f"LLM Council v{__version__}")


if __name__ == "__main__":
    app()
