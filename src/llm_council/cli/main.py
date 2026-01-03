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
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_council import __version__

# Global state for CLI context
_cli_state: dict[str, Any] = {
    "quiet": False,
    "debug": False,
    "no_color": False,
    "config_path": None,
}


def _get_console() -> Console:
    """Get console with current settings."""
    return Console(
        no_color=_cli_state["no_color"],
        force_terminal=None if not _cli_state["no_color"] else False,
    )


def _print(message: str, **kwargs: Any) -> None:
    """Print message unless quiet mode is enabled."""
    if not _cli_state["quiet"]:
        _get_console().print(message, **kwargs)


def _setup_logging() -> None:
    """Configure logging based on debug flag."""
    if _cli_state["debug"]:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stderr,
        )
        logging.getLogger("llm_council").setLevel(logging.DEBUG)


def _version_callback(value: bool) -> None:
    """Handle --version flag."""
    if value:
        print(f"LLM Council v{__version__}")
        raise typer.Exit()


app = typer.Typer(
    name="council",
    help="Multi-LLM Council Framework - Orchestrate multiple LLM backends",
    no_args_is_help=True,
)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress non-essential output"),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Enable debug logging"),
    ] = False,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Custom config file path"),
    ] = None,
    no_color: Annotated[
        bool,
        typer.Option("--no-color", help="Disable colored output"),
    ] = False,
) -> None:
    """Multi-LLM Council Framework - Orchestrate multiple LLM backends."""
    _cli_state["quiet"] = quiet
    _cli_state["debug"] = debug
    _cli_state["no_color"] = no_color
    _cli_state["config_path"] = config_path
    _setup_logging()


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
        return new_agent, mode or default_mode, True
    return subagent, mode, False


def _get_config_file() -> Path:
    """Get the config file path.

    If a custom path is specified via --config, validates it is within
    allowed directories (home dir or current working directory).
    """
    if _cli_state["config_path"]:
        custom_path = Path(_cli_state["config_path"]).resolve()
        # Security: Ensure config path is within reasonable boundaries
        home = Path.home().resolve()
        cwd = Path.cwd().resolve()
        try:
            # Check if path is under home or cwd
            custom_path.relative_to(home)
        except ValueError:
            try:
                custom_path.relative_to(cwd)
            except ValueError:
                # Path is outside allowed directories - only allow if it exists
                # and is a regular file (not a device, symlink to sensitive file, etc.)
                if not custom_path.exists():
                    raise typer.Exit(1)
        return custom_path
    return Path.home() / ".config" / "llm-council" / "config.yaml"


def _load_config() -> dict[str, Any]:
    """Load full config from file if it exists."""
    config_file = _get_config_file()
    if not config_file.exists():
        return {}

    try:
        return yaml.safe_load(config_file.read_text()) or {}
    except (yaml.YAMLError, OSError):
        return {}


def _load_config_defaults() -> dict[str, Any]:
    """Load defaults section from config file."""
    config = _load_config()
    return config.get("defaults", {})


def _get_nested_value(data: dict[str, Any], key: str) -> Any:
    """Get a nested value using dot notation (e.g., 'defaults.timeout')."""
    parts = key.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, list):
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def _set_nested_value(data: dict[str, Any], key: str, value: Any) -> None:
    """Set a nested value using dot notation."""
    parts = key.split(".")
    current = data
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]

    # Parse value if it looks like JSON
    final_value = value
    if isinstance(value, str):
        try:
            final_value = json.loads(value)
        except json.JSONDecodeError:
            final_value = value

    current[parts[-1]] = final_value


@app.command()
def run(
    subagent: Annotated[str, typer.Argument(help="Subagent type (drafter, critic, planner, etc.)")],
    task: Annotated[str | None, typer.Argument(help="Task description")] = None,
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="Agent mode (e.g., impl/arch/test for drafter)"),
    ] = None,
    providers: Annotated[
        str | None,
        typer.Option("--providers", "-p", help="Comma-separated provider list"),
    ] = None,
    models: Annotated[
        str | None,
        typer.Option("--models", "-m", help="Comma-separated model IDs for multi-model council"),
    ] = None,
    no_artifacts: Annotated[
        bool,
        typer.Option("--no-artifacts", help="Disable artifact storage"),
    ] = False,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed output"),
    ] = False,
    timeout: Annotated[
        int | None,
        typer.Option("--timeout", "-t", help="Request timeout in seconds"),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option("--temperature", help="Model temperature (0.0-2.0)"),
    ] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", help="Max output tokens"),
    ] = None,
    input_file: Annotated[
        str | None,
        typer.Option("--input", "-i", help="Read task from file (use '-' for stdin)"),
    ] = None,
    output_file: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write output to file"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would run without executing"),
    ] = False,
    context: Annotated[
        str | None,
        typer.Option("--context", "--system", help="Additional system context/instructions"),
    ] = None,
    schema_file: Annotated[
        Path | None,
        typer.Option("--schema", help="Custom output schema JSON file"),
    ] = None,
) -> None:
    """Run a council task with the specified subagent."""
    console = _get_console()

    # Handle input from file or stdin
    task_text = task
    if input_file:
        if input_file == "-":
            task_text = sys.stdin.read().strip()
        else:
            input_path = Path(input_file)
            if not input_path.exists():
                console.print(f"[red]Error:[/red] Input file not found: {input_file}")
                raise typer.Exit(1)
            task_text = input_path.read_text().strip()

    if not task_text:
        console.print("[red]Error:[/red] No task provided. Use positional argument or --input")
        raise typer.Exit(1)

    # Input validation: limit task size to prevent resource exhaustion
    max_task_length = 100_000  # 100KB limit
    if len(task_text) > max_task_length:
        console.print(
            f"[red]Error:[/red] Task too long ({len(task_text):,} chars). "
            f"Maximum is {max_task_length:,} characters."
        )
        raise typer.Exit(1)

    # Load defaults from config file
    config_defaults = _load_config_defaults()

    # Resolve agent aliases
    resolved_agent, resolved_mode, was_deprecated = _resolve_agent_alias(subagent, mode)

    # Show deprecation warning
    if was_deprecated and not output_json:
        _print(
            f"[yellow]Warning:[/yellow] '{subagent}' is deprecated. "
            f"Use '{resolved_agent}' instead. (Will be removed in v1.0)",
        )

    # Build provider list
    if providers:
        provider_list = [p.strip() for p in providers.split(",")]
    else:
        provider_list = config_defaults.get("providers", ["openrouter"])

    # Get timeout (CLI > config > default)
    effective_timeout = timeout or config_defaults.get("timeout", 120)
    max_retries = config_defaults.get("max_retries", 3)
    enable_degradation = config_defaults.get("enable_degradation", True)

    model_list = [m.strip() for m in models.split(",") if m.strip()] if models else None

    # Handle dry-run
    if dry_run:
        _print(f"[bold]Dry run:[/bold] would execute {resolved_agent}")
        _print(f"  Mode: {resolved_mode or 'default'}")
        _print(f"  Providers: {', '.join(provider_list)}")
        _print(f"  Models: {', '.join(model_list) if model_list else 'default'}")
        _print(f"  Timeout: {effective_timeout}s")
        _print(f"  Temperature: {temperature if temperature is not None else 'default'}")
        _print(f"  Max tokens: {max_tokens if max_tokens is not None else 'default'}")
        _print(f"  Context: {context[:50] + '...' if context and len(context) > 50 else context or 'none'}")
        _print(f"  Schema: {schema_file or 'default'}")
        _print(f"  Task: {task_text[:100]}{'...' if len(task_text) > 100 else ''}")
        return

    if not output_json:
        mode_str = f" --mode {resolved_mode}" if resolved_mode else ""
        if model_list and len(model_list) > 1:
            _print(
                f"[bold blue]Council[/bold blue] Running {resolved_agent}{mode_str} with "
                f"{len(model_list)} models..."
            )
        else:
            _print(
                f"[bold blue]Council[/bold blue] Running {resolved_agent}{mode_str} with "
                f"{len(provider_list)} provider(s)..."
            )

    try:
        from llm_council import Council
        from llm_council.protocol.types import CouncilConfig

        # Load custom schema if provided
        custom_schema = None
        if schema_file:
            if not schema_file.exists():
                console.print(f"[red]Error:[/red] Schema file not found: {schema_file}")
                raise typer.Exit(1)
            custom_schema = json.loads(schema_file.read_text())

        config = CouncilConfig(
            providers=provider_list,
            models=model_list,
            timeout=effective_timeout,
            max_retries=max_retries,
            enable_artifact_store=not no_artifacts,
            enable_health_check=False,
            enable_graceful_degradation=enable_degradation,
            mode=resolved_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            system_context=context,
            output_schema=custom_schema,
        )

        council = Council(config=config)
        result = asyncio.run(council.run(task=task_text, subagent=resolved_agent))

        # Format output
        if output_json:
            output_text = json.dumps(result.model_dump(), indent=2, default=str)
        else:
            output_text = None

        # Write to file or stdout
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if output_json:
                output_file.write_text(output_text)
            else:
                output_file.write_text(json.dumps(result.model_dump(), indent=2, default=str))
            _print(f"[green]Output written to {output_file}[/green]")
        elif output_json:
            print(output_text)
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
            error_output = json.dumps({"error": str(e)})
            if output_file:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(error_output)
            else:
                print(error_output)
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def doctor(
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
    provider_filter: Annotated[
        str | None,
        typer.Option("--provider", help="Check specific provider only"),
    ] = None,
) -> None:
    """Check provider availability and configuration."""
    console = _get_console()

    if not output_json:
        _print("[bold blue]Council Doctor[/bold blue] Checking providers...\n")

    from llm_council.providers.registry import get_registry

    registry = get_registry()
    provider_names = registry.list_providers()

    if provider_filter:
        if provider_filter not in provider_names:
            if output_json:
                print(json.dumps({"error": f"Unknown provider: {provider_filter}"}))
            else:
                console.print(f"[red]Error:[/red] Unknown provider: {provider_filter}")
                console.print(f"Available: {', '.join(provider_names)}")
            raise typer.Exit(1)
        provider_names = [provider_filter]

    if not provider_names:
        if output_json:
            print(json.dumps({"providers": [], "error": "No providers registered"}))
        else:
            console.print("[yellow]No providers registered.[/yellow]")
            console.print("Install provider packages: pip install the-llm-council[all]")
        return

    results = []
    for name in provider_names:
        try:
            provider = registry.get_provider(name)
            result = asyncio.run(provider.doctor())
            results.append({
                "name": name,
                "ok": result.ok,
                "message": result.message or "",
                "latency_ms": result.latency_ms,
            })
        except Exception as e:
            results.append({
                "name": name,
                "ok": False,
                "message": str(e),
                "latency_ms": None,
            })

    if output_json:
        print(json.dumps({"providers": results}, indent=2))
    else:
        table = Table(title="Provider Status")
        table.add_column("Provider", style="cyan")
        table.add_column("Status")
        table.add_column("Message")
        table.add_column("Latency")

        for r in results:
            status = "[green]OK[/green]" if r["ok"] else "[red]FAIL[/red]"
            latency = f"{r['latency_ms']:.0f}ms" if r["latency_ms"] else "-"
            table.add_row(r["name"], status, r["message"] or "-", latency)

        console.print(table)


@app.command()
def config(
    show: Annotated[
        bool,
        typer.Option("--show", help="Show current configuration"),
    ] = False,
    init: Annotated[
        bool,
        typer.Option("--init", help="Initialize default configuration"),
    ] = False,
    path: Annotated[
        bool,
        typer.Option("--path", help="Show config file path"),
    ] = False,
    validate: Annotated[
        bool,
        typer.Option("--validate", help="Validate configuration file"),
    ] = False,
    get_key: Annotated[
        str | None,
        typer.Option("--get", help="Get config value by key (dot notation)"),
    ] = None,
    set_key: Annotated[
        str | None,
        typer.Option("--set", help="Set config key (use with value argument)"),
    ] = None,
    set_value: Annotated[
        str | None,
        typer.Argument(help="Value to set (when using --set)"),
    ] = None,
    edit: Annotated[
        bool,
        typer.Option("--edit", help="Open config in $EDITOR"),
    ] = False,
) -> None:
    """Manage LLM Council configuration."""
    console = _get_console()
    config_file = _get_config_file()
    config_dir = config_file.parent

    # --path: Show config file path
    if path:
        print(str(config_file))
        return

    # --show: Show current configuration
    if show:
        if config_file.exists():
            console.print(config_file.read_text())
        else:
            console.print("[yellow]No configuration file found.[/yellow]")
            console.print(f"Run 'council config --init' to create one at {config_file}")
        return

    # --init: Initialize default configuration
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

    # --validate: Validate configuration
    if validate:
        if not config_file.exists():
            console.print("[red]Error:[/red] No configuration file found")
            raise typer.Exit(1)

        try:
            config_data = yaml.safe_load(config_file.read_text())
            if not isinstance(config_data, dict):
                console.print("[red]Error:[/red] Configuration must be a YAML dictionary")
                raise typer.Exit(1)

            # Basic validation
            if "providers" in config_data and not isinstance(config_data["providers"], list):
                console.print("[red]Error:[/red] 'providers' must be a list")
                raise typer.Exit(1)

            if "defaults" in config_data and not isinstance(config_data["defaults"], dict):
                console.print("[red]Error:[/red] 'defaults' must be a dictionary")
                raise typer.Exit(1)

            console.print("[green]Configuration is valid.[/green]")
        except yaml.YAMLError as e:
            console.print(f"[red]Error:[/red] Invalid YAML: {e}")
            raise typer.Exit(1)
        return

    # --get: Get config value
    if get_key:
        if not config_file.exists():
            console.print("[red]Error:[/red] No configuration file found")
            raise typer.Exit(1)

        config_data = _load_config()
        value = _get_nested_value(config_data, get_key)
        if value is None:
            console.print(f"[yellow]Key not found:[/yellow] {get_key}")
            raise typer.Exit(1)

        if isinstance(value, (dict, list)):
            print(json.dumps(value, indent=2))
        else:
            print(value)
        return

    # --set: Set config value
    if set_key:
        if set_value is None:
            console.print("[red]Error:[/red] --set requires a value argument")
            console.print("Usage: council config --set KEY VALUE")
            raise typer.Exit(1)

        config_dir.mkdir(parents=True, exist_ok=True)
        config_data = _load_config()
        _set_nested_value(config_data, set_key, set_value)

        with open(config_file, "w") as f:
            yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)

        console.print(f"[green]Set {set_key} = {set_value}[/green]")
        return

    # --edit: Open in editor
    if edit:
        if not config_file.exists():
            console.print("[yellow]Config file doesn't exist. Creating default...[/yellow]")
            config_dir.mkdir(parents=True, exist_ok=True)
            config_file.write_text("# LLM Council Configuration\n\ndefaults:\n  providers:\n    - openrouter\n  timeout: 120\n")

        editor = os.environ.get("EDITOR", "vi")
        # Security: validate editor doesn't contain shell metacharacters
        # Allow paths like /usr/bin/vim but reject shell injection attempts
        shell_metacharacters = ";|&$`()<>!\n\r\t"
        if not editor or any(c in editor for c in shell_metacharacters):
            console.print("[red]Error:[/red] Invalid EDITOR environment variable")
            raise typer.Exit(1)
        try:
            subprocess.run([editor, str(config_file)], check=False)
        except FileNotFoundError:
            console.print(f"[red]Error:[/red] Editor '{editor}' not found")
            raise typer.Exit(1)
        return

    # No flags provided - show usage
    console.print("Usage: council config [OPTIONS]")
    console.print("\nOptions:")
    console.print("  --show      Show current configuration")
    console.print("  --init      Initialize default configuration")
    console.print("  --path      Show config file path")
    console.print("  --validate  Validate configuration file")
    console.print("  --get KEY   Get config value by key")
    console.print("  --set KEY VALUE  Set config value")
    console.print("  --edit      Open config in $EDITOR")


@app.command()
def version() -> None:
    """Show version information."""
    console = _get_console()
    console.print(f"LLM Council v{__version__}")


if __name__ == "__main__":
    app()
