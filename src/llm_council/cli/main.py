"""
CLI entry point for LLM Council.

Commands:
    council run <subagent> <task>  - Run a council task
    council eval <dataset>         - Run a dataset-driven evaluation
    council eval-compare           - Compare named runtime variants on one dataset
    council eval-import-pr         - Import PR review material into local-only storage
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
from typing import TYPE_CHECKING, Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_council import __version__
from llm_council.eval_import import import_github_pr_review
from llm_council.evaluation import (
    EvalComparisonReport,
    EvalReport,
    load_eval_dataset,
    load_eval_variants,
    run_eval_comparison,
    run_eval_dataset,
)
from llm_council.protocol.types import ReasoningProfile, RuntimeProfile
from llm_council.providers.concurrency import provider_call_slot

if TYPE_CHECKING:
    from llm_council.protocol.types import CouncilConfig

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
    defaults = config.get("defaults", {})
    return defaults if isinstance(defaults, dict) else {}


def _load_provider_configs() -> dict[str, dict[str, Any]]:
    """Extract per-provider configs from config file.

    Reads the ``providers`` list from config.yaml and returns a
    dict keyed by provider name with constructor kwargs
    (e.g. ``default_model``).

    Example config.yaml::

        providers:
          - name: openai
            default_model: gpt-5.4
          - name: gemini
            default_model: gemini-3.1-pro-preview

    Returns::

        {"openai": {"default_model": "gpt-5.4"},
         "gemini": {"default_model": "gemini-3.1-pro-preview"}}
    """
    config = _load_config()
    providers_list = config.get("providers", [])
    if not isinstance(providers_list, list):
        return {}

    result: dict[str, dict[str, Any]] = {}
    for entry in providers_list:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name or not isinstance(name, str):
            continue
        # Forward known constructor kwargs only
        kwargs: dict[str, Any] = {}
        if "default_model" in entry:
            kwargs["default_model"] = entry["default_model"]
        if kwargs:
            result[name] = kwargs
    return result


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
    runtime_profile: Annotated[
        RuntimeProfile,
        typer.Option(
            "--runtime-profile",
            help="High-level runtime budget override: default or bounded",
        ),
    ] = RuntimeProfile.DEFAULT,
    reasoning_profile: Annotated[
        ReasoningProfile,
        typer.Option(
            "--reasoning-profile",
            help="High-level reasoning override: default, off, or light",
        ),
    ] = ReasoningProfile.DEFAULT,
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
    files: Annotated[
        list[str] | None,
        typer.Option(
            "--files",
            "-f",
            help="File paths to include as context (repeatable, or comma-separated)",
        ),
    ] = None,
    schema_file: Annotated[
        Path | None,
        typer.Option("--schema", help="Custom output schema JSON file"),
    ] = None,
    route: Annotated[
        bool,
        typer.Option(
            "--route",
            help="When running router, execute the routed subagent and mode after classification",
        ),
    ] = False,
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

    # Read --files and merge into context
    if files:
        file_context_parts: list[str] = []
        max_total_bytes = 200_000  # 200KB total cap
        max_per_file = 50_000  # 50KB per file
        total_bytes = 0
        # Flatten: each entry can be comma-separated or a single path
        all_paths = [p.strip() for entry in files for p in entry.split(",")]
        for fpath in all_paths:
            if not fpath:
                continue
            p = Path(fpath)
            if not p.exists():
                _print(f"[yellow]Warning:[/yellow] File not found: {fpath}")
                continue
            content = p.read_text(encoding="utf-8", errors="replace")
            if len(content) > max_per_file:
                content = content[:max_per_file] + f"\n... [truncated at {max_per_file // 1000}KB]"
            if total_bytes + len(content) > max_total_bytes:
                _print(f"[yellow]Warning:[/yellow] Skipping {fpath} (200KB total limit)")
                continue
            total_bytes += len(content)
            file_context_parts.append(f"=== FILE: {fpath} ===\n{content}\n=== END: {fpath} ===")
        if file_context_parts:
            file_block = "\n\n".join(file_context_parts)
            context = f"{context}\n\n{file_block}" if context else file_block

    # Load defaults from config file
    config_defaults = _load_config_defaults()

    # Apply output_format default from config (CLI flag takes precedence)
    if not output_json:
        output_json = config_defaults.get("output_format") == "json"

    # Resolve agent aliases
    resolved_agent, resolved_mode, was_deprecated = _resolve_agent_alias(subagent, mode)
    if route and resolved_agent != "router":
        console.print("[red]Error:[/red] --route can only be used with the router subagent")
        raise typer.Exit(1)

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
        effective_schema: str = str(schema_file) if schema_file else "default"
        if not schema_file:
            try:
                from llm_council.subagents import get_effective_schema, load_subagent

                effective_schema = (
                    get_effective_schema(load_subagent(resolved_agent), resolved_mode) or "default"
                )
            except Exception:
                effective_schema = "default"

        _print(f"[bold]Dry run:[/bold] would execute {resolved_agent}")
        _print(f"  Mode: {resolved_mode or 'default'}")
        _print(f"  Providers: {', '.join(provider_list)}")
        _print(f"  Models: {', '.join(model_list) if model_list else 'default'}")
        _print(f"  Timeout: {effective_timeout}s")
        _print(f"  Temperature: {temperature if temperature is not None else 'default'}")
        _print(f"  Max tokens: {max_tokens if max_tokens is not None else 'default'}")
        _print(f"  Runtime profile: {runtime_profile.value}")
        _print(f"  Reasoning profile: {reasoning_profile.value}")
        ctx_display = context[:50] + "..." if context and len(context) > 50 else context or "none"
        _print(f"  Context: {ctx_display}")
        _print(f"  Schema: {effective_schema}")
        _print(f"  Follow router: {'yes' if route else 'no'}")
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
            runtime_profile=runtime_profile,
            reasoning_profile=reasoning_profile,
            system_context=context,
            output_schema=custom_schema,
            follow_router=route,
            provider_configs=_load_provider_configs(),
        )

        council = Council(config=config)
        result = asyncio.run(
            council.run(task=task_text, subagent=resolved_agent, follow_router=route)
        )

        output_payload = json.dumps(result.model_dump(), indent=2, default=str)

        # Write to file or stdout
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(output_payload)
            _print(f"[green]Output written to {output_file}[/green]")
        elif output_json:
            print(output_payload)
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
                if result.execution_plan:
                    console.print("\n[bold]Execution Plan:[/bold]")
                    console.print(
                        f"  Mode: {result.execution_plan.get('mode') or resolved_mode or 'default'}"
                    )
                    console.print(
                        f"  Schema: {result.execution_plan.get('schema_name') or 'none'} "
                        f"({result.execution_plan.get('schema_source') or 'unknown'})"
                    )
                    console.print(
                        f"  Model pack: {result.execution_plan.get('model_pack') or 'default'} "
                        f"({result.execution_plan.get('model_pack_source') or 'unknown'})"
                    )
                    console.print(
                        "  Execution profile: "
                        f"{result.execution_plan.get('execution_profile') or 'prompt_only'}"
                    )
                    console.print(
                        f"  Runtime profile: {result.execution_plan.get('runtime_profile') or 'default'}"
                    )
                    console.print(
                        f"  Budget class: {result.execution_plan.get('budget_class') or 'normal'}"
                    )
                    providers_display = result.execution_plan.get("providers") or provider_list
                    console.print(f"  Providers: {', '.join(providers_display)}")
                    required_capabilities = result.execution_plan.get("required_capabilities") or []
                    if required_capabilities:
                        console.print(
                            "  Required capabilities: " + ", ".join(required_capabilities)
                        )
                    registered_tools = result.execution_plan.get("registered_tools") or []
                    if registered_tools:
                        console.print("  Registered tools: " + ", ".join(registered_tools))
                    evidence_items = result.execution_plan.get("evidence_items")
                    if evidence_items:
                        console.print(f"  Evidence items: {evidence_items}")
                    executed_capabilities = result.execution_plan.get("executed_capabilities") or []
                    if executed_capabilities:
                        console.print(
                            "  Executed capabilities: " + ", ".join(executed_capabilities)
                        )
                    pending_capabilities = result.execution_plan.get("pending_capabilities") or []
                    if pending_capabilities:
                        console.print("  Pending capabilities: " + ", ".join(pending_capabilities))
                    model_overrides = result.execution_plan.get("model_overrides") or {}
                    if model_overrides:
                        console.print(
                            "  Model overrides: "
                            + ", ".join(
                                f"{provider}={model}"
                                for provider, model in sorted(model_overrides.items())
                            )
                        )
                    reasoning = result.execution_plan.get("reasoning")
                    reasoning_profile_used = result.execution_plan.get("reasoning_profile")
                    if reasoning_profile_used:
                        console.print(f"  Reasoning profile: {reasoning_profile_used}")
                    phase_token_budgets = result.execution_plan.get("phase_token_budgets") or {}
                    if phase_token_budgets:
                        console.print(
                            "  Phase token budgets: "
                            + ", ".join(
                                f"{phase}={tokens}" for phase, tokens in phase_token_budgets.items()
                            )
                        )
                    if reasoning:
                        console.print(
                            "  Reasoning: "
                            + ", ".join(
                                f"{key}={value}"
                                for key, value in reasoning.items()
                                if value is not None
                            )
                        )
                if result.routed and result.routing_decision:
                    routing_mode = result.routing_decision.get("mode")
                    routing_mode_suffix = (
                        f" --mode {routing_mode}" if isinstance(routing_mode, str) else ""
                    )
                    console.print("\n[bold]Routing:[/bold]")
                    console.print(
                        "  Routed to: "
                        f"{result.routing_decision.get('subagent_to_run')}"
                        f"{routing_mode_suffix}"
                    )
                    console.print(
                        f"  Router reasoning: {result.routing_decision.get('reasoning') or 'n/a'}"
                    )

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
    provider_filters: Annotated[
        list[str] | None,
        typer.Option("--provider", help="Check one or more specific providers"),
    ] = None,
    deep: Annotated[
        bool,
        typer.Option(
            "--deep",
            help=(
                "Run a low-cost non-interactive generation probe after the normal provider health "
                "check. This may incur API/CLI usage."
            ),
        ),
    ] = False,
    probe_timeout: Annotated[
        float,
        typer.Option(
            "--probe-timeout",
            min=1.0,
            help="Timeout in seconds for the deep readiness probe",
        ),
    ] = 5.0,
) -> None:
    """Check provider availability and configuration."""
    console = _get_console()

    # Apply output_format default from config (CLI flag takes precedence)
    if not output_json:
        config_defaults = _load_config_defaults()
        output_json = config_defaults.get("output_format") == "json"

    if not output_json:
        _print("[bold blue]Council Doctor[/bold blue] Checking providers...\n")

    from llm_council.providers.registry import get_registry

    registry = get_registry()
    provider_names = registry.list_providers()

    if provider_filters:
        requested = [
            item.strip() for raw in provider_filters for item in raw.split(",") if item.strip()
        ]
        selected: list[str] = []
        for requested_name in requested:
            normalized = registry.resolve_name(requested_name)
            if normalized not in provider_names:
                if output_json:
                    print(
                        json.dumps(
                            {
                                "error": f"Unknown provider: {requested_name}",
                                "available": provider_names,
                            }
                        )
                    )
                else:
                    console.print(f"[red]Error:[/red] Unknown provider: {requested_name}")
                    console.print(f"Available: {', '.join(provider_names)}")
                raise typer.Exit(1)
            if normalized not in selected:
                selected.append(normalized)
        provider_names = selected

    if not provider_names:
        if output_json:
            print(json.dumps({"providers": [], "error": "No providers registered"}))
        else:
            console.print("[yellow]No providers registered.[/yellow]")
            console.print("Install provider packages: pip install the-llm-council[all]")
        return

    # Load per-provider config so doctor shows configured models
    provider_cfgs = _load_provider_configs()

    async def _run_deep_probe(name: str, provider: Any) -> dict[str, Any]:
        from time import perf_counter

        from llm_council.providers.base import GenerateRequest

        start = perf_counter()
        async with provider_call_slot(name, timeout_seconds=max(float(probe_timeout), 1.0)):
            response = await provider.generate(
                GenerateRequest(
                    prompt="Reply with OK and nothing else.",
                    timeout_seconds=probe_timeout,
                )
            )
        latency_ms = round((perf_counter() - start) * 1000, 1)
        text = (response.text if hasattr(response, "text") else str(response)).strip()
        return {
            "probe_ok": True,
            "probe_message": text[:120] or "Probe completed",
            "probe_latency_ms": latency_ms,
        }

    async def _check_provider(name: str) -> dict[str, Any]:
        try:
            kwargs = provider_cfgs.get(name, {})
            provider = registry.get_provider(name, **kwargs)
            result = await provider.doctor()
            payload = {
                "name": name,
                "ok": result.ok,
                "message": result.message or "",
                "latency_ms": result.latency_ms,
            }
            if deep:
                if result.ok:
                    try:
                        payload.update(await _run_deep_probe(name, provider))
                    except Exception as e:
                        payload.update(
                            {
                                "probe_ok": False,
                                "probe_message": str(e),
                                "probe_latency_ms": None,
                            }
                        )
                else:
                    payload.update(
                        {
                            "probe_ok": False,
                            "probe_message": "Skipped because base doctor failed",
                            "probe_latency_ms": None,
                        }
                    )
            return payload
        except Exception as e:
            payload = {
                "name": name,
                "ok": False,
                "message": str(e),
                "latency_ms": None,
            }
            if deep:
                payload.update(
                    {
                        "probe_ok": False,
                        "probe_message": "Skipped because provider initialization failed",
                        "probe_latency_ms": None,
                    }
                )
            return payload

    async def _run_all_checks() -> list[dict[str, Any]]:
        if deep:
            results: list[dict[str, Any]] = []
            for name in provider_names:
                results.append(await _check_provider(name))
            return results
        return await asyncio.gather(*[_check_provider(n) for n in provider_names])

    results = asyncio.run(_run_all_checks())

    if output_json:
        print(json.dumps({"providers": results}, indent=2))
    else:
        table = Table(title="Provider Status")
        table.add_column("Provider", style="cyan")
        table.add_column("Status")
        table.add_column("Message")
        table.add_column("Latency")
        if deep:
            table.add_column("Probe")
            table.add_column("Probe Latency")

        for r in results:
            status = "[green]OK[/green]" if r["ok"] else "[red]FAIL[/red]"
            latency = f"{r['latency_ms']:.0f}ms" if r["latency_ms"] else "-"
            row = [r["name"], status, r["message"] or "-", latency]
            if deep:
                probe_status = "[green]OK[/green]" if r.get("probe_ok") else "[red]FAIL[/red]"
                probe_latency = (
                    f"{r['probe_latency_ms']:.0f}ms" if r.get("probe_latency_ms") else "-"
                )
                row.extend([f"{probe_status} {r.get('probe_message') or '-'}", probe_latency])
            table.add_row(*row)

        console.print(table)


@app.command("eval")
def evaluate(
    dataset_path: Annotated[
        Path,
        typer.Argument(help="Path to an evaluation dataset (.yaml, .yml, or .json)"),
    ],
    providers: Annotated[
        str | None,
        typer.Option("--providers", "-p", help="Comma-separated provider list"),
    ] = None,
    models: Annotated[
        str | None,
        typer.Option("--models", "-m", help="Comma-separated model IDs for multi-model council"),
    ] = None,
    timeout: Annotated[
        int | None,
        typer.Option("--timeout", help="Override per-provider timeout in seconds"),
    ] = None,
    max_retries: Annotated[
        int | None,
        typer.Option("--max-retries", help="Override synthesis retry count"),
    ] = None,
    runtime_profile: Annotated[
        RuntimeProfile,
        typer.Option(
            "--runtime-profile",
            help="High-level runtime budget override for eval runs: default or bounded",
        ),
    ] = RuntimeProfile.DEFAULT,
    reasoning_profile: Annotated[
        ReasoningProfile,
        typer.Option(
            "--reasoning-profile",
            help="High-level reasoning override for eval runs: default, off, or light",
        ),
    ] = ReasoningProfile.DEFAULT,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
    output_file: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write report to file"),
    ] = None,
    case_ids: Annotated[
        list[str] | None,
        typer.Option("--case", help="Restrict eval to one or more case IDs"),
    ] = None,
    max_cases: Annotated[
        int | None,
        typer.Option("--max-cases", help="Run only the first N selected cases"),
    ] = None,
    fail_fast: Annotated[
        bool,
        typer.Option("--fail-fast", help="Stop on the first failing case"),
    ] = False,
) -> None:
    """Run a dataset-driven evaluation and emit per-mode scorecards."""
    console = _get_console()

    # Apply output_format default from config (CLI flag takes precedence)
    if not output_json:
        config_defaults = _load_config_defaults()
        output_json = config_defaults.get("output_format") == "json"
    else:
        config_defaults = _load_config_defaults()

    if providers:
        provider_list = [p.strip() for p in providers.split(",") if p.strip()]
    else:
        provider_list = config_defaults.get("providers", ["openrouter"])

    try:
        dataset = load_eval_dataset(dataset_path)
        base_config = _build_eval_base_config(
            provider_list,
            models,
            config_defaults,
            timeout_override=timeout,
            max_retries_override=max_retries,
            runtime_profile=runtime_profile,
            reasoning_profile=reasoning_profile,
        )
        report = asyncio.run(
            run_eval_dataset(
                dataset,
                base_config=base_config,
                case_ids=case_ids,
                max_cases=max_cases,
                fail_fast=fail_fast,
            )
        )
    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    _emit_eval_report(report, output_json=output_json, output_file=output_file)
    if report.failed_cases:
        raise typer.Exit(1)


@app.command("eval-compare")
def evaluate_compare(
    dataset_path: Annotated[
        Path,
        typer.Argument(help="Path to an evaluation dataset (.yaml, .yml, or .json)"),
    ],
    variants_path: Annotated[
        Path,
        typer.Argument(help="Path to a named variant file (.yaml, .yml, or .json)"),
    ],
    providers: Annotated[
        str | None,
        typer.Option("--providers", "-p", help="Base comma-separated provider list"),
    ] = None,
    models: Annotated[
        str | None,
        typer.Option("--models", "-m", help="Base comma-separated model IDs"),
    ] = None,
    timeout: Annotated[
        int | None,
        typer.Option("--timeout", help="Override per-provider timeout in seconds"),
    ] = None,
    max_retries: Annotated[
        int | None,
        typer.Option("--max-retries", help="Override synthesis retry count"),
    ] = None,
    runtime_profile: Annotated[
        RuntimeProfile,
        typer.Option(
            "--runtime-profile",
            help="High-level runtime budget override for compare runs: default or bounded",
        ),
    ] = RuntimeProfile.DEFAULT,
    reasoning_profile: Annotated[
        ReasoningProfile,
        typer.Option(
            "--reasoning-profile",
            help="High-level reasoning override for compare runs: default, off, or light",
        ),
    ] = ReasoningProfile.DEFAULT,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
    output_file: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write report to file"),
    ] = None,
    variant_names: Annotated[
        list[str] | None,
        typer.Option("--variant", help="Restrict compare to one or more named variants"),
    ] = None,
    case_ids: Annotated[
        list[str] | None,
        typer.Option("--case", help="Restrict eval to one or more case IDs"),
    ] = None,
    max_cases: Annotated[
        int | None,
        typer.Option("--max-cases", help="Run only the first N selected cases"),
    ] = None,
    fail_fast: Annotated[
        bool,
        typer.Option("--fail-fast", help="Stop on the first failing case"),
    ] = False,
) -> None:
    """Compare multiple named runtime variants on the same evaluation dataset."""
    console = _get_console()

    if not output_json:
        config_defaults = _load_config_defaults()
        output_json = config_defaults.get("output_format") == "json"
    else:
        config_defaults = _load_config_defaults()

    if providers:
        provider_list = [p.strip() for p in providers.split(",") if p.strip()]
    else:
        provider_list = config_defaults.get("providers", ["openrouter"])

    try:
        dataset = load_eval_dataset(dataset_path)
        variants = load_eval_variants(variants_path)
        base_config = _build_eval_base_config(
            provider_list,
            models,
            config_defaults,
            timeout_override=timeout,
            max_retries_override=max_retries,
            runtime_profile=runtime_profile,
            reasoning_profile=reasoning_profile,
        )
        report = asyncio.run(
            run_eval_comparison(
                dataset,
                variants,
                base_config=base_config,
                variant_names=variant_names,
                case_ids=case_ids,
                max_cases=max_cases,
                fail_fast=fail_fast,
            )
        )
    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    _emit_eval_comparison_report(report, output_json=output_json, output_file=output_file)
    if any(item.report.failed_cases for item in report.variant_results):
        raise typer.Exit(1)


def _build_eval_base_config(
    provider_list: list[str],
    models: str | None,
    config_defaults: dict[str, Any],
    *,
    timeout_override: int | None = None,
    max_retries_override: int | None = None,
    runtime_profile: RuntimeProfile = RuntimeProfile.DEFAULT,
    reasoning_profile: ReasoningProfile = ReasoningProfile.DEFAULT,
) -> CouncilConfig:
    """Build the shared CouncilConfig used by eval commands."""

    from llm_council.protocol.types import CouncilConfig

    model_list = [m.strip() for m in models.split(",") if m.strip()] if models else None
    effective_timeout = (
        timeout_override if timeout_override is not None else config_defaults.get("timeout", 120)
    )
    max_retries = (
        max_retries_override
        if max_retries_override is not None
        else config_defaults.get("max_retries", 3)
    )
    enable_degradation = config_defaults.get("enable_degradation", True)

    return CouncilConfig(
        providers=provider_list,
        models=model_list,
        timeout=effective_timeout,
        max_retries=max_retries,
        enable_artifact_store=True,
        enable_health_check=False,
        enable_graceful_degradation=enable_degradation,
        runtime_profile=runtime_profile,
        reasoning_profile=reasoning_profile,
        provider_configs=_load_provider_configs(),
    )


def _emit_eval_report(
    report: EvalReport,
    *,
    output_json: bool,
    output_file: Path | None,
) -> None:
    """Render an evaluation report to stdout and optionally to a file."""

    console = _get_console()
    payload = report.model_dump()

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_json:
            output_file.write_text(json.dumps(payload, indent=2, default=str))
        else:
            output_file.write_text(json.dumps(payload, indent=2, default=str))

    if output_json:
        print(json.dumps(payload, indent=2, default=str))
        return

    summary = (
        f"Cases: {report.passed_cases}/{report.total_cases} passed\n"
        f"Criteria: {report.passed_criteria}/{report.total_criteria} passed\n"
        f"Duration: {report.duration_ms}ms"
    )
    title = f"[green]Eval: {report.dataset_name}[/green]"
    border_style = "green" if report.failed_cases == 0 else "yellow"
    console.print(Panel(summary, title=title, border_style=border_style))

    table = Table(title="Per-Mode Scorecards")
    table.add_column("Mode", style="cyan")
    table.add_column("Cases")
    table.add_column("Case Pass Rate")
    table.add_column("Criteria Pass Rate")
    table.add_column("Avg Duration")

    for item in report.mode_scorecards:
        table.add_row(
            item.mode_key,
            f"{item.passed_cases}/{item.total_cases}",
            f"{item.case_pass_rate:.0%}",
            f"{item.criteria_pass_rate:.0%}",
            f"{item.average_duration_ms}ms",
        )
    console.print(table)

    failing_cases = [case for case in report.case_results if not case.passed]
    if failing_cases:
        console.print("\n[bold]Failing Cases:[/bold]")
        for case in failing_cases:
            failures = [criterion for criterion in case.criteria if not criterion.passed]
            failure_summary = "; ".join(
                f"{criterion.name}: {criterion.message or f'expected {criterion.expected!r}, got {criterion.actual!r}'}"
                for criterion in failures
            )
            console.print(f"  - {case.case_id} ({case.mode_key}): {failure_summary}")


def _emit_eval_comparison_report(
    report: EvalComparisonReport,
    *,
    output_json: bool,
    output_file: Path | None,
) -> None:
    """Render a comparison report across named runtime variants."""

    console = _get_console()
    payload = report.model_dump()

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(payload, indent=2, default=str))

    if output_json:
        print(json.dumps(payload, indent=2, default=str))
        return

    summary = (
        f"Dataset: {report.dataset_name}\n"
        f"Variants: {len(report.variant_results)}\n"
        f"Best: {report.best_variant or '-'}"
    )
    console.print(
        Panel(
            summary,
            title="[green]Eval Comparison[/green]",
            border_style="green" if report.variant_results else "yellow",
        )
    )

    table = Table(title="Variant Ranking")
    table.add_column("Rank")
    table.add_column("Variant", style="cyan")
    table.add_column("Cases")
    table.add_column("Case Pass Rate")
    table.add_column("Criteria Pass Rate")
    table.add_column("Duration")
    table.add_column("Providers")

    results_by_name = {item.variant_name: item for item in report.variant_results}
    for index, variant_name in enumerate(report.ranking, start=1):
        item = results_by_name[variant_name]
        variant_report = item.report
        table.add_row(
            str(index),
            variant_name,
            f"{variant_report.passed_cases}/{variant_report.total_cases}",
            f"{variant_report.case_pass_rate:.0%}",
            f"{variant_report.criteria_pass_rate:.0%}",
            f"{variant_report.duration_ms}ms",
            ", ".join(item.providers),
        )
    console.print(table)


@app.command()
def eval_import_pr(
    repo: Annotated[
        str,
        typer.Argument(help="GitHub repo slug, e.g. sherifkozman/eve"),
    ],
    pr_number: Annotated[
        int,
        typer.Argument(help="Pull request number to import"),
    ],
    output_root: Annotated[
        Path | None,
        typer.Option(
            "--output-root",
            help="Local-only output root. Defaults to .council-private/imports/github",
        ),
    ] = None,
    max_diff_lines: Annotated[
        int | None,
        typer.Option("--max-diff-lines", help="Optionally truncate imported patch to N lines"),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
) -> None:
    """Import a PR diff and Greptile review labels into local-only storage."""
    console = _get_console()

    try:
        imported = import_github_pr_review(
            repo,
            pr_number,
            output_root=output_root,
            max_diff_lines=max_diff_lines,
        )
    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    payload = imported.model_dump()
    if output_json:
        print(json.dumps(payload, indent=2, default=str))
        return

    console.print(
        Panel(
            "\n".join(
                [
                    f"Repo: {imported.repo}",
                    f"PR: {imported.pr_number}",
                    f"Title: {imported.title}",
                    f"Imported Greptile comments: {imported.imported_comment_count}",
                    f"Import root: {imported.import_root}",
                    f"Review input: {imported.review_input_path}",
                    f"Labels: {imported.greptile_labels_path}",
                ]
            ),
            title="[green]PR Import Complete[/green]",
            border_style="green",
        )
    )


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
    default_model: anthropic/claude-opus-4-6

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
            default_cfg = (
                "# LLM Council Configuration\n\n"
                "defaults:\n  providers:\n"
                "    - openrouter\n  timeout: 120\n"
            )
            config_file.write_text(default_cfg)

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
