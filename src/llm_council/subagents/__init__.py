"""
Subagent configurations.

YAML-based configurations for each subagent role:
- router, planner, assessor, researcher, architect
- implementer, reviewer, test-designer, shipper, red-team
"""

import re
from pathlib import Path
from typing import Any

import yaml

SUBAGENTS_DIR = Path(__file__).parent

# Strict allowlist pattern: lowercase alphanumeric, hyphens, underscores only
_VALID_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _validate_name(name: str, resource_type: str = "subagent") -> None:
    """Validate resource name against strict allowlist to prevent path traversal."""
    if not name:
        raise ValueError(f"{resource_type} name cannot be empty")
    if not _VALID_NAME_PATTERN.match(name):
        raise ValueError(
            f"Invalid {resource_type} name '{name}': must match pattern "
            f"'^[a-z0-9][a-z0-9_-]*$' (lowercase alphanumeric, hyphens, underscores)"
        )


def _ensure_path_containment(path: Path, base_dir: Path, resource_type: str) -> None:
    """Ensure resolved path stays within the base directory."""
    resolved = path.resolve()
    base_resolved = base_dir.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        raise ValueError(f"{resource_type} path escapes allowed directory: {path}")


def load_subagent(name: str) -> dict[str, Any]:
    """Load a subagent configuration by name.

    Args:
        name: Subagent name (must match ^[a-z0-9][a-z0-9_-]*$)

    Returns:
        Parsed YAML configuration dict

    Raises:
        ValueError: If name is invalid or path escapes subagents directory
        FileNotFoundError: If config file doesn't exist
    """
    _validate_name(name, "subagent")
    config_path = SUBAGENTS_DIR / f"{name}.yaml"
    _ensure_path_containment(config_path, SUBAGENTS_DIR, "Subagent config")

    if not config_path.exists():
        raise FileNotFoundError(f"Subagent config not found: {name}")
    with open(config_path) as f:
        loaded = yaml.safe_load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"Subagent config for '{name}' must be a YAML dictionary")
        return loaded


def list_subagents() -> list[str]:
    """List available subagent names."""
    return [p.stem for p in SUBAGENTS_DIR.glob("*.yaml")]


__all__ = ["load_subagent", "list_subagents", "SUBAGENTS_DIR"]
