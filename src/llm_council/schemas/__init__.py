"""
JSON schemas for subagent outputs.

Each subagent has a corresponding schema for structured output validation.
"""

import json
import re
from pathlib import Path
from typing import Any

SCHEMAS_DIR = Path(__file__).parent

# Strict allowlist pattern: lowercase alphanumeric, hyphens, underscores only
_VALID_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _validate_schema_name(name: str) -> None:
    """Validate schema name against strict allowlist to prevent path traversal."""
    if not name:
        raise ValueError("Schema name cannot be empty")
    if not _VALID_NAME_PATTERN.match(name):
        raise ValueError(
            f"Invalid schema name '{name}': must match pattern "
            f"'^[a-z0-9][a-z0-9_-]*$' (lowercase alphanumeric, hyphens, underscores)"
        )


def _ensure_path_containment(path: Path, base_dir: Path) -> None:
    """Ensure resolved path stays within the base directory."""
    resolved = path.resolve()
    base_resolved = base_dir.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        raise ValueError(f"Schema path escapes allowed directory: {path}")


def load_schema(name: str) -> dict[str, Any]:
    """Load a schema by name.

    Args:
        name: Schema name (must match ^[a-z0-9][a-z0-9_-]*$)

    Returns:
        Parsed JSON schema dict

    Raises:
        ValueError: If name is invalid or path escapes schemas directory
        FileNotFoundError: If schema file doesn't exist
    """
    _validate_schema_name(name)
    schema_path = SCHEMAS_DIR / f"{name}.json"
    _ensure_path_containment(schema_path, SCHEMAS_DIR)

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {name}")
    with open(schema_path) as f:
        loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"Schema '{name}' must be a JSON object")
        return loaded


def list_schemas() -> list[str]:
    """List available schema names."""
    return [p.stem for p in SCHEMAS_DIR.glob("*.json")]


__all__ = ["load_schema", "list_schemas", "SCHEMAS_DIR"]
