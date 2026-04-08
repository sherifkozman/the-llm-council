"""llm-council package."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = None

from .council import Council
from .engine.orchestrator import CostEstimate, CouncilResult, OrchestratorConfig
from .protocol.types import CouncilConfig
from .providers.base import (
    DoctorResult,
    GenerateRequest,
    GenerateResponse,
    GenerateResult,
    Message,
    ProviderAdapter,
    ProviderCapabilities,
    ProviderCapabilityName,
)
from .providers.registry import ProviderRegistry, get_registry

_FALLBACK_VERSION = "0.0.0+unknown"
_DIST_NAME = "the-llm-council"
_PROJECT_NAME_RE = re.compile(r'^name\s*=\s*["\']([^"\']+)["\']\s*$')
_PROJECT_VERSION_RE = re.compile(r'^version\s*=\s*["\']([^"\']+)["\']\s*$')


def _read_local_project_version(start: Path | None = None) -> str | None:
    """Read the repo version directly from the nearest pyproject.toml."""
    module_path = (start or Path(__file__)).resolve()
    for parent in module_path.parents:
        pyproject = parent / "pyproject.toml"
        if not pyproject.exists():
            continue
        try:
            text = pyproject.read_text(encoding="utf-8")
        except OSError:
            continue

        parsed_version = _read_project_version_from_toml(text)
        if parsed_version is not None:
            return parsed_version

        in_project_section = False
        project_name: str | None = None
        project_version: str | None = None
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                if in_project_section and project_name == _DIST_NAME and project_version:
                    return project_version
                in_project_section = stripped == "[project]"
                continue
            if not in_project_section:
                continue
            name_match = _PROJECT_NAME_RE.match(stripped)
            if name_match:
                project_name = name_match.group(1)
                continue
            match = _PROJECT_VERSION_RE.match(stripped)
            if match:
                project_version = match.group(1)
        if project_name == _DIST_NAME and project_version:
            return project_version
    return None


def _read_project_version_from_toml(text: str) -> str | None:
    """Parse pyproject content with tomllib when available."""
    if tomllib is None:
        return None
    try:
        data = tomllib.loads(text)
    except Exception:
        return None
    project = data.get("project")
    if not isinstance(project, dict):
        return None
    if project.get("name") != _DIST_NAME:
        return None
    version = project.get("version")
    return version if isinstance(version, str) else None


def _detect_version(module_path: Path | None = None) -> str:
    """Prefer the local source tree version before installed metadata."""
    local_version = _read_local_project_version(module_path)
    if local_version:
        return local_version

    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version(_DIST_NAME)
    except Exception:
        return _FALLBACK_VERSION


__version__ = _detect_version()

__all__ = [
    "__version__",
    "Council",
    "CouncilConfig",
    "CouncilResult",
    "CostEstimate",
    "DoctorResult",
    "GenerateRequest",
    "GenerateResponse",
    "GenerateResult",
    "Message",
    "OrchestratorConfig",
    "ProviderAdapter",
    "ProviderCapabilityName",
    "ProviderCapabilities",
    "ProviderRegistry",
    "get_registry",
]
