"""Shared native CLI path and identity helpers."""

from __future__ import annotations

import os
import shutil


def selected_cli_path(path: str | None, command: str) -> str | None:
    """Resolve before isolated children change cwd; retain the selected entrypoint."""
    selected = path or shutil.which(command)
    if selected is None:
        return None
    return os.path.abspath(shutil.which(selected) or selected)


def native_identity(path: str | None) -> dict[str, str | None]:
    """Safe, request-local identity, not a dump of subprocess output or environment."""
    return {
        "cli_path": path,
        "cli_realpath": os.path.realpath(path) if path else None,
        "cli_version": None,
    }
