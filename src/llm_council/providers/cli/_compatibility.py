"""Admit only native releases covered by the synthetic contract matrix."""

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


def check_verified_version(
    observed: str, verified: tuple[str, ...], *, path: str, code: int | None
) -> None:
    """Version/help compatibility cannot prove tool or instruction isolation."""
    if code != 0 or observed not in verified:
        raise RuntimeError(
            f"unsupported or unverified CLI version at {path!r}: observed {observed!r} "
            f"(exit {code}); native-tested versions: {', '.join(verified)}. "
            "Untested updates remain unsupported and require native contract verification."
        )
