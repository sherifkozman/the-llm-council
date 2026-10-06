"""Admit only native releases covered by the synthetic contract matrix."""

from __future__ import annotations


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
