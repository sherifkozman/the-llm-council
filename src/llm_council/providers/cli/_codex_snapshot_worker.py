"""Copy a record-selected Codex runtime; invoked as an isolated helper process."""

import hashlib
import json
import os
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any

_MAX_FILE_BYTES = 512 * 1024 * 1024
_MAX_TOTAL_BYTES = 768 * 1024 * 1024


def _windows_source(source: Path, relative: PurePosixPath) -> Path:
    """Reject links/junctions and Windows path aliases in the selected tree."""
    path = source
    for part in ("", *relative.parts):
        if part:
            stem = part.split(".")[0].upper()
            if (
                part.rstrip(" .") != part
                or any(c in part for c in '<>"|?*')
                or stem
                in {
                    "CON",
                    "PRN",
                    "AUX",
                    "NUL",
                    *(f"COM{i}" for i in range(1, 10)),
                    *(f"LPT{i}" for i in range(1, 10)),
                }
            ):
                raise ValueError("unsupported Windows path")
            path = path / part
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("runtime source contains a reparse point")
    return path


def copy_windows_runtime(
    source: Path,
    destination: Path,
    expected: dict[str, str],
    executables: list[str],
    *,
    source_entrypoint: str = "bin/codex.exe",
) -> None:
    """Copy verified bytes under the existing same-user trust boundary on Windows."""
    entrypoint = _source_entrypoint(source_entrypoint)
    if (
        not isinstance(expected, dict)
        or not 1 <= len(expected) <= 64
        or "bin/codex" not in expected
        or "bin/codex.exe" in expected
        or not isinstance(executables, list)
        or not all(isinstance(n, str) for n in executables)
        or executables != sorted(set(executables))
        or "bin/codex" not in executables
        or not set(executables) <= set(expected)
    ):
        raise ValueError("invalid Windows file inventory")
    total = 0
    seen: set[str] = set()
    for name, digest in sorted(expected.items()):
        if (
            not isinstance(name, str)
            or len(name) > 240
            or any(c in name for c in ("\0", "\\", ":"))
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("invalid file record")
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or str(relative) != name
            or any(p in (".", "..") for p in name.split("/"))
            or name.casefold() in seen
        ):
            raise ValueError("invalid file path")
        seen.add(name.casefold())
        source_relative = entrypoint if name == "bin/codex" else relative
        path = _windows_source(source, source_relative)
        before = path.lstat()
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_size > _MAX_FILE_BYTES
                or (info.st_dev, info.st_ino) != (before.st_dev, before.st_ino)
            ):
                raise ValueError("unsupported or changed source file")
            target = destination / ("bin/codex.exe" if name == "bin/codex" else name)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            checksum = hashlib.sha256()
            size = 0
            with target.open("xb") as output:
                while chunk := os.read(fd, 1024 * 1024):
                    size += len(chunk)
                    total += len(chunk)
                    if size > _MAX_FILE_BYTES or total > _MAX_TOTAL_BYTES:
                        raise ValueError("runtime exceeds size limit")
                    checksum.update(chunk)
                    output.write(chunk)
            after = _windows_source(source, source_relative).lstat()
            if (after.st_dev, after.st_ino) != (info.st_dev, info.st_ino):
                raise ValueError("source file changed while copying")
            if checksum.hexdigest() != digest:
                raise ValueError("file digest mismatch")
            os.chmod(target, 0o400)
        finally:
            os.close(fd)


def _executable_is_accessible(parent_fd: int, name: str, info: os.stat_result) -> bool:
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (current.st_dev, current.st_ino, current.st_mode) != (
        info.st_dev,
        info.st_ino,
        info.st_mode,
    ):
        return False
    return os.access(name, os.X_OK, dir_fd=parent_fd, effective_ids=True, follow_symlinks=False)


def _source_entrypoint(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or len(value) > 240:
        raise ValueError("invalid selected executable path")
    path = PurePosixPath(value)
    if (
        any(char in value for char in ("\x00", "\\", ":"))
        or path.is_absolute()
        or str(path) != value
        or any(part in (".", "..") for part in value.split("/"))
        or not (len(path.parts) == 1 or (len(path.parts) == 2 and path.parts[0] == "bin"))
    ):
        raise ValueError("invalid selected executable path")
    return path


def copy_runtime(
    source: Path,
    destination: Path,
    expected: dict[str, str],
    executables: list[str],
    *,
    source_entrypoint: str = "bin/codex",
) -> None:
    entrypoint = _source_entrypoint(source_entrypoint)
    if (
        not isinstance(expected, dict)
        or not (1 <= len(expected) <= 64)
        or "bin/codex" not in expected
    ):
        raise ValueError("invalid file inventory")
    if (
        not isinstance(executables, list)
        or not all(isinstance(name, str) for name in executables)
        or executables != sorted(set(executables))
        or "bin/codex" not in executables
        or not set(executables) <= set(expected)
    ):
        raise ValueError("invalid executable inventory")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(source, flags)
    total = 0
    try:
        for name, digest in sorted(expected.items()):
            if (
                not isinstance(name, str)
                or len(name) > 240
                or any(x in name for x in ("\x00", "\\", ":"))
            ):
                raise ValueError("invalid path")
            relative = PurePosixPath(name)
            if (
                relative.is_absolute()
                or str(relative) != name
                or any(part in (".", "..") for part in name.split("/"))
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("invalid file record")
            source_relative = entrypoint if name == "bin/codex" else relative
            parent_fd = os.dup(root_fd)
            try:
                for part in source_relative.parts[:-1]:
                    child_fd = os.open(part, flags, dir_fd=parent_fd)
                    os.close(parent_fd)
                    parent_fd = child_fd
                file_fd = os.open(
                    source_relative.name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=parent_fd,
                )
                try:
                    info = os.fstat(file_fd)
                    if (
                        not stat.S_ISREG(info.st_mode)
                        or info.st_size > _MAX_FILE_BYTES
                        or bool(info.st_mode & 0o111) != (name in executables)
                        or (
                            name in executables
                            and not _executable_is_accessible(parent_fd, source_relative.name, info)
                        )
                    ):
                        raise ValueError("unsupported source file or executable mode")
                    target = destination.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    target_fd = os.open(
                        target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
                    )
                    try:
                        output = os.fdopen(target_fd, "wb")
                    except BaseException:
                        os.close(target_fd)
                        raise
                    checksum = hashlib.sha256()
                    size = 0
                    with output:
                        while chunk := os.read(file_fd, 1024 * 1024):
                            size += len(chunk)
                            total += len(chunk)
                            if size > _MAX_FILE_BYTES or total > _MAX_TOTAL_BYTES:
                                raise ValueError("runtime exceeds size limit")
                            output.write(chunk)
                            checksum.update(chunk)
                    if checksum.hexdigest() != digest:
                        raise ValueError("file digest mismatch")
                    if name in executables and not _executable_is_accessible(
                        parent_fd, source_relative.name, info
                    ):
                        raise ValueError("source executable changed while copying")
                    os.chmod(target, 0o500 if name in executables else 0o400)
                finally:
                    os.close(file_fd)
            finally:
                os.close(parent_fd)
    finally:
        os.close(root_fd)


def copy_candidate_runtime(
    source: Path,
    destination: Path,
    candidates: list[dict[str, Any]],
    *,
    source_entrypoint: str = "bin/codex",
) -> str:
    """Choose by file digest, reject ambiguous inventories, then copy and reverify."""
    entrypoint = _source_entrypoint(source_entrypoint)
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 256:
        raise ValueError("invalid candidate inventory")
    for candidate in candidates:
        if (
            not isinstance(candidate, dict)
            or set(candidate) != {"files", "executables"}
            or not isinstance(candidate["files"], dict)
            or not isinstance(candidate["files"].get("bin/codex"), str)
            or not isinstance(candidate["executables"], list)
        ):
            raise ValueError("invalid candidate inventory")
    if sys.platform == "win32":
        path = _windows_source(source, entrypoint)
        checksum = hashlib.sha256()
        size = 0
        with path.open("rb") as file:
            if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
                raise ValueError("unsupported candidate executable")
            while chunk := file.read(1024 * 1024):
                size += len(chunk)
                if size > _MAX_FILE_BYTES:
                    raise ValueError("candidate executable exceeds size limit")
                checksum.update(chunk)
        digest = checksum.hexdigest()
        matches = [
            candidate for candidate in candidates if candidate["files"]["bin/codex"] == digest
        ]
        if not matches or any(candidate != matches[0] for candidate in matches[1:]):
            raise ValueError("unknown or ambiguous candidate executable")
        copy_windows_runtime(
            source,
            destination,
            matches[0]["files"],
            matches[0]["executables"],
            source_entrypoint=source_entrypoint,
        )
        return digest
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(source, flags)
    try:
        parent_fd = os.dup(root_fd)
        try:
            for part in entrypoint.parts[:-1]:
                child_fd = os.open(part, flags, dir_fd=parent_fd)
                os.close(parent_fd)
                parent_fd = child_fd
            executable_fd = os.open(
                entrypoint.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd
            )
            try:
                info = os.fstat(executable_fd)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_size > _MAX_FILE_BYTES
                    or not _executable_is_accessible(parent_fd, entrypoint.name, info)
                ):
                    raise ValueError("unsupported candidate executable")
                checksum = hashlib.sha256()
                size = 0
                while chunk := os.read(executable_fd, 1024 * 1024):
                    size += len(chunk)
                    if size > _MAX_FILE_BYTES:
                        raise ValueError("candidate executable exceeds size limit")
                    checksum.update(chunk)
                if not _executable_is_accessible(parent_fd, entrypoint.name, info):
                    raise ValueError("candidate executable changed while hashing")
            finally:
                os.close(executable_fd)
        finally:
            os.close(parent_fd)
    finally:
        os.close(root_fd)
    digest = checksum.hexdigest()
    matches = [candidate for candidate in candidates if candidate["files"]["bin/codex"] == digest]
    if not matches or any(candidate != matches[0] for candidate in matches[1:]):
        raise ValueError("unknown or ambiguous candidate executable")
    chosen = matches[0]
    copy_runtime(
        source,
        destination,
        chosen["files"],
        chosen["executables"],
        source_entrypoint=source_entrypoint,
    )
    return digest


if __name__ == "__main__":
    try:
        data = json.load(sys.stdin)
        digest = copy_candidate_runtime(
            Path(data["source"]),
            Path(data["destination"]),
            data["candidates"],
            source_entrypoint=data.get("source_entrypoint", "bin/codex"),
        )
        print(digest)
    except (OSError, ValueError, TypeError, KeyError, RecursionError, NotImplementedError):
        print("Codex runtime snapshot rejected", file=sys.stderr)
        sys.exit(1)
