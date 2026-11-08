"""Compare pinned npm and Homebrew Codex runtimes on one snapshot record.

This starts the native probe twice. Run only on the approved disposable runner.
"""

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

EXPECTED_PACKAGE_VERSION = "0.167.0"
EXPECTED_VERSION_OUTPUT = f"codex-cli {EXPECTED_PACKAGE_VERSION}"
EXPECTED_SHARED_FILES = 40


def inventory(root: Path) -> tuple[dict[str, str], set[str]]:
    files: dict[str, str] = {}
    executables: set[str] = set()
    for directory, dirs, names in os.walk(root, followlinks=False):
        base = Path(directory)
        if any((base / name).is_symlink() for name in dirs):
            raise ValueError("runtime package contains a symlink directory")
        for name in names:
            path = base / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("runtime package contains a non-regular file")
            relative = path.relative_to(root).as_posix()
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            files[relative] = digest.hexdigest()
            if info.st_mode & 0o111:
                executables.add(relative)
    return files, executables


def common_inventory(npm_root: Path, homebrew_root: Path) -> dict[str, object]:
    npm_files, npm_executables = inventory(npm_root)
    brew_files, brew_executables = inventory(homebrew_root)
    common = {
        name: digest
        for name, digest in npm_files.items()
        if brew_files.get(name) == digest
        and (name in npm_executables) == (name in brew_executables)
    }
    if len(common) != EXPECTED_SHARED_FILES:
        raise ValueError(f"expected {EXPECTED_SHARED_FILES} common files, found {len(common)}")
    executables = sorted(common.keys() & npm_executables)
    if "bin/codex" not in executables:
        raise ValueError("common inventory does not contain executable bin/codex")
    return {"files": dict(sorted(common.items())), "executables": executables}


def require_version(source: str, result: dict[str, object]) -> None:
    if result.get("version") != EXPECTED_VERSION_OUTPUT:
        raise ValueError(f"{source} runtime did not report {EXPECTED_VERSION_OUTPUT}")


def main() -> None:
    if len(sys.argv) != 6:
        raise SystemExit(
            "usage: codex_homebrew_qualification.py NPM_LAUNCHER NPM_RUNTIME_ROOT "
            "HOMEBREW_LAUNCHER HOMEBREW_RUNTIME_ROOT OUTPUT_DIR"
        )
    npm_launcher, npm_root, brew_launcher, brew_root, output = map(Path, sys.argv[1:])
    output.mkdir(parents=True, exist_ok=True)
    shared_path = output / "shared-inventory.json"
    shared_path.write_text(json.dumps(common_inventory(npm_root, brew_root), indent=2) + "\n")
    probe = Path(__file__).with_name("codex_native_platform_probe.py")
    results = {}
    for source, launcher in (("npm", npm_launcher), ("homebrew", brew_launcher)):
        record_path = output / f"{source}-qualification.json"
        subprocess.run(
            [sys.executable, str(probe), str(launcher), str(record_path), str(shared_path)],
            check=True,
            timeout=900,
        )
        result = json.loads(record_path.read_text())
        require_version(source, result)
        results[source] = result
    final_path = output / "qualification.json"
    final_path.write_text(
        json.dumps(
            {
                "version": EXPECTED_PACKAGE_VERSION,
                "common_inventory_sha256": hashlib.sha256(shared_path.read_bytes()).hexdigest(),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
