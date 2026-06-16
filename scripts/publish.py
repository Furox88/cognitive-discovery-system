#!/usr/bin/env python3
"""Local PyPI publish script for cognitive-discovery-system.

Usage:
    python scripts/publish.py                  # build + test + upload
    python scripts/publish.py --version 0.8.1 # verify version matches
    python scripts/publish.py --dry-run        # build but do not upload
    python scripts/publish.py --skip-tests     # skip the pytest step

Why local-only: this project does not auto-publish from CI. Tag pushes stay
clean. The wheel + sdist are pushed manually after a green local test pass.

Requirements (install once):
    pip install build twine
    # Token: stored at ~/.pypi-token (PyPI API token, pypi-...).
    #   echo 'pypi-Ag...' > ~/.pypi-token && chmod 600 ~/.pypi-token
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKEN_FILE = Path.home() / ".pypi-token"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess, inheriting stdout/stderr. Raise on non-zero exit."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=True, **kwargs)
    return result


def run_capture(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess, capture output, return result without raising."""
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def git(*args: str) -> str:
    return run_capture(["git", *args], cwd=PROJECT_ROOT).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--version", help="Verify pyproject version matches")
    parser.add_argument("--dry-run", action="store_true", help="Build but skip upload")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest step")
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Skip mkdocs gh-deploy (Pages not updated after PyPI upload).",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    # 1. Sanity: clean working tree, on main
    status = git("status", "--porcelain")
    if status:
        print("Working tree is dirty. Commit/stash first:\n", status, file=sys.stderr)
        return 1

    branch = git("branch", "--show-current")
    if branch != "main":
        print(f"Not on main (currently on {branch!r}). Publish from main only.", file=sys.stderr)
        return 1

    # 2. Verify pyproject version
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = next(
        (line for line in pyproject.splitlines() if line.startswith("version = ")),
        None,
    )
    if not match:
        print("Could not find version in pyproject.toml", file=sys.stderr)
        return 1
    pyproject_version = match.split("=", 1)[1].strip().strip('"')

    if args.version and args.version != pyproject_version:
        print(
            f"Version mismatch: pyproject={pyproject_version}, requested={args.version}",
            file=sys.stderr,
        )
        return 1

    print(f"Version: {pyproject_version}", flush=True)

    # 3. Tests
    if not args.skip_tests:
        print("\n=== Running tests ===", flush=True)
        run([sys.executable, "-m", "pytest", "-q"])

    # 4. Clean dist, build
    dist = PROJECT_ROOT / "dist"
    if dist.exists():
        shutil.rmtree(dist)

    print("\n=== Building sdist + wheel ===", flush=True)
    run([sys.executable, "-m", "build"])

    artifacts = sorted(dist.iterdir())
    print("\nBuilt artifacts:", flush=True)
    for path in artifacts:
        size_kb = path.stat().st_size / 1024
        print(f"  {path.name} ({size_kb:.1f} kB)")

    # 5. Upload (unless dry-run)
    if args.dry_run:
        print("\nDry run: skipping upload.", flush=True)
        return 0

    if not TOKEN_FILE.exists():
        print(
            f"No token file at {TOKEN_FILE}. See script header for setup.",
            file=sys.stderr,
        )
        return 1

    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    env = os.environ.copy()
    env["TWINE_USERNAME"] = "__token__"
    env["TWINE_PASSWORD"] = token

    print("\n=== Uploading to PyPI ===", flush=True)
    subprocess.run(
        [sys.executable, "-m", "twine", "upload", "dist/*"],
        check=True,
        env=env,
    )

    print(f"\nDone. v{pyproject_version} published.", flush=True)

    # 6. Deploy docs to GitHub Pages (legacy mode: push to gh-pages branch)
    if not args.skip_docs:
        print("\n=== Deploying docs to GitHub Pages ===", flush=True)
        try:
            run([sys.executable, "-m", "mkdocs", "gh-deploy", "--force", "--no-history"])
        except subprocess.CalledProcessError as exc:
            print(
                f"  mkdocs gh-deploy failed (exit {exc.returncode}). "
                "PyPI upload already succeeded; you can retry with --skip-docs off later.",
                file=sys.stderr,
            )

    print("\nNext: tag the release and create a GitHub release.", flush=True)
    print(f"  git tag -a v{pyproject_version} -m 'Release v{pyproject_version}'")
    print(f"  git push origin v{pyproject_version}")
    print(f"  gh release create v{pyproject_version} --generate-notes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
