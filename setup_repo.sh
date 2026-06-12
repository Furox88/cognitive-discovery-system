#!/usr/bin/env bash
#
# CDS (Cognitive Discovery System) - One-command development setup
#
# This script sets up a clean development environment for contributors
# and maintainers. It creates a virtualenv, installs the package in
# editable mode with dev dependencies, and runs the test suite.
#
# Usage:
#   ./setup_repo.sh
#
# After running you can use:
#   source .venv/bin/activate
#   cds --help
#   pytest
#

set -euo pipefail

echo "=== Cognitive Discovery System - Developer Setup ==="
echo

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo "Using existing .venv"
fi

# Activate venv for the rest of the script
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip -q

echo "Installing package in editable mode with [dev] extras..."
pip install -e ".[dev]" -q

echo
echo "Running tests to verify the environment..."
pytest --tb=no -q

echo
echo "=== Setup complete! ==="
echo
echo "Activate the environment with:"
echo "    source .venv/bin/activate"
echo
echo "Useful commands:"
echo "    cds --help"
echo "    cds modules"
echo "    cds hypothesis \"What causes the Hubble tension?\""
echo "    cds hypothesize \"Your research question here\" -n 3"
echo "    cds constants"
echo "    pytest"
echo
echo "Happy hacking! Contributions welcome."

