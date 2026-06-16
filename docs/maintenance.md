# Maintenance & Distribution Guide

This guide explains how to maintain the Cognitive Discovery Platform and publish new versions to PyPI.

## Local Development Setup

To install all dependencies required for development, testing, and documentation:

```bash
pip install -e ".[dev,test,docs]"
```

## Running Quality Checks

Before submitting a PR, ensure all checks pass:

```bash
# Linting
ruff check .

# Type checking
mypy src/

# Testing
pytest tests/
```

## Publishing to PyPI

CDS uses **GitHub Actions** with **Trusted Publishers** (OIDC) to automate releases.

### 1. Update Version
Update the `version` field in `pyproject.toml` and `src/cds/__init__.py`.

### 2. Commit and Tag
Commit your changes and push a new version tag:

```bash
git add .
git commit -m "chore: bump version to vX.Y.Z"
git push origin main

git tag vX.Y.Z
git push origin vX.Y.Z
```

### 3. Automated Release
The `PyPI Publish` workflow will trigger automatically on the tag push. It will:
- Build the source and wheel distributions.
- Securely upload them to PyPI using the OIDC token.

## Documentation Deployment

The MkDocs site is automatically built and deployed to GitHub Pages via the `CI` workflow.
