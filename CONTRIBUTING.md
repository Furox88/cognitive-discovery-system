# Contributing to Cognitive Discovery Platform

Thank you for your interest! CDS is an open-source computational science platform. Your input — whether as a researcher, engineer, or domain expert — is valuable.

## How to Contribute

1. **Open an Issue** for:
   - Bugs
   - Feature requests
   - New module ideas
   - Research use cases

2. **Pull Requests**:
   - Fork the repo
   - Create a feature branch (`git checkout -b feature/amazing-idea`)
   - Make focused changes with clear commit messages
   - Add or update tests where applicable
   - Run `ruff check` and `pytest` before submitting
   - Open a PR with a good description

3. **Discussions**:
   - Use GitHub Discussions for broader ideas

## Development Setup

```bash
git clone https://github.com/Furox88/cognitive-discovery-system.git
cd cognitive-discovery-system
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,test]"

# Install pre-commit hooks (run on every commit)
pip install pre-commit
pre-commit install

# Run tests (802 tests, see CI)
pytest

# Run linter
ruff check src/ tests/
ruff format --check src/ tests/

# Try the CLI
cds --help
cds constants
```

## Pre-commit Hooks

The repo uses pre-commit to run Ruff (lint + format) and Mypy (type check) on every commit. CI runs the same checks, so locally catching them first saves a CI round-trip.

```bash
# Install hooks (one-time, after clone)
pre-commit install

# Run on all files manually
pre-commit run --all-files

# Skip hooks for a single commit (use sparingly)
git commit --no-verify -m "wip: ..."
```

## Signed Commits (recommended)

Signed commits give cryptographic proof that you authored the change. The maintainer's commits are signed; we encourage contributors to do the same but do not require it.

### Option A: SSH signing (simplest)

```bash
# 1. Generate an SSH key (or reuse an existing one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2. Add the PUBLIC key to GitHub: Settings → SSH and GPG keys → New SSH key
#    Paste the contents of ~/.ssh/id_ed25519.pub

# 3. Tell Git to use this key for signing
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub
# Or for SSH auth keys: ~/.ssh/id_ed25519

# 4. Sign your commits
git commit -S -m "fix: ..."
```

### Option B: GPG signing

```bash
# 1. Generate a GPG key
gpg --full-generate-key
# Choose RSA 4096, set Name/Email matching your git config.

# 2. Get the key ID
gpg --list-secret-keys --keyid-format=long

# 3. Export the public key
gpg --armor --export <KEY_ID> > gpg-pubkey.asc
# Add this to GitHub: Settings → SSH and GPG keys → New GPG key

# 4. Tell Git to use this key
git config --global user.signingkey <KEY_ID>

# 5. Sign your commits
git commit -S -m "fix: ..."
```

### Verifying signatures

After pushing, your commits will show a "Verified" badge on GitHub. To verify locally:

```bash
git log --show-signature -1
# gpg: Good signature from ...
```

CI does not enforce signed commits; this is a courtesy recommendation.


git commit --no-verify
```

## Dependency Lockfile

`requirements.lock` (production) and `requirements-dev.lock` (dev/test/docs) are committed to the repo. To regenerate after editing `pyproject.toml`:

```bash
pip install pip-tools
pip-compile pyproject.toml --output-file requirements.lock
pip-compile pyproject.toml --extra dev --extra test --extra docs --output-file requirements-dev.lock
```

## Package Name vs Repository Name

The **PyPI package** is published as `cognitive-discovery-platform` (installed via `pip install cognitive-discovery-platform`), while the **GitHub repository** is named `cognitive-discovery-system`. This is intentional — the repo name was chosen first, and the PyPI name was adjusted to better describe the platform's scope. Both refer to the same project.

## Code Style

- Ruff for linting (configured in `pyproject.toml`)
- Line length: 100 characters
- Prefer clarity and explicitness over cleverness
- All modules are **pure Python** — avoid adding heavy dependencies (NumPy, SciPy)
- Include docstrings with Args/Returns/Raises sections
- Add tests for new functionality

## Module Ideas

Looking for contributions in:
- Machine learning basics (k-means, kNN, decision trees)
- Symbolic math
- Partial differential equations (PDE) solvers
- Information theory (entropy, mutual information)
- Numerical interpolation (Lagrange, splines)
- Combinatorics and number theory

## Running Examples

```bash
python examples/quantum_demo.py
python examples/optimization_demo.py
python examples/signals_demo.py
python examples/stats_demo.py
python examples/hypothesis_demo.py
python examples/hypothesis_custom_generator.py
# See docs/research-workflows.md for embedding patterns and custom generators
```

## Contributing a New or Improved Module

- Keep it **pure Python** (no NumPy, SciPy, etc.)
- Add clear docstrings (Args, Returns, Raises)
- Write tests in `tests/test_*.py`
- Add a runnable example in `examples/`
- Update `docs/api-reference.md` and README module table if needed
- Add entry to CHANGELOG under Unreleased
- Run `ruff check` and `pytest` before PR

## CLI and Hypothesis Contributions

- CLI uses Typer + Rich. New commands go in `src/cds/cli.py`
- Keep help text concise and useful (`cds --help`)
- For the hypothesis module (core to "cognitive discovery"):
  - Improve `PromptTemplate` or offline generator
  - Add domain-specific ideas while keeping the Protocol clean
  - Researchers typically provide their own generator for production use cases
- Add or update smoke tests in `tests/test_cli.py` for new commands

## Documentation & Release Hygiene

- Update relevant docs/ and README when changing behavior
- Keep test count references consistent (exact count via CI and pytest --collect-only)
- Always update CHANGELOG for user-facing changes
- Follow the PR template checklist (tests, lint, changelog)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Open an issue or discussion.

— Furox88 + contributors
