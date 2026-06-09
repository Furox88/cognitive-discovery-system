# Contributing to Cognitive Discovery System

Thank you for your interest! CDS is an open-source computational science toolkit. Your input — whether as a researcher, engineer, or domain expert — is valuable.

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
pip install -e ".[dev]"

# Run tests (267 tests)
pytest

# Run linter
ruff check src/ tests/

# Try the CLI
cds --help
cds constants
```

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
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Open an issue or discussion.

— Furox88 + contributors
