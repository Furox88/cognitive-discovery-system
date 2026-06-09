# Contributing to Cognitive Discovery System

Thank you for your interest in contributing! This guide will help you get started.

## Getting Started

1. **Fork** the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/cognitive-discovery-system.git
   cd cognitive-discovery-system
   ```

2. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[test]"
   ```

3. **Run the test suite** to make sure everything works:
   ```bash
   pytest
   ```

## Development Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the code style of the existing codebase.

3. Add or update tests for your changes. We aim for **≥80% coverage**.

4. Run the test suite:
   ```bash
   pytest
   ```

5. Commit your changes with a clear message:
   ```bash
   git commit -m "Add: brief description of your change"
   ```

6. Push and open a Pull Request against `main`.

## Code Guidelines

- **Type hints**: All public functions and methods must have type annotations.
- **Docstrings**: Use Google-style docstrings for modules, classes, and public functions.
- **Tests**: Every new module must include a corresponding test file in `tests/`.
- **Coverage**: The CI enforces a minimum of 80% branch coverage.

## Reporting Issues

- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) for bugs.
- Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) for new ideas.

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
