# Contributing

Thanks for wanting to contribute!

## Setup

```bash
git clone https://github.com/Furox88/cognitive-discovery-system.git
cd cognitive-discovery-system
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest  # make sure everything passes
```

## Workflow

1. Fork & create a branch: `git checkout -b feature/my-thing`
2. Write code + tests (aim for 80%+ coverage)
3. Run `pytest` before pushing
4. Open a PR against `main`

## Guidelines

- Type hints on public functions
- Keep docstrings brief
- Every new module needs tests in `tests/`

## Issues

Use the bug report or feature request templates when opening issues.

## License

Contributions are under MIT License.
