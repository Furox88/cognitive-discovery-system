# Cognitive Discovery System (CDS)

[![CI](https://github.com/Furox88/cognitive-discovery-system/actions/workflows/ci.yml/badge.svg)](https://github.com/Furox88/cognitive-discovery-system/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)

**Open-source research assistant for scientific discovery, structured reasoning, and workflow automation.**

CDS helps researchers with AI-powered tools for hypothesis generation, knowledge organization, and reproducible research workflows.

> Currently in **early development**. Contributions welcome!

## Goals

- **Hypothesis generation** — Generate, critique, and refine research hypotheses grounded in existing knowledge
- **Knowledge organization** — Concept graphs, note management, literature linking, and traceable reasoning
- **Research workflow automation** — Reproducible pipelines from idea to analysis to reporting
- **LLM provider agnostic** — Support for OpenAI, xAI, Anthropic, local models, etc.

## Planned Features

| Phase | Feature                      | Status    |
|-------|------------------------------|-----------|
| 1     | AI-assisted idea generation  | In design |
| 1     | Structured research planning | Planned   |
| 2     | Concept relationship mapping | Planned   |
| 2     | Scientific note management   | Planned   |
| 3     | Workflow automation          | Planned   |
| Later | Multi-agent collaboration    | Future    |

## Quick Start

```bash
git clone https://github.com/Furox88/cognitive-discovery-system.git
cd cognitive-discovery-system
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run CLI
cds --help

# Run tests
pytest
```

## Architecture

```
src/cds/
├── core/           # Shared models, config
├── hypothesis/     # Hypothesis generation & evaluation
├── knowledge/      # Notes, concepts, retrieval
├── agents/         # LLM agent abstractions
└── cli.py          # Command-line interface
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

Looking for:
- Researchers with domain expertise
- LLM/agent engineers
- People building similar tools

## License

MIT — see [LICENSE](LICENSE).

## Contact

- Maintainer: [@Furox88](https://github.com/Furox88)
- Issues & Discussions: [GitHub](https://github.com/Furox88/cognitive-discovery-system/issues)
