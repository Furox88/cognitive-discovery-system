# Contributing to Cognitive Discovery System

Thank you for your interest! CDS is still in the very early stages. Your input — whether as a researcher, engineer, or domain expert — is extremely valuable.

## How to Contribute

1. **Open an Issue** for:
   - Bugs
   - Feature requests
   - Research use cases / domain pain points
   - Architecture discussions

2. **Pull Requests**:
   - Fork the repo
   - Create a feature branch (`git checkout -b feature/amazing-idea`)
   - Make focused changes with clear commit messages
   - Add or update tests where applicable
   - Open a PR with a good description (what, why, how it was tested)

3. **Discussions**:
   - Use GitHub Discussions for broader ideas (e.g. "Should we integrate with SymPy deeply?" or "Best way to represent concept graphs?")

## Development Setup (Early)

```bash
git clone https://github.com/Furox88/cognitive-discovery-system.git
cd cognitive-discovery-system

# Recommended: use uv or a venv
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run the CLI
cds --help
cds hypothesize "what drives the Hubble tension?" --domain cosmology -n 2 --show-prompt

# Run tests (when we have them)
pytest
```

## Code Style

- Ruff + mypy (configured in pyproject.toml)
- Prefer clarity and explicitness over cleverness
- All scientific claims or assumptions in code should be documented

## Research Contributions

If you are a physicist, cosmologist, mathematician, etc.:

- Share real research questions you face
- Provide examples of good vs. bad hypotheses
- Suggest literature, datasets, or existing tools we should integrate with
- Help define evaluation criteria for generated hypotheses (what makes one "good"?)

## LLM / Agent Integration

We want to be **provider agnostic**.

Priority order for early integrations:
1. xAI Grok (since this is built by someone talking to Grok daily)
2. OpenAI (especially relevant for Codex for OSS program participants)
3. Local models (Ollama, llama.cpp, etc.)
4. Anthropic, Google, etc.

When adding LLM support, always support:
- Dry-run / prompt-only mode
- Structured output (JSON mode or strict schemas)
- Full traceability (log the exact prompt + model + params used)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Open an issue or discussion. We're figuring this out together.

— Furox88 + contributors
