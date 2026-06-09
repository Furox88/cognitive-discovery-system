# Getting Started

## Installation

```bash
git clone https://github.com/Furox88/cognitive-discovery-system.git
cd cognitive-discovery-system
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

```bash
# Show version and info
cds --help

# Generate hypotheses (requires LLM API key)
cds hypothesize "what causes the Hubble tension?" --domain cosmology
```

## Running Tests

```bash
pytest
```

## Project Structure

```
src/cds/
├── core/           # Shared models and config
├── hypothesis/     # Hypothesis generation
├── knowledge/      # Notes, concepts, retrieval
├── agents/         # LLM agent abstractions
└── cli.py          # Command-line interface
```
