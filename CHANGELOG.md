# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-06-09

### Added

- **Hypothesis module** (`cds.hypothesis`): Create, evaluate, and manage scientific hypotheses with lifecycle tracking (draft → proposed → testing → supported/refuted).
- **Mathematical modeling** (`cds.modeling`): Symbolic expression trees with evaluation, numerical differentiation (central differences), and numerical integration (trapezoidal rule).
- **Note management** (`cds.notes`): Research note CRUD with full-text search, tagging, and cross-referencing.
- **Concept mapping** (`cds.concept_map`): Directed graph for concept relationships with BFS pathfinding and search.
- **CLI** (`cds.cli`): Command-line interface for creating hypotheses, notes, and concepts.
- **Test infrastructure**: pytest + pytest-cov with 80% minimum branch coverage.
- **CI**: GitHub Actions workflow running tests on Python 3.9–3.12.
- **Community files**: CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, issue/PR templates.

[0.1.0]: https://github.com/Furox88/cognitive-discovery-system/releases/tag/v0.1.0
