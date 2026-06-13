# Roadmap

This document outlines the planned direction for CDS. Priorities focus on making the toolkit more useful for research, education, and computational discovery while keeping it lightweight and pure Python.

## Near Term

- Expand the hypothesis module with more domain-specific templates and better offline generation
- Add more runnable demos and examples, especially for combining modules (e.g., generate hypotheses then validate with stats or Monte Carlo)
- Improve performance notes and add more benchmark comparisons for common use cases
- Enhance the CLI with additional utilities (e.g., quick module exploration, result export)

## Medium Term

- Add support for more numerical methods and algorithms useful in scientific workflows
- Improve documentation with more detailed usage guides and research-oriented tutorials
- Add optional (but still lightweight) extensions for common research tasks while preserving the core no-heavy-deps philosophy
- Better integration examples between the discovery tools and the numerical modules

## Longer Term

- Explore community contributions for new modules (ML basics, PDEs, etc.)
- Continue strengthening automation around releases, testing, and documentation so maintainer time stays focused on research features
- Maintain the goal of being an accessible, readable toolkit for students, researchers, and independent developers

Contributions and ideas are welcome, especially around the hypothesis generation and scientific discovery aspects.

## Additional Ideas
- More cross-module demos (hypothesis + stats, quantum + montecarlo, etc.)
- Simple web UI or notebook templates for non-coders
- Better error messages and input validation in core modules
- Integration examples with common research data formats (CSV, HDF5 placeholders)
