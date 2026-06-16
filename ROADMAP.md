# Roadmap

Planned direction for CDS. All work maintains the **zero-dependency, pure Python** philosophy.

## v0.7.0 — Coverage & Polish

- [x] ~~Push test coverage to **97%+** across every module~~ — achieved **100%**
- [x] ~~Add `typing.Protocol`-based extension points for custom hypothesis generators~~ — shipped as `HypothesisGenerator` Protocol
- [x] ~~Document the PyPI package name (`cognitive-discovery-platform`) vs repo name (`cognitive-discovery-system`) distinction in CONTRIBUTING.md~~ — already documented in §"Package Name vs Repository Name"
- [x] ~~Replace remaining generic error messages with actionable guidance~~ — updated 11 messages in linalg, signals, and stats modules

## v0.8.0 — Performance & Benchmarks (Released 2026-06-16)

- [x] ~~Publish automated benchmark suite (FFT speed, Monte Carlo convergence, LU decomposition vs naive)~~ — see `benchmarks/run_benchmarks.py` & `docs/benchmarks.md`

## v0.8.1 — Performance Polish (Next)

- [ ] Add optional `--num-workers` flag to all parallel Monte Carlo functions
- [ ] Implement caching layer for repeated simulations with identical parameters
- [ ] Reduce overhead in `MultiQubitRegister.measure_shots()` via batched sampling

## v0.9.0 — New Modules & Integration

- [ ] Add `cds.pde` — 1D/2D finite difference solvers for heat, wave, and Laplace equations
- [ ] Extend `cds.optimization.line_search` with Wolfe conditions and strong/weak variants (basic golden-section `line_search` already shipped)
- [ ] Cross-module demos: hypothesis → stats validation, quantum → ML feature pipelines, Monte Carlo → integration benchmarks
- [ ] CSV/JSON round-trip support in `cds.data_analysis.DataSet`

## v1.0.0 — Stability Release

- [ ] Freeze public API — backward-compatible guarantees for all `cds.*` exports
- [ ] Full API reference documentation on GitHub Pages
- [ ] Type stubs (`.pyi`) for IDE autocompletion
- [ ] Security audit (dependency pinning, signed releases)
- [ ] Mark as stable — remove alpha/beta labels

## Longer Term

- Optional lightweight extras: `cds[plot]` for matplotlib integration, `cds[pandas]` for DataFrame interop
- Notebook templates (Jupyter) for non-CLI users
- Community-contributed domain modules (bioinformatics, finance)
- Education-focused "tour of numerical methods" guide
- **Mathematical Modeling Framework** — model creation, analysis, equation development ([#2](https://github.com/Furox88/cognitive-discovery-system/issues/2))
- **Knowledge Organization System** — knowledge graph, concept mapping, structured research notes ([#3](https://github.com/Furox88/cognitive-discovery-system/issues/3))

---

Contributions and ideas are welcome — especially for cross-module demos, new algorithms, and the hypothesis generation system.
