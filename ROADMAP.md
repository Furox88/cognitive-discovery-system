# Roadmap

Planned direction for CDS. All work maintains the **zero-dependency, pure Python** philosophy.

## v0.7.0 — Coverage & Polish (Next)

- [ ] Push test coverage to **97%+** across every module
- [ ] Add `typing.Protocol`-based extension points for custom hypothesis generators
- [ ] Document the PyPI package name (`cognitive-discovery-platform`) vs repo name (`cognitive-discovery-system`) distinction in CONTRIBUTING.md
- [ ] Replace remaining generic error messages with actionable guidance

## v0.8.0 — Performance & Benchmarks

- [ ] Publish automated benchmark suite (FFT speed, Monte Carlo convergence, LU decomposition vs naive)
- [ ] Add optional `--num-workers` flag to all parallel Monte Carlo functions
- [ ] Implement caching layer for repeated simulations with identical parameters
- [ ] Reduce overhead in `MultiQubitRegister.measure_shots()` via batched sampling

## v0.9.0 — New Modules & Integration

- [ ] Add `cds.pde` — 1D/2D finite difference solvers for heat, wave, and Laplace equations
- [ ] Add `cds.optimization.line_search` — Wolfe conditions, strong/weak line search
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

---

Contributions and ideas are welcome — especially for cross-module demos, new algorithms, and the hypothesis generation system.
