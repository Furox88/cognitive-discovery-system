# Roadmap

Planned direction for CDS. All work maintains the **zero-dependency, pure Python** philosophy.

## v0.7.0 — Coverage & Polish (Released)

- [x] ~~Push test coverage to **97%+** across every module~~ — achieved **100%**
- [x] ~~Add `typing.Protocol`-based extension points for custom hypothesis generators~~ — shipped as `HypothesisGenerator` Protocol
- [x] ~~Document the PyPI package name (`cognitive-discovery-platform`) vs repo name (`cognitive-discovery-system`) distinction in CONTRIBUTING.md~~ — already documented in §"Package Name vs Repository Name"
- [x] ~~Replace remaining generic error messages with actionable guidance~~ — updated 11 messages in linalg, signals, and stats modules

## v0.8.x — Performance & Benchmarks (Released 2026-06-16)

- [x] ~~Publish automated benchmark suite (FFT speed, Monte Carlo convergence, LU decomposition vs naive)~~ — see `benchmarks/run_benchmarks.py` & `docs/benchmarks.md`
- [x] ~~Add `cds[all]` meta-extras~~ (single-install convenience)
- [x] ~~Bump pydantic 2.12 → 2.13~~ (security patch)

## v0.9.x — Beta Milestone (Released 2026-06-17)

- [x] ~~Classifier `3 - Alpha` → `4 - Beta`~~ (maturity signal)
- [x] ~~Multi-OS CI matrix~~ (Linux + Windows + macOS × Python 3.10/3.11/3.12)
- [x] ~~Branch coverage enabled~~ (`branch = true` in coverage config)
- [x] ~~Mypy `--strict` on both `src/` and `tests/`~~ (0 issues, 73 files)
- [x] ~~Hatch-vcs dynamic versioning~~ (version derived from git tags, no manual bumps)
- [x] ~~Git-cliff auto-CHANGELOG on tag push~~ (no manual CHANGELOG edits)
- [x] ~~Sigstore release attestation~~ (`actions/attest-build-provenance@v2`)
- [x] ~~Branch protection on main~~ (1 PR reviewer, linear history, no force push, no deletes)
- [x] ~~Threat model in SECURITY.md~~ (in-scope vs out-of-scope, user best practices)
- [x] ~~Türkçe başlangıç rehberi~~ (`docs/getting-started.tr.md`)
- [x] ~~Signed commits guide in CONTRIBUTING.md~~ (SSH or GPG)

## v1.0.0 — Stability Release (Next)

- [ ] Freeze public API — backward-compatible guarantees for all `cds.*` exports
- [ ] Full API reference documentation on GitHub Pages (auto-generated via mkdocstrings; needs polish)
- [ ] Type stubs (`.pyi`) for IDE autocompletion (deferred — pure-Python with full type hints doesn't need them per project policy)
- [ ] Security audit (dependency pinning, signed releases — `requirements.lock` already pinned, sigstore attestation live)
- [ ] **Mark as stable** — remove alpha/beta labels, bump to 1.0.0
- [ ] Optional: enable `required_signatures` in branch protection (after maintainer configures GPG/SSH signing key)
- [ ] Optional: Enable Dependabot security-only updates (currently weekly, can switch to security-only mode)

## Longer Term

- Optional lightweight extras: `cds[plot]` for matplotlib integration, `cds[pandas]` for DataFrame interop
- Notebook templates (Jupyter) for non-CLI users
- Community-contributed domain modules (bioinformatics, finance)
- Education-focused "tour of numerical methods" guide
- **Educational NLP track** (Sprint 1+ — see `cds.nlp`):
  - Sprint 1 ✅ BPE tokenizer + token / positional embeddings (v0.9.0b5)
  - Sprint 2 Multi-head self-attention block (Sprint 2 — pure-Python, slow on purpose)
  - Sprint 3 Pure-Python autograd with optional `cds[fast-jit]` Numba backend (kept opt-in so the core stays zero-dep)
  - Sprint 4 Tiny GPT-from-scratch training run on a public char-level corpus (Karpathy Shakespeare-style)
  - Sprint 5 Attention / embedding visualisations for the educational narrative
  - Scope explicitly excludes production-scale training — that belongs in PyTorch / JAX / MLX.
- **Mathematical Modeling Framework** — model creation, analysis, equation development ([#2](https://github.com/Furox88/cognitive-discovery-system/issues/2))
- **Knowledge Organization System** — knowledge graph, concept mapping, structured research notes ([#3](https://github.com/Furox88/cognitive-discovery-system/issues/3))
- **CDS Script Templates** — domain-specific scientific workflows (quantum chemistry, signal processing demos)

---

Contributions and ideas are welcome — especially for cross-module demos, new algorithms, and the hypothesis generation system.

**Status legend:**
- ✅ Released
- 🔄 In progress
- 📋 Planned
