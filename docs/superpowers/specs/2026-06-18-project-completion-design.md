# Project Completion Spec — v0.10.0b2 → v1.0.0 Readiness

**Date:** 2026-06-18
**Author:** ZCode (brainstorming session)
**Status:** Approved by maintainer (Furkan)

## Objective

Close the five documented gaps between the current `0.10.0b2` release and the
`v1.0.0` stability milestone defined in `ROADMAP.md`. All work preserves the
project's core philosophy: **zero-dependency, pure Python**.

## Scope Decomposition

Five independent subsystems, each with isolated file boundaries. Four run fully
in parallel; one (`mkdocs.yml`) is coordinated via a single owning agent.

| ID | Subsystem                  | Touches                                  | Conflict risk |
|----|----------------------------|------------------------------------------|---------------|
| A  | Documentation examples     | `examples/*.py`, `docs/tutorials/*.md`   | via mkdocs (B) |
| B  | API reference polish       | `docs/api.md`, `mkdocs.yml`              | owns mkdocs   |
| C  | Coverage restoration       | `tests/test_*.py` (new only)             | none          |
| D  | Sprint 5: NLP visualisation| `src/cds/nlp/viz.py` (new), `__init__.py`| `__init__.py` |
| E  | CI automated benchmarks    | `.github/workflows/benchmarks.yml` (new) | none          |

## Detailed Designs

### A — Documentation Examples & Tutorials

**Goal:** Every public submodule has a runnable example and a tutorial.

**Affected modules (9):** `core`, `data_analysis`, `diffeq`, `graph`,
`math_utils`, `montecarlo`, `numerical_integration`, `probability`, `scientific`.

**Deliverables per module:**
1. `examples/<module>_demo.py` — runnable, mirrors the `stats_demo.py` pattern:
   a `main()` function, clear section banners via `print`, `if __name__ == "__main__"`.
2. `docs/tutorials/<module>_demo.md` — quick-start narrative with copyable
   snippets (matches `docs/tutorials/quick_start.md` style).

**mkdocs nav:** Agent B adds the new tutorial entries to `mkdocs.yml`.

**Quality bar:**
- Every example runs end-to-end without error (`python examples/<x>_demo.py`).
- No network access, no optional dependencies required to run the default path.
- Output is deterministic (fixed seeds where randomness is involved).

### B — API Reference Polish

**Goal:** `mkdocs build --strict` produces a complete, navigable API reference.

**Current state:** `docs/api.md` already contains `:::` directives for every
module; the polish is navigation and per-module framing.

**Deliverables:**
1. Add a one-paragraph lead-in above each `:::` directive in `docs/api.md`
   describing what the module does and when to reach for it.
2. Restructure `mkdocs.yml` nav: group tutorials by domain, add the 9 new
   tutorial entries from subsystem A, add the NLP viz tutorial from subsystem D.
3. Verify `mkdocs build --strict` passes locally.

**Ownership rule:** Subsystem B is the **sole owner** of `mkdocs.yml` to
avoid merge conflicts during parallel execution.

### C — Coverage Restoration (98.59% → ≥99%)

**Goal:** Restore the `fail_under = 99` gate in CI to green.

**Identified uncovered branches (from coverage report):**
- `src/cds/nlp/bpe.py`: lines 222→225, 230→232, 301, 403→402, 426→429, 431
- `src/cds/nlp/layers.py`: lines 175, 179
- `src/cds/nlp/model.py`: lines 239, 262, 373→382, 466
- `src/cds/nlp/optim.py`: lines 124, 130

**Deliverables:**
1. New test files (or additions to existing test files) that exercise the
   missed branches via edge cases: empty inputs, boundary tokens, malformed
   training configs, optimizer edge conditions.
2. Verification: `pytest tests/ --cov=cds --cov-branch --cov-report=term`
   reports ≥99.00% with no `FAIL Required test coverage` message.

**Constraint:** Tests must not alter production behaviour; they only observe it.

### D — Sprint 5: NLP Visualisation

**Goal:** Complete the educational NLP track with visualisation primitives.

**Deliverables:**
1. `src/cds/nlp/viz.py` (new) with three pure-Python functions:
   - `render_attention_heatmap(attn_weights, row_tokens, col_tokens)` — ASCII grid.
   - `render_embedding_projection(embeddings, labels=None, top_n=10)` — ASCII
     2D scatter. Projection via `cds.math_utils.linalg.power_iteration` applied
     to the covariance matrix of the embedding matrix (top-2 eigenvectors).
     This keeps the renderer zero-dependency.
   - `render_training_curve(losses, width=50, height=10)` — ASCII loss curve.
2. Update `src/cds/nlp/__init__.py` to export the three functions under a
   documented `viz` namespace (or directly — matching existing `nlp` export style).
3. `examples/nlp_viz_demo.py` — runnable demonstration of all three.
4. `docs/tutorials/nlp_viz.md` — narrative tutorial (added to nav by B).

**Zero-dependency rule:** ASCII rendering is the default path; matplotlib stays
optional and is **not** imported in the default code path.

### E — CI Automated Benchmarks

**Goal:** Benchmarks regenerate on a schedule and on release tags, with a
machine-readable trail for regression detection.

**Deliverables:**
1. `benchmarks/run_benchmarks.py` extended: emit a `benchmarks/results.json`
   artifact with timestamp, git SHA, and all measured metrics.
2. `.github/workflows/benchmarks.yml` (new) — runs on:
   - push of `v*` tags (release)
   - weekly schedule (Monday 03:00 UTC)
   Workflow: checkout → install → run `run_benchmarks.py` → commit regenerated
   `docs/benchmarks.md` and `benchmarks/results.json` back to `main`.
3. `docs/benchmarks.md` gains a "Last measured" line (commit SHA + date).

## Cross-Cutting Constraints

1. **Zero-dependency** — no new required imports in `src/cds/`. Optional deps
   remain opt-in via extras.
2. **No coverage regression** — subsystems A, B, D, E must not lower the
   measured coverage below 98.59%.
3. **Commit hygiene** — each subsystem lands as its own commit with a
   conventional-commit message (`docs:`, `test:`, `feat(nlp):`, `ci:`).
4. **Verifiable success** — after all five land, the following must pass:
   `ruff check . && ruff format --check . && mypy src/ && mypy tests/ &&
   pytest tests/ --cov=cds --cov-branch && mkdocs build --strict`.

## Out of Scope (Deferred)

- Issues #2 (Mathematical Modeling Framework) and #3 (Knowledge Organization
  System) — separate feature tracks, not part of this completion spec.
- The remaining v1.0.0 ROADMAP items (API freeze policy, security audit,
  removing the beta label) — these are governance/maturity steps that follow
  after the codebase is complete.

## Success Criteria

After this spec is implemented:
- All 16 submodules have examples and tutorials.
- `mkdocs build --strict` passes with a polished, navigable API reference.
- CI coverage gate is green at ≥99%.
- The educational NLP track (Sprints 1–5) is complete.
- Benchmarks are regenerated automatically on releases and weekly.
