# API Reference Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `mkdocs build --strict` produce a polished, navigable API reference by adding module lead-ins and wiring all new tutorials into the nav.

**Architecture:** Subsystem B is the **sole owner** of `mkdocs.yml` and `docs/api.md`. No other plan touches these files. Work is purely documentation editing — no production code.

**Tech Stack:** Markdown, mkdocstrings, mkdocs Material theme.

**Spec reference:** `docs/superpowers/specs/2026-06-18-project-completion-design.md` §B

**Coordination note:** Plan A (doc examples) creates `docs/tutorials/<module>_demo.md` files; Plan D creates `docs/tutorials/nlp_viz.md`. This plan wires them into `mkdocs.yml`. To avoid serialisation, **run this plan after A and D have landed** (or, if run in parallel, apply the nav additions in a follow-up integration commit once those files exist).

---

## File Structure

**Modified:**
- `docs/api.md` — add a 1–2 sentence lead-in above each `:::` directive
- `mkdocs.yml` — restructure `nav`, add 10 new tutorial entries

**Created:** none.

---

### Task 1: Add module lead-ins to api.md

**Files:**
- Modify: `docs/api.md`

- [ ] **Step 1: Rewrite `docs/api.md` with lead-ins**

The current file already has `:::` directives for every module. Replace it with a version that adds a one-paragraph lead-in above each section. Use this content:

```markdown
# API Reference

Auto-generated reference for every public CDS module. Each entry below is rendered from the module's own docstrings by mkdocstrings.

## Core Data Models

The shared `Domain`, `Hypothesis`, and `HypothesisStatus` types used throughout CDS — the foundation the hypothesis engine builds on.

::: cds.core

## Hypothesis Generation

The cognitive-discovery centrepiece: structured hypothesis generation from a research question, plus a statistical evaluator.

::: cds.hypothesis

## Statistics

Descriptive statistics, regression, and frequentist hypothesis tests (t-test, chi-square, ANOVA, Mann-Whitney, …).

::: cds.stats

## Probability

Continuous PDFs (Gaussian, uniform, exponential) and discrete PMFs (binomial, Poisson) with reproducible sampling.

::: cds.probability

## Mathematical Utilities

Calculus (derivative, integral, gradient) and a compact linear-algebra toolkit (PLU, QR, Cholesky, eigenvalues via power iteration).

::: cds.math_utils

## Numerical Integration

Deterministic quadrature rules: trapezoid, Simpson 1/3 and 3/8, Gauss–Legendre, Romberg, and adaptive Simpson.

::: cds.numerical_integration

## Differential Equations

Initial-value-problem solvers: Euler, midpoint, RK4, adaptive RK45, and a system-of-ODEs integrator.

::: cds.diffeq

## Monte Carlo Methods

Stochastic integration: π estimation, generic Monte-Carlo integration, 1D/2D random walks, and Buffon's needle.

::: cds.montecarlo

## Optimization

Gradient descent, Newton's method, Adam, and golden-section line search.

::: cds.optimization

## Machine Learning

Pure-Python neural networks: an MLP with Adam-based training.

::: cds.ml

## Signal Processing

DFT, radix-2 FFT/IFFT, convolution, and digital filters.

::: cds.signals

## Quantum Computing

Single- and multi-qubit state-vector simulation with O(1) sampling.

::: cds.quantum

## Scientific Computing

Curated physical constants and classical physics formulas (mechanics, waves, relativity, thermo).

::: cds.scientific

## Graph Theory

BFS, DFS, Dijkstra shortest paths, Kruskal MST, topological sort, cycle detection.

::: cds.graph

## Data Analysis

CSV loading, normalisation, smoothing, and ASCII visualisation.

::: cds.data_analysis

## Educational NLP

From-scratch transformer primitives: BPE tokeniser, sinusoidal embeddings, attention, autograd, MiniGPT.

::: cds.nlp
```

- [ ] **Step 2: Verify it builds (strict mode)**

Run: `mkdocs build --strict`
Expected: build succeeds with no warnings about missing references. (If a `:::` target is missing, the module name in pyproject's packages list doesn't match — fix the directive, not the code.)

- [ ] **Step 3: Commit**

```bash
git add docs/api.md
git commit -m "docs(api): add module lead-ins to API reference"
```

---

### Task 2: Restructure mkdocs.yml nav with all tutorials

**Files:**
- Modify: `mkdocs.yml`

- [ ] **Step 1: Replace the `nav:` block**

Open `mkdocs.yml` and replace the existing `nav:` section with this one. It groups tutorials by domain and includes every tutorial produced by Plans A and D.

```yaml
nav:
  - Home: index.md
  - Getting Started:
      - English: getting-started.md
      - Türkçe: getting-started.tr.md
  - Tutorials:
      - Quick Start: tutorials/quick_start.md
      - Core Data Models: tutorials/core_demo.md
      - Scientific Formulas: tutorials/scientific_demo.md
      - Math Utilities: tutorials/math_utils_demo.md
      - Numerical Integration: tutorials/numerical_integration_demo.md
      - Differential Equations: tutorials/diffeq_demo.md
      - Monte Carlo: tutorials/montecarlo_demo.md
      - Probability: tutorials/probability_demo.md
      - Graph Algorithms: tutorials/graph_demo.md
      - Data Analysis: tutorials/data_analysis_demo.md
      - Hypothesis Engine: tutorials/hypothesis_demo.md
      - Quantum Lab: tutorials/quantum_demo.md
      - NLP Visualisation: tutorials/nlp_viz.md
  - API Reference: api.md
  - Case Studies:
      - Hubble Tension: CASE_STUDY_HUBBLE.md
      - Quantum-ML: CASE_STUDY_QUANTUM_ML.md
  - Research Workflows: research-workflows.md
  - Performance: benchmarks.md
  - Maintenance: maintenance.md
```

- [ ] **Step 2: Build with strict mode**

Run: `mkdocs build --strict`
Expected: success. If it fails with `File 'docs/tutorials/X.md' not found`, that tutorial hasn't landed yet — **defer wiring that single entry** until the owning plan finishes, but keep the rest. Record which entries are deferred in the commit message.

- [ ] **Step 3: Commit**

```bash
git add mkdocs.yml
git commit -m "docs(nav): restructure tutorials by domain, add 10 new entries"
```

---

### Task 3: Verify mkdocstrings handler options and final strict build

**Files:**
- Modify: `mkdocs.yml` (only if handler options need tightening)

- [ ] **Step 1: Review current handler options**

Read the `plugins.mkdocstrings.handlers.python.options` block in `mkdocs.yml`. The current options are:
```yaml
options:
  show_source: true
  show_root_heading: true
  show_category_heading: true
  import_column_cols: 0
```

Note: `import_column_cols` is **not a valid mkdocstrings option** (typo for `show_imports`/`members`). Verify by checking mkdocstrings-python docs.

- [ ] **Step 2: Fix the handler options**

Replace the `options:` block with a valid, well-formed one:

```yaml
          options:
            show_source: true
            show_root_heading: true
            show_category_heading: true
            show_symbol_type_heading: true
            show_symbol_type_toc: true
            members_order: source
            separate_signature: true
            show_signature_annotations: true
```

- [ ] **Step 3: Strict build**

Run: `mkdocs build --strict`
Expected: success, no warnings.

- [ ] **Step 4: Commit**

```bash
git add mkdocs.yml
git commit -m "docs(mkdocs): fix invalid handler options, tighten API rendering"
```

---

### Task 4: Cross-check that index.md module list is complete

**Files:**
- Modify: `docs/index.md`

- [ ] **Step 1: Diff the index.md table against the 16 submodules**

The table in `docs/index.md` currently lists 15 modules. Run:
```bash
python -c "import cds, pkgutil; mods=sorted(m.name for m in pkgutil.iter_modules(cds.__path__)); print(len(mods), mods)"
```
and compare to the table.

- [ ] **Step 2: Add any missing module row**

If `core` is missing from the table, add:
```markdown
| `cds.core` | Shared data models (`Domain`, `Hypothesis`, `HypothesisStatus`) |
```

Place it as the first row (it's the foundation). Also bump the test count if the "803 tests" line in `index.md` is stale after Plan C lands more tests — but only touch the test count once Plan C is done.

- [ ] **Step 3: Final strict build**

Run: `mkdocs build --strict`
Expected: success.

- [ ] **Step 4: Commit**

```bash
git add docs/index.md
git commit -m "docs(index): complete module table (core was missing)"
```
