# Changelog

All notable changes to **cognitive-discovery-system** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v1.1.3] - 2026-06-19

### Patch — type-safety cleanup

A patch release: no API changes, no behavior changes. Resolves the final
block of `mypy --strict` errors so the type checker reports **0 errors
across 136 files**. No public surface changed; all 1165 tests pass and
the suite is unchanged.

### Fixed

- **43 `mypy --strict` errors resolved** across `examples/`, `dashboard/`,
  and `scripts/`. The fixes fall into five categories, none of which
  alter runtime behavior:
  - **`int` → `float` argument types** — numeric literals passed to APIs
    typed as `float` (e.g. integration tolerances, attention scaling)
    now use explicit `float` literals so the call matches the signature.
  - **`Domain` enum usage** — iteration and membership checks against the
    `Domain` enum now go through `Domain.__members__.values()` / explicit
    member references, matching the pattern established in v1.1.2 for
    CodeQL compatibility.
  - **Attention tensor rank** — reshape/transpose calls in
    `examples/nlp_attention_demo.py` and `examples/nlp_mini_gpt_demo.py`
    annotated to reflect the actual tensor rank rather than an
    over-narrow `Any`/scalar guess.
  - **Invariant list widening** — a few demo helper functions that
    return heterogeneous sequences now declare `list[object]` (or the
    specific union) instead of an impossible precise element type.
  - **`Protocol` signature alignment** — a duck-typed callback protocol
    in `scripts/publish.py` had a signature that diverged from its
    concrete implementations; the protocol and call sites now agree.

### Changed

- **Version bump `1.1.2` → `1.1.3`** in `pyproject.toml` and
  `src/cds/_version.py`.

## [v1.1.2] - 2026-06-19

### Patch — security hardening & CodeQL closure

A patch release: no API changes, no behavior changes. Closes the remaining
CodeQL code-scanning alerts with a fix (not a workaround),
tightens `main` branch protection, and improves the discoverability of the
vulnerability-reporting flow. Brings the repo to **0 open** code-scanning
alerts (22/22 closed: 20 fixed by code, 2 dismissed as documented false
positives).

### Security

- **CodeQL `py/non-iterable-in-for-loop` — fix.** Enum iteration
  in `examples/core_demo.py` and `tests/test_hypothesis.py` now uses
  `Domain.__members__.values()` instead of `for x in Domain:`. This is the
  Enum API's standard member-collection view: functionally identical (same
  members, same insertion order — all 1164 tests pass) but the return type
  is an explicitly iterable mapping view that CodeQL's type resolver can
  follow. Previous attempts (idiomatic iteration, direct import from the
  defining module) did not satisfy the analyzer; this does, with no
  camouflage (`list()` wrapping) and no behavior change.
- **`main` branch protection strengthened.** Added `required_status_checks`
  for `CI` and `CodeQL` — a PR can no longer merge into `main` while those
  checks are red. `enforce_admins` is left disabled so the maintainer
  retains a direct-push path; everything else (linear history, conversation
  resolution, no force-pushes, no deletions) is unchanged.
- **Security-reporting guidance.** The README now links to `SECURITY.md`
  and the private-advisory flow; the bug-report issue template warns against
  opening public issues for vulnerabilities and redirects to private
  advisories.

### Changed

- **`CITATION.cff` version refresh.** Two stale `1.1.0` references (root and
  `preferred-citation` blocks) bumped to `1.1.2`. This was missed in v1.1.1.

## [v1.1.1] - 2026-06-19

### Patch — supply-chain & CI hardening

A patch release: no API changes, no behavior changes. Strengthens release
integrity and CI correctness so downstream consumers and contributors get
verifiable, low-churn workflows.

### Security

- **CodeQL static analysis** (`codeql.yml`) — runs `security-and-quality`
  query pack on every push/PR into `main` plus a weekly schedule; findings
  surface as code-scanning alerts under the Security tab. Complements
  `attest.yml` (which signs release artifacts with build provenance) and
  `dependabot.yml` (dependency CVEs).
- **README security badge** — live CodeQL status added next to the CI badge.

### Changed

- **`changelog.yml` made idempotent** — all dispatches now target a single
  fixed branch (`chore/changelog-regen`); `peter-evans/create-pull-request`
  updates the existing open PR instead of opening a duplicate per tag.
  Previously up to 6 stale changelog PRs accumulated; now only one is ever
  open. Also bumped the action `v6 → v7` and added a `concurrency` group to
  serialize simultaneous dispatches.
- **`benchmarks.yml` respects branch protection** — switched from a direct
  bot push (rejected on protected `main`) to opening a PR via
  `peter-evans/create-pull-request@v7`, so regenerated artifacts merge
  through the normal review flow without breaking protection rules.

### Removed

- 6 stale auto-generated changelog PRs (#10, #19, #20, #21, #22, #23) and
  their per-tag branches — all superseded by the hand-curated CHANGELOG on
  `main`.

## [v1.1.0] - 2026-06-19

### Minor — two new modules (symbolic math + knowledge organization)

A backward-compatible feature release. Adds two new public subpackages
(`cds.modeling`, `cds.knowledge`) without changing any existing API. The
platform now spans **17 modules**, **1164 tests**, and **99.59%** coverage.

### Added

- **`cds.modeling`** — symbolic algebra for equation development:
  - An expression tree (`+`, `-`, `*`, `/`, `**`, unary `-`, variables,
    numbers) built by overloading Python operators on a `Var`/`Const` AST.
  - Symbolic differentiation (`diff`), algebraic simplification
    (`simplify`), substitution (`subs`), evaluation (`evalf`), and
    LaTeX export (`to_latex`).
  - A `MathModel` for equation systems with `solve_equation` (bisection
    / Newton-style root finding) and `fit_parameters` (least-squares
    parameter fitting to observations).
  - Runnable demo (`examples/modeling_demo.md`) + tutorial.
- **`cds.knowledge`** — a knowledge-organization layer in three pure-Python,
  dependency-free files:
  - `graph.py`: `Concept`, `Relation` (typed directed edges), and a
    `KnowledgeGraph` with undirected traversal (shortest path via BFS,
    transitive closure, cycle detection), neighbors, and JSON persistence.
  - `notes.py`: `Note` + `Notebook` research notebook with tag and
    concept lookups, and JSON persistence.
  - `retrieval.py`: `search()` producing ranked `SearchResult` hits across
    both concepts and notes (matched field + normalized score).
  - Runnable demo (`examples/knowledge_demo.py`) + tutorial.
- Both modules are wired into the package `__all__`, the CLI `modules`
  listing, and the API reference (`docs/api.md`).

### Maintenance

- Docs: resync README, `docs/index.md`, getting-started (EN + TR), and
  CITATION.cff to v1.1.0 — module count 16→17, test count 883→1164,
  coverage 99.48%→99.59%. Keep `pyproject.toml` + `src/cds/_version.py`
  lockstepped at 1.1.0.
- Chore(repo): ignore the `_demo_*.json` runtime artifacts the
  knowledge demo writes next to its script, so a demo run no longer
  dirties the working tree.

## [v1.0.4] - 2026-06-18

### Security & Tooling — CI hardening

A follow-up patch to v1.0.3. No behavior changes to the published package;
fully backward compatible. Adds continuous dependency-vulnerability scanning
to CI, tightens the type-check configuration, and resynchronizes docs that
had drifted after v1.0.0.

### Maintenance

- Chore(ci): add a `pip-audit` job to the CI workflow (`tests.yml`). Runs
  once on the reference cell (Linux + Python 3.12) against the PyPI Advisory
  database. Currently `continue-on-error: true` — the dev/docs/test toolchain
  (jupyter, mkdocs, build, twine, pillow, lxml, ...) carries transitive deps
  with open CVEs unrelated to the published runtime (typer/pydantic/rich,
  which audit clean). Surfacing them every run gives visibility without
  red-flashing the matrix on a fresh advisory.
- Chore(types): remove the global `ignore_missing_imports = true` from
  `[tool.mypy]`. CDS only depends on PEP 561 typed runtime libs (typer,
  pydantic, rich), all of which resolve cleanly under strict mypy (98 files,
  0 errors). Dropping the escape hatch means a future un-stubbed dependency
  surfaces a real error instead of being silently typed as `Any`.
- Chore(repo): delete a stray Windows `nul` artifact left in the working tree.
- Docs: resync README + `docs/` after v1.0.0 — test count 845→878, version
  strings updated, `CITATION.cff` bumped to 1.0.4. Keep `pyproject.toml` +
  `src/cds/_version.py` lockstepped at 1.0.4.

## [v1.0.3] - 2026-06-18

### Hygiene — test-suite type correctness

A follow-up patch to v1.0.2. No behavior changes to the published package;
fully backward compatible. Brings `tests/` to the same mypy baseline as
`src/` so the full type-check (`mypy src/ tests/`) is green, and resolves
post-release lint/test/publish findings caught after v1.0.2 shipped.

### Testing

- Test(types): clear all remaining mypy errors across `tests/` (39 files,
  0 errors). Fixes list-variance issues (`Parameter <: Tensor`), redundant
  `type: ignore` comments, and intentional error-path `**` operands.
- Test(benchmarks): isolate benchmark test artifacts — `run_all()` /
  `_write_json()` accept an `output_dir` parameter so `pytest` no longer
  clobbers the committed `benchmarks/results.json`.

### Maintenance

- Chore(ci): resolve post-1.0.2 lint/test/publish findings (commit 5e429ec).
- Docs: keep `pyproject.toml` + `src/cds/_version.py` lockstepped at 1.0.3.

## [v1.0.2] - 2026-06-18

### Hygiene — test isolation for benchmark artifacts

A follow-up patch to v1.0.1. No behavior changes to the published package;
fully backward compatible. Resolves a repo-hygiene issue where running the
benchmark test suite silently clobbered the committed
`benchmarks/results.json` artifact (and `docs/benchmarks.md`) on every
`pytest` invocation.

### Bug Fixes

- Fix(tests): `test_run_all_generates_report` no longer writes into the
  working tree. `run_all()` and `_write_json()` now accept an optional
  `output_dir` parameter; the test passes its `tmp_path`, so the committed
  `benchmarks/results.json` and `docs/benchmarks.md` are never modified by a
  test run. Verified: object hash of `results.json` is byte-identical before
  and after `pytest`.

### Testing

- Test(benchmarks): added a regression assertion that `results.json` lands
  in `tmp_path` (not `benchmarks/`) after `run_all(output_dir=...)`.

### Features

- Feat(benchmarks): `run_all(output_dir=None)` and
  `_write_json(record, output_dir=None)` — new optional parameter for
  redirecting artifact output. Defaults preserve existing CLI behaviour.

## [v1.0.1] - 2026-06-18

### Maintenance — typed, documented, hardened

A follow-up to the v1.0 stable release focused on packaging quality and
numerical correctness guarantees. No behavior changes; fully backward
compatible.

### Features

- Feat(packaging): declare PEP 561 type information — ship `src/cds/py.typed`
  marker and force-include it in the wheel so downstream users get full
  static type-checking support out of the box.
- Feat(docs): add "type checked" badge to the README.

### Testing

- Test(invariants): add `tests/test_numerical_invariants.py` — 32
  property-based numerical invariants across 8 modules (linalg, signals,
  stats, quadrature, quantum, diffeq, monte carlo, probability). Fixed
  seed, fully reproducible, zero new dev dependencies.

### Documentation

- Docs(api): add docstrings to all 51 previously-undocumented public
  functions identified via AST scan (3 empty-docstring gaps closed).

### Bug Fixes

- Chore: bump version to 1.0.1.

## [v1.0.0] - 2026-06-18

### 1.0 — Stable release

Cognitive Discovery Platform reaches its first stable release. All five
subsystems of the v1.0 completion spec are implemented, gated, and
verified: 99.16% statement coverage (845 tests passing), zero mypy
errors, clean ruff, and a strict mkdocs build.

### Features

- Feat(nlp): NLP visualisation module — attention heatmap, PCA projection, training curve ([7956597](7956597))
- Feat(benchmarks): emit results.json with timestamp + git SHA provenance, add CI workflow ([4e18500](4e18500))

### Testing

- Test(coverage): close NLP edge-case gaps — tensor/ops/optim/bpe/layers/model/attention; reach 99.16% statement coverage ([6f0a487](6f0a487))
- Style(tests): ruff format cleanup — collapse over-split listcomp, expand train() signature, remove duplicate inline imports ([36f4ee4](36f4ee4))

### Documentation

- Docs(examples): add 9 module demos + matching tutorials ([9f00fc5](9f00fc5))
- Docs(api): restructure mkdocs nav, add api.md lead-ins, fix stale test counts ([4cad8be](4cad8be))
- Docs(nlp): fix griffe warnings — split combined params in docstrings ([3aedff4](3aedff4))
- Docs(benchmarks): regenerate results.json with current SHA provenance ([69c5a88](69c5a88))

### Bug Fixes

- Chore: bump version to 1.0.0 + Production/Stable classifier; sync _version.py

### Other

- Promoted from `Development Status :: 4 - Beta` to `Development Status :: 5 - Production/Stable`.

## [v0.10.0b1] - 2026-06-17


### Features


- Feat(nlp): autograd engine + MiniGPT-from-scratch ([5e0d308](5e0d308))


### Documentation


- Docs(changelog): regenerate for v0.9.0b8 (#17) ([2b0a7d3](2b0a7d3))


## [v0.9.0b8] - 2026-06-17


### Features


- Feat(nlp): add multi-head attention + transformer block ([4f38cae](4f38cae))


### Documentation


- Docs(changelog): regenerate for v0.9.0b7 (#16) ([44ec93c](44ec93c))


## [v0.9.0b7] - 2026-06-17


### Bug Fixes


- Fix: catch two more stale surfaces (CITATION.cff, mkdocs api.md) ([b59b6e3](b59b6e3))


### Documentation


- Docs(changelog): regenerate for v0.9.0b6 (#15) ([e1244ce](e1244ce))


## [v0.9.0b6] - 2026-06-17


### Bug Fixes


- Fix: sync test count 572 to 655 across all surfaces ([952ffe0](952ffe0))


### Documentation


- Docs(changelog): regenerate for v0.9.0b5 (#14) ([f7a037b](f7a037b))


## [v0.9.0b5] - 2026-06-17


### Features


- Feat(nlp): add cds.nlp module — BPE tokenizer + sinusoidal embeddings ([045b9d1](045b9d1))


### Other


- Build: drop hatch-vcs + setuptools_scm, switch to static versioning ([6e04d31](6e04d31))


### Documentation


- Docs(changelog): regenerate for v0.9.0b4 (#13) ([da52ee0](da52ee0))


## [v0.9.0b4] - 2026-06-17


### Bug Fixes


- Fix: drop stale 'auditor' comment + sync 570 to 572 across docs and cli ([0dcaba6](0dcaba6))


### Documentation


- Docs(changelog): regenerate for v0.9.0b4 (#11) ([b1eda78](b1eda78))


## [v0.9.0b3] - 2026-06-17


### Bug Fixes


- Fix(docs): include getting-started.tr.md in mkdocs nav; fix broken CONTRIBUTING link ([85316fe](85316fe))

- Fix: bump CITATION.cff top-level version to 0.9.0b2 ([80cc3b4](80cc3b4))

- Fix: bump CITATION.cff top-level version to 0.9.0b2 (was 0.8.5, missed during release) ([adee53c](adee53c))

- Fix: keep cast() in test_legendre (Python 3.10/3.11 mypy needs it, 3.12 ignores) ([bcd20fc](bcd20fc))

- Fix: resolve 34 mypy errors (matrix/fft type unions, plot float types, missing confidence arg, cli.Path) ([eea36a3](eea36a3))

- Fix: type local _legendre import as Callable[[int, float], tuple[float, float]] (cross-Python mypy fix) ([486bad5](486bad5))

- Fix: use # type: ignore[no-any-return] instead of cast() for cross-Python compat ([bd3aaa8](bd3aaa8))


### Other


- Build: add setuptools_scm local-scheme config (no-local-version, fallback) ([82d07c9](82d07c9))

- Build: exclude hatch-vcs generated _version.py from ruff ([76760d3](76760d3))

- Build: publish.py ignores hatch-vcs regenerated _version.py in dirty check ([bea5f71](bea5f71))

- Build: setuptools_scm version_scheme=no-guess-dev; fix wheel version regex ([8ccb292](8ccb292))

- Release: v0.9.0b1 - Beta milestone, CI multi-OS, hatch-vcs, git-cliff, attestation ([3e3928f](3e3928f))

- Release: v0.9.0b2 - same code, re-tag to trigger attest workflow (checkout fix) ([c54e900](c54e900))

- Release: v0.9.0b3 - branch protection, threat model, i18n, signed commits guide ([31983fe](31983fe))


### Documentation


- Docs(changelog): regenerate for v0.9.0b1 ([44f7823](44f7823))

- Docs(changelog): regenerate for v0.9.0b2 ([404532c](404532c))

- Docs(changelog): regenerate for v0.9.0b3 (#8) ([3a8ef95](3a8ef95))

- Docs(changelog): regenerate for v0.9.0b3 (#9) ([c4fcb9b](c4fcb9b))

- Docs: add threat model, signed commits guide, Why CDS comparison, Turkish i18n; update ROADMAP ([0ba07cf](0ba07cf))

- Docs: update SECURITY.md supported versions (0.6.x → 0.8.x) ([b42cf5f](b42cf5f))


### Styling


- Style: pre-commit auto-fix (trailing whitespace, import order, ruff format) ([9adbf5a](9adbf5a))

- Style: pre-commit auto-fix _version.py (add blank line before __version__) ([277b535](277b535))


### Miscellaneous Tasks


- Ci: changelog PR action needs explicit base branch (checked out on tag) ([b96d322](b96d322))

- Ci: changelog action checks out main (not tag) to avoid detached HEAD issues ([af3cd12](af3cd12))

- Ci: changelog workflow needs write contents + pull-requests perms ([a319a9f](a319a9f))

- Ci: changelog workflow uses PR instead of direct push (branch protection compat) ([3f593cb](3f593cb))

- Ci: fix attest + changelog workflows (checkout step, detached HEAD push) ([d2c9d6e](d2c9d6e))

- Ci: quote step name with colon (YAML parse error) ([0731ca8](0731ca8))

- Ci: rewrite tests.yml with simple bash conditional (no GitHub conditional parse issues) ([2cc8483](2cc8483))

- Ci: split test step into 2 by matrix (gate vs report-only) - fix missing fi ([9e68ff0](9e68ff0))


## [v0.8.5] - 2026-06-16


### Other


- Release: v0.8.5 - cds[all] extras, auto-generated API ref, docs deploy ([12b03b0](12b03b0))


## [v0.8.4] - 2026-06-16


### Other


- Release: v0.8.4 - pydantic upgrade ([58ee8d9](58ee8d9))


## [v0.8.3] - 2026-06-16


### Other


- Release: v0.8.3 - example guard consistency ([5017680](5017680))


## [v0.8.2] - 2026-06-16


### Other


- Release: v0.8.2 - docstrings, knowledge/ removal, CHANGELOG backfill ([f22150c](f22150c))

- Scripts: replace publish.ps1 with pure-Python publish.py ([c7c08e2](c7c08e2))


### Miscellaneous Tasks


- Ci: fix pre-commit and coverage gate for stable CI ([ca66c9f](ca66c9f))

- Ci: remove Docs workflow; relax coverage gate; re-enable legacy Pages ([cdcf5d0](cdcf5d0))


## [v0.8.1] - 2026-06-16


### Other


- Release: v0.8.1 ([4dc49e2](4dc49e2))


### Documentation


- Docs(roadmap): link open enhancement issues #2 and #3 under Longer Term ([ee6103b](ee6103b))

- Docs: mark v0.8.0 as released; move 3 pending perf items to v0.8.1 ([80e6afe](80e6afe))


### Testing


- Test: hit 100% statement coverage; keep branch coverage off ([5c606c4](5c606c4))


### Miscellaneous Tasks


- Ci(pypi): fix pypi-publish workflow ([8335cb6](8335cb6))

- Ci(scripts): add publish.ps1 for local PyPI releases ([57e81a4](57e81a4))

- Ci: harden repo to 10/10 - pre-commit, lockfiles, coverage gate ([7c3b18f](7c3b18f))

- Ci: remove pypi-publish workflow ([043b7f5](043b7f5))


## [v0.8.0] - 2026-06-16


### Other


- Squashed: collapse 108 commits into a single cohesive history ([d4e8aee](d4e8aee))

- Release: v0.8.0 - Performance & Benchmarks ([83211ef](83211ef))


<!-- generated by git-cliff -->
