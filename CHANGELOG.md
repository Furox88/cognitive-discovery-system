# Changelog

All notable changes to **cognitive-discovery-system** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.9.0b4] - 2026-06-17


### <!-- 10 -->💼 Other


- Build: drop hatch-vcs + setuptools_scm, switch to static versioning ([6e04d31](6e04d31))


## [v0.9.0b4] - 2026-06-17


### <!-- 1 -->🐛 Bug Fixes


- Fix: drop stale 'auditor' comment + sync 570 to 572 across docs and cli ([0dcaba6](0dcaba6))


### <!-- 3 -->📚 Documentation


- Docs(changelog): regenerate for v0.9.0b4 (#11) ([b1eda78](b1eda78))


## [v0.9.0b3] - 2026-06-17


### <!-- 1 -->🐛 Bug Fixes


- Fix(docs): include getting-started.tr.md in mkdocs nav; fix broken CONTRIBUTING link ([85316fe](85316fe))

- Fix: bump CITATION.cff top-level version to 0.9.0b2 ([80cc3b4](80cc3b4))

- Fix: bump CITATION.cff top-level version to 0.9.0b2 (was 0.8.5, missed during release) ([adee53c](adee53c))

- Fix: keep cast() in test_legendre (Python 3.10/3.11 mypy needs it, 3.12 ignores) ([bcd20fc](bcd20fc))

- Fix: resolve 34 mypy errors (matrix/fft type unions, plot float types, missing confidence arg, cli.Path) ([eea36a3](eea36a3))

- Fix: type local _legendre import as Callable[[int, float], tuple[float, float]] (cross-Python mypy fix) ([486bad5](486bad5))

- Fix: use # type: ignore[no-any-return] instead of cast() for cross-Python compat ([bd3aaa8](bd3aaa8))


### <!-- 10 -->💼 Other


- Build: add setuptools_scm local-scheme config (no-local-version, fallback) ([82d07c9](82d07c9))

- Build: exclude hatch-vcs generated _version.py from ruff ([76760d3](76760d3))

- Build: publish.py ignores hatch-vcs regenerated _version.py in dirty check ([bea5f71](bea5f71))

- Build: setuptools_scm version_scheme=no-guess-dev; fix wheel version regex ([8ccb292](8ccb292))

- Release: v0.9.0b1 - Beta milestone, CI multi-OS, hatch-vcs, git-cliff, attestation ([3e3928f](3e3928f))

- Release: v0.9.0b2 - same code, re-tag to trigger attest workflow (checkout fix) ([c54e900](c54e900))

- Release: v0.9.0b3 - branch protection, threat model, i18n, signed commits guide ([31983fe](31983fe))


### <!-- 3 -->📚 Documentation


- Docs(changelog): regenerate for v0.9.0b1 ([44f7823](44f7823))

- Docs(changelog): regenerate for v0.9.0b2 ([404532c](404532c))

- Docs(changelog): regenerate for v0.9.0b3 (#8) ([3a8ef95](3a8ef95))

- Docs(changelog): regenerate for v0.9.0b3 (#9) ([c4fcb9b](c4fcb9b))

- Docs: add threat model, signed commits guide, Why CDS comparison, Turkish i18n; update ROADMAP ([0ba07cf](0ba07cf))

- Docs: update SECURITY.md supported versions (0.6.x → 0.8.x) ([b42cf5f](b42cf5f))


### <!-- 5 -->🎨 Styling


- Style: pre-commit auto-fix (trailing whitespace, import order, ruff format) ([9adbf5a](9adbf5a))

- Style: pre-commit auto-fix _version.py (add blank line before __version__) ([277b535](277b535))


### <!-- 7 -->⚙️ Miscellaneous Tasks


- Ci: changelog PR action needs explicit base branch (checked out on tag) ([b96d322](b96d322))

- Ci: changelog action checks out main (not tag) to avoid detached HEAD issues ([af3cd12](af3cd12))

- Ci: changelog workflow needs write contents + pull-requests perms ([a319a9f](a319a9f))

- Ci: changelog workflow uses PR instead of direct push (branch protection compat) ([3f593cb](3f593cb))

- Ci: fix attest + changelog workflows (checkout step, detached HEAD push) ([d2c9d6e](d2c9d6e))

- Ci: quote step name with colon (YAML parse error) ([0731ca8](0731ca8))

- Ci: rewrite tests.yml with simple bash conditional (no GitHub conditional parse issues) ([2cc8483](2cc8483))

- Ci: split test step into 2 by matrix (gate vs report-only) - fix missing fi ([9e68ff0](9e68ff0))


## [v0.8.5] - 2026-06-16


### <!-- 10 -->💼 Other


- Release: v0.8.5 - cds[all] extras, auto-generated API ref, docs deploy ([12b03b0](12b03b0))


## [v0.8.4] - 2026-06-16


### <!-- 10 -->💼 Other


- Release: v0.8.4 - pydantic upgrade ([58ee8d9](58ee8d9))


## [v0.8.3] - 2026-06-16


### <!-- 10 -->💼 Other


- Release: v0.8.3 - example guard consistency ([5017680](5017680))


## [v0.8.2] - 2026-06-16


### <!-- 10 -->💼 Other


- Release: v0.8.2 - docstrings, knowledge/ removal, CHANGELOG backfill ([f22150c](f22150c))

- Scripts: replace publish.ps1 with pure-Python publish.py ([c7c08e2](c7c08e2))


### <!-- 7 -->⚙️ Miscellaneous Tasks


- Ci: fix pre-commit and coverage gate for stable CI ([ca66c9f](ca66c9f))

- Ci: remove Docs workflow; relax coverage gate; re-enable legacy Pages ([cdcf5d0](cdcf5d0))


## [v0.8.1] - 2026-06-16


### <!-- 10 -->💼 Other


- Release: v0.8.1 ([4dc49e2](4dc49e2))


### <!-- 3 -->📚 Documentation


- Docs(roadmap): link open enhancement issues #2 and #3 under Longer Term ([ee6103b](ee6103b))

- Docs: mark v0.8.0 as released; move 3 pending perf items to v0.8.1 ([80e6afe](80e6afe))


### <!-- 6 -->🧪 Testing


- Test: hit 100% statement coverage; keep branch coverage off ([5c606c4](5c606c4))


### <!-- 7 -->⚙️ Miscellaneous Tasks


- Ci(pypi): fix pypi-publish workflow ([8335cb6](8335cb6))

- Ci(scripts): add publish.ps1 for local PyPI releases ([57e81a4](57e81a4))

- Ci: harden repo to 10/10 - pre-commit, lockfiles, coverage gate ([7c3b18f](7c3b18f))

- Ci: remove pypi-publish workflow ([043b7f5](043b7f5))


## [v0.8.0] - 2026-06-16


### <!-- 10 -->💼 Other


- Squashed: collapse 108 commits into a single cohesive history ([d4e8aee](d4e8aee))

- Release: v0.8.0 - Performance & Benchmarks ([83211ef](83211ef))


<!-- generated by git-cliff -->
