# CI Automated Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the benchmark suite regenerate itself on a schedule (weekly) and on release tags, producing a machine-readable `benchmarks/results.json` artifact alongside the human-readable `docs/benchmarks.md`, so performance regressions leave a queryable trail without anyone having to remember to run it by hand.

**Architecture:** Two edits, one new file.

1. **Extend `benchmarks/run_benchmarks.py`** — add a `run_all(emit_json: bool = True)` mode that, in addition to writing `docs/benchmarks.md`, writes `benchmarks/results.json` containing a timestamp, the git SHA (captured via `subprocess`, fails soft when git is absent), and the structured metric tree. The existing CLI behaviour (run → regenerate `docs/benchmarks.md`) is preserved; JSON emission is additive and on by default.
2. **New `.github/workflows/benchmarks.yml`** — triggers on `v*` tag pushes and a weekly cron (Monday 03:00 UTC), checks out, installs, runs `run_benchmarks.py`, then commits the regenerated `docs/benchmarks.md` + `benchmarks/results.json` back to `main` via `stefanzweifel/git-auto-commit-action@v5` (matches the repo's commit-back idiom; the repo already uses `peter-evans/create-pull-request` for the changelog — for benchmarks a direct commit to `main` is appropriate because the artifacts are generated, not authored, and the schedule makes PR-per-run noise unjustified).

**Tech Stack:** stdlib `json`/`datetime`/`subprocess` in the script; GitHub Actions for orchestration. No new Python dependency.

**Spec reference:** `docs/superpowers/specs/2026-06-18-project-completion-design.md` §E

**Cross-cutting constraints honoured:**
- Zero-dependency: `run_benchmarks.py` already imports numpy optionally; we add only stdlib imports.
- No coverage regression: `benchmarks/` is **not** under `--cov=cds` (the script imports `cds` but is not part of the package), and it has no unit tests today. We keep that boundary — the JSON emitter is exercised by a smoke-test invocation in CI itself (Task 3), not by the pytest suite.
- Commit hygiene: workflow commits use `ci(benchmarks):` conventional-commit prefix and a `github-actions[bot]` author.

---

## File Structure

**Created (1 file):**
- `.github/workflows/benchmarks.yml` — the scheduled + on-tag benchmark job

**Modified (1 file):**
- `benchmarks/run_benchmarks.py` — add `results.json` emission + timestamp/SHA provenance

**Generated (2 artifacts, committed by CI, gitignored-or-not — see Task 4):**
- `benchmarks/results.json` — machine-readable
- `docs/benchmarks.md` — human-readable (already generated today)

---

### Task 1: Add JSON + provenance emission to `run_benchmarks.py`

**Files:**
- Modify: `benchmarks/run_benchmarks.py`

- [ ] **Step 1: Read the current `run_all` and the module head**

Run: `head -20 benchmarks/run_benchmarks.py` and `sed -n '210,270p' benchmarks/run_benchmarks.py`
Confirm the structure: `run_all()` builds a `results: dict[str, OrderedDict[str,str]]`, renders the markdown report, writes `docs/benchmarks.md`. The existing `bench_*` functions return string-valued `OrderedDict`s.

- [ ] **Step 2: Add the JSON helper functions near the top**

After the existing `HAS_NUMPY` block (around line 20), add three helpers:

```python
import json
import subprocess
from datetime import datetime, timezone


def _git_sha() -> str:
    """Return the current commit SHA, or ``"unknown"`` if git is unavailable.

    CI runs inside a checkout; local runs may not have git on PATH. Failure
    must never break a benchmark run, so every error path returns a sentinel.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "unknown"


def _coerce_metric(value: str) -> float | str:
    """Best-effort parse of a metric string into a number.

    Benchmark values are formatted as ``"0.0423s"``, ``"727.3x"``, ``"22"``
    (CPU cores), ``"8.0x"``, etc. We strip a trailing unit and try ``float``;
    anything that doesn't parse stays a string (e.g. ``"O(N^3) PLU"``).
    """
    s = value.strip()
    # Strip a single trailing alphabetic unit (s, x) but keep "e" so scientific
    # notation like "1.5e-3s" still parses after the strip.
    stripped = s.rstrip()
    if stripped and stripped[-1].isalpha():
        stripped = stripped[:-1].strip()
    try:
        return float(stripped)
    except ValueError:
        return s


def _build_json_record(
    results: dict[str, "OrderedDict[str, str]"],
) -> dict[str, object]:
    """Assemble the machine-readable artifact: provenance + metric tree."""
    numeric: dict[str, dict[str, float | str]] = {}
    for category, metrics in results.items():
        numeric[category] = {k: _coerce_metric(v) for k, v in metrics.items()}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "metrics": numeric,
    }


def _write_json(record: dict[str, object]) -> Path:
    """Write the JSON artifact next to the script and return its path."""
    out = Path(__file__).resolve().parent / "results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=False)
        f.write("\n")
    return out
```

> Place `import json` / `import subprocess` / `from datetime import ...` with the other top-level imports (merge into the existing block at lines 11–19) rather than scattering them — the snippet above groups them for readability only.

- [ ] **Step 3: Wire JSON emission into `run_all`**

At the end of `run_all()` (currently lines 260–265, the block that writes `docs/benchmarks.md`), extend it:

```python
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    with open(docs_dir / "benchmarks.md", "w", encoding="utf-8") as f:
        f.write(report)

    # Machine-readable artifact for regression tracking (spec §E).
    record = _build_json_record(results)
    json_path = _write_json(record)

    print(f"Benchmarks completed. Report saved to {docs_dir / 'benchmarks.md'}")
    print(f"JSON artifact saved to {json_path}")
```

This keeps the existing CLI contract intact (running the script still regenerates `docs/benchmarks.md`) and **additionally** writes `benchmarks/results.json`.

- [ ] **Step 4: Add a "Last measured" line to the markdown report header**

In the report-builder section (just after `report = "# CDS Performance & Intelligence Report\n\n"`), insert a provenance line so the human-readable doc self-documents when it was last regenerated:

```python
    sha = _git_sha()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = "# CDS Performance & Intelligence Report\n\n"
    report += (
        f"> **Last measured:** `{sha}` on {ts}. Regenerated automatically by "
        f"the `benchmarks` GitHub Actions workflow (weekly + on release tags). "
        f"Raw data: `benchmarks/results.json`.\n\n"
    )
    report += (
        "This report measures both raw speed and algorithmic scaling. ..."
    )
```

(Replace the existing standalone `report += ("This report measures...")` line; don't duplicate it.)

- [ ] **Step 5: Smoke-test the script locally**

Run: `python benchmarks/run_benchmarks.py`
Expected:
- prints the two "saved to ..." lines,
- `docs/benchmarks.md` now begins with a `> **Last measured:**` blockquote,
- `benchmarks/results.json` exists and is valid JSON.

- [ ] **Step 6: Validate the JSON shape**

Run:
```bash
python -c "import json; d=json.load(open('benchmarks/results.json')); assert 'timestamp_utc' in d and 'git_sha' in d and 'metrics' in d; assert 'Linear Algebra (Approaching C-Speed)' in d['metrics']; print('ok', d['git_sha'])"
```
Expected: prints `ok <sha>` (or `ok unknown` if run outside a git repo). If the category key doesn't match, adjust the assert to a category that actually exists in the report.

- [ ] **Step 7: Type-check (best-effort — benchmarks/ is not in the mypy config, but a quick check prevents CI surprises)**

Run: `python -m mypy benchmarks/run_benchmarks.py --ignore-missing-imports`
Expected: no errors in the new helpers. (Pre-existing untyped `OrderedDict` usage may emit notes; those are out of scope.)

- [ ] **Step 8: Commit**

```bash
git add benchmarks/run_benchmarks.py
git commit -m "feat(benchmarks): emit results.json with timestamp + git SHA provenance"
```

---

### Task 2: Create `.github/workflows/benchmarks.yml`

**Files:**
- Create: `.github/workflows/benchmarks.yml`

- [ ] **Step 1: Write the workflow**

Create `.github/workflows/benchmarks.yml`:

```yaml
name: Benchmarks

on:
  schedule:
    # Monday 03:00 UTC — picked off-peak and offset from the default
    # GH Actions cron rush (top of the hour).
    - cron: "0 3 * * 1"
  push:
    tags:
      - "v*"
  # Allow manual runs from the Actions tab for ad-hoc regeneration.
  workflow_dispatch:

permissions:
  contents: write

concurrency:
  group: benchmarks-${{ github.ref }}
  cancel-in-progress: false

jobs:
  regenerate:
    name: Regenerate benchmark report
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
      - name: Checkout (full history for SHA)
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install package
        run: |
          python -m pip install --upgrade pip
          # Install with numpy so the baseline comparison columns populate.
          pip install ".[all]"

      - name: Run benchmarks
        run: python benchmarks/run_benchmarks.py

      - name: Verify artifacts changed
        id: changed
        run: |
          if git diff --quiet docs/benchmarks.md benchmarks/results.json; then
            echo "changed=false" >> "$GITHUB_OUTPUT"
          else
            echo "changed=true" >> "$GITHUB_OUTPUT"
          fi

      - name: Commit regenerated artifacts
        if: steps.changed.outputs.changed == 'true'
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "ci(benchmarks): regenerate report and results.json"
          file_pattern: "docs/benchmarks.md benchmarks/results.json"
          commit_user_name: "github-actions[bot]"
          commit_user_email: "41898282+github-actions[bot]@users.noreply.github.com"
```

Design notes baked into the workflow:
- **Triggers:** weekly cron + `v*` tags + manual `workflow_dispatch` (cheap to add, valuable for "regenerate now" after a perf fix).
- **`fetch-depth: 0`** so the `_git_sha()` helper returns the real commit, not a detached shallow SHA.
- **`pip install ".[all]"`** so the optional-numpy baseline columns in `bench_linalg()` actually populate instead of being silently skipped.
- **`concurrency`** with `cancel-in-progress: false` so a tag push that races the weekly cron never truncates a commit-back.
- **Diff guard** — if nothing changed (e.g. a no-op re-run), skip the commit so we don't generate empty churn commits on `main`.
- **Direct commit to `main`** (not a PR): the artifacts are generated, not authored, and a weekly PR would be pure noise. This matches how generated artifacts are normally handled; if the maintainer later prefers PR review, swap `git-auto-commit-action` for `peter-evans/create-pull-request@v6` (the same action `changelog.yml` uses) and add a `branch:`/`base:` block.

- [ ] **Step 2: Lint the workflow YAML locally**

Run:
```bash
python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/benchmarks.yml')); print('yaml ok')" 2>&1 || \
echo "PyYAML not installed — skip local lint; the GH Actions parser will validate on push."
```
Expected: `yaml ok`, or the skip message. (PyYAML is already a transitive dep of mkdocs; if the install has it, this is a free check.)

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/benchmarks.yml
git commit -m "ci(benchmarks): weekly + on-tag benchmark regeneration workflow"
```

---

### Task 3: Local end-to-end dry run of what CI will do

**Files:** none.

This task de-risks the workflow before it ever runs in GH: it runs the exact sequence CI will run, on the local machine, and asserts the artifacts land where the commit step expects them.

- [ ] **Step 1: Clean slate the generated artifacts**

```bash
# Remove the generated files so we can confirm they're recreated.
rm -f docs/benchmarks.md benchmarks/results.json
```

- [ ] **Step 2: Run the install + benchmark sequence CI runs**

```bash
pip install -e ".[all]" >/dev/null 2>&1 && python benchmarks/run_benchmarks.py
```
Expected: both artifacts recreated, both "saved to ..." lines printed.

- [ ] **Step 3: Confirm both artifacts are present and well-formed**

```bash
test -f docs/benchmarks.md && test -f benchmarks/results.json && \
python -c "import json; json.load(open('benchmarks/results.json')); print('both ok')"
```
Expected: `both ok`.

- [ ] **Step 4: Confirm `git status` shows exactly the two artifacts**

Run: `git status --porcelain docs/benchmarks.md benchmarks/results.json`
Expected: two lines (one per file). If `results.json` does not appear, check that Task 1 Step 3's `_write_json` call is inside `run_all()` and runs unconditionally (not behind a flag).

- [ ] **Step 5: If the dry run needed fixes, fold them in and re-commit**

If Step 1–4 surfaced a bug in `run_benchmarks.py` (e.g. JSON not written, provenance line malformed), fix it now and:

```bash
git add benchmarks/run_benchmarks.py
git commit -m "fix(benchmarks): correct JSON emission flagged by CI dry run"
```

---

### Task 4: Decide the `results.json` gitignore policy

**Files:**
- Modify: `.gitignore` (conditionally)

This is a deliberate decision point, not a rote step. The spec wants a "machine-readable trail for regression detection," which means `results.json` must be **committed** (so each weekly regeneration is a diffable historical point). But `docs/benchmarks.md` is already committed today and shows up clean in `git status`, so by symmetry `results.json` should be committed too.

- [ ] **Step 1: Check whether `.gitignore` currently excludes either artifact**

Run:
```bash
git check-ignore -v docs/benchmarks.md benchmarks/results.json || echo "neither ignored"
```
Expected: `neither ignored` (or, if one is ignored, the matching `.gitignore` line — which would then block the CI commit step and must be fixed).

- [ ] **Step 2: If `results.json` is ignored, allowlist it**

In `.gitignore`, add (negation pattern) near any `*.json` or `results*` rule:

```
# Benchmark artifacts are committed for regression tracking (spec §E).
!benchmarks/results.json
```

- [ ] **Step 3: If a change was needed, commit it**

```bash
git add .gitignore
git commit -m "build: track benchmarks/results.json for regression history"
```

If Step 1 reported `neither ignored`, skip this task entirely — nothing to commit.

---

### Task 5: Document the workflow in the benchmark report + ROADMAP

**Files:**
- Modify: `benchmarks/run_benchmarks.py` (docstring), `ROADMAP.md` (optional checkbox)

- [ ] **Step 1: Update the script module docstring**

In `benchmarks/run_benchmarks.py`, extend the top docstring's "Run directly..." note (around line 8) so users know about the CI path and the JSON artifact:

```python
"""
...
Run directly to regenerate ``docs/benchmarks.md`` and the machine-readable
``benchmarks/results.json`` (timestamped + git-SHA-stamped for regression
tracking):

    python benchmarks/run_benchmarks.py

In CI, the ``Benchmarks`` workflow (``.github/workflows/benchmarks.yml``)
runs this automatically on ``v*`` tag pushes and weekly (Monday 03:00 UTC),
committing both artifacts back to ``main``.
...
"""
```

- [ ] **Step 2: Tick the relevant ROADMAP line (it already tracks this under v0.8.x, but the *automated* part is new)**

In `ROADMAP.md`, under **v1.0.0 — Stability Release**, add a bullet so the new automation is visible (place it among the unchecked items):

```markdown
- [ ] Automated benchmark regeneration on releases + weekly schedule (`.github/workflows/benchmarks.yml`) — CI artifact `benchmarks/results.json` for regression tracking
```

(Leave it unchecked: the workflow file existing doesn't mean it has successfully run end-to-end on `main` yet — that completes on the first tag push or Monday cron after merge.)

- [ ] **Step 3: Commit**

```bash
git add benchmarks/run_benchmarks.py ROADMAP.md
git commit -m "docs(benchmarks): document the CI regeneration workflow + JSON artifact"
```

---

### Task 6: Final verification (the spec's verifiable-success bar)

**Files:** none.

- [ ] **Step 1: Run the full project gate, confirming Task 1 didn't break the package import path**

Run:
```bash
ruff check benchmarks/run_benchmarks.py && \
ruff format --check benchmarks/run_benchmarks.py && \
python benchmarks/run_benchmarks.py && \
test -f benchmarks/results.json && \
test -f docs/benchmarks.md
```
Expected: all green, both artifacts present.

- [ ] **Step 2: Confirm the test suite + coverage gate are unaffected**

Run: `python -m pytest tests/ --cov=cds --cov-branch -q`
Expected: no `FAIL Required test coverage`, count unchanged from baseline. (`benchmarks/` is outside `--cov=cds`, so this plan cannot move the number.)

- [ ] **Step 3: Validate the workflow file is syntactically sound and has the required triggers**

Run:
```bash
python - <<'PY'
import yaml
wf = yaml.safe_load(open('.github/workflows/benchmarks.yml'))
on = wf['on'] if 'on' in wf else wf[True]  # PyYAML parses bare `on:` as True
assert 'schedule' in on, "missing schedule trigger"
assert 'push' in on and 'tags' in on['push'], "missing tag push trigger"
assert wf['jobs']['regenerate']['steps'], "no steps defined"
print('workflow triggers ok:', list(on.keys()))
PY
```
Expected: `workflow triggers ok: ['schedule', 'push', 'workflow_dispatch']`. If PyYAML isn't installed, run `pip install pyyaml` first (it's already a transitive dep of mkdocs-material) or fall back to a visual review of the file.

- [ ] **Step 4: Confirm the "Last measured" line rendered**

Run: `head -5 docs/benchmarks.md`
Expected: the first non-title line is a `> **Last measured:**` blockquote containing a git SHA and a UTC timestamp. If it's missing, Task 1 Step 4's header rewrite didn't land — re-apply it.

- [ ] **Step 5: Note for the maintainer (not a failure)**

The workflow will not actually run until the branch with `.github/workflows/benchmarks.yml` is on the default branch (or a tag is pushed). The first *real* end-to-end CI run happens on the next `v*` tag or the next Monday 03:00 UTC after merge. That is expected and matches how `changelog.yml` already behaves. Surface this in the PR description so the reviewer knows the green local dry run (Task 3) is the acceptance signal, not a green Actions check.
