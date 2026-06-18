# Coverage Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the `fail_under = 99` coverage gate to green by adding targeted edge-case tests for the remaining uncovered lines.

**Architecture:** Add tests only — never modify production code. Follow the existing `tests/test_coverage_complete.py` pattern: class-based tests, a docstring naming the target line, `pytest.raises`/mock for defensive branches. Aggregate all new tests into a single new file `tests/test_coverage_nlp_final.py` to keep the coverage series discoverable.

**Tech Stack:** pytest, unittest.mock (for defensive error paths).

**Spec reference:** `docs/superpowers/specs/2026-06-18-project-completion-design.md` §C

**Measured gap (from `pytest --cov=cds.nlp --cov-branch --cov-report=term-missing`):**
- `nlp/autograd/tensor.py` — lines 56, 62, 74, 77, 80, 153, 157, 158, 255; branches 152→153, 156→157, 157→156, 157→158, 253→255 (9 stmts + 5 branches)
- `nlp/optim.py` — lines 124, 130; branches 123→124, 129→130 (Adam validation)
- `nlp/bpe.py` — lines 301, 431; branches 222→225, 230→232, 292→301, 403→402, 426→429, 430→431 (decode non-EOS token + progress print)
- `nlp/layers.py` — lines 175, 179; branches 174→175, 178→179 (`_add` empty + shape mismatch)
- `nlp/autograd/ops.py` — line 39; branch 38→39 (`log(0)` gradient)
- `nlp/model.py` — lines 239, 262, 466; branches 238→239, 261→262, 373→382 (sample temperature + mask branch + `sample()` wrapper)
- `nlp/attention.py` — line 214; branch 213→214 (`merge_heads` empty-rows edge)

Each task below covers one file's gap with concrete tests. Run coverage after each task; commit when that file's missing list is empty.

---

## File Structure

**Created (1 file):**
- `tests/test_coverage_nlp_final.py` — all new edge-case tests

**Modified:** none (production code is read-only).

---

### Task 1: tensor.py — reflected operators, `__pos__`, backward cycle skip

**Files:**
- Create: `tests/test_coverage_nlp_final.py`

- [ ] **Step 1: Write the failing tests for tensor.py reflected ops**

Create `tests/test_coverage_nlp_final.py` with this content (Task 1 of 7):

```python
"""Coverage completion for the educational NLP module.

Targets the last uncovered lines/branches reported by
`pytest --cov=cds.nlp --cov-branch --cov-report=term-missing`:

- nlp/autograd/tensor.py  — reflected dunders (__radd__, __rsub__, ...),
                            __pos__, backward() cycle-skip, TypeError path
- nlp/autograd/ops.py     — log(0) gradient guard
- nlp/optim.py            — Adam validator branches
- nlp/bpe.py              — decode non-EOS token, train progress print
- nlp/layers.py           — _add empty-list and shape-mismatch branches
- nlp/model.py            — sample temperature, attention mask, sample() fn
- nlp/attention.py        — merge_heads empty-rows branch
"""

from __future__ import annotations

import pytest

from cds.nlp import BPEMerge, BPETokenizer, Tensor
from cds.nlp.autograd import Tensor as AGTensor
from cds.nlp.autograd import log as ag_log


# ---------------------------------------------------------------------------
# 1. tensor.py — reflected operators + __pos__ + cycle skip + TypeError
# ---------------------------------------------------------------------------


class TestTensorReflectedOps:
    """Cover __radd__/__rsub__/__rmul__/__rtruediv__ (lines 56,62,74,77)."""

    def test_radd(self) -> None:
        t = AGTensor(data=3.0)
        out = 2.0 + t  # calls __radd__
        assert out.data == 5.0

    def test_rsub(self) -> None:
        t = AGTensor(data=3.0)
        out = 10.0 - t  # calls __rsub__
        assert out.data == 7.0

    def test_rmul(self) -> None:
        t = AGTensor(data=3.0)
        out = 4.0 * t  # calls __rmul__
        assert out.data == 12.0

    def test_rtruediv(self) -> None:
        t = AGTensor(data=2.0)
        out = 10.0 / t  # calls __rtruediv__
        assert out.data == 5.0


class TestTensorPosAndPowType:
    """Cover __pos__ (line 80) and __pow__ TypeError (line 255)."""

    def test_pos_returns_self(self) -> None:
        t = AGTensor(data=5.0)
        assert (+t).data == 5.0

    def test_pow_rejects_tensor_exponent(self) -> None:
        t = AGTensor(data=2.0)
        with pytest.raises(TypeError):
            _ = t ** AGTensor(data=3.0)


class TestBackwardCycleSkip:
    """Cover the `if node in visited: continue` branch (lines 153,157,158).

    Construct a graph where a child is reachable by two paths so the DFS
    encounters a node already in `visited`.
    """

    def test_diamond_graph_visits_each_node_once(self) -> None:
        a = AGTensor(data=2.0, requires_grad=True)
        b = a * AGTensor(data=3.0)
        c = a * AGTensor(data=4.0)
        d = b + c
        d.backward()
        # a appears in two paths; grad accumulates from both b and c
        # db/da = 3, dc/da = 4, so dd/da = 7
        assert a.grad == pytest.approx(7.0)
```

- [ ] **Step 2: Run the tests, verify they pass**

Run: `python -m pytest tests/test_coverage_nlp_final.py -v`
Expected: 7 passed.

- [ ] **Step 3: Verify tensor.py gap closed**

Run: `python -m pytest tests/test_coverage_nlp_final.py --cov=cds.nlp.autograd.tensor --cov-branch --cov-report=term-missing -q`
Expected: `tensor.py` shows no `Missing` lines (lines 56,62,74,77,80,153,157,158,255 all covered).

- [ ] **Step 4: Commit**

```bash
git add tests/test_coverage_nlp_final.py
git commit -m "test(coverage): tensor.py reflected ops, __pos__, cycle skip, TypeError"
```

---

### Task 2: ops.py — log(0) gradient guard

**Files:**
- Modify: `tests/test_coverage_nlp_final.py` (append)

- [ ] **Step 1: Append the ops.py test class**

Append to `tests/test_coverage_nlp_final.py`:

```python
# ---------------------------------------------------------------------------
# 2. autograd/ops.py — log(0) gradient undefined (line 39)
# ---------------------------------------------------------------------------


class TestLogZeroGradient:
    """Cover the `if a.data == 0.0: raise ValueError` branch."""

    def test_log_of_zero_raises_on_backward(self) -> None:
        zero = AGTensor(data=0.0, requires_grad=True)
        y = ag_log(zero)
        with pytest.raises(ValueError, match="log\\(0\\)"):
            y.backward()
```

- [ ] **Step 2: Run, verify**

Run: `python -m pytest tests/test_coverage_nlp_final.py::TestLogZeroGradient -v`
Expected: 1 passed.

- [ ] **Step 3: Verify ops.py gap closed**

Run: `python -m pytest tests/test_coverage_nlp_final.py --cov=cds.nlp.autograd.ops --cov-branch --cov-report=term-missing -q`
Expected: `ops.py` Missing list empty.

- [ ] **Step 4: Commit**

```bash
git add tests/test_coverage_nlp_final.py
git commit -m "test(coverage): ops.py log(0) gradient guard"
```

---

### Task 3: optim.py — Adam validation branches

**Files:**
- Modify: `tests/test_coverage_nlp_final.py` (append)

- [ ] **Step 1: Append the optim.py test class**

```python
# ---------------------------------------------------------------------------
# 3. optim.py — Adam.__post_init__ validation (lines 124, 130)
# ---------------------------------------------------------------------------


from cds.nlp.optim import Adam


class TestAdamValidation:
    """Cover Adam.__post_init__ error branches."""

    def test_rejects_zero_lr(self) -> None:
        with pytest.raises(ValueError, match="lr must be > 0"):
            Adam(params=[], lr=0.0)

    def test_rejects_negative_weight_decay(self) -> None:
        with pytest.raises(ValueError, match="weight_decay must be >= 0"):
            Adam(params=[], lr=0.01, weight_decay=-0.1)
```

- [ ] **Step 2: Run, verify**

Run: `python -m pytest tests/test_coverage_nlp_final.py::TestAdamValidation -v`
Expected: 2 passed.

- [ ] **Step 3: Verify optim.py gap closed**

Run: `python -m pytest tests/test_coverage_nlp_final.py --cov=cds.nlp.optim --cov-branch --cov-report=term-missing -q`
Expected: `optim.py` Missing list empty.

- [ ] **Step 4: Commit**

```bash
git add tests/test_coverage_nlp_final.py
git commit -m "test(coverage): optim.py Adam validation branches"
```

---

### Task 4: layers.py — `_add` empty + shape mismatch

**Files:**
- Modify: `tests/test_coverage_nlp_final.py` (append)

- [ ] **Step 1: Append the layers.py test class**

```python
# ---------------------------------------------------------------------------
# 4. layers.py — _add internal helper (lines 175, 179)
# ---------------------------------------------------------------------------


from cds.nlp.layers import _add


class TestLayersAddHelper:
    """Cover _add empty-list early return and shape-mismatch error."""

    def test_empty_returns_empty(self) -> None:
        assert _add([], []) == []

    def test_shape_mismatch_raises(self) -> None:
        a = [[1.0, 2.0]]
        b = [[1.0, 2.0], [3.0, 4.0]]  # different row count
        with pytest.raises(ValueError, match="add shape mismatch"):
            _add(a, b)
```

- [ ] **Step 2: Run, verify**

Run: `python -m pytest tests/test_coverage_nlp_final.py::TestLayersAddHelper -v`
Expected: 2 passed.

- [ ] **Step 3: Verify layers.py gap closed**

Run: `python -m pytest tests/test_coverage_nlp_final.py --cov=cds.nlp.layers --cov-branch --cov-report=term-missing -q`
Expected: `layers.py` Missing list empty.

- [ ] **Step 4: Commit**

```bash
git add tests/test_coverage_nlp_final.py
git commit -m "test(coverage): layers.py _add empty and mismatch branches"
```

---

### Task 5: bpe.py — decode non-EOS token + train progress print

**Files:**
- Modify: `tests/test_coverage_nlp_final.py` (append)

- [ ] **Step 1: Append the bpe.py test class**

```python
# ---------------------------------------------------------------------------
# 5. bpe.py — decode non-EOS branch (line 301) + train progress print (431)
# ---------------------------------------------------------------------------


from cds.nlp.bpe import train_bpe


class TestBPEDecodeProgress:
    """Cover decode path that appends a non-terminator token, and the
    progress-print branch in train_bpe (line 431)."""

    def test_train_bpe_prints_progress(self, capsys: pytest.CaptureFixture[str]) -> None:
        corpus = ["low lower lowest"]
        train_bpe(corpus, vocab_size=20, show_progress=True)
        out = capsys.readouterr().out
        assert "merge" in out

    def test_decode_reconstructs_multi_token_word(self) -> None:
        # Build a minimal tokenizer whose vocab maps two sub-tokens that
        # decode back to a single word (no EOS terminator between them).
        vocab = {"l": 1, "o": 2, "w": 3, "lo": 4, "low": 5, "</w>": 6, "low</w>": 7}
        merges = [
            BPEMerge(pair=("l", "o"), rank=0, new_token="lo"),
            BPEMerge(pair=("lo", "w"), rank=1, new_token="low"),
            BPEMerge(pair=("low", "</w>"), rank=2, new_token="low</w>"),
        ]
        tok = BPETokenizer(vocab=vocab, merges=merges)
        # Decode the single merged token; exercises the `else: current.append(tok)`
        # branch (line 301) for tokens that are not word-terminators.
        decoded = tok.decode([7])
        assert "low" in decoded
```

- [ ] **Step 2: Run, verify**

Run: `python -m pytest tests/test_coverage_nlp_final.py::TestBPEDecodeProgress -v`
Expected: 2 passed. If the decode test fails because the real vocab/decode API differs, read `src/cds/nlp/bpe.py:280-310` and adjust the token ids so the decoded sequence hits line 301.

- [ ] **Step 3: Verify bpe.py gap closed**

Run: `python -m pytest tests/test_coverage_nlp_final.py --cov=cds.nlp.bpe --cov-branch --cov-report=term-missing -q`
Expected: `bpe.py` Missing list empty.

- [ ] **Step 4: Commit**

```bash
git add tests/test_coverage_nlp_final.py
git commit -m "test(coverage): bpe.py decode non-EOS token + progress print"
```

---

### Task 6: model.py — sample temperature, attention mask, `sample()` fn

**Files:**
- Modify: `tests/test_coverage_nlp_final.py` (append)

- [ ] **Step 1: Read the model.py public surface**

Run: `python -c "from cds.nlp.model import MiniGPT, sample; import inspect; print(inspect.signature(MiniGPT.__init__)); print(inspect.signature(MiniGPT.sample))"`

Record the actual constructor signature so the test builds a valid model.

- [ ] **Step 2: Append the model.py test class**

```python
# ---------------------------------------------------------------------------
# 6. model.py — sample temperature branch (239), attention mask (373->382),
#               sample() wrapper fn (466)
# ---------------------------------------------------------------------------


from cds.nlp.model import MiniGPT, sample


class TestMiniGPTSampling:
    """Cover temperature != 1.0 in sample() and the module-level sample()."""

    def _tiny_model(self) -> MiniGPT:
        # Use the real signature captured in Step 1.
        return MiniGPT(vocab_size=5, d_model=4, n_heads=2, max_len=8, seed=0)

    def test_sample_with_temperature(self) -> None:
        m = self._tiny_model()
        ids = m.sample(prompt_ids=[0, 1], n_tokens=3, temperature=0.5)
        assert isinstance(ids, list)
        assert len(ids) >= 2

    def test_sample_module_function(self) -> None:
        m = self._tiny_model()
        ids = sample(m, prompt_ids=[0], n_tokens=2)
        assert isinstance(ids, list)


class TestMiniGPTAttentionMaskBranch:
    """Cover the masked-attention branch in _scaled_dot_product_attention
    inside model.py (line 373->382) by driving a forward pass with a mask."""

    def test_forward_with_causal_mask(self) -> None:
        m = self._tiny_model_ref()
        # A forward pass on a short prompt exercises the mask branch when
        # the internal attention uses causal masking.
        logits = m.forward([0, 1, 2])
        assert len(logits) == 3

    def _tiny_model_ref(self) -> MiniGPT:
        return MiniGPT(vocab_size=5, d_model=4, n_heads=2, max_len=8, seed=0)
```

If the real `MiniGPT` constructor uses different kwarg names, **adjust `_tiny_model`/`_tiny_model_ref` to match** before running. If the mask branch (373→382) is not reached by a plain forward, inspect `src/cds/nlp/model.py:340-390` to find what argument enables the mask and pass it.

- [ ] **Step 3: Run, verify**

Run: `python -m pytest tests/test_coverage_nlp_final.py::TestMiniGPTSampling tests/test_coverage_nlp_final.py::TestMiniGPTAttentionMaskBranch -v`
Expected: 3 passed.

- [ ] **Step 4: Verify model.py gap closed**

Run: `python -m pytest tests/test_coverage_nlp_final.py --cov=cds.nlp.model --cov-branch --cov-report=term-missing -q`
Expected: `model.py` Missing list empty.

- [ ] **Step 5: Commit**

```bash
git add tests/test_coverage_nlp_final.py
git commit -m "test(coverage): model.py sample temperature, mask, sample() fn"
```

---

### Task 7: attention.py — merge_heads empty-rows branch

**Files:**
- Modify: `tests/test_coverage_nlp_final.py` (append)

- [ ] **Step 1: Read attention.py to understand the empty branch**

Run: `python -c "import inspect,cds.nlp.attention as a; print(inspect.getsource(a.merge_heads))"`

The branch `if n == 0: return [[] for _ in range(n)]` (line 214) fires when the input has rows but each row is empty.

- [ ] **Step 2: Append the attention.py test**

```python
# ---------------------------------------------------------------------------
# 7. attention.py — merge_heads empty-rows branch (line 214)
# ---------------------------------------------------------------------------


from cds.nlp.attention import merge_heads


class TestMergeHeadsEmptyRows:
    """Cover the `if n == 0` branch inside merge_heads."""

    def test_heads_with_empty_rows(self) -> None:
        # n_heads=2, each row is empty -> n == 0 branch
        heads: list[list[list[float]]] = [[[]], [[]]]
        out = merge_heads(heads)
        assert out == []
```

- [ ] **Step 3: Run, verify**

Run: `python -m pytest tests/test_coverage_nlp_final.py::TestMergeHeadsEmptyRows -v`
Expected: 1 passed. (Note: the `n == 0` branch is logically `return []` which is the same as the `not heads` early return — if the branch cannot fire, mark it with a `# pragma: no cover` comment in attention.py instead of a test, and note this in the commit message.)

- [ ] **Step 4: Verify attention.py gap closed**

Run: `python -m pytest tests/test_coverage_nlp_final.py --cov=cds.nlp.attention --cov-branch --cov-report=term-missing -q`
Expected: `attention.py` Missing list empty.

- [ ] **Step 5: Commit**

```bash
git add tests/test_coverage_nlp_final.py
git commit -m "test(coverage): attention.py merge_heads empty-rows branch"
```

---

### Task 8: Final whole-suite coverage check

**Files:** none.

- [ ] **Step 1: Run the full suite with the gate**

Run: `python -m pytest tests/ --cov=cds --cov-branch -q`
Expected: `802 + 17 new = 819 passed, 1 skipped` and **no** `FAIL Required test coverage` message. Coverage should be ≥99.00%.

- [ ] **Step 2: If still below 99%, diagnose**

Run: `python -m pytest tests/ --cov=cds --cov-branch --cov-report=term-missing -q`
Look for remaining `Missing` lines. If they're in NLP files we didn't target, add one more test class following the established pattern. If they're in unrelated modules (e.g. `core/models.py` branch 64→66), add a minimal test:

```python
class TestCoreModelsBranches:
    def test_hypothesis_with_rationale_only(self) -> None:
        from cds.core import Domain, Hypothesis, HypothesisStatus
        h = Hypothesis(
            id="x", domain=Domain.PHYSICS, research_question="q", statement="s",
            status=HypothesisStatus.PROPOSED, confidence=0.5, rationale="because",
        )
        # to_markdown exercises the rationale branch but not the empty-assumptions
        # branch in sequence — driving each optional field independently covers each.
        assert "## Rationale" in h.to_markdown()
```

- [ ] **Step 3: Final commit if Task 8 added anything**

```bash
git add tests/test_coverage_nlp_final.py
git commit -m "test(coverage): close remaining branches to hit 99% gate"
```

- [ ] **Step 4: Confirm gate is green**

Run: `python -m pytest tests/ --cov=cds --cov-branch -q 2>&1 | tail -3`
Expected line present: `Total coverage: 99.0%` or higher, and no `FAIL`.
