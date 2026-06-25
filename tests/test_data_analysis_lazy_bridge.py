"""Tests for the optional-extras lazy-loading machinery.

These tests deliberately do **not** import pandas, so they run in *every*
environment — including the minimal-deps CI cell where no optional extra is
installed. They pin two core contracts that must hold even when pandas is
absent:

- The package ``__getattr__`` bridge (``cds/data_analysis/__init__.py``)
  resolves optional symbols on demand and rejects unknown names.
- ``pandas_io._require_pandas`` raises a helpful, install-hint-bearing
  ``ImportError`` when pandas is missing — the path that matters *most*
  precisely when pandas is not installed.

The pandas-exercising round-trip tests live in
``test_data_analysis_pandas_io.py`` and are ``importorskip``'d there, so they
only run when the ``[pandas]`` extra is present. Keeping the lazy-machinery
tests here (rather than in the skip'd module) guarantees the bridge code is
covered even in a zero-dependency environment.
"""

from __future__ import annotations

import builtins
import sys

import pytest

# Use ``import`` (not ``from ... import``) for cds.data_analysis so CodeQL's
# py/import-and-import-from query stays quiet: the lazy-attribute tests below
# also reach the package via `import cds.data_analysis as da`, and mixing the
# two forms is exactly what that query flags. The submodule is accessed as an
# attribute of the package either way.
import cds.data_analysis.pandas_io as pandas_io


# --------------------------------------------------------------------------- #
# Import-guard: missing pandas raises a helpful ImportError
# --------------------------------------------------------------------------- #
def test_missing_pandas_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the lazy import inside _require_pandas to fail.
    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "pandas":
            raise ImportError("simulated missing pandas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Drop any cached pandas module so the import is re-attempted.
    monkeypatch.delitem(sys.modules, "pandas", raising=False)

    with pytest.raises(ImportError, match="optional dependency"):
        pandas_io._require_pandas()


def test_module_attribute_error_for_unknown_name() -> None:
    import cds.data_analysis as da

    with pytest.raises(AttributeError):
        _ = da.nonexistent_function  # noqa: B018


def test_lazy_attribute_exposes_bridge() -> None:
    import cds.data_analysis as da

    # Accessing through the package __getattr__ should resolve to the module fn.
    assert callable(da.to_dataframe)
    assert callable(da.from_dataframe)
