"""Sample tests demonstrating the pytest + coverage scaffolding."""

import pytest

import cds
from cds.example import greeting


def test_version_is_exposed():
    assert cds.__version__ == "0.0.1"


def test_greeting_default():
    assert greeting() == "Hello, researcher!"


def test_greeting_with_name():
    assert greeting("Ada") == "Hello, Ada!"


def test_greeting_strips_whitespace():
    assert greeting("  Ada  ") == "Hello, Ada!"


@pytest.mark.parametrize("bad_name", ["", "   "])
def test_greeting_rejects_empty(bad_name):
    with pytest.raises(ValueError):
        greeting(bad_name)
