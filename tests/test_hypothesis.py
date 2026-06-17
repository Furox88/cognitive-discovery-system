"""Tests for hypothesis generation module."""

from cds.core.models import Domain, Hypothesis, HypothesisStatus
from cds.hypothesis.generator import (
    PromptTemplate,
    SimpleOfflineGenerator,
    generate_hypotheses,
)


def test_generate_hypotheses_returns_list() -> None:
    hypos = generate_hypotheses(
        "Hubble tension resolution mechanisms",
        domain=Domain.COSMOLOGY,
        n=2,
    )
    assert len(hypos) == 2
    for h in hypos:
        assert isinstance(h, Hypothesis)
        assert h.research_question == "Hubble tension resolution mechanisms"
        assert h.domain == Domain.COSMOLOGY
        assert h.status == HypothesisStatus.NEW
        assert 0.0 <= h.confidence <= 1.0


def test_prompt_template_renders() -> None:
    prompt = PromptTemplate.render("Test question about dark energy", Domain.COSMOLOGY, n=3)
    assert "Research Question: Test question about dark energy" in prompt
    assert "Domain focus: cosmology" in prompt
    assert "Generate 3 distinct hypotheses" in prompt


def test_offline_generator_is_deterministic_enough() -> None:
    gen = SimpleOfflineGenerator()
    h1 = gen.generate("foo", domain=Domain.COSMOLOGY, n=1)
    h2 = gen.generate("foo", domain=Domain.COSMOLOGY, n=1)
    assert len(h1) == 1 and len(h2) == 1
    assert h1[0].domain == Domain.COSMOLOGY


def test_generate_single_hypothesis() -> None:
    hypos = generate_hypotheses("dark matter", domain=Domain.PHYSICS, n=1)
    assert len(hypos) == 1
    assert hypos[0].domain == Domain.PHYSICS


def test_generate_multiple_hypotheses() -> None:
    hypos = generate_hypotheses("protein folding", domain=Domain.BIOLOGY, n=5)
    assert len(hypos) == 5
    for h in hypos:
        assert h.domain == Domain.BIOLOGY


def test_hypothesis_to_markdown() -> None:
    h = Hypothesis(
        id="test-1",
        statement="Test statement",
        domain=Domain.PHYSICS,
        research_question="test question",
        rationale="test rationale",
        confidence=0.8,
        assumptions=["a1", "a2"],
        predictions=["p1"],
    )
    md = h.to_markdown()
    assert "test-1" in md
    assert "Test statement" in md
    assert "test rationale" in md
    assert "a1" in md
    assert "p1" in md


def test_hypothesis_model_fields() -> None:
    h = Hypothesis(
        id="h-1",
        statement="stmt",
        domain=Domain.CHEMISTRY,
        research_question="q",
        confidence=0.5,
    )
    assert h.confidence == 0.5
    assert h.status == HypothesisStatus.NEW
    assert h.tags == []
    assert h.sources == []


def test_prompt_template_different_domains() -> None:
    for domain in Domain:
        prompt = PromptTemplate.render("test", domain, n=1)
        assert domain.value in prompt


def test_generate_general_science() -> None:
    hypos = generate_hypotheses("climate", domain=Domain.GENERAL_SCIENCE, n=2)
    assert len(hypos) == 2
    assert hypos[0].domain == Domain.GENERAL_SCIENCE
