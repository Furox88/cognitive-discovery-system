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


def test_hypothesis_to_markdown_minimal() -> None:
    """to_markdown() with no rationale / assumptions / predictions / tags.

    Exercises every ``if <field>:`` False-branch in the renderer so the
    minimal hypothesis (only the required fields) renders without the
    optional sections.
    """
    h = Hypothesis(
        id="minimal-1",
        statement="Bare statement",
        domain=Domain.MATHEMATICS,
        research_question="bare question",
        confidence=0.5,
    )
    md = h.to_markdown()
    assert "Bare statement" in md
    assert "mathematics" in md
    # None of the optional sections should appear
    assert "## Rationale" not in md
    assert "## Assumptions" not in md
    assert "## Predictions" not in md
    assert "**Tags**" not in md


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
    # __members__.values() is the Enum API's explicit member-collection view.
    # Same members/order as iterating the EnumType, but a mapping view whose
    # type CodeQL's non-iterable query can resolve (the EnumType iterator
    # could not).
    for domain in Domain.__members__.values():
        prompt = PromptTemplate.render("test", domain, n=1)
        assert domain.value in prompt


def test_generate_general_science() -> None:
    hypos = generate_hypotheses("climate", domain=Domain.GENERAL_SCIENCE, n=2)
    assert len(hypos) == 2
    assert hypos[0].domain == Domain.GENERAL_SCIENCE


def test_generate_with_domain_enum_instance_skips_str_mapping() -> None:
    # Passing a Domain enum instance (not a str) leaves the `isinstance(domain, str)`
    # guard False, so the try/except str→Domain mapping is skipped entirely
    # (generator.py 149 -> 156 edge). The returned hypotheses carry that domain.
    gen = SimpleOfflineGenerator()
    hypos = gen.generate("neutrino mass ordering", domain=Domain.PHYSICS, n=2)
    assert len(hypos) == 2
    assert all(h.domain == Domain.PHYSICS for h in hypos)


def test_evaluator_goodness_of_fit_with_explicit_expected() -> None:
    # Supplying a non-None `expected` list skips the uniform-distribution
    # fallback (evaluator.py 164 -> 168 edge) and feeds the caller's counts
    # straight into chi_square_gof.
    from cds.hypothesis.evaluator import HypothesisEvaluator

    h = Hypothesis(
        id="gof-1",
        statement="Loaded die is fair",
        domain=Domain.MATHEMATICS,
        research_question="die fairness",
        confidence=0.5,
    )
    evaluator = HypothesisEvaluator()
    result = evaluator.goodness_of_fit(h, observed=[10, 20, 30], expected=[20, 20, 20])
    assert result.hypothesis_id == "gof-1"
    assert "Chi-square goodness-of-fit" in result.test_name
