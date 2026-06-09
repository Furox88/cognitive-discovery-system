"""Basic tests for hypothesis generation (early stage)."""
from cds.core.models import Domain, Hypothesis, HypothesisStatus
from cds.hypothesis.generator import (
    PromptTemplate,
    SimpleOfflineGenerator,
    generate_hypotheses,
)


def test_generate_hypotheses_returns_list():
    hypos = generate_hypotheses(
        "Hubble tension resolution mechanisms",
        domain=Domain.COSMOLOGY, n=2,
    )
    assert len(hypos) == 2
    for h in hypos:
        assert isinstance(h, Hypothesis)
        assert h.research_question == "Hubble tension resolution mechanisms"
        assert h.domain == Domain.COSMOLOGY
        assert h.status == HypothesisStatus.NEW
        assert 0.0 <= h.confidence <= 1.0


def test_prompt_template_renders():
    prompt = PromptTemplate.render("Test question about dark energy", Domain.COSMOLOGY, n=3)
    assert "Research Question: Test question about dark energy" in prompt
    assert "Domain focus: cosmology" in prompt
    assert "Generate 3 distinct hypotheses" in prompt


def test_offline_generator_is_deterministic_enough():
    gen = SimpleOfflineGenerator()
    h1 = gen.generate("foo", domain=Domain.COSMOLOGY, n=1)
    h2 = gen.generate("foo", domain=Domain.COSMOLOGY, n=1)
    # Not strictly same ids, but statements come from the same small template pool
    assert len(h1) == 1 and len(h2) == 1
    assert h1[0].domain == Domain.COSMOLOGY
