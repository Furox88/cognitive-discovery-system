"""
Scientific hypothesis generator for the Cognitive Discovery System.

This module turns research questions into structured, falsifiable
hypotheses with clear assumptions and predictions.

It provides ready-to-use prompt templates and a simple offline
generator for demos and testing. The HypothesisGenerator Protocol
makes it straightforward for researchers to supply their own
generator implementations for specialized workflows.

Key goals:
- High-quality, reusable prompt templates for scientific reasoning
- Clean separation between templates and generation logic
- Simple Protocol for custom generator implementations
- Usable offline as a starting point

See the HypothesisGenerator protocol and examples for how to
provide a custom generator.
"""
from __future__ import annotations

import uuid
from typing import Any, Protocol

from cds.core.models import Domain, Hypothesis, HypothesisStatus


class HypothesisGenerator(Protocol):
    """Interface for hypothesis generators."""

    def generate(
        self,
        research_question: str,
        domain: Domain | str = Domain.GENERAL_SCIENCE,
        n: int = 3,
        **kwargs: Any,
    ) -> list[Hypothesis]:
        ...


class PromptTemplate:
    """Prompt templates for different providers / use cases."""

    SYSTEM = (
        "You are an expert research scientist and rigorous thinker. "
        "Your goal is to propose high-quality, falsifiable, "
        "novel-yet-grounded scientific hypotheses. "
        "Always make assumptions explicit. Prioritize testability and clarity. "
        "Respond ONLY in the requested structured format."
    )

    USER_BASE = """Research Question: {research_question}

Domain focus: {domain}

Generate {n} distinct hypotheses.

For each hypothesis provide:
- Clear one-sentence statement
- Short rationale (2-4 sentences) connecting to known science
- Key assumptions (bullet list)
- Specific, measurable predictions or consequences (bullet list)
- Estimated confidence (0-1) with brief justification

Format each as:
ID: H-<number>
Statement: ...
Rationale: ...
Assumptions:
- ...
Predictions:
- ...
Confidence: 0.xx
"""

    @classmethod
    def render(cls, research_question: str, domain: Domain, n: int = 3) -> str:
        return cls.USER_BASE.format(
            research_question=research_question,
            domain=domain.value,
            n=n,
        )


class SimpleOfflineGenerator:
    """
    A deterministic offline generator for demos and early development.

    It creates plausible but generic hypotheses. Researchers can replace
    or wrap it with a custom implementation of HypothesisGenerator
    tailored to their domain or data sources.
    """

    def __init__(self) -> None:
        self.templates = {
            Domain.COSMOLOGY: [
                ("Late-time modifications to gravity can mimic "
                 "dark energy while altering structure growth."),
                ("A time-varying dark energy equation of state "
                 "w(a) with a sharp transition at z~0.5 "
                 "explains current tensions."),
                ("Primordial non-Gaussianity of local type at "
                 "f_NL ~ 5-10 is detectable with next-gen "
                 "surveys and resolves sigma8 tension."),
            ],
            Domain.PHYSICS: [
                ("A hidden sector with light mediators can "
                 "resolve the muon g-2 anomaly without "
                 "conflicting with collider bounds."),
                ("Modified dispersion relations at Planck scale "
                 "suppress high-energy cosmic rays in a "
                 "characteristic energy-dependent way."),
            ],
            Domain.MATHEMATICS: [
                ("A new family of special functions between "
                 "hypergeometric and q-hypergeometric satisfies "
                 "a novel functional equation."),
            ],
        }

    def generate(
        self,
        research_question: str,
        domain: Domain | str = Domain.GENERAL_SCIENCE,
        n: int = 3,
        **kwargs: Any,
    ) -> list[Hypothesis]:
        # Ensure domain is a Domain enum instance
        if isinstance(domain, str):
            try:
                # Case-insensitive mapping for better UX
                domain = Domain(domain.lower())
            except ValueError:
                domain = Domain.GENERAL_SCIENCE

        ideas = self.templates.get(domain, self.templates[Domain.PHYSICS])[:n]
        if len(ideas) < n:
            # Pad with generic
            ideas += [
                f"Variant of mechanism X applied to "
                f"{research_question} yields new observable Y.",
            ] * (n - len(ideas))

        hypos: list[Hypothesis] = []
        for i, idea in enumerate(ideas[:n], 1):
            h = Hypothesis(
                id=f"H-{uuid.uuid4().hex[:8]}",
                statement=idea,
                domain=domain,
                research_question=research_question,
                rationale=(
                    "Builds on known tensions in the literature "
                    "and proposes a falsifiable deviation."
                ),
                assumptions=[
                    "Background model is approximately correct at low energies.",
                    "New physics at observable scales doesn't violate existing constraints.",
                ],
                predictions=[
                    "A measurable deviation in observable O at scale S with amplitude A.",
                    "Correlation between two previously uncorrelated datasets D1 and D2.",
                ],
                status=HypothesisStatus.NEW,
                confidence=0.45 + (i * 0.05),
                tags=[domain.value, "early-draft"],
            )
            hypos.append(h)
        return hypos


def generate_hypotheses(
    research_question: str,
    domain: Domain = Domain.GENERAL_SCIENCE,
    n: int = 3,
    generator: HypothesisGenerator | None = None,
) -> list[Hypothesis]:
    """Convenience entrypoint."""
    gen = generator or SimpleOfflineGenerator()
    return gen.generate(research_question=research_question, domain=domain, n=n)
