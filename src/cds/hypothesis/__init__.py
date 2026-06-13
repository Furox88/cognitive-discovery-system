"""
Hypothesis generation and evaluation module for Cognitive Discovery.

Provides tools to generate structured scientific hypotheses from
research questions. Includes prompt templates and an offline
generator for immediate use, plus a clear Protocol for supplying
custom generator implementations for specialized research needs.

The focus is on making hypotheses falsifiable, with explicit
assumptions, predictions, and confidence estimates.

Example:
    from cds.hypothesis import generate_hypotheses

    hypos = generate_hypotheses(
        "Why do we observe the Hubble tension?",
        domain="cosmology",
        n=3
    )
"""

from cds.hypothesis.generator import (
    Domain,
    Hypothesis,
    HypothesisGenerator,
    HypothesisStatus,
    PromptTemplate,
    SimpleOfflineGenerator,
    generate_hypotheses,
)

__all__ = [
    "Domain",
    "Hypothesis",
    "HypothesisGenerator",
    "HypothesisStatus",
    "PromptTemplate",
    "SimpleOfflineGenerator",
    "generate_hypotheses",
]
