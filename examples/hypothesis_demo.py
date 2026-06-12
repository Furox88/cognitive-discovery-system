"""Demonstration of the hypothesis generation module for cognitive discovery."""

from cds.hypothesis import generate_hypotheses, PromptTemplate

print("=== CDS Hypothesis Generation Demo ===\n")

# Example research question
question = "What causes the observed tensions in cosmology (e.g. Hubble tension)?"

print(f"Research question: {question}\n")

# Generate hypotheses using the built-in offline generator
hypos = generate_hypotheses(question, domain="cosmology", n=2)

for h in hypos:
    print(f"ID: {h.id}")
    print(f"Statement: {h.statement}")
    print(f"Domain: {h.domain.value}")
    print(f"Confidence: {h.confidence:.2f}")
    print(f"Rationale: {h.rationale}")
    if h.assumptions:
        print("Assumptions:")
        for a in h.assumptions:
            print(f"  - {a}")
    if h.predictions:
        print("Predictions:")
        for p in h.predictions:
            print(f"  - {p}")
    print()

# Show how to get the raw prompt for use with an external LLM backend (various models or local setups).
print("=== Prompt template for external LLM ===")
prompt = PromptTemplate.render(question, hypos[0].domain, n=2)
print(prompt[:500] + "...\n")

print("This module provides structured, falsifiable hypotheses with explicit assumptions and predictions.")
print("The offline generator is for demos; for real discovery work, plug in a real LLM backend.")
