"""Example combining hypothesis generation with stats validation.

This shows how the cognitive discovery tools can be paired with
statistical modules for quick research prototyping.
"""

from cds.hypothesis import generate_hypotheses
from cds.stats import mean, one_sample_ttest, stdev

print("=== Hypothesis + Stats Demo ===\n")

question = "Does a new catalyst improve reaction yield?"

print(f"Research question: {question}\n")

hypos = generate_hypotheses(question, domain="chemistry", n=2)

for h in hypos:
    print(f"Hypothesis: {h.statement}")
    print(f"Predictions: {h.predictions}\n")

# Simulate some experimental data (e.g. yields with/without catalyst)
control_yields = [72.1, 71.8, 73.2, 70.9, 72.5]
catalyst_yields = [78.4, 79.1, 77.8, 80.2, 78.9]

print("Control group mean yield:", mean(control_yields))
print("Catalyst group mean yield:", mean(catalyst_yields))
print("Std dev (catalyst):", stdev(catalyst_yields))

# Quick statistical check (toy example)
result = one_sample_ttest(catalyst_yields, popmean=mean(control_yields))
print(f"One-sample t-test vs control mean: t={result.statistic:.2f}, p={result.p_value:.4f}")

print("\nThis kind of pairing lets researchers quickly turn a generated hypothesis into a testable experiment sketch.")
