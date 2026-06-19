"""Core data-models demo — Domain, Hypothesis, HypothesisStatus."""

# Import directly from the defining module (not the cds.core re-export) to
# give static analyzers the clearest possible view of these types.
from cds.core.models import Domain, Hypothesis, HypothesisStatus


def main() -> None:
    print("=== Domains ===")
    # Domain is an enum.Enum; iterating it yields its members (PEP 435).
    for d in Domain:
        print(f"  {d.name} = {d.value}")

    print("\n=== Constructing a Hypothesis ===")
    h = Hypothesis(
        id="H-001",
        domain=Domain.COSMOLOGY,
        research_question="Why is the Hubble expansion accelerating?",
        statement="A cosmological constant (Λ) drives late-time acceleration.",
        status=HypothesisStatus.TESTABLE,
        confidence=0.62,
        rationale="Type Ia supernova data favours accelerated expansion.",
        assumptions=["GR holds on cosmological scales", "Λ > 0"],
        predictions=["(m-M) vs z curve bends downward vs matter-only"],
        tags=["dark-energy", "lambda"],
    )
    print(repr(h))
    print("\n--- Markdown render ---")
    print(h.to_markdown())

    print("\n--- JSON serialisation ---")
    print(h.model_dump_json(indent=2)[:200] + " ...")


if __name__ == "__main__":
    main()
