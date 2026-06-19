"""Core data-models demo — Domain, Hypothesis, HypothesisStatus."""

# Import directly from the defining module (not the cds.core re-export) to
# give static analyzers the clearest possible view of these types.
from cds.core.models import Domain, Hypothesis, HypothesisStatus


def main() -> None:
    print("=== Domains ===")
    # __members__.values() is the Enum API's explicit member-collection view
    # (a Mapping's ValuesView). Functionally identical to iterating the
    # EnumType itself (same 6 members, same order), but the return type is an
    # unambiguously iterable mapping view — CodeQL's non-iterable-in-for-loop
    # query can resolve it where the EnumType iterator could not.
    for d in Domain.__members__.values():
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
