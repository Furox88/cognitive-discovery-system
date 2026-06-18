"""Knowledge organization demo — concept graphs, notes, and structured retrieval.

Builds a small research knowledge base (cosmology concepts + lab notes),
exercises the graph traversal, persistence, and ranked search APIs, and
prints everything to stdout. Run with ``python examples/knowledge_demo.py``.
"""

from cds.knowledge import (
    KnowledgeGraph,
    Notebook,
    search,
)


def main() -> None:
    # --- 1. Build a concept graph --------------------------------------- #
    kg = KnowledgeGraph(name="Cosmology")

    kg.add_concept(
        "Dark Energy",
        description="Hypothetical form of energy driving the accelerated expansion of the universe.",
        tags=["cosmology", "energy"],
        metadata={"first_proposed": "1998"},
    )
    kg.add_concept(
        "Hubble Constant",
        description="Rate of expansion of the universe, in km/s/Mpc.",
        tags=["cosmology", "measurement"],
    )
    kg.add_concept("Cosmic Microwave Background", tags=["cosmology", "radiation"])
    kg.add_concept(
        "Dark Matter", description="Non-luminous matter inferred from gravitational effects."
    )

    # Typed, directed relations: source --kind--> target
    kg.add_relation("Dark Energy", "Hubble Constant", kind="affects")
    kg.add_relation("Hubble Constant", "Cosmic Microwave Background", kind="constrains")
    kg.add_relation("Dark Matter", "Cosmic Microwave Background", kind="affects")

    print("=== Concept Graph ===")
    print(kg.to_markdown())

    # --- 2. Traverse the graph ------------------------------------------ #
    print("=== Traversal ===")
    print(f"neighbors of 'Hubble Constant': {kg.neighbors('Hubble Constant')}")
    print(f"outgoing from 'Dark Energy':    {kg.neighbors_out('Dark Energy')}")
    print(f"incoming to 'CMB':               {kg.neighbors_in('Cosmic Microwave Background')}")

    path = kg.find_path("Dark Energy", "Cosmic Microwave Background")
    print(f"path Dark Energy -> CMB: {path}")

    print(f"reachable from 'Dark Energy': {sorted(kg.reachable('Dark Energy'))}")
    print(f"cycles: {kg.find_cycles()}")

    # --- 3. Keep a notebook of research notes --------------------------- #
    notebook = Notebook(name="Cosmology Lab Book")
    notebook.add_note(
        note_id="n1",
        title="Hubble Tension",
        body="The local and CMB measurements of the Hubble Constant disagree at the ~5 sigma level.",
        tags=["experiment", "tension"],
        linked_concepts=["Hubble Constant", "Cosmic Microwave Background"],
        created="2024-03-12",
    )
    notebook.add_note(
        note_id="n2",
        title="Dark Energy candidates",
        body="Cosmological constant vs. quintessence field — both still viable.",
        tags=["theory"],
        linked_concepts=["Dark Energy"],
        created="2024-04-01",
    )

    print("\n=== Notebook ===")
    print(notebook.to_markdown())
    print("notes tagged 'experiment':", [n.id for n in notebook.notes_by_tag("experiment")])
    print(
        "notes about 'Hubble Constant':",
        [n.id for n in notebook.notes_for_concept("Hubble Constant")],
    )

    # --- 4. Ranked retrieval across both -------------------------------- #
    print("\n=== Search: 'hubble' ===")
    for hit in search(kg, notebook, query="hubble"):
        kind = "concept" if hit.concept_name else "note"
        ident = hit.concept_name or hit.note_id
        print(f"  [{kind}] {ident}  score={hit.score}  matched_on={hit.matched_on}")

    # --- 5. Persistence (JSON round-trip) ------------------------------- #
    print("\n=== Persistence ===")
    kg.save("_demo_knowledge_graph.json")
    notebook.save("_demo_knowledge_notebook.json")
    reloaded = KnowledgeGraph.load("_demo_knowledge_graph.json")
    print(
        f"reloaded graph has {len(reloaded.concepts)} concepts, {len(reloaded.relations)} relations"
    )


if __name__ == "__main__":
    main()
