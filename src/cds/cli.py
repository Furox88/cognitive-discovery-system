"""CLI for the Cognitive Discovery System."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from cds import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cds",
        description="CDS — research assistant CLI",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command")

    hyp = sub.add_parser("hypothesis", help="Manage hypotheses")
    hyp_sub = hyp.add_subparsers(dest="action")
    hyp_new = hyp_sub.add_parser("new", help="Create a hypothesis")
    hyp_new.add_argument("statement")
    hyp_new.add_argument("rationale")

    note = sub.add_parser("note", help="Manage notes")
    note_sub = note.add_subparsers(dest="action")
    note_new = note_sub.add_parser("new", help="Create a note")
    note_new.add_argument("title")
    note_new.add_argument("body")

    concept = sub.add_parser("concept", help="Manage concepts")
    concept_sub = concept.add_subparsers(dest="action")
    concept_new = concept_sub.add_parser("new", help="Add a concept")
    concept_new.add_argument("name")
    concept_new.add_argument("--description", default="")

    sub.add_parser("info", help="Show system info")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "info":
        print(f"Cognitive Discovery System v{__version__}")
        print("Modules: hypothesis, notes, concept_map, modeling")
        return 0

    if args.command == "hypothesis" and args.action == "new":
        from cds.hypothesis import Hypothesis
        h = Hypothesis(statement=args.statement, rationale=args.rationale)
        print(f"Created hypothesis {h.id}: {h.summary()}")
        return 0

    if args.command == "note" and args.action == "new":
        from cds.notes import Note
        n = Note(title=args.title, body=args.body)
        print(f"Created note {n.id}: {n.title}")
        return 0

    if args.command == "concept" and args.action == "new":
        from cds.concept_map import Concept
        c = Concept(name=args.name, description=args.description)
        print(f"Created concept: {c.name}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
