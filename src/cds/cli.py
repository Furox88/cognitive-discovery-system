"""Command-line interface for the Cognitive Discovery System.

Provides a simple REPL for interacting with hypotheses, notes, and
concept maps from the terminal.
"""

from __future__ import annotations

import argparse
import sys

from cds import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cds",
        description="Cognitive Discovery System — research assistant CLI",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    sub = parser.add_subparsers(dest="command")

    # --- hypothesis ---
    hyp = sub.add_parser("hypothesis", help="Manage research hypotheses")
    hyp_sub = hyp.add_subparsers(dest="action")
    hyp_new = hyp_sub.add_parser("new", help="Create a new hypothesis")
    hyp_new.add_argument("statement", help="The hypothesis statement")
    hyp_new.add_argument("rationale", help="The rationale for the hypothesis")

    # --- note ---
    note = sub.add_parser("note", help="Manage research notes")
    note_sub = note.add_subparsers(dest="action")
    note_new = note_sub.add_parser("new", help="Create a new note")
    note_new.add_argument("title", help="Note title")
    note_new.add_argument("body", help="Note body")

    # --- concept ---
    concept = sub.add_parser("concept", help="Manage concept maps")
    concept_sub = concept.add_subparsers(dest="action")
    concept_new = concept_sub.add_parser("new", help="Add a new concept")
    concept_new.add_argument("name", help="Concept name")
    concept_new.add_argument("--description", default="", help="Concept description")

    # --- info ---
    sub.add_parser("info", help="Show system information")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the CLI."""
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
