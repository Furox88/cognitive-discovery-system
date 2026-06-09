"""CDS Command Line Interface."""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from cds.core.models import Domain
from cds.hypothesis.generator import PromptTemplate, generate_hypotheses

app = typer.Typer(
    name="cds",
    help="Cognitive Discovery System — AI-assisted scientific research tooling.",
    add_completion=False,
)
console = Console()


class DomainChoice(str, Enum):
    physics = "physics"
    cosmology = "cosmology"
    mathematics = "mathematics"
    biology = "biology"
    chemistry = "chemistry"
    general = "general_science"


@app.command()
def version() -> None:
    """Show CDS version."""
    from cds import __version__

    console.print(f"[bold]CDS[/] version [cyan]{__version__}[/]")


@app.command()
def hypothesize(
    question: Annotated[str, typer.Argument(help="The core research question or problem")],
    domain: Annotated[
        DomainChoice,
        typer.Option("--domain", "-d", help="Scientific domain focus"),
    ] = DomainChoice.general,
    n: Annotated[int, typer.Option("--num", "-n", help="Number of hypotheses to propose")] = 3,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Save results as JSON"),
    ] = None,
    show_prompt: Annotated[
        bool,
        typer.Option("--show-prompt", help="Print the exact prompt template (for LLM use)"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Do not run generation logic, just show what would happen"),
    ] = False,
) -> None:
    """
    Generate scientific hypotheses for a research question.

    This is an early demo. It currently uses an offline generator.
    In the future this will call real LLMs (Grok, OpenAI, local models, etc.).
    """
    dom = Domain(domain.value)

    if show_prompt:
        prompt = PromptTemplate.render(question, dom, n)
        console.print(Panel.fit(prompt, title="Prompt Template (ready for any LLM)", border_style="blue"))
        return

    if dry_run:
        console.print("[yellow]Dry run mode — no generation performed.[/]")
        console.print(f"Would generate {n} hypotheses for: [bold]{question}[/] in domain [cyan]{dom.value}[/]")
        return

    console.print(f"[bold]Generating hypotheses[/] for: [italic]{question}[/]")
    console.print(f"Domain: [cyan]{dom.value}[/] | Count: {n}\n")

    hypos = generate_hypotheses(question, domain=dom, n=n)

    table = Table(title="Generated Hypotheses", show_lines=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Statement", style="white")
    table.add_column("Confidence", justify="right")

    for h in hypos:
        table.add_row(h.id, h.statement[:90] + ("..." if len(h.statement) > 90 else ""), f"{h.confidence:.2f}")

    console.print(table)

    # Show full details for the first one
    if hypos:
        console.print("\n[bold]Detailed view of first hypothesis:[/]\n")
        console.print(Panel(hypos[0].to_markdown(), title=hypos[0].id, border_style="green"))

    if output:
        data = [h.model_dump(mode="json") for h in hypos]
        output.write_text(json.dumps(data, indent=2, default=str))
        console.print(f"\n[green]Saved to {output}[/]")


@app.command()
def prompt(
    question: Annotated[str, typer.Argument(help="Research question")],
    domain: Annotated[DomainChoice, typer.Option("--domain", "-d")] = DomainChoice.general,
    n: Annotated[int, typer.Option("--num", "-n")] = 3,
) -> None:
    """Print a ready-to-use prompt for an external LLM (Grok, Claude, GPT, local...)."""
    dom = Domain(domain.value)
    prompt_text = PromptTemplate.render(question, dom, n)
    console.print(Syntax(prompt_text, "markdown", theme="monokai", line_numbers=False))


@app.command()
def info() -> None:
    """Show project info and status."""
    console.print(Panel.fit(
        "[bold]Cognitive Discovery System (CDS)[/]\n\n"
        "Open-source research assistant for scientific discovery.\n"
        "Status: Early development / Architecture phase.\n\n"
        "Next steps: Real LLM integration, knowledge graph, mathematical modeling tools.",
        title="CDS",
        border_style="magenta",
    ))


if __name__ == "__main__":
    app()
