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
    help="Cognitive Discovery System — computational science toolkit.",
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
    """Generate scientific hypotheses for a research question."""
    dom = Domain(domain.value)

    if show_prompt:
        prompt = PromptTemplate.render(question, dom, n)
        console.print(Panel.fit(prompt, title="Prompt Template", border_style="blue"))
        return

    if dry_run:
        console.print("[yellow]Dry run mode — no generation performed.[/]")
        console.print(
            f"Would generate {n} hypotheses for: "
            f"[bold]{question}[/] in domain [cyan]{dom.value}[/]"
        )
        return

    console.print(f"[bold]Generating hypotheses[/] for: [italic]{question}[/]")
    console.print(f"Domain: [cyan]{dom.value}[/] | Count: {n}\n")

    hypos = generate_hypotheses(question, domain=dom, n=n)

    table = Table(title="Generated Hypotheses", show_lines=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Statement", style="white")
    table.add_column("Confidence", justify="right")

    for h in hypos:
        stmt = h.statement[:90] + ("..." if len(h.statement) > 90 else "")
        table.add_row(h.id, stmt, f"{h.confidence:.2f}")

    console.print(table)

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
    """Print a ready-to-use prompt for an external LLM."""
    dom = Domain(domain.value)
    prompt_text = PromptTemplate.render(question, dom, n)
    console.print(Syntax(prompt_text, "markdown", theme="monokai", line_numbers=False))


@app.command()
def info() -> None:
    """Show project info and status."""
    console.print(Panel.fit(
        "[bold]Cognitive Discovery System (CDS)[/]\n\n"
        "Pure Python computational science toolkit.\n"
        "Modules: quantum, optimization, signals, probability, stats,\n"
        "  math_utils, data_analysis, scientific, graph, montecarlo,\n"
        "  diffeq, hypothesis\n\n"
        "Status: Alpha v0.1.0 | Tests: 267 | Dependencies: pure Python",
        title="CDS",
        border_style="magenta",
    ))


@app.command()
def constants() -> None:
    """List available physical constants."""
    from cds.scientific.constants import CONSTANTS

    table = Table(title="Physical Constants")
    table.add_column("Name", style="cyan")
    table.add_column("Value")
    table.add_column("Description")
    for name, (val, desc) in CONSTANTS.items():
        table.add_row(name, f"{val:.6e}" if val < 0.01 or val > 1e4 else f"{val}", desc)
    console.print(table)


@app.command()
def calc(
    formula: Annotated[str, typer.Argument(help="Formula: ke, gravity, wave, gas")],
) -> None:
    """Quick physics calculations."""
    from cds.scientific import formulas

    if formula == "ke":
        console.print("KE = 0.5 * m * v²")
        m = float(typer.prompt("mass (kg)"))
        v = float(typer.prompt("velocity (m/s)"))
        console.print(f"[green]Kinetic Energy = {formulas.kinetic_energy(m, v):.4f} J[/]")
    elif formula == "gravity":
        console.print("F = G * m1 * m2 / r²")
        m1 = float(typer.prompt("mass 1 (kg)"))
        m2 = float(typer.prompt("mass 2 (kg)"))
        r = float(typer.prompt("distance (m)"))
        console.print(f"[green]Force = {formulas.gravitational_force(m1, m2, r):.6e} N[/]")
    elif formula == "wave":
        wl = float(typer.prompt("wavelength (m)"))
        console.print(f"[green]Frequency = {formulas.wave_frequency(wl):.4e} Hz[/]")
    elif formula == "gas":
        n = float(typer.prompt("moles"))
        t = float(typer.prompt("temperature (K)"))
        v = float(typer.prompt("volume (m³)"))
        console.print(f"[green]Pressure = {formulas.ideal_gas_pressure(n, t, v):.2f} Pa[/]")
    else:
        console.print(f"[red]Unknown formula '{formula}'. Options: ke, gravity, wave, gas[/]")


if __name__ == "__main__":
    app()
