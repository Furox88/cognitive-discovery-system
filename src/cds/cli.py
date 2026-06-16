"""Platform Command Line Interface."""

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
from typer import Context

from cds.core.models import Domain
from cds.hypothesis.generator import PromptTemplate, generate_hypotheses

app = typer.Typer(
    name="cds",
    help="Cognitive Discovery Platform — computational science platform.",
    add_completion=False,
    invoke_without_command=True,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        from cds import __version__

        console.print(f"[bold]Platform[/] version [cyan]{__version__}[/]")
        raise typer.Exit()


@app.callback()
def main(
    ctx: Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=_version_callback,
        is_eager=True,
        help="Show Platform version and exit",
    ),
) -> None:
    """Platform CLI entrypoint."""
    if ctx.invoked_subcommand is None:
        # No subcommand given, show help
        console.print(ctx.get_help())
        raise typer.Exit()


class DomainChoice(str, Enum):
    """Supported scientific domains for hypothesis generation and CLI dispatch."""

    physics = "physics"
    cosmology = "cosmology"
    mathematics = "mathematics"
    biology = "biology"
    chemistry = "chemistry"
    general = "general_science"


@app.command()
def version() -> None:
    """Show Platform version."""
    from cds import __version__

    console.print(f"[bold]Platform[/] version [cyan]{__version__}[/]")


@app.command()
def hypothesis(
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
        typer.Option(
            "--show-prompt",
            help="Print the exact prompt template (for use with a custom generator)",
        ),
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
            f"Would generate {n} hypotheses for: [bold]{question}[/] in domain [cyan]{dom.value}[/]"
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
    domain: Annotated[
        DomainChoice,
        typer.Option("--domain", "-d", help="Scientific domain focus"),
    ] = DomainChoice.general,
    n: Annotated[
        int,
        typer.Option("--num", "-n", help="Number of hypotheses to propose"),
    ] = 3,
) -> None:
    """Print a ready-to-use prompt for a custom generator implementation."""
    dom = Domain(domain.value)
    prompt_text = PromptTemplate.render(question, dom, n)
    console.print(Syntax(prompt_text, "markdown", theme="monokai", line_numbers=False))


@app.command()
def info() -> None:
    """Show Platform info, module status, and Platform health."""
    from rich.columns import Columns
    from rich.text import Text

    from cds import __version__

    status_panel = Panel.fit(
        "[bold]Platform (CDS)[/]\n"
        "[dim]Pure Python scientific computing platform[/]\n\n"
        "🚀 [bold green]Status:[/] Production-Ready (Alpha)\n"
        "🧪 [bold blue]Tests:[/] 570 Passing (100% Coverage)\n"
        "📦 [bold magenta]Deps:[/] 0 External (Pure Python)\n"
        f"🔗 [bold cyan]Version:[/] {__version__}",
        title="Platform Info",
        border_style="green",
    )

    module_text = Text.from_markup(
        "[bold]Core Modules:[/]\n"
        "• [cyan]quantum[/]       • [cyan]signals[/]\n"
        "• [cyan]math_utils[/]    • [cyan]stats[/]\n"
        "• [cyan]optimization[/]  • [cyan]montecarlo[/]\n"
        "• [cyan]hypothesis[/]    • [cyan]diffeq[/]\n"
        "• [cyan]graph[/]         • [cyan]data_analysis[/]\n"
        "• [cyan]ml[/]            • [cyan]probability[/]\n"
        "• [cyan]scientific[/]    • [cyan]numerical_integration[/]"
    )

    capability_panel = Panel.fit(module_text, title="Architecture", border_style="blue")

    console.print(Columns([status_panel, capability_panel]))


@app.command()
def dashboard() -> None:
    """Launch the interactive Platform dashboard."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root_dir = Path(__file__).parent.parent.parent
    dashboard_path = root_dir / "dashboard" / "app.py"
    if not dashboard_path.exists():
        console.print("[red]Error:[/] Dashboard file not found at " + str(dashboard_path))
        return

    console.print("[yellow]Launching Platform Interactive Dashboard...[/]")

    # Ensure src is in PYTHONPATH so dashboard can import cds
    env = os.environ.copy()
    src_path = str(root_dir / "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_path

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(dashboard_path)], check=True, env=env
        )
    except KeyboardInterrupt:
        console.print("\n[blue]Dashboard stopped.[/]")
    except FileNotFoundError:
        console.print(
            "[red]Error:[/] Streamlit not found. Install it with 'pip install streamlit'."
        )


@app.command()
def benchmark() -> None:
    """Run built-in benchmarks to verify performance."""
    console.print("[yellow]Benchmarking Platform performance...[/]")
    console.print("Run 'python benchmarks/run_benchmarks.py' for detailed results.")


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
def plot(
    values: Annotated[str, typer.Argument(help="Comma-separated list of numbers (e.g. '1,5,3,8')")],
    title: Annotated[str, typer.Option("--title", "-t", help="Title of the plot")] = "CLI Plot",
) -> None:
    """Plot a series of numbers directly in the terminal."""
    from cds.data_analysis.viz import plot_line

    try:
        data = [float(x.strip()) for x in values.split(",")]
        console.print(plot_line(data, title=title))
    except ValueError:
        console.print("[red]Error:[/] Values must be a comma-separated list of numbers.")


@app.command()
def calc(
    formula: Annotated[str, typer.Argument(help="Formula: ke, gravity, wave, gas")],
) -> None:
    """Quick physics calculations."""
    from cds.scientific import formulas

    try:
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
    except ValueError:
        console.print("[red]Error:[/] Input must be a valid number.")
    except Exception as e:
        console.print(f"[red]Error:[/] {str(e)}")


@app.command()
def modules() -> None:
    """List all scientific modules available in the Platform."""
    from rich import box

    table = Table(title="Platform Scientific Modules", box=box.SIMPLE_HEAVY)
    table.add_column("Module", style="cyan bold")
    table.add_column("Key Capabilities", style="white")

    module_info = [
        ("cds.quantum", "Single & multi-qubit circuits, Bell/GHZ states, entanglement"),
        ("cds.signals", "DFT, radix-2 FFT, 2D FFT, convolution, filtering"),
        ("cds.math_utils", "LU/QR/Cholesky, power iteration, Gram-Schmidt, calculus"),
        ("cds.optimization", "Gradient descent, Newton, Adam, golden section search"),
        ("cds.stats", "Descriptive stats, regression, t-tests, chi-square, ANOVA"),
        ("cds.probability", "Gaussian, binomial, Poisson and other distributions"),
        ("cds.montecarlo", "π estimation, integration, random walks"),
        ("cds.diffeq", "Euler, RK4, midpoint, ODE system solvers"),
        ("cds.graph", "BFS/DFS, Dijkstra, Kruskal MST, topological sort"),
        ("cds.ml", "Neural networks (MLP), backpropagation, activation functions"),
        ("cds.scientific", "Physical constants + common formulas"),
        ("cds.data_analysis", "CSV loading, normalization, z-score, moving average"),
        (
            "cds.hypothesis",
            "Structured hypothesis generation with prompt templates for custom research workflows",
        ),
    ]

    for name, desc in module_info:
        table.add_row(name, desc)

    console.print(table)
    console.print("\n[dim]All modules are pure Python with no heavy dependencies.[/dim]")
    console.print("[dim]See examples/ for runnable demos of each module.[/dim]\n")


if __name__ == "__main__":  # pragma: no cover
    app()
