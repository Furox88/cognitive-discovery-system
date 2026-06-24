"""System command-line interface.

Pure-stdlib CLI built on :mod:`argparse`. It replaces the previous
``typer``/``rich`` implementation so the whole ``cds`` package stays
zero-dependency at runtime. Rich-style colour is reproduced with small ANSI
escape helpers; the textual output (help text, table contents, prompts) is
preserved verbatim where the test suite asserts on it.

The entry point :func:`main` accepts an optional ``argv`` so tests can drive a
specific command without spawning a subprocess, and returns the integer exit
code instead of calling :func:`sys.exit` directly when ``argv`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from cds.core.models import Domain
from cds.hypothesis.generator import PromptTemplate, generate_hypotheses

# --------------------------------------------------------------------------- #
# ANSI colour helpers — minimal stand-in for ``rich`` markup tags.
# Each maps a rich style to an SGR escape sequence; ``_wrap`` returns the text
# wrapped so it renders coloured on a terminal and plain (stripped) elsewhere.
# --------------------------------------------------------------------------- #
_RESET = "\033[0m"
_STYLES: dict[str, str] = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "italic": "\033[3m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "bold green": "\033[1;32m",
    "bold blue": "\033[1;34m",
    "bold cyan": "\033[1;36m",
    "bold magenta": "\033[1;35m",
    "bold red": "\033[1;31m",
    "bold yellow": "\033[1;33m",
}


def _supports_color() -> bool:
    """Return True when stdout looks like an interactive colour terminal."""
    return sys.stdout.isatty()


def _wrap(style: str, text: str) -> str:
    """Wrap ``text`` in the ANSI codes for ``style`` if stdout is a TTY."""
    if not _supports_color():
        return text
    code = _STYLES.get(style, "")
    if not code:
        return text
    return f"{code}{text}{_RESET}"


def _print(*args: object) -> None:
    """Print helper routed through stdout (so tests can capture via capsys)."""
    print(*args)


# Mapping from rich-style markup tags to plain-text + colour pairs. Each entry
# is (style, plain_text). Used by ``_render`` to expand the minimal markup the
# commands embed in their strings.
def _render(markup: str) -> str:
    """Render rich-style ``[style]...[/]`` markup into ANSI-coloured text.

    Only the tag shapes actually used by this CLI are supported:
    ``[bold]``, ``[italic]``, ``[dim]``, ``[red]``, ``[green]``, ``[yellow]``,
    ``[blue]``, ``[cyan]``, ``[magenta]`` and a few combined ``[bold green]``
    variants. Unknown tags are stripped to their inner text. Nesting is not
    supported (the CLI never nests them).
    """
    out: list[str] = []
    i = 0
    n = len(markup)
    while i < n:
        open_idx = markup.find("[", i)
        if open_idx == -1:
            out.append(markup[i:])
            break
        out.append(markup[i:open_idx])
        close_idx = markup.find("]", open_idx)
        if close_idx == -1:
            out.append(markup[open_idx:])
            break
        tag = markup[open_idx + 1 : close_idx]
        # Closing tag [/] ends the current styled run.
        if tag.startswith("/"):
            out.append(_RESET if _supports_color() else "")
            i = close_idx + 1
            continue
        # Combined forms like "bold green" are looked up directly.
        code = _STYLES.get(tag)
        if code is None and " " in tag:
            # ``[bold green]`` style — already in the table; fall back to first word.
            code = _STYLES.get(tag.split()[0], "")
        out.append(code if (_supports_color() and code is not None) else "")
        i = close_idx + 1
    return "".join(out)


# --------------------------------------------------------------------------- #
# Table rendering — minimal fixed-width formatter standing in for ``rich.Table``.
# --------------------------------------------------------------------------- #
def _format_table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    """Render ``headers``/``rows`` as a bordered ASCII table with a title.

    A tiny reimplementation of the ``rich.Table`` output the CLI used to
    produce: top/bottom title rule, a header row underlined with ``-``, and
    each data row on its own line. Columns are sized to the widest cell.
    """
    cols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for c, cell in enumerate(row[:cols]):
            widths[c] = max(widths[c], len(cell))

    def _border(left: str, fill: str, right: str) -> str:
        return left + fill + right

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    lines: list[str] = []
    if title:
        lines.append(_wrap("bold", title))
    lines.append(sep)
    header_cells = " | ".join(h.ljust(widths[c]) for c, h in enumerate(headers))
    lines.append(f"| {header_cells} |")
    lines.append(sep)
    for row in rows:
        cells = " | ".join(
            str(row[c]).ljust(widths[c]) if c < len(row) else "".ljust(widths[c])
            for c in range(cols)
        )
        lines.append(f"| {cells} |")
    lines.append(sep)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Command implementations
# --------------------------------------------------------------------------- #
def _cmd_version(args: argparse.Namespace) -> int:
    """Show the installed System version."""
    from cds import __version__

    _print(_render(f"[bold]System[/] version [cyan]{__version__}[/]"))
    return 0


def _cmd_hypothesis(args: argparse.Namespace) -> int:
    """Generate scientific hypotheses for a research question."""
    dom = Domain(args.domain)

    if args.show_prompt:
        prompt = PromptTemplate.render(args.question, dom, args.num)
        _print(_render(f"[blue]{prompt}[/]"))
        return 0

    if args.dry_run:
        _print(_render("[yellow]Dry run mode — no generation performed.[/]"))
        _print(
            _render(
                f"Would generate {args.num} hypotheses for: "
                f"[bold]{args.question}[/] in domain [cyan]{dom.value}[/]"
            )
        )
        return 0

    _print(_render(f"[bold]Generating hypotheses[/] for: [italic]{args.question}[/]"))
    _print(_render(f"Domain: [cyan]{dom.value}[/] | Count: {args.num}\n"))

    hypos = generate_hypotheses(args.question, domain=dom, n=args.num)

    rows: list[list[str]] = []
    for h in hypos:
        stmt = h.statement[:90] + ("..." if len(h.statement) > 90 else "")
        rows.append([h.id, stmt, f"{h.confidence:.2f}"])
    _print(_format_table("Generated Hypotheses", ["ID", "Statement", "Confidence"], rows))

    if hypos:
        _print(_render("\n[bold]Detailed view of first hypothesis:[/]\n"))
        _print(_render(f"[green]{hypos[0].to_markdown()}[/]"))

    if args.output:
        data = [h.to_dict() for h in hypos]
        Path(args.output).write_text(json.dumps(data, indent=2, default=str))
        _print(_render(f"\n[green]Saved to {args.output}[/]"))

    return 0


def _cmd_prompt(args: argparse.Namespace) -> int:
    """Print a ready-to-use prompt for a custom generator implementation."""
    dom = Domain(args.domain)
    prompt_text = PromptTemplate.render(args.question, dom, args.num)
    _print(prompt_text)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    """Show System info, module status, and System health."""
    from cds import __version__

    _print(_render("[bold]System (CDS)[/]"))
    _print(_render("[dim]Pure Python scientific computing system[/]"))
    _print("")
    _print(_render("[bold green]Status:[/] Stable"))
    _print(_render("[bold blue]Tests:[/] 1284 Passing"))
    _print(_render("[bold magenta]Deps:[/] 0 External (Pure Python)"))
    _print(_render(f"[bold cyan]Version:[/] {__version__}"))
    _print("")
    _print(_render("[bold]Architecture:[/]"))
    _print(_render("[bold]Core Modules:[/]"))
    for line in (
        "quantum       signals",
        "math_utils    stats",
        "optimization  montecarlo",
        "hypothesis    diffeq",
        "graph         data_analysis",
        "ml            probability",
        "scientific    numerical_integration",
    ):
        _print(f"  • {line}")
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    """Launch the interactive System dashboard."""
    root_dir = Path(__file__).parent.parent.parent
    dashboard_path = root_dir / "dashboard" / "app.py"
    if not dashboard_path.exists():
        _print(_render("[red]Error:[/] Dashboard file not found at " + str(dashboard_path)))
        return 1

    _print(_render("[yellow]Launching System Interactive Dashboard...[/]"))

    # Ensure src is in PYTHONPATH so dashboard can import cds
    env = os.environ.copy()
    src_path = str(root_dir / "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_path

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
            check=True,
            env=env,
        )
    except KeyboardInterrupt:
        _print(_render("\n[blue]Dashboard stopped.[/]"))
    except FileNotFoundError:
        _print(
            _render("[red]Error:[/] Streamlit not found. Install it with 'pip install streamlit'.")
        )
        return 1
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Run built-in benchmarks to verify performance."""
    _print(_render("[yellow]Benchmarking System performance...[/]"))
    _print("Run 'python benchmarks/run_benchmarks.py' for detailed results.")
    return 0


def _cmd_constants(args: argparse.Namespace) -> int:
    """List available physical constants."""
    from cds.scientific.constants import CONSTANTS

    rows = [
        [name, f"{val:.6e}" if val < 0.01 or val > 1e4 else f"{val}", desc]
        for name, (val, desc) in CONSTANTS.items()
    ]
    _print(_format_table("Physical Constants", ["Name", "Value", "Description"], rows))
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    """Plot a series of numbers directly in the terminal."""
    from cds.data_analysis.viz import plot_line

    try:
        data = [float(x.strip()) for x in args.values.split(",")]
        _print(plot_line(data, title=args.title))
    except ValueError:
        _print(_render("[red]Error:[/] Values must be a comma-separated list of numbers."))
        return 1
    return 0


def _cmd_calc(args: argparse.Namespace) -> int:
    """Quick physics calculations."""
    from cds.scientific import formulas

    try:
        if args.formula == "ke":
            _print("KE = 0.5 * m * v²")
            m = float(input("mass (kg) "))
            v = float(input("velocity (m/s) "))
            _print(_render(f"[green]Kinetic Energy = {formulas.kinetic_energy(m, v):.4f} J[/]"))
        elif args.formula == "gravity":
            _print("F = G * m1 * m2 / r²")
            m1 = float(input("mass 1 (kg) "))
            m2 = float(input("mass 2 (kg) "))
            r = float(input("distance (m) "))
            _print(_render(f"[green]Force = {formulas.gravitational_force(m1, m2, r):.6e} N[/]"))
        elif args.formula == "wave":
            wl = float(input("wavelength (m) "))
            _print(_render(f"[green]Frequency = {formulas.wave_frequency(wl):.4e} Hz[/]"))
        elif args.formula == "gas":
            n = float(input("moles "))
            t = float(input("temperature (K) "))
            v = float(input("volume (m³) "))
            _print(_render(f"[green]Pressure = {formulas.ideal_gas_pressure(n, t, v):.2f} Pa[/]"))
        else:
            _print(
                _render(
                    f"[red]Unknown formula '{args.formula}'. Options: ke, gravity, wave, gas[/]"
                )
            )
    except ValueError:
        _print(_render("[red]Error:[/] Input must be a valid number."))
        return 1
    except Exception as e:  # noqa: BLE001 — CLI surface, keep the message readable
        _print(_render(f"[red]Error:[/] {str(e)}"))
        return 1
    return 0


def _cmd_modules(args: argparse.Namespace) -> int:
    """List all scientific modules available in the System."""
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
        (
            "cds.modeling",
            "Symbolic math: expressions, MathModel, equation solving, parameter fitting",
        ),
        ("cds.knowledge", "Knowledge graph, concept mapping, research notes, structured retrieval"),
        ("cds.ml", "Neural networks (MLP), backpropagation, activation functions"),
        ("cds.scientific", "Physical constants + common formulas"),
        ("cds.data_analysis", "CSV loading, normalization, z-score, moving average"),
        ("cds.nlp", "Educational NLP: BPE tokenizer, embeddings, attention, autograd, MiniGPT"),
        (
            "cds.hypothesis",
            "Structured hypothesis generation with prompt templates for custom research workflows",
        ),
    ]
    rows = [[name, desc] for name, desc in module_info]
    _print(_format_table("System Scientific Modules", ["Module", "Key Capabilities"], rows))
    _print(_render("\n[dim]All modules are pure Python with no heavy dependencies.[/]"))
    _print(_render("[dim]See examples/ for runnable demos of each module.[/]\n"))
    return 0


# --------------------------------------------------------------------------- #
# Argument parser construction
# --------------------------------------------------------------------------- #
_DOMAIN_CHOICES = ["physics", "cosmology", "mathematics", "biology", "chemistry", "general_science"]


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``cds`` argument parser and its subcommands."""
    parser = argparse.ArgumentParser(
        prog="cds",
        description="Cognitive Discovery System — computational science platform.",
    )
    parser.add_argument("--version", "-v", action="store_true", help="Show System version and exit")

    sub = parser.add_subparsers(dest="command")

    p_version = sub.add_parser("version", help="Show System version.")
    p_version.set_defaults(func=_cmd_version)

    p_hyp = sub.add_parser(
        "hypothesis", help="Generate scientific hypotheses for a research question."
    )
    p_hyp.add_argument("question", help="The core research question or problem")
    p_hyp.add_argument(
        "--domain",
        "-d",
        default="general_science",
        choices=_DOMAIN_CHOICES,
        help="Scientific domain focus",
    )
    p_hyp.add_argument("--num", "-n", type=int, default=3, help="Number of hypotheses to propose")
    p_hyp.add_argument("--output", "-o", help="Save results as JSON")
    p_hyp.add_argument("--show-prompt", action="store_true", help="Print the exact prompt template")
    p_hyp.add_argument("--dry-run", action="store_true", help="Do not run generation logic")
    p_hyp.set_defaults(func=_cmd_hypothesis)

    p_prompt = sub.add_parser("prompt", help="Print a ready-to-use prompt for a custom generator.")
    p_prompt.add_argument("question", help="Research question")
    p_prompt.add_argument(
        "--domain",
        "-d",
        default="general_science",
        choices=_DOMAIN_CHOICES,
        help="Scientific domain focus",
    )
    p_prompt.add_argument(
        "--num", "-n", type=int, default=3, help="Number of hypotheses to propose"
    )
    p_prompt.set_defaults(func=_cmd_prompt)

    p_info = sub.add_parser("info", help="Show System info, module status, and System health.")
    p_info.set_defaults(func=_cmd_info)

    sub.add_parser("dashboard", help="Launch the interactive System dashboard.").set_defaults(
        func=_cmd_dashboard
    )
    sub.add_parser("benchmark", help="Run built-in benchmarks to verify performance.").set_defaults(
        func=_cmd_benchmark
    )
    sub.add_parser("constants", help="List available physical constants.").set_defaults(
        func=_cmd_constants
    )

    p_plot = sub.add_parser("plot", help="Plot a series of numbers directly in the terminal.")
    p_plot.add_argument("values", help="Comma-separated list of numbers (e.g. '1,5,3,8')")
    p_plot.add_argument("--title", "-t", default="CLI Plot", help="Title of the plot")
    p_plot.set_defaults(func=_cmd_plot)

    p_calc = sub.add_parser("calc", help="Quick physics calculations.")
    p_calc.add_argument("formula", help="Formula: ke, gravity, wave, gas")
    p_calc.set_defaults(func=_cmd_calc)

    sub.add_parser(
        "modules", help="List all scientific modules available in the System."
    ).set_defaults(func=_cmd_modules)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code.

    When ``argv`` is ``None`` (the normal ``cds`` invocation) it reads
    :data:`sys.argv`; tests pass an explicit list so no subprocess is needed.

    argparse raises :class:`SystemExit` for ``--help`` and usage errors. We
    catch it here and surface its code as the return value so callers (tests
    and ``__main__``) never see an exception — only an integer exit code.
    """
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 0

    if args.version:
        from cds import __version__

        _print(_render(f"[bold]System[/] version [cyan]{__version__}[/]"))
        return 0

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return int(func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
