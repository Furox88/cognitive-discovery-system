"""Named systems of equations — :class:`MathModel`.

A :class:`MathModel` bundles a set of named symbolic expressions with shared
parameters and variables. It is the natural unit for "equation development":
you write ``v = u + a*t``, then ask the model for the Jacobian column
``dv/da``, render the whole system to Markdown or LaTeX, and finally hand a
single equation to :func:`cds.modeling.solver.solve_equation` to find a root.

This module is deliberately thin — all symbolic machinery lives in
:mod:`cds.modeling.expression`. :class:`MathModel` only adds bookkeeping
(name → expression, parameters, declared variables) and rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cds.modeling.expression import Expression


@dataclass
class MathModel:
    """A named system of symbolic equations sharing parameters and variables.

    Attributes:
        name: human-readable model title (used in :meth:`to_markdown`).
        equations: ordered ``(label, expression)`` pairs. Labels are the
            equation names callers refer to (e.g. ``"velocity"``);
            expressions are symbolic :class:`Expression` trees.
        parameters: constant values substituted during evaluation. They
            shadow variables of the same name, mirroring how a physicist
            treats ``g`` vs ``t``.
        variables: the declared free variables. Recorded explicitly so the
            model is self-describing even before any equation is inspected.
        description: optional one-line summary.
    """

    name: str
    equations: list[tuple[str, Expression]] = field(default_factory=list)
    parameters: dict[str, float] = field(default_factory=dict)
    variables: list[str] = field(default_factory=list)
    description: str | None = None

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def add_equation(self, label: str, expr: Expression) -> None:
        """Append a named equation to the system."""
        self.equations.append((label, expr))

    def set_parameter(self, name: str, value: float) -> None:
        """Bind or update a named parameter value."""
        self.parameters[name] = float(value)

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    def evaluate(self, env: dict[str, float]) -> dict[str, float]:
        """Evaluate every equation, merging parameters into the bindings.

        Args:
            env: values for the free variables (parameters override these
                if a name collides).

        Returns:
            mapping of equation label to its evaluated numeric value.

        Raises:
            ValueError: if a free variable has no binding (propagated from
                :meth:`Expression.evaluate`).
        """
        merged: dict[str, float] = {**env, **self.parameters}
        return {label: expr.evaluate(merged) for label, expr in self.equations}

    def equation(self, label: str) -> Expression:
        """Return the expression for a named equation.

        Raises:
            KeyError: if ``label`` is not in this model.
        """
        for name, expr in self.equations:
            if name == label:
                return expr
        raise KeyError(f"no equation labelled {label!r} in model {self.name!r}")

    def gradient(self, label: str, var: str) -> Expression:
        """Symbolic partial derivative of one equation w.r.t. one variable."""
        return self.equation(label).diff(var)

    def jacobian(self, var: str) -> dict[str, Expression]:
        """Symbolic partial derivative of *every* equation w.r.t. ``var``.

        Returns:
            mapping of equation label to its derivative expression.
        """
        return {label: expr.diff(var) for label, expr in self.equations}

    def free_variables(self) -> set[str]:
        """All variable names that actually appear in some equation."""
        seen: set[str] = set()
        for _, expr in self.equations:
            seen |= expr.variables()
        return seen - set(self.parameters)

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def to_markdown(self) -> str:
        """Render this model as a structured Markdown document."""
        lines: list[str] = [f"# Model: {self.name}", ""]
        if self.description:
            lines += [self.description, ""]
        if self.parameters:
            lines.append("## Parameters")
            for name, value in self.parameters.items():
                lines.append(f"- `{name}` = {value}")
            lines.append("")
        if self.variables:
            lines += [
                "## Variables",
                ", ".join(f"`{v}`" for v in self.variables),
                "",
            ]
        lines += ["## Equations"]
        for label, expr in self.equations:
            lines.append(f"- **{label}**: `{expr.to_str()}`")
        lines.append("")
        return "\n".join(lines)

    def to_latex(self) -> str:
        """Render the model's equations as a LaTeX align block."""
        rows = [f"{label} &= {expr.to_latex()}" for label, expr in self.equations]
        body = " \\\\ ".join(rows)
        return f"\\begin{{align*}}\n{body}\n\\end{{align*}}"
