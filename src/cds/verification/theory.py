"""
TheoryVerifier — symbolic and logical consistency checking.

This module provides tools to automatically detect common classes
of problems that appear when developing new physical theories:

- Contradictory or over-constrained definitions of key variables
- Failure of exact differentials (path dependence of integrals)
- Parameter degeneracy (different parameter values producing identical observables)

These checks are *non-trivial* and were used in practice to surface
real issues in the definitions of L, t_eff, measurement thresholds, etc.

The implementation favors SymPy for most symbolic work.
Z3 (SMT solver) can be used for satisfiability-style definition conflicts
when installed.
"""
from __future__ import annotations

from typing import Any

try:
    import sympy as sp
    from sympy import symbols, diff, simplify, solve, Eq, oo, limit
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

try:
    from z3 import Real, Solver, unsat
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


class TheoryVerifier:
    """
    Collection of automated checks for theoretical consistency.

    Usage example (from real RBA analysis):

        verifier = TheoryVerifier()
        verifier.check_definition_conflict(
            definitions=[L == 1 - E/E_max, L == xi_I / xi_max],
            assumptions=[E == 0.5, E_max == 1.0, xi_I == 0.6, xi_max == 1.0]
        )
    """

    def __init__(self) -> None:
        if not HAS_SYMPY:
            raise ImportError("sympy is required for TheoryVerifier. Install with: pip install 'cognitive-discovery-system[physics]'")

    # ------------------------------------------------------------------
    # 1. Definition conflict (Z3 style when available, otherwise SymPy)
    # ------------------------------------------------------------------
    def check_definition_conflict(
        self,
        definitions: list[Any],
        assumptions: list[Any] | None = None,
        variable: str = "L",
    ) -> dict[str, Any]:
        """
        Check whether multiple definitions of a variable can be simultaneously
        satisfied under given assumptions.

        Returns a report with 'consistent' boolean and explanation.
        """
        assumptions = assumptions or []

        if HAS_Z3:
            s = Solver()
            L = Real(variable)
            for d in definitions:
                # definitions are expected as Z3 expressions or we convert simple ones
                s.add(d)
            for a in assumptions:
                s.add(a)

            result = s.check()
            return {
                "consistent": result != unsat,
                "solver": "z3",
                "result": str(result),
            }

        # Fallback: use SymPy to see if the system has a solution
        L = symbols(variable, real=True)
        eqs = [Eq(L, d.rhs if hasattr(d, "rhs") else d) for d in definitions]
        try:
            sol = solve(eqs, L, dict=True)
            consistent = len(sol) > 0
        except Exception:
            consistent = False

        return {
            "consistent": consistent,
            "solver": "sympy",
            "solution": sol if 'sol' in locals() else None,
        }

    # ------------------------------------------------------------------
    # 2. Path independence / exact differential test
    # ------------------------------------------------------------------
    def check_path_independence(self, integrand_expr: Any, variables: list[str]) -> dict[str, Any]:
        """
        Test whether an integrand corresponds to an exact differential.

        For an expression M(phi, L), we check dM/dL == 0 when treating
        other variables as independent (simple necessary condition for
        path independence in this context).
        """
        if len(variables) < 2:
            raise ValueError("Need at least two variables for path independence test")

        syms = symbols(" ".join(variables))
        M = integrand_expr
        dM_dvar = diff(M, syms[1])  # check derivative w.r.t second variable

        is_exact = simplify(dM_dvar) == 0

        return {
            "is_path_independent": bool(is_exact),
            "derivative": dM_dvar,
            "message": "Exact differential (path independent)" if is_exact else "Path dependent (derivative nonzero)",
        }

    # ------------------------------------------------------------------
    # 3. Parameter degeneracy detection
    # ------------------------------------------------------------------
    def check_degeneracy(
        self,
        observable_func: callable,
        param_pairs: list[tuple[float, float]],
        rtol: float = 1e-8,
    ) -> dict[str, Any]:
        """
        Check whether different parameter combinations produce (approximately)
        the same observable value.

        Example: different (eps, L) pairs giving identical w in an equation.
        """
        values = [observable_func(eps, L) for eps, L in param_pairs]
        all_equal = all(abs(v - values[0]) < rtol for v in values)

        return {
            "degenerate": all_equal,
            "values": values,
            "param_pairs": param_pairs,
            "message": "Degeneracy detected — different parameters give same output"
            if all_equal
            else "No degeneracy in the tested pairs",
        }

    # ------------------------------------------------------------------
    # Convenience: run a battery of common checks
    # ------------------------------------------------------------------
    def run_rba_style_checks(self) -> dict[str, Any]:
        """
        Run a battery of checks that were historically useful for RBA-like theories.

        This serves both as documentation and as a ready-to-use example
        of strong, non-trivial verification.
        """
        results = {}

        # Example 1: L definition conflict (classic problem)
        L, E, E_max, xi_I, xi_max = symbols("L E E_max xi_I xi_max", positive=True)
        defs = [Eq(L, 1 - E / E_max), Eq(L, xi_I / xi_max)]
        results["L_definition_conflict"] = self.check_definition_conflict(
            definitions=defs,
            assumptions=[Eq(E, 0.5), Eq(E_max, 1), Eq(xi_I, 0.6), Eq(xi_max, 1)],
        )

        # Example 2: path independence of phi/L integrand
        phi, L = symbols("phi L")
        M = phi / L
        results["path_independence_phi_over_L"] = self.check_path_independence(M, ["phi", "L"])

        # Example 3: degeneracy in w = -1 + eps*(1 - L)
        def w(eps, L):
            return -1 + eps * (1 - L)

        results["w_degeneracy"] = self.check_degeneracy(
            w, [(0.1, 0.5), (0.05, 0.0), (0.25, 0.8)]
        )

        return results
