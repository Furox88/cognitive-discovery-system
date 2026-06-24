"""Tests for cds.modeling — symbolic expressions, MathModel, and solvers."""

from __future__ import annotations

import math

import pytest

from cds.modeling import (
    Add,
    Constant,
    Cos,
    Div,
    Exp,
    Expression,
    FitResult,
    Log,
    MathModel,
    Mul,
    Pow,
    Sin,
    SolveResult,
    Sqrt,
    Sub,
    Variable,
    fit_parameters,
    solve_equation,
)

# ---------------------------------------------------------------------- #
# Evaluation
# ---------------------------------------------------------------------- #


class TestEvaluation:
    def test_constant(self) -> None:
        assert Constant(3.5).evaluate({}) == 3.5

    def test_variable(self) -> None:
        assert Variable("x").evaluate({"x": 2.0}) == 2.0

    def test_variable_missing_binding_raises(self) -> None:
        with pytest.raises(ValueError, match="no value bound"):
            Variable("x").evaluate({})

    def test_add(self) -> None:
        assert (Variable("x") + 3).evaluate({"x": 4}) == 7.0

    def test_sub(self) -> None:
        assert (Variable("x") - 3).evaluate({"x": 4}) == 1.0

    def test_mul(self) -> None:
        assert (Variable("x") * 3).evaluate({"x": 4}) == 12.0

    def test_div(self) -> None:
        assert (Variable("x") / 2).evaluate({"x": 10}) == 5.0

    def test_pow(self) -> None:
        assert (Variable("x") ** 3).evaluate({"x": 2}) == 8.0

    def test_neg(self) -> None:
        assert (-Variable("x")).evaluate({"x": 5}) == -5.0

    def test_pos_is_identity(self) -> None:
        assert (+Variable("x")).evaluate({"x": 5}) == 5.0

    def test_radd(self) -> None:
        assert (3 + Variable("x")).evaluate({"x": 4}) == 7.0

    def test_rsub(self) -> None:
        assert (10 - Variable("x")).evaluate({"x": 4}) == 6.0

    def test_rmul(self) -> None:
        assert (3 * Variable("x")).evaluate({"x": 4}) == 12.0

    def test_rtruediv(self) -> None:
        assert (12 / Variable("x")).evaluate({"x": 4}) == 3.0

    def test_rpow(self) -> None:
        assert (2 ** Variable("x")).evaluate({"x": 3}) == 8.0

    def test_sin(self) -> None:
        assert abs(Sin(Variable("x")).evaluate({"x": math.pi / 2}) - 1.0) < 1e-12

    def test_cos(self) -> None:
        assert abs(Cos(Variable("x")).evaluate({"x": 0.0}) - 1.0) < 1e-12

    def test_exp(self) -> None:
        assert abs(Exp(Variable("x")).evaluate({"x": 1.0}) - math.e) < 1e-12

    def test_log(self) -> None:
        assert abs(Log(Variable("x")).evaluate({"x": math.e}) - 1.0) < 1e-12

    def test_sqrt(self) -> None:
        assert abs(Sqrt(Variable("x")).evaluate({"x": 9.0}) - 3.0) < 1e-12

    def test_nested_expression(self) -> None:
        # x^2 + 3x at x=2 => 4 + 6 = 10
        x = Variable("x")
        expr = x**2 + 3 * x
        assert expr.evaluate({"x": 2.0}) == 10.0

    def test_unsupported_operand_raises(self) -> None:
        with pytest.raises(TypeError, match="unsupported operand"):
            _ = Variable("x") + "foo"  # type: ignore[operator]


# ---------------------------------------------------------------------- #
# Differentiation
# ---------------------------------------------------------------------- #


class TestDifferentiation:
    def test_constant_derivative_is_zero(self) -> None:
        assert Constant(5).diff("x") == Constant(0.0)

    def test_variable_derivative_wrt_self(self) -> None:
        assert Variable("x").diff("x") == Constant(1.0)

    def test_variable_derivative_wrt_other(self) -> None:
        assert Variable("x").diff("y") == Constant(0.0)

    def test_sum_rule(self) -> None:
        # d/dx (x^2 + 3x) at x=2 => 2x + 3 = 7
        x = Variable("x")
        df = (x**2 + 3 * x).diff("x")
        assert abs(df.evaluate({"x": 2.0}) - 7.0) < 1e-6

    def test_difference_rule(self) -> None:
        # d/dx (x^2 - x) at x=3 => 2x - 1 = 5
        x = Variable("x")
        df = (x**2 - x).diff("x")
        assert abs(df.evaluate({"x": 3.0}) - 5.0) < 1e-6

    def test_product_rule(self) -> None:
        # d/dx (x * sin(x)) at x=pi/2 => sin(x) + x*cos(x) = 1 + 0 = 1
        x = Variable("x")
        df = (x * Sin(x)).diff("x")
        assert abs(df.evaluate({"x": math.pi / 2}) - 1.0) < 1e-6

    def test_quotient_rule(self) -> None:
        # d/dx (sin(x) / x) at x=1 => (cos(1)*1 - sin(1)*0)/1 ... use numeric check
        x = Variable("x")
        df = (Sin(x) / x).diff("x")
        h = 1e-6
        numeric = (math.sin(1 + h) / (1 + h) - math.sin(1 - h) / (1 - h)) / (2 * h)
        assert abs(df.evaluate({"x": 1.0}) - numeric) < 1e-4

    def test_power_constant_exponent(self) -> None:
        # d/dx x^3 at x=2 => 3x^2 = 12
        x = Variable("x")
        df = (x**3).diff("x")
        assert abs(df.evaluate({"x": 2.0}) - 12.0) < 1e-6

    def test_power_constant_base(self) -> None:
        # d/dx 2^x at x=0 => 2^0 * ln(2) * 1 = ln(2)
        x = Variable("x")
        df = (2**x).diff("x")
        assert abs(df.evaluate({"x": 0.0}) - math.log(2)) < 1e-6

    def test_power_general_case(self) -> None:
        # d/dx x^x at x=2 => x^x * (ln(x) + 1) = 4 * (ln2 + 1)
        x = Variable("x")
        df = (x**x).diff("x")
        expected = (2**2) * (math.log(2) + 1)
        assert abs(df.evaluate({"x": 2.0}) - expected) < 1e-6

    def test_sin_derivative(self) -> None:
        # d/dx sin(x) = cos(x); at x=0 => 1
        assert abs(Sin(Variable("x")).diff("x").evaluate({"x": 0.0}) - 1.0) < 1e-6

    def test_cos_derivative(self) -> None:
        # d/dx cos(x) = -sin(x); at x=pi/2 => -1
        df = Cos(Variable("x")).diff("x")
        assert abs(df.evaluate({"x": math.pi / 2}) - (-1.0)) < 1e-6

    def test_exp_derivative(self) -> None:
        # d/dx e^x = e^x; at x=1 => e
        assert abs(Exp(Variable("x")).diff("x").evaluate({"x": 1.0}) - math.e) < 1e-6

    def test_log_derivative(self) -> None:
        # d/dx ln(x) = 1/x; at x=2 => 0.5
        assert abs(Log(Variable("x")).diff("x").evaluate({"x": 2.0}) - 0.5) < 1e-6

    def test_sqrt_derivative(self) -> None:
        # d/dx sqrt(x) = 1/(2*sqrt(x)); at x=4 => 0.25
        assert abs(Sqrt(Variable("x")).diff("x").evaluate({"x": 4.0}) - 0.25) < 1e-6

    def test_chain_rule_nested(self) -> None:
        # d/dx sin(x^2) at x=1 => cos(1) * 2
        x = Variable("x")
        df = Sin(x**2).diff("x")
        assert abs(df.evaluate({"x": 1.0}) - math.cos(1) * 2) < 1e-6

    def test_constant_expression_derivative(self) -> None:
        # d/dx (5 + 3) => derivative of a constant sub-expression evaluates to 0
        df = (Constant(5.0) + Constant(3.0)).diff("x")
        assert abs(df.evaluate({}) - 0.0) < 1e-12

    def test_power_no_variables_derivative_is_zero(self) -> None:
        # d/dx (2^3) where neither base nor exponent depends on x => 0
        df = (Constant(2.0) ** Constant(3.0)).diff("x")
        assert df == Constant(0.0)


# ---------------------------------------------------------------------- #
# Simplification
# ---------------------------------------------------------------------- #


class TestSimplification:
    def test_add_zero_left(self) -> None:
        assert (Constant(0.0) + Variable("x")).simplify() == Variable("x")

    def test_add_zero_right(self) -> None:
        assert (Variable("x") + Constant(0.0)).simplify() == Variable("x")

    def test_sub_zero_right(self) -> None:
        assert (Variable("x") - Constant(0.0)).simplify() == Variable("x")

    def test_mul_one_left(self) -> None:
        assert (Constant(1.0) * Variable("x")).simplify() == Variable("x")

    def test_mul_one_right(self) -> None:
        assert (Variable("x") * Constant(1.0)).simplify() == Variable("x")

    def test_mul_zero(self) -> None:
        assert (Variable("x") * Constant(0.0)).simplify() == Constant(0.0)

    def test_div_one(self) -> None:
        assert (Variable("x") / Constant(1.0)).simplify() == Variable("x")

    def test_div_zero_numerator(self) -> None:
        assert (Constant(0.0) / Variable("x")).simplify() == Constant(0.0)

    def test_pow_exponent_zero(self) -> None:
        assert (Variable("x") ** Constant(0.0)).simplify() == Constant(1.0)

    def test_pow_exponent_one(self) -> None:
        assert (Variable("x") ** Constant(1.0)).simplify() == Variable("x")

    def test_constant_folding_add(self) -> None:
        assert (Constant(2.0) + Constant(3.0)).simplify() == Constant(5.0)

    def test_constant_folding_mul(self) -> None:
        assert (Constant(2.0) * Constant(3.0)).simplify() == Constant(6.0)

    def test_constant_folding_sub(self) -> None:
        assert (Constant(5.0) - Constant(2.0)).simplify() == Constant(3.0)

    def test_constant_folding_div(self) -> None:
        assert (Constant(6.0) / Constant(2.0)).simplify() == Constant(3.0)

    def test_constant_folding_pow(self) -> None:
        assert (Constant(2.0) ** Constant(3.0)).simplify() == Constant(8.0)

    def test_constant_folding_sin(self) -> None:
        assert abs(Sin(Constant(math.pi / 2)).simplify().evaluate({}) - 1.0) < 1e-12

    def test_constant_folding_cos(self) -> None:
        assert abs(Cos(Constant(0.0)).simplify().evaluate({}) - 1.0) < 1e-12

    def test_constant_folding_exp(self) -> None:
        assert abs(Exp(Constant(0.0)).simplify().evaluate({}) - 1.0) < 1e-12

    def test_constant_folding_log(self) -> None:
        assert abs(Log(Constant(1.0)).simplify().evaluate({}) - 0.0) < 1e-12

    def test_constant_folding_sqrt(self) -> None:
        assert abs(Sqrt(Constant(16.0)).simplify().evaluate({}) - 4.0) < 1e-12

    def test_mul_left_constant_other_than_zero_one(self) -> None:
        # left Constant(2.0): neither 0 nor 1 => stays a Mul
        s = (Constant(2.0) * Variable("x")).simplify()
        assert isinstance(s, Mul)

    def test_mul_right_constant_other_than_zero_one(self) -> None:
        # right Constant(3.0): neither 0 nor 1 => stays a Mul
        s = (Variable("x") * Constant(3.0)).simplify()
        assert isinstance(s, Mul)

    def test_mul_left_constant_zero_collapses(self) -> None:
        # left Constant(0.0) * x => 0
        assert (Constant(0.0) * Variable("x")).simplify() == Constant(0.0)

    def test_div_right_constant_other_than_one(self) -> None:
        # right Constant(3.0): not 1 => stays a Div
        s = (Variable("x") / Constant(3.0)).simplify()
        assert isinstance(s, Div)

    def test_sub_cannot_simplify(self) -> None:
        # x - y cannot be simplified — returns a Sub of the same shape
        s = (Variable("x") - Variable("y")).simplify()
        assert isinstance(s, Sub)

    def test_pow_constant_base_and_variable_exp(self) -> None:
        # 2 ** x: neither base nor exp folds — returns Pow unchanged
        x = Variable("x")
        s = (Constant(2.0) ** x).simplify()
        assert isinstance(s, Pow)

    def test_pow_constant_exponent_other_than_zero_one(self) -> None:
        # x ** 3: exponent is a constant that is neither 0 nor 1 => Pow unchanged
        s = (Variable("x") ** Constant(3.0)).simplify()
        assert isinstance(s, Pow)

    def test_simplify_sin_keeps_variable(self) -> None:
        assert isinstance(Sin(Variable("x")).simplify(), Sin)

    def test_simplify_cos_keeps_variable(self) -> None:
        assert isinstance(Cos(Variable("x")).simplify(), Cos)

    def test_simplify_exp_keeps_variable(self) -> None:
        assert isinstance(Exp(Variable("x")).simplify(), Exp)

    def test_simplify_log_keeps_variable(self) -> None:
        assert isinstance(Log(Variable("x")).simplify(), Log)

    def test_simplify_sqrt_keeps_variable(self) -> None:
        assert isinstance(Sqrt(Variable("x")).simplify(), Sqrt)

    def test_simplify_preserves_nontrivial(self) -> None:
        # x + y cannot be simplified — returns an Add of the same shape
        s = (Variable("x") + Variable("y")).simplify()
        assert isinstance(s, Add)

    def test_simplify_recurse_into_tree(self) -> None:
        # (x + 0) * 1 => simplify inner add to x, then mul-1 to x
        expr = (Variable("x") + Constant(0.0)) * Constant(1.0)
        assert expr.simplify() == Variable("x")

    def test_base_simplify_is_identity(self) -> None:
        # The Expression base simplify() returns self unchanged.
        base = Expression()
        assert base.simplify() is base

    # ------------------------------------------------------------------ #
    # simplify() is a pure rewrite: it never raises. A constant fold that
    # would be undefined at evaluation time is left in place so the error
    # surfaces when the expression is actually evaluated, not during the
    # (side-effect-free) simplification pass.
    # ------------------------------------------------------------------ #

    def test_div_zero_constant_divisor_is_left_in_place(self) -> None:
        # 5 / 0 must not fold: left as a Div, error deferred to evaluate().
        s = (Constant(5.0) / Constant(0.0)).simplify()
        assert isinstance(s, Div)

    def test_div_zero_constant_divisor_surfaces_at_evaluate(self) -> None:
        expr = (Constant(5.0) / Constant(0.0)).simplify()
        with pytest.raises(ZeroDivisionError):
            expr.evaluate({})

    def test_log_nonpositive_constant_is_left_in_place(self) -> None:
        # log(0) and log(-1) must not fold: left as a Log.
        assert isinstance(Log(Constant(0.0)).simplify(), Log)
        assert isinstance(Log(Constant(-1.0)).simplify(), Log)

    def test_log_negative_constant_surfaces_at_evaluate(self) -> None:
        expr = Log(Constant(-1.0)).simplify()
        with pytest.raises(ValueError):
            expr.evaluate({})

    def test_sqrt_negative_constant_is_left_in_place(self) -> None:
        # sqrt(-4) is undefined over the reals: left as a Sqrt.
        assert isinstance(Sqrt(Constant(-4.0)).simplify(), Sqrt)

    def test_sqrt_negative_constant_surfaces_at_evaluate(self) -> None:
        expr = Sqrt(Constant(-4.0)).simplify()
        with pytest.raises(ValueError):
            expr.evaluate({})

    def test_pow_negative_base_fractional_exp_is_left_in_place(self) -> None:
        # (-8) ** (1/3) is complex over the reals: left as a Pow.
        s = (Constant(-8.0) ** Constant(1.0 / 3.0)).simplify()
        assert isinstance(s, Pow)

    def test_pow_negative_base_fractional_exp_surfaces_at_evaluate(self) -> None:
        expr = (Constant(-8.0) ** Constant(1.0 / 3.0)).simplify()
        with pytest.raises((ValueError, TypeError)):
            expr.evaluate({})


# ---------------------------------------------------------------------- #
# Substitution & variables
# ---------------------------------------------------------------------- #


class TestSubsAndVariables:
    def test_variables_constant(self) -> None:
        assert Constant(3.0).variables() == set()

    def test_variables_variable(self) -> None:
        assert Variable("x").variables() == {"x"}

    def test_variables_binary(self) -> None:
        assert (Variable("x") + Variable("y")).variables() == {"x", "y"}

    def test_variables_unary(self) -> None:
        assert Sin(Variable("x")).variables() == {"x"}

    def test_subs_replaces_variable(self) -> None:
        assert (Variable("x") ** 2).subs(x=3.0).evaluate({}) == 9.0

    def test_subs_partial(self) -> None:
        # Only x is substituted; y remains free
        expr = (Variable("x") + Variable("y")).subs(x=2.0)
        assert expr.evaluate({"y": 3.0}) == 5.0

    def test_subs_unknown_name_ignored(self) -> None:
        assert (Variable("x")).subs(z=5.0).evaluate({"x": 1.0}) == 1.0

    def test_subs_into_unary(self) -> None:
        assert abs(Sin(Variable("x")).subs(x=0.0).evaluate({}) - 0.0) < 1e-12


# ---------------------------------------------------------------------- #
# Rendering: to_str / to_latex / repr
# ---------------------------------------------------------------------- #


class TestRendering:
    def test_constant_int_str(self) -> None:
        assert Constant(3.0).to_str() == "3"

    def test_constant_float_str(self) -> None:
        assert "." in Constant(3.5).to_str()

    def test_constant_float_latex(self) -> None:
        # Non-integer constant renders through the repr() branch of to_latex.
        assert "." in Constant(3.5).to_latex()

    def test_constant_int_latex(self) -> None:
        assert Constant(3.0).to_latex() == "3"

    def test_sub_str_and_latex(self) -> None:
        # Direct Sub rendering (x - y), covering the subtract branches.
        sub = Variable("x") - Variable("y")
        assert "(x - y)" == sub.to_str()
        assert "x - y" in sub.to_latex()

    def test_variable_str(self) -> None:
        assert Variable("theta").to_str() == "theta"

    def test_add_str_has_parens(self) -> None:
        assert "(x + 3)" in (Variable("x") + 3).to_str()

    def test_mul_str(self) -> None:
        assert "*" in (Variable("x") * Variable("y")).to_str()

    def test_div_str(self) -> None:
        assert "/" in (Variable("x") / Variable("y")).to_str()

    def test_pow_str(self) -> None:
        assert "**" in (Variable("x") ** 2).to_str()

    def test_sin_str(self) -> None:
        assert Sin(Variable("x")).to_str() == "sin(x)"

    def test_cos_str(self) -> None:
        assert Cos(Variable("x")).to_str() == "cos(x)"

    def test_exp_str(self) -> None:
        assert Exp(Variable("x")).to_str() == "exp(x)"

    def test_log_str(self) -> None:
        assert Log(Variable("x")).to_str() == "log(x)"

    def test_sqrt_str(self) -> None:
        assert Sqrt(Variable("x")).to_str() == "sqrt(x)"

    def test_latex_frac(self) -> None:
        assert "\\frac{" in (Variable("x") / Variable("y")).to_latex()

    def test_latex_pow(self) -> None:
        assert "^{" in (Variable("x") ** 2).to_latex()

    def test_latex_sin(self) -> None:
        assert "\\sin" in Sin(Variable("x")).to_latex()

    def test_latex_cos(self) -> None:
        assert "\\cos" in Cos(Variable("x")).to_latex()

    def test_latex_exp(self) -> None:
        latex = Exp(Variable("x")).to_latex()
        assert "e^{" in latex

    def test_latex_log(self) -> None:
        assert "\\ln" in Log(Variable("x")).to_latex()

    def test_latex_sqrt(self) -> None:
        assert "\\sqrt{" in Sqrt(Variable("x")).to_latex()

    def test_latex_mul_cdot(self) -> None:
        assert "\\cdot" in (Variable("x") * Variable("y")).to_latex()

    def test_repr(self) -> None:
        assert "Sin" in repr(Sin(Variable("x")))


# ---------------------------------------------------------------------- #
# to_func compilation
# ---------------------------------------------------------------------- #


class TestToFunc:
    def test_to_func_single_var(self) -> None:
        f = (Variable("x") ** 2).to_func("x")
        assert f(3.0) == 9.0

    def test_to_func_multi_var(self) -> None:
        f = (Variable("x") + Variable("y")).to_func("x", "y")
        assert f(2.0, 3.0) == 5.0

    def test_to_func_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="not in expression"):
            (Variable("x") ** 2).to_func("z")

    def test_to_func_missing_var_raises(self) -> None:
        with pytest.raises(ValueError, match="missing variable"):
            (Variable("x") + Variable("y")).to_func("x")

    def test_to_func_wrong_arity_raises(self) -> None:
        f = (Variable("x") ** 2).to_func("x")
        with pytest.raises(ValueError, match="expected 1 args"):
            f(1.0, 2.0)


# ---------------------------------------------------------------------- #
# Equality and hashing
# ---------------------------------------------------------------------- #


class TestEquality:
    def test_constant_equality(self) -> None:
        assert Constant(2.0) == Constant(2.0)
        assert Constant(2.0) != Constant(3.0)

    def test_variable_equality(self) -> None:
        assert Variable("x") == Variable("x")
        assert Variable("x") != Variable("y")

    def test_constant_hash(self) -> None:
        assert hash(Constant(2.0)) == hash(Constant(2.0))

    def test_variable_hash(self) -> None:
        assert hash(Variable("x")) == hash(Variable("x"))

    def test_constant_not_equal_to_nonconstant(self) -> None:
        assert Constant(1.0) != "1.0"
        assert Variable("x") != 42


# ---------------------------------------------------------------------- #
# MathModel
# ---------------------------------------------------------------------- #


class TestMathModel:
    def _kinematics(self) -> MathModel:
        t = Variable("t")
        v0 = Variable("v0")
        model = MathModel(
            name="Kinematics",
            parameters={"a": 9.81},
            variables=["t", "v0"],
        )
        model.add_equation("velocity", v0 + Variable("a") * t)
        return model

    def test_add_equation_and_iterate(self) -> None:
        model = MathModel(name="m")
        model.add_equation("eq1", Variable("x"))
        assert model.equations[0][0] == "eq1"

    def test_set_parameter(self) -> None:
        model = MathModel(name="m")
        model.set_parameter("g", 9.81)
        assert model.parameters["g"] == 9.81

    def test_evaluate_merges_parameters(self) -> None:
        model = self._kinematics()
        # v = v0 + a*t, with a=9.81 (param), t=2, v0=0 => 19.62
        result = model.evaluate({"t": 2.0, "v0": 0.0})
        assert abs(result["velocity"] - 19.62) < 1e-9

    def test_equation_lookup(self) -> None:
        model = self._kinematics()
        assert model.equation("velocity").variables() == {"t", "v0", "a"}

    def test_equation_lookup_missing_raises(self) -> None:
        model = self._kinematics()
        with pytest.raises(KeyError, match="no equation labelled"):
            model.equation("nope")

    def test_gradient(self) -> None:
        model = self._kinematics()
        # dv/dt = a => 9.81
        g = model.gradient("velocity", "t")
        assert abs(g.evaluate({"t": 1.0, "v0": 0.0, "a": 9.81}) - 9.81) < 1e-9

    def test_jacobian(self) -> None:
        model = self._kinematics()
        jac = model.jacobian("t")
        assert "velocity" in jac
        # dv/dt = a (constant w.r.t. t except through 'a'); bind a only since
        # the derivative collapses to the parameter 'a'.
        assert abs(jac["velocity"].evaluate({"a": 9.81, "t": 1.0}) - 9.81) < 1e-9

    def test_free_variables_excludes_parameters(self) -> None:
        model = self._kinematics()
        # 'a' is a parameter, so it should not appear in free variables
        assert "a" not in model.free_variables()
        assert "t" in model.free_variables()

    def test_to_markdown_includes_name(self) -> None:
        md = self._kinematics().to_markdown()
        assert "Kinematics" in md
        assert "Parameters" in md
        assert "Equations" in md

    def test_to_markdown_with_description(self) -> None:
        model = MathModel(name="m", description="hello world")
        assert "hello world" in model.to_markdown()

    def test_to_markdown_with_variables_section(self) -> None:
        model = self._kinematics()
        assert "Variables" in model.to_markdown()

    def test_to_latex_align_block(self) -> None:
        latex = self._kinematics().to_latex()
        assert "\\begin{align*}" in latex
        assert "\\end{align*}" in latex
        assert "velocity" in latex

    def test_evaluate_missing_variable_raises(self) -> None:
        model = self._kinematics()
        with pytest.raises(ValueError, match="no value bound"):
            model.evaluate({"t": 2.0})  # v0 missing


# ---------------------------------------------------------------------- #
# Solver: solve_equation
# ---------------------------------------------------------------------- #


class TestSolveEquation:
    def test_sqrt_two(self) -> None:
        # x^2 - 2 = 0 => x = sqrt(2)
        result = solve_equation(Variable("x") ** 2 - 2, variable="x", x0=1.0)
        assert isinstance(result, SolveResult)
        assert result.converged
        assert abs(result.x - math.sqrt(2)) < 1e-6

    def test_residual_near_zero(self) -> None:
        result = solve_equation(Variable("x") ** 2 - 4, variable="x", x0=1.0)
        assert result.residual < 1e-6

    def test_returns_iterations(self) -> None:
        result = solve_equation(Variable("x") - 5, variable="x", x0=1.0)
        assert result.iterations >= 0
        assert abs(result.x - 5.0) < 1e-6

    def test_non_converging_low_max_iter(self) -> None:
        # A hard function with max_iter=1 likely won't converge
        result = solve_equation(Variable("x") ** 5 - 1000, variable="x", x0=1.0, max_iter=1)
        assert result.converged is False


# ---------------------------------------------------------------------- #
# Solver: fit_parameters
# ---------------------------------------------------------------------- #


class TestFitParameters:
    def test_linear_fit(self) -> None:
        # y = m*x; data (1,2),(2,4) => m=2
        m, x = Variable("m"), Variable("x")
        model = MathModel(name="linear", parameters={"m": 0.0})
        model.add_equation("y", m * x)
        data = [({"x": 1.0}, 2.0), ({"x": 2.0}, 4.0)]
        result = fit_parameters(model, data, ["m"], x0=[0.1])
        assert isinstance(result, FitResult)
        assert abs(result.parameters["m"] - 2.0) < 1e-3

    def test_fit_converged_flag(self) -> None:
        m, x = Variable("m"), Variable("x")
        model = MathModel(name="linear")
        model.add_equation("y", m * x)
        data = [({"x": 1.0}, 3.0)]
        result = fit_parameters(model, data, ["m"], x0=[1.0])
        assert result.converged

    def test_fit_uses_target_label(self) -> None:
        m, x = Variable("m"), Variable("x")
        b = Variable("b")
        model = MathModel(name="lin")
        model.add_equation("aux", m * x)  # first, ignored
        model.add_equation("target", m * x + b)
        data = [({"x": 1.0}, 5.0)]  # if m=2, b=3 => 5
        result = fit_parameters(model, data, ["m", "b"], x0=[0.1, 0.1], target_label="target")
        # Both m and b returned; exact split underdetermined but objective ~0
        assert result.residual < 1e-3

    def test_fit_empty_parameters_raises(self) -> None:
        model = MathModel(name="m")
        model.add_equation("y", Variable("x"))
        with pytest.raises(ValueError, match="at least one parameter"):
            fit_parameters(model, [({"x": 1.0}, 1.0)], [])

    def test_fit_no_observations_raises(self) -> None:
        m, x = Variable("m"), Variable("x")
        model = MathModel(name="m")
        model.add_equation("y", m * x)
        with pytest.raises(ValueError, match="at least one"):
            fit_parameters(model, [], ["m"])

    def test_fit_residual_decreases(self) -> None:
        # With a bad starting guess, fitting should reduce the residual vs naive
        m, x = Variable("m"), Variable("x")
        model = MathModel(name="m")
        model.add_equation("y", m * x)
        data = [({"x": 1.0}, 10.0)]
        result = fit_parameters(model, data, ["m"], x0=[0.0])
        # residual at m=10 should be ~0
        assert result.residual < 1.0
