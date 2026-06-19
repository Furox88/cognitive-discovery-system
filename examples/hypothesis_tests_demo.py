"""Classical hypothesis testing demo (t-test, chi-square, ANOVA)."""

from cds.stats import (
    chi_square_gof,
    chi_square_independence,
    one_sample_ttest,
    one_way_anova,
    two_sample_ttest,
)

# --- One-sample t-test (Student, 1908) ---


def main() -> None:
    print("=== One-sample t-test ===")
    sample = [5.1, 4.8, 5.3, 4.9, 5.0, 5.2, 4.7]
    res = one_sample_ttest(sample, popmean=4.5)
    print(f"H0: mean == 4.5  ->  t = {res.statistic:.3f}, df = {res.df}, p = {res.p_value:.4f}")

    # --- Two-sample t-test (pooled vs Welch) ---
    print("\n=== Two-sample t-test ===")
    group_a = [20.0, 22.0, 19.0, 24.0, 23.0, 21.0]
    group_b = [28.0, 31.0, 26.0, 30.0, 29.0, 27.0]
    pooled = two_sample_ttest(group_a, group_b, equal_var=True)
    welch = two_sample_ttest(group_a, group_b, equal_var=False)
    print(f"Pooled: t = {pooled.statistic:.3f}, df = {pooled.df:.1f}, p = {pooled.p_value:.5f}")
    print(f"Welch : t = {welch.statistic:.3f}, df = {welch.df:.2f}, p = {welch.p_value:.5f}")

    # --- Chi-square goodness-of-fit (Pearson, 1900) ---
    print("\n=== Chi-square goodness-of-fit ===")
    observed = [16.0, 18.0, 16.0, 14.0, 12.0, 12.0]
    expected = [16.0, 16.0, 16.0, 16.0, 16.0, 8.0]
    gof = chi_square_gof(observed, expected)
    print(f"chi2 = {gof.statistic:.3f}, df = {gof.df}, p = {gof.p_value:.4f}")

    # --- Chi-square test of independence ---
    print("\n=== Chi-square independence ===")
    table = [[10.0, 20.0, 30.0], [6.0, 9.0, 17.0]]
    ind = chi_square_independence(table)
    print(f"chi2 = {ind.statistic:.3f}, df = {ind.df}, p = {ind.p_value:.4f}")

    # --- One-way ANOVA (Fisher, 1925) ---
    print("\n=== One-way ANOVA ===")
    g1 = [8.0, 9.0, 7.0, 8.0, 9.0]
    g2 = [10.0, 11.0, 9.0, 12.0, 10.0]
    g3 = [13.0, 12.0, 14.0, 13.0, 15.0]
    anova = one_way_anova(g1, g2, g3)
    print(f"F = {anova.statistic:.3f}, df_between = {anova.df}, p = {anova.p_value:.6f}")


if __name__ == "__main__":
    main()
