import pytest

from cds.stats import (
    chi2_sf,
    chi_square_gof,
    chi_square_independence,
    f_sf,
    one_sample_ttest,
    one_way_anova,
    t_sf,
    two_sample_ttest,
)


class TestDistributionTails:
    def test_t_sf_known_value(self):
        # P(|T| >= 2) for t(10) ~ 0.07339
        assert abs(t_sf(2.0, 10) - 0.07339) < 1e-4

    def test_t_sf_zero_is_one(self):
        assert abs(t_sf(0.0, 5) - 1.0) < 1e-9

    def test_t_sf_symmetric(self):
        assert abs(t_sf(1.5, 8) - t_sf(-1.5, 8)) < 1e-12

    def test_chi2_sf_critical_value(self):
        # chi2 0.95 quantile for df=1 is ~3.841
        assert abs(chi2_sf(3.841, 1) - 0.05) < 1e-3

    def test_chi2_sf_df5(self):
        assert abs(chi2_sf(11.0705, 5) - 0.05) < 1e-3

    def test_chi2_sf_zero_is_one(self):
        assert chi2_sf(0.0, 3) == 1.0

    def test_f_sf_known_value(self):
        # F=4.0, df1=2, df2=12 -> ~0.0466
        assert abs(f_sf(4.0, 2, 12) - 0.04663) < 1e-3

    def test_f_sf_zero_is_one(self):
        assert f_sf(0.0, 3, 10) == 1.0


class TestOneSampleTTest:
    def test_zero_difference(self):
        # sample mean equals popmean -> t == 0, p == 1
        r = one_sample_ttest([4.9, 5.1, 5.0, 5.2, 4.8], 5.0)
        assert abs(r.statistic) < 1e-9
        assert abs(r.p_value - 1.0) < 1e-9

    def test_zero_variance_raises(self):
        with pytest.raises(ValueError):
            one_sample_ttest([5.0, 5.0, 5.0, 5.0], 5.0)

    def test_significant_shift(self):
        r = one_sample_ttest([10.0, 11.0, 9.0, 10.5, 9.5], 5.0)
        assert r.statistic > 0
        assert r.p_value < 0.01
        assert r.df == 4

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            one_sample_ttest([1.0], 0.0)


class TestTwoSampleTTest:
    def test_pooled_known(self):
        # a=[1..5], b=[2,4,6,8,10] -> t=-1.897, df=8
        r = two_sample_ttest([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert abs(r.statistic - (-1.8974)) < 1e-3
        assert r.df == 8
        assert abs(r.p_value - 0.0943) < 1e-3

    def test_welch_df_noninteger(self):
        r = two_sample_ttest(
            [1, 2, 3, 4, 5],
            [10, 20, 30],
            equal_var=False,
        )
        assert r.df != int(r.df) or r.df < 6

    def test_identical_groups_p_one(self):
        r = two_sample_ttest([1, 2, 3, 4], [1, 2, 3, 4])
        assert abs(r.statistic) < 1e-9
        assert abs(r.p_value - 1.0) < 1e-9

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            two_sample_ttest([1.0], [1.0, 2.0])


class TestChiSquare:
    def test_gof_known(self):
        r = chi_square_gof(
            [16, 18, 16, 14, 12, 12],
            [16, 16, 16, 16, 16, 8],
        )
        assert abs(r.statistic - 3.5) < 1e-9
        assert r.df == 5
        assert abs(r.p_value - 0.6234) < 1e-3

    def test_gof_perfect_fit(self):
        r = chi_square_gof([10, 20, 30], [10, 20, 30])
        assert r.statistic == 0.0
        assert abs(r.p_value - 1.0) < 1e-9

    def test_gof_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            chi_square_gof([1, 2], [1, 2, 3])

    def test_independence_table(self):
        # classic 2x2 table
        table = [[10, 20], [30, 40]]
        r = chi_square_independence(table)
        assert r.df == 1
        assert r.statistic >= 0
        assert 0.0 <= r.p_value <= 1.0

    def test_independence_independent_data(self):
        # rows proportional -> chi2 ~ 0
        table = [[10, 20], [20, 40]]
        r = chi_square_independence(table)
        assert r.statistic < 1e-9

    def test_independence_too_small_raises(self):
        with pytest.raises(ValueError):
            chi_square_independence([[1, 2]])

    def test_independence_zero_row_total(self):
        # Row with all zeros -> some expected frequencies are 0 ->
        # the `if exp > 0` branch in chi_square_independence skips them.
        # Result should still be a valid TestResult.
        table = [[10, 20], [0, 0]]
        r = chi_square_independence(table)
        assert r.df == 1
        assert r.statistic >= 0
        assert 0.0 <= r.p_value <= 1.0

    def test_independence_zero_col_total(self):
        # Column with all zeros -> some expected frequencies are 0.
        table = [[10, 0], [20, 0]]
        r = chi_square_independence(table)
        assert r.df == 1
        assert r.statistic >= 0


class TestANOVA:
    def test_known_f(self):
        r = one_way_anova([1, 2, 3], [2, 3, 4], [4, 5, 6])
        assert abs(r.statistic - 7.0) < 1e-9
        assert abs(r.p_value - 0.02702) < 1e-3

    def test_identical_groups(self):
        r = one_way_anova([1, 2, 3], [1, 2, 3])
        assert abs(r.statistic) < 1e-9

    def test_too_few_groups_raises(self):
        with pytest.raises(ValueError):
            one_way_anova([1, 2, 3])
