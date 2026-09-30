"""Unit tests for src/population.py (Mini-Project 1).

Run with:  pytest -q
"""

import math
import statistics

import numpy as np
import pytest

from src.population import (
    BootstrapPredictionInterval,
    ClassroomPlanner,
    DistrictPopulation,
    ExponentialGrowthForecaster,
    FibonacciRatioForecaster,
    LinearTrendForecaster,
    ModelSelector,
    fibonacci,
    mae,
    mape,
    rmse,
)

YEARS = list(range(2015, 2025))
KAMPALA = [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800]


@pytest.fixture
def kampala() -> DistrictPopulation:
    return DistrictPopulation("Kampala", YEARS, KAMPALA)


# --- DistrictPopulation -----------------------------------------------------
class TestDistrictPopulation:
    def test_len_and_repr(self, kampala):
        assert len(kampala) == 10
        text = repr(kampala)
        assert "Kampala" in text and "2015-2024" in text

    def test_arrays_are_numpy_and_read_only(self, kampala):
        assert isinstance(kampala.populations, np.ndarray)
        with pytest.raises(ValueError):
            kampala.populations[0] = 0  # frozen

    @pytest.mark.parametrize(
        "years, pops",
        [
            ([], []),                          # empty input
            ([2015, 2016], [100]),             # unequal lengths
            ([2015, 2016], [100, -5]),         # negative value
            ([2015, 2017], [100, 110]),        # gap in years
            ([2015, 2016], [100, float("nan")]),
        ],
    )
    def test_invalid_input_rejected(self, years, pops):
        with pytest.raises(ValueError):
            DistrictPopulation("X", years, pops)

    def test_blank_name_rejected(self):
        with pytest.raises(ValueError):
            DistrictPopulation("  ", [2015], [1])

    def test_zero_population_is_allowed_but_growth_is_not(self):
        d = DistrictPopulation("Empty", [2015, 2016], [0, 10])
        with pytest.raises(ValueError):
            d.yoy_growth()

    def test_statistics_matches_numpy_with_ddof1(self, kampala):
        s = kampala.stats_with_statistics()
        n1 = kampala.stats_with_numpy(ddof=1)
        n0 = kampala.stats_with_numpy(ddof=0)
        assert s["variance"] == pytest.approx(n1["variance"])
        # Population variance = sample variance * (n-1)/n
        assert n0["variance"] == pytest.approx(s["variance"] * 9 / 10)
        assert s["median"] == pytest.approx(1460.0)

    def test_cagr_hand_calculation(self, kampala):
        # (1800/1200)^(1/9) - 1
        assert kampala.cagr() == pytest.approx(1.5 ** (1 / 9) - 1)

    def test_cagr_equals_geometric_mean_of_yoy(self, kampala):
        g = kampala.yoy_growth()
        geo = statistics.geometric_mean((1 + g).tolist()) - 1
        assert kampala.cagr() == pytest.approx(geo)

    def test_split(self, kampala):
        train, test = kampala.split(2021)
        assert len(train) == 7 and len(test) == 3
        assert test.years.tolist() == [2022, 2023, 2024]
        with pytest.raises(ValueError):
            kampala.split(2030)


# --- Forecasters ------------------------------------------------------------
class TestForecasters:
    def test_linear_recovers_exact_line(self):
        years = np.arange(2000, 2006)
        values = 100 + 7 * (years - 2000)
        model = LinearTrendForecaster().fit(years, values)
        np.testing.assert_allclose(model.predict(3), [142, 149, 156])
        np.testing.assert_allclose(model.residuals(), 0, atol=1e-9)

    def test_exponential_recovers_geometric_series(self):
        years = np.arange(5)
        values = 100 * 1.1 ** years
        model = ExponentialGrowthForecaster().fit(years, values)
        assert model.growth_rate_ == pytest.approx(0.10)
        np.testing.assert_allclose(model.predict(2), [100 * 1.1**5, 100 * 1.1**6])

    def test_fibonacci_sequence_and_ratios(self):
        assert fibonacci(7) == [1, 1, 2, 3, 5, 8, 13]
        assert fibonacci(0) == []
        model = FibonacciRatioForecaster(start_index=1).fit([2024], [100])
        # ratios 1/1, 2/1, 3/2 -> cumulative 1, 2, 3
        np.testing.assert_allclose(model.predict(3), [100, 200, 300])

    def test_fibonacci_ratios_converge_to_golden_ratio(self):
        ratios = FibonacciRatioForecaster(start_index=30).ratios(1)
        assert ratios[0] == pytest.approx((1 + math.sqrt(5)) / 2)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            LinearTrendForecaster().predict(3)

    @pytest.mark.parametrize("horizon", [0, -1, 2.5])
    def test_bad_horizon(self, kampala, horizon):
        model = LinearTrendForecaster().fit(kampala.years, kampala.populations)
        with pytest.raises(ValueError):
            model.predict(horizon)

    def test_too_few_points(self):
        with pytest.raises(ValueError):
            LinearTrendForecaster().fit([2024], [10])


# --- Metrics ----------------------------------------------------------------
class TestMetrics:
    def test_known_values(self):
        a, p = [100, 200, 300], [110, 190, 300]
        assert mae(a, p) == pytest.approx(20 / 3)
        assert rmse(a, p) == pytest.approx(math.sqrt(200 / 3))
        assert mape(a, p) == pytest.approx((10 + 5 + 0) / 3)

    def test_perfect_forecast_scores_zero(self):
        assert mae([1, 2], [1, 2]) == rmse([1, 2], [1, 2]) == mape([1, 2], [1, 2]) == 0

    def test_edge_cases(self):
        with pytest.raises(ValueError):
            mae([], [])
        with pytest.raises(ValueError):
            rmse([1, 2], [1])
        with pytest.raises(ValueError):
            mape([0, 1], [0, 1])


# --- Selector, bootstrap, planner --------------------------------------------
def test_selector_picks_linear_for_linear_data():
    d = DistrictPopulation("Line", YEARS, [500 + 20 * i for i in range(10)])
    sel = ModelSelector(
        [LinearTrendForecaster(), ExponentialGrowthForecaster(), FibonacciRatioForecaster()]
    )
    results = sel.validate(d, 2021)
    assert sel.best(results).model == "Linear trend"
    assert len(results) == 3


def test_bootstrap_is_reproducible_and_brackets_point(kampala):
    boot = BootstrapPredictionInterval(n_resamples=200, seed=7)
    a = boot.run(LinearTrendForecaster(), kampala, 5)
    b = boot.run(LinearTrendForecaster(), kampala, 5)
    np.testing.assert_allclose(a.lower, b.lower)
    assert a.paths.shape == (200, 5)
    assert np.all(a.lower <= a.point) and np.all(a.point <= a.upper)


def test_classroom_planner_hand_calculation():
    planner = ClassroomPlanner(school_age_share=0.18, pupils_per_classroom=53)
    # 100k people -> 18,000 pupils -> ceil(339.6) = 340 classrooms
    assert planner.classrooms(100) == 340
    # growth from 100k to 110k -> 374 - 340 = 34 extra
    assert planner.additional_classrooms(100, 110) == 34
    # shrinking population needs no new classrooms
    assert planner.additional_classrooms(110, 100) == 0
    assert planner.classrooms(0) == 0


def test_classroom_planner_rejects_invalid():
    with pytest.raises(ValueError):
        ClassroomPlanner(pupils_per_classroom=0)
    with pytest.raises(ValueError):
        ClassroomPlanner(school_age_share=1.5)
    with pytest.raises(ValueError):
        ClassroomPlanner().pupils(-1)
