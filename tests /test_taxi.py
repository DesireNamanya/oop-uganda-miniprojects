"""Unit tests for src/taxi.py (Mini-Project 5).

Run with:  pytest -q
"""

import statistics

import numpy as np
import pytest

from src.taxi import (
    Backtester,
    FleetPlanner,
    LinearMarket,
    LinearTrend,
    MovingAverage,
    NaiveForecaster,
    Route,
    SeasonalNaive,
    SeriesForecaster,
    SimpleExponentialSmoothing,
    simulate_weekly_demand,
)

NTINDA = [35, 40, 42, 50, 55, 60, 48, 52, 47, 45]


@pytest.fixture
def ntinda() -> Route:
    return Route("Kampala-Ntinda", NTINDA, 2000)


# --- Route -------------------------------------------------------------------
def test_route_revenue_and_stats(ntinda):
    assert ntinda.daily_revenue()[0] == 70_000
    assert ntinda.total_revenue() == sum(NTINDA) * 2000
    d = ntinda.describe()
    assert d["mean"] == pytest.approx(47.4)
    assert d["variance"] == pytest.approx(statistics.variance(NTINDA))
    assert len(ntinda) == 10


@pytest.mark.parametrize("args", [("R", [], 1000), ("R", [10, -1], 1000), ("R", [10], 0), ("", [10], 1000),
                                  ("R", [10, float("nan")], 1000)])
def test_route_rejects_invalid(args):
    with pytest.raises(ValueError):
        Route(*args)


def test_route_zero_passengers_allowed():
    r = Route("Quiet", [0, 0, 0], 1000)
    assert r.total_revenue() == 0


# --- market -------------------------------------------------------------------
def test_equilibrium_by_hand():
    market = LinearMarket(a=120, b=0.02, c=10, d=0.03)
    q, p = market.equilibrium()
    # 120 - 0.02P = 10 + 0.03P  ->  P = 110 / 0.05 = 2200, Q = 76
    assert p == pytest.approx(2200) and q == pytest.approx(76)
    assert market.excess_demand(2000) == pytest.approx(10)      # 80 demanded vs 70 supplied
    assert market.excess_demand(p) == pytest.approx(0)
    assert market.price_elasticity(2000) == pytest.approx(-0.5)


def test_market_rejects_wrong_slopes():
    with pytest.raises(ValueError):
        LinearMarket(120, -0.02, 10, 0.03)


def test_market_without_meaningful_equilibrium():
    with pytest.raises(ValueError):
        LinearMarket(a=10, b=0.02, c=50, d=0.03).equilibrium()   # supply exceeds demand even at P=0


# --- forecasters ----------------------------------------------------------------
def test_forecasters_by_hand():
    h = [10, 20, 30, 40]
    assert MovingAverage(3).fit(h).predict() == pytest.approx(30)
    assert NaiveForecaster().fit(h).predict() == 40
    assert LinearTrend().fit(h).predict() == pytest.approx(50)
    # SES alpha=0.5: 10 -> 15 -> 22.5 -> 31.25
    assert SimpleExponentialSmoothing(0.5).fit(h).predict() == pytest.approx(31.25)
    assert SimpleExponentialSmoothing(1.0).fit(h).predict() == 40      # alpha=1 is naive
    assert SeasonalNaive(2).fit(h).predict() == 30


def test_forecasters_share_the_abstract_interface():
    for m in (MovingAverage(), SimpleExponentialSmoothing(), LinearTrend(), NaiveForecaster(), SeasonalNaive()):
        assert isinstance(m, SeriesForecaster)
    with pytest.raises(TypeError):
        SeriesForecaster()


def test_forecast_never_negative():
    assert LinearTrend().fit([30, 20, 10, 0]).predict() == 0.0


@pytest.mark.parametrize("model, history", [(MovingAverage(3), [1, 2]), (SeasonalNaive(7), [1] * 6),
                                            (LinearTrend(), [5]), (NaiveForecaster(), [])])
def test_insufficient_history(model, history):
    with pytest.raises(ValueError):
        model.fit(history)


def test_invalid_parameters_and_unfitted():
    with pytest.raises(ValueError):
        SimpleExponentialSmoothing(0)
    with pytest.raises(ValueError):
        MovingAverage(0)
    with pytest.raises(RuntimeError):
        NaiveForecaster().predict()


# --- backtesting ----------------------------------------------------------------
def test_backtest_uses_only_past_data():
    series = [10, 20, 30, 40, 50]
    res = Backtester(first_target=3).run(NaiveForecaster(), series)
    assert res.target_index.tolist() == [3, 4]
    assert res.forecasts.tolist() == [30, 40]        # each forecast = previous day
    assert res.mae == pytest.approx(10)


def test_tune_alpha_picks_one_for_a_random_walk_like_series():
    best, scores = Backtester(3).tune_alpha(NTINDA)
    assert best == 1.0
    assert scores[1.0] == pytest.approx(Backtester(3).run(NaiveForecaster(), NTINDA).mae)


def test_backtest_too_short():
    with pytest.raises(ValueError):
        Backtester(first_target=5).run(NaiveForecaster(), [1, 2, 3])


# --- fleet ----------------------------------------------------------------------
def test_fleet_planner():
    fp = FleetPlanner(seats=14, trips_per_day=8, buffer=0.15)
    assert fp.daily_capacity == 112
    assert fp.vehicles(45) == 1                        # 51.75 / 112 -> 1
    assert fp.vehicles(100) == 2                       # 115 / 112 -> 2 (round UP)
    assert fp.vehicles(0) == 1                         # keep a minimum service
    assert fp.vehicles(0, min_vehicles=0) == 0
    assert fp.load_factor(56, 1) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        fp.vehicles(-5)
    with pytest.raises(ValueError):
        FleetPlanner(buffer=1.5)


# --- simulation -----------------------------------------------------------------
def test_simulated_weekly_pattern():
    y = simulate_weekly_demand(50, n_days=70, seed=1, noise_sd=0.0)
    weekday_means = y.reshape(10, 7).mean(axis=0)     # starts on Monday
    assert np.argmax(weekday_means) == 4 and np.argmin(weekday_means) == 6   # Friday busiest, Sunday quietest
    np.testing.assert_array_equal(simulate_weekly_demand(50, seed=3), simulate_weekly_demand(50, seed=3))


def test_seasonal_naive_beats_moving_average_on_weekly_data():
    y = simulate_weekly_demand(60, n_days=60, seed=11)
    bt = Backtester(first_target=14)
    assert bt.run(SeasonalNaive(7), y).mae < bt.run(MovingAverage(3), y).mae
