"""Unit tests for src/fisheries.py (Mini-Project 3).

Run with:  pytest -q
"""

import statistics

import numpy as np
import pytest

from src.fisheries import (
    ClosedSeasonPolicy,
    FishStock,
    HarvestPolicy,
    PriceModel,
    RiskAssessor,
    fibonacci_stock,
    weekly_revenue,
)


# --- baseline ---------------------------------------------------------------
def test_fibonacci_baseline():
    seq = fibonacci_stock(15)
    assert len(seq) == 15 and seq[:5] == [1, 1, 2, 3, 5] and seq[-1] == 610
    assert fibonacci_stock(0) == []
    with pytest.raises(ValueError):
        fibonacci_stock(-1)


# --- stock dynamics ---------------------------------------------------------
def test_one_step_by_hand():
    stock = FishStock(r=0.4, K=10_000, n0=4_000, policy=0.1)
    nxt, harvest = stock.step(4_000, 0.1)
    # 4000 + 0.4*4000*(1-0.4) - 0.1*4000 = 4000 + 960 - 400
    assert nxt == pytest.approx(4_560)
    assert harvest == pytest.approx(400)


@pytest.mark.parametrize("h", [0.05, 0.10, 0.20, 0.30])
def test_converges_to_analytic_equilibrium(h):
    stock = FishStock(policy=h)
    traj = stock.simulate(300)
    assert traj.final_stock == pytest.approx(stock.equilibrium(), rel=1e-6)
    assert traj.harvest[-1] == pytest.approx(stock.sustainable_yield(), rel=1e-6)


def test_msy_benchmarks():
    stock = FishStock(r=0.4, K=10_000)
    assert stock.msy == pytest.approx(1_000)
    assert stock.h_msy == pytest.approx(0.2)
    rates = np.linspace(0, 0.39, 40)
    yields = [stock.sustainable_yield(h) for h in rates]
    assert rates[int(np.argmax(yields))] == pytest.approx(0.2)


def test_zero_stock_stays_zero_and_overharvest_never_negative():
    assert FishStock(n0=0, policy=0.2).simulate(10).final_stock == 0
    traj = FishStock(policy=0.9).simulate(200)
    assert np.all(traj.stock >= 0)
    assert traj.final_stock < 1e-3


def test_no_harvest_grows_to_carrying_capacity():
    assert FishStock(policy=0.0).simulate(200).final_stock == pytest.approx(10_000, rel=1e-6)


@pytest.mark.parametrize("kwargs", [{"r": 0}, {"K": -1}, {"n0": -5}, {"policy": 1.0}, {"policy": -0.1}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        FishStock(**kwargs)


def test_simulate_rejects_zero_weeks():
    with pytest.raises(ValueError):
        FishStock().simulate(0)


# --- policies ---------------------------------------------------------------
def test_closed_season_policy():
    policy = ClosedSeasonPolicy(0.2, closed_weeks=range(14, 22))
    assert isinstance(policy, HarvestPolicy)
    assert policy.rate_at(13) == 0.2 and policy.rate_at(14) == 0.0 and policy.rate_at(21) == 0.0
    assert policy.rate_at(52 + 15) == 0.0         # repeats every year
    traj = FishStock(policy=policy).simulate(52)
    assert np.all(traj.harvest[14:22] == 0) and traj.harvest[22] > 0
    with pytest.raises(ValueError):
        ClosedSeasonPolicy(0.2, closed_weeks=[60])


# --- prices -----------------------------------------------------------------
@pytest.mark.parametrize("boundary", ["reflect", "clip"])
def test_prices_stay_in_band_and_are_reproducible(boundary):
    a = PriceModel(step_sd=3_000, boundary=boundary, seed=5).simulate(52, 200)
    b = PriceModel(step_sd=3_000, boundary=boundary, seed=5).simulate(52, 200)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (200, 52)
    assert np.all(a[:, 0] == 12_000)
    assert a.min() >= 9_000 and a.max() <= 16_000


def test_zero_volatility_gives_flat_price():
    p = PriceModel(step_sd=0).simulate(10, 3)
    assert np.all(p == 12_000)


@pytest.mark.parametrize("kwargs", [{"lower": 20_000}, {"start": 20_000}, {"step_sd": -1}, {"boundary": "wrap"}])
def test_invalid_price_model(kwargs):
    with pytest.raises(ValueError):
        PriceModel(**kwargs)


def test_revenue_units():
    # 2 t at 12,000 UGX/kg = 2,000 kg * 12,000 = 24,000,000 UGX
    assert weekly_revenue(np.array([2.0]), np.array([12_000.0]))[0] == pytest.approx(24_000_000)
    with pytest.raises(ValueError):
        weekly_revenue(np.array([1.0, 2.0]), np.array([1.0]))


# --- risk -------------------------------------------------------------------
def test_describe_matches_statistics():
    data = [10.0, 12.0, 14.0, 20.0]
    d = RiskAssessor.describe(data)
    assert d["variance"] == pytest.approx(statistics.variance(data))
    assert d["cv"] == pytest.approx(statistics.stdev(data) / statistics.mean(data))
    with pytest.raises(ValueError):
        RiskAssessor.describe([1.0])


def test_cv_is_scale_free_but_variance_is_not():
    data = np.array([100.0, 120.0, 90.0, 110.0])
    small, big = RiskAssessor.describe(data), RiskAssessor.describe(data * 1_000_000)
    assert big["cv"] == pytest.approx(small["cv"])
    assert big["variance"] == pytest.approx(small["variance"] * 1e12)


@pytest.mark.parametrize("cv, expected", [(0.0, "Low"), (0.099, "Low"), (0.10, "Moderate"),
                                          (0.19, "Moderate"), (0.20, "High"), (1.5, "High")])
def test_classify_boundaries(cv, expected):
    assert RiskAssessor().classify(cv) == expected


def test_value_at_risk():
    revenues = np.arange(1, 101, dtype=float)   # 1..100
    level, var = RiskAssessor(alpha=0.05).value_at_risk(revenues)
    assert level == pytest.approx(np.percentile(revenues, 5))
    assert var == pytest.approx(revenues.mean() - level)
    with pytest.raises(ValueError):
        RiskAssessor().value_at_risk([])


def test_overall_class_includes_stock_status():
    assessor = RiskAssessor()
    steady = np.full(100, 1e9) * (1 + np.linspace(-0.01, 0.01, 100))  # CV ~ 0.6 %
    assert assessor.assess(steady, FishStock(policy=0.2)).overall_class == "Low"
    assert assessor.assess(steady, FishStock(policy=0.3)).overall_class == "Moderate"
    report = assessor.assess(steady, FishStock(policy=0.45))
    assert report.stock_status == "Collapsing" and report.overall_class == "High"
    assert report.financial_class == "Low"
