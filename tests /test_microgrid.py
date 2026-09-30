"""Unit tests for src/microgrid.py (Mini-Project 2).

Run with:  pytest -q
"""

import numpy as np
import pytest

from src.microgrid import (
    DemandSeries,
    HybridMicroGrid,
    MicroGrid,
    SingularSystemError,
    parse_demand,
    prompt_demand,
)


@pytest.fixture
def grid() -> MicroGrid:
    return MicroGrid()


# --- well-posedness and solving ------------------------------------------
def test_determinant_and_condition_number(grid):
    assert grid.determinant == pytest.approx(-5.0)
    s = np.linalg.svd(grid.coefficients, compute_uv=False)
    assert grid.condition_number == pytest.approx(s.max() / s.min())
    assert grid.is_well_posed()


def test_solve_day_matches_hand_solution(grid):
    # Cramer's rule: x = (2*D2 - D1)/5, y = (4*D1 - 3*D2)/5
    d1, d2 = 100.0, 90.0
    x, y = grid.solve_day(d1, d2)
    assert x == pytest.approx((2 * d2 - d1) / 5)
    assert y == pytest.approx((4 * d1 - 3 * d2) / 5)
    np.testing.assert_allclose(grid.coefficients @ [x, y], [d1, d2])


def test_zero_demand_gives_zero_dispatch(grid):
    np.testing.assert_allclose(grid.solve_day(0, 0), [0, 0])


def test_loop_and_vectorised_agree(grid):
    rhs = DemandSeries.synthetic(n_days=30, seed=1).rhs
    np.testing.assert_allclose(grid.solve_loop(rhs), grid.solve_vectorised(rhs))


@pytest.mark.parametrize("bad", [(-1, 50), (np.nan, 50), (10,), (10, 20, 30)])
def test_bad_demands_rejected(grid, bad):
    with pytest.raises(ValueError):
        grid.solve_day(*bad)


def test_invalid_matrices_rejected():
    with pytest.raises(ValueError):
        MicroGrid([[1, 2, 3], [4, 5, 6]])          # not square
    with pytest.raises(ValueError):
        MicroGrid([[1, 2], [3, 4]], {"solar": 1})  # cost/source mismatch
    with pytest.raises(ValueError):
        MicroGrid(unit_costs={"solar": -1, "battery": 2})


def test_singular_matrix_refuses_to_solve():
    singular = MicroGrid([[1, 2], [2, 4]])
    assert not singular.is_well_posed()
    with pytest.raises(SingularSystemError):
        singular.solve_day(10, 20)


# --- infeasibility handling -------------------------------------------------
def test_infeasible_day_is_flagged_and_repaired(grid):
    # D1/D2 < 0.75 forces negative battery use
    rhs = np.array([[100.0, 60.0], [90.0, 100.0]])
    for strategy in ("clip", "nnls", "cover"):
        plan = grid.dispatch(rhs, strategy)
        assert plan.infeasible_days.tolist() == [1]
        assert np.all(plan.adjusted >= -1e-9)
        np.testing.assert_allclose(plan.adjusted[:, 0], plan.raw[:, 0])  # feasible day untouched


def test_cover_strategy_never_leaves_load_unmet(grid):
    rhs = np.array([[60.0], [100.0]])
    plan = grid.dispatch(rhs, "cover")
    assert np.all(plan.residual <= 1e-6)             # demand - supply <= 0
    # Cheapest cover here is solar only: x = max(60/3, 100/4) = 25
    np.testing.assert_allclose(plan.adjusted[:, 0], [25.0, 0.0], atol=1e-6)


def test_unknown_strategy(grid):
    with pytest.raises(ValueError):
        grid.dispatch(np.array([[1.0], [1.0]]), "magic")


def test_daily_cost(grid):
    assert grid.daily_cost(np.array([10.0, 2.0])) == pytest.approx(10 * 150 + 2 * 450)
    np.testing.assert_allclose(grid.daily_cost(np.array([[1.0, 0.0], [0.0, 1.0]])), [150, 450])


# --- input handling ---------------------------------------------------------
@pytest.mark.parametrize("text, expected", [("42", 42.0), (" 3.5 ", 3.5), ("1,200", 1200.0), ("0", 0.0)])
def test_parse_demand_valid(text, expected):
    assert parse_demand(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "abc", "-5", "nan", "inf"])
def test_parse_demand_invalid(text):
    with pytest.raises(ValueError):
        parse_demand(text)


def test_prompt_reprompts_until_valid():
    answers = iter(["", "ten", "-3", "95.5"])
    messages = []
    value = prompt_demand("D1: ", input_fn=lambda _: next(answers), output_fn=messages.append)
    assert value == 95.5
    assert len(messages) == 3


def test_prompt_gives_up_after_max_attempts():
    with pytest.raises(RuntimeError):
        prompt_demand("D1: ", input_fn=lambda _: "x", output_fn=lambda _: None, max_attempts=2)


# --- demand data -------------------------------------------------------------
def test_synthetic_is_reproducible_and_csv_round_trips(tmp_path):
    a = DemandSeries.synthetic(seed=7)
    b = DemandSeries.synthetic(seed=7)
    np.testing.assert_array_equal(a.rhs, b.rhs)
    assert len(a) == 30
    loaded = DemandSeries.from_csv(a.to_csv(tmp_path / "demand.csv"))
    np.testing.assert_allclose(loaded.rhs, a.rhs)
    assert loaded.dates == a.dates


def test_csv_with_bad_value_reports_line(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("date,d1_kwh,d2_kwh\n2026-09-01,90,100\n2026-09-02,,100\n")
    with pytest.raises(ValueError, match="Line 3"):
        DemandSeries.from_csv(path)


def test_empty_series_rejected():
    with pytest.raises(ValueError):
        DemandSeries([], {"d1_kwh": []})


# --- extension: hybrid grid and sensitivity ----------------------------------
def test_hybrid_solves_3x3():
    hybrid = HybridMicroGrid()
    assert isinstance(hybrid, MicroGrid)
    sol = hybrid.solve_day(95, 100, 40)
    np.testing.assert_allclose(hybrid.coefficients @ sol, [95, 100, 40])
    assert hybrid.determinant == pytest.approx(-7.0)


def test_hybrid_dependent_row_classification():
    dep = HybridMicroGrid.with_dependent_constraint((1, 1))
    assert dep.rank == 2
    assert dep.classify(95, 100, 195) == "infinitely many"   # D3 = D1 + D2: consistent
    assert dep.classify(95, 100, 40) == "none"               # contradictory
    with pytest.raises(SingularSystemError):
        dep.solve_day(95, 100, 195)
    sol, resid = dep.solve_least_squares(95, 100, 195)
    assert resid == pytest.approx(0, abs=1e-8)


def test_sensitivity_bounded_by_condition_number(grid):
    result = grid.sensitivity([95.0, 100.0], rel_perturbation=0.05, n_draws=500, seed=3)
    assert result.solutions.shape == (500, 2)
    assert np.all(result.amplification <= grid.condition_number + 1e-9)
    with pytest.raises(ValueError):
        grid.sensitivity([95.0, 100.0], rel_perturbation=0)
