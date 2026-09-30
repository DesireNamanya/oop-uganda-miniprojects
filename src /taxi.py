"""Matatu route revenue, fare equilibrium, forecasting and fleet planning (Mini-Project 5).

Design overview
---------------
* :class:`Route` - passenger counts (NumPy array) and fare; revenue and
  descriptive statistics via :mod:`statistics`.
* :class:`LinearMarket` - linear demand and supply curves solved as a 2x2
  system with :func:`scipy.linalg.solve`.
* :class:`SeriesForecaster` (abstract) - one-step-ahead forecasters sharing
  ``fit(history)`` / ``predict()``. Subclasses: :class:`MovingAverage`,
  :class:`SimpleExponentialSmoothing`, :class:`LinearTrend`,
  :class:`NaiveForecaster` and :class:`SeasonalNaive`.
* :class:`Backtester` - rolling-origin (walk-forward) evaluation and grid
  search; it *has* forecasters (composition) and works with any subclass.
* :class:`FleetPlanner` - converts a passenger forecast into vehicles.
* :func:`simulate_weekly_demand` - seeded 60-day series with a weekly pattern.

Days are numbered from 1 in the notebook; arrays are 0-based internally.
"""

from __future__ import annotations

import copy
import math
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import linalg

from src.population import mae  # reuse the validated metric from Project 1

__all__ = [
    "Route",
    "LinearMarket",
    "SeriesForecaster",
    "MovingAverage",
    "SimpleExponentialSmoothing",
    "LinearTrend",
    "NaiveForecaster",
    "SeasonalNaive",
    "BacktestResult",
    "Backtester",
    "FleetPlanner",
    "simulate_weekly_demand",
]


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------
class Route:
    """A matatu route with daily passenger counts and a flat fare.

    Parameters
    ----------
    name : str
    passengers : Sequence[float]
        Passengers per day (non-negative).
    fare : float
        Fare per passenger in UGX (positive).
    """

    def __init__(self, name: str, passengers: Sequence[float], fare: float) -> None:
        if not name or not name.strip():
            raise ValueError("Route name must be non-empty.")
        p = np.asarray(passengers, dtype=float)
        if p.ndim != 1 or p.size == 0:
            raise ValueError("Need a non-empty 1-D sequence of passenger counts.")
        if not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError("Passenger counts must be finite and non-negative.")
        if not math.isfinite(fare) or fare <= 0:
            raise ValueError("Fare must be a positive number.")
        self.name = name.strip()
        self._p = p
        self._p.setflags(write=False)
        self.fare = float(fare)

    def __repr__(self) -> str:
        return f"Route({self.name!r}, days={len(self)}, fare=UGX {self.fare:,.0f})"

    def __len__(self) -> int:
        return int(self._p.size)

    @property
    def passengers(self) -> np.ndarray:
        """Read-only daily passenger counts."""
        return self._p

    def daily_revenue(self) -> np.ndarray:
        """Revenue per day (UGX) = passengers x fare."""
        return self._p * self.fare

    def total_revenue(self) -> float:
        """Revenue over all days (UGX)."""
        return float(self.daily_revenue().sum())

    def describe(self) -> dict[str, float]:
        """Mean, median, sample variance and sample std of passengers (``statistics``)."""
        data = self._p.tolist()
        if len(data) < 2:
            raise ValueError("Need at least two days for a sample variance.")
        mean = statistics.mean(data)
        sd = statistics.stdev(data)
        return {"mean": mean, "median": statistics.median(data),
                "variance": statistics.variance(data), "std": sd, "cv": sd / mean if mean else math.nan}


# ---------------------------------------------------------------------------
# Market equilibrium
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LinearMarket:
    """Linear demand ``Qd = a - b P`` and supply ``Qs = c + d P``.

    Parameters
    ----------
    a, b : float
        Demand intercept and slope (``b > 0``).
    c, d : float
        Supply intercept and slope (``d > 0``).
    """

    a: float
    b: float
    c: float
    d: float

    def __post_init__(self) -> None:
        if self.b <= 0 or self.d <= 0:
            raise ValueError("Demand must slope down (b > 0) and supply up (d > 0).")

    def system(self) -> tuple[np.ndarray, np.ndarray]:
        """Coefficient matrix and RHS for unknowns ``[Q, P]``.

        ``Q + bP = a`` (demand) and ``Q - dP = c`` (supply).
        """
        return np.array([[1.0, self.b], [1.0, -self.d]]), np.array([self.a, self.c])

    def equilibrium(self) -> tuple[float, float]:
        """``(Q*, P*)`` from :func:`scipy.linalg.solve`."""
        A, rhs = self.system()
        q, p = linalg.solve(A, rhs)
        if q < 0 or p < 0:
            raise ValueError("Equilibrium is not economically meaningful (negative Q or P).")
        return float(q), float(p)

    def demand(self, price: float) -> float:
        return max(0.0, self.a - self.b * price)

    def supply(self, price: float) -> float:
        return max(0.0, self.c + self.d * price)

    def excess_demand(self, price: float) -> float:
        """``Qd - Qs`` at ``price``: positive = shortage, negative = surplus."""
        return self.demand(price) - self.supply(price)

    def price_elasticity(self, price: float) -> float:
        """Point elasticity of demand ``(dQ/dP)(P/Q)``."""
        q = self.demand(price)
        if q == 0:
            raise ValueError("Elasticity undefined where demand is zero.")
        return -self.b * price / q


# ---------------------------------------------------------------------------
# Forecasters
# ---------------------------------------------------------------------------
class SeriesForecaster(ABC):
    """One-step-ahead forecaster for a daily series.

    Subclasses implement :meth:`_forecast` (history -> next value). The base
    class validates the history, enforces ``min_history`` and never returns
    a negative passenger count.
    """

    label = "Forecaster"
    min_history = 1

    def __init__(self) -> None:
        self._history: np.ndarray | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def fit(self, history: Sequence[float]) -> "SeriesForecaster":
        h = np.asarray(history, dtype=float)
        if h.ndim != 1 or h.size < self.min_history:
            raise ValueError(f"{type(self).__name__} needs at least {self.min_history} observations.")
        if not np.all(np.isfinite(h)):
            raise ValueError("History contains non-finite values.")
        self._history = h
        return self

    def predict(self) -> float:
        """Forecast for the day after the fitted history."""
        if self._history is None:
            raise RuntimeError("Call fit() before predict().")
        return max(0.0, float(self._forecast(self._history)))

    @abstractmethod
    def _forecast(self, history: np.ndarray) -> float:
        """Return the next value given ``history``."""


class MovingAverage(SeriesForecaster):
    """Mean of the last ``window`` days (the original method uses 3)."""

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        if window < 1:
            raise ValueError("window must be >= 1.")
        self.window = window
        self.min_history = window
        self.label = f"{window}-day moving average"

    def __repr__(self) -> str:
        return f"MovingAverage(window={self.window})"

    def _forecast(self, history: np.ndarray) -> float:
        return float(history[-self.window:].mean())


class SimpleExponentialSmoothing(SeriesForecaster):
    """``level_t = alpha * y_t + (1 - alpha) * level_{t-1}``, started at ``y_1``.

    ``alpha = 1`` reproduces the naive forecast; small ``alpha`` smooths heavily.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = float(alpha)
        self.label = f"Exp. smoothing (alpha={self.alpha:.2f})"

    def __repr__(self) -> str:
        return f"SimpleExponentialSmoothing(alpha={self.alpha})"

    def _forecast(self, history: np.ndarray) -> float:
        level = history[0]
        for y in history[1:]:
            level = self.alpha * y + (1 - self.alpha) * level
        return float(level)


class LinearTrend(SeriesForecaster):
    """Least-squares line through the history (``np.polyfit``), extended one day."""

    label = "Linear trend"
    min_history = 2

    def _forecast(self, history: np.ndarray) -> float:
        t = np.arange(history.size)
        slope, intercept = np.polyfit(t, history, 1)
        return float(intercept + slope * history.size)


class NaiveForecaster(SeriesForecaster):
    """Tomorrow = today. The benchmark every model should beat."""

    label = "Naive (last value)"


    def _forecast(self, history: np.ndarray) -> float:
        return float(history[-1])


class SeasonalNaive(SeriesForecaster):
    """Tomorrow = the same weekday last week (``period`` days ago)."""

    def __init__(self, period: int = 7) -> None:
        super().__init__()
        if period < 1:
            raise ValueError("period must be >= 1.")
        self.period = period
        self.min_history = period
        self.label = f"Seasonal naive (period {period})"

    def __repr__(self) -> str:
        return f"SeasonalNaive(period={self.period})"

    def _forecast(self, history: np.ndarray) -> float:
        return float(history[-self.period])


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    """Walk-forward forecasts for one model on one series."""

    label: str
    target_index: np.ndarray  # 0-based indices of forecast days
    forecasts: np.ndarray
    actuals: np.ndarray

    @property
    def errors(self) -> np.ndarray:
        return self.actuals - self.forecasts

    @property
    def mae(self) -> float:
        return mae(self.actuals, self.forecasts)


class Backtester:
    """Rolling-origin (walk-forward) evaluation.

    For every target day ``t`` from ``first_target`` to the end of the
    series, a fresh copy of the model is fitted on days ``0 .. t-1`` only and
    asked for day ``t``. No future information leaks into any forecast.

    Parameters
    ----------
    first_target : int
        0-based index of the first day to forecast (3 = "day 4").
    """

    def __init__(self, first_target: int = 3) -> None:
        if first_target < 1:
            raise ValueError("first_target must be >= 1.")
        self.first_target = first_target

    def run(self, model: SeriesForecaster, series: Sequence[float]) -> BacktestResult:
        y = np.asarray(series, dtype=float)
        if y.size <= self.first_target:
            raise ValueError("Series too short for the chosen first_target.")
        idx = np.arange(self.first_target, y.size)
        preds = np.array([copy.deepcopy(model).fit(y[:t]).predict() for t in idx])
        return BacktestResult(model.label, idx, preds, y[idx])

    def compare(self, models: Iterable[SeriesForecaster], series: Sequence[float]) -> list[BacktestResult]:
        return [self.run(m, series) for m in models]

    def tune_alpha(self, series: Sequence[float], grid: Sequence[float] | None = None) -> tuple[float, dict[float, float]]:
        """Grid-search the SES ``alpha`` minimising backtest MAE.

        Returns ``(best_alpha, {alpha: mae})``. Ties go to the smaller alpha.
        """
        grid = np.round(np.arange(0.05, 1.0001, 0.05), 2) if grid is None else grid
        scores = {float(a): self.run(SimpleExponentialSmoothing(a), series).mae for a in grid}
        best = min(scores, key=lambda a: (round(scores[a], 10), a))
        return best, scores


# ---------------------------------------------------------------------------
# Fleet planning
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FleetPlanner:
    """Vehicles needed to carry a forecast number of passengers.

    Parameters
    ----------
    seats : int, default 14
    trips_per_day : int, default 8
        One-way trips per vehicle per day.
    buffer : float, default 0.15
        Safety margin added to the forecast.
    """

    seats: int = 14
    trips_per_day: int = 8
    buffer: float = 0.15

    def __post_init__(self) -> None:
        if self.seats <= 0 or self.trips_per_day <= 0:
            raise ValueError("seats and trips_per_day must be positive.")
        if not 0 <= self.buffer < 1:
            raise ValueError("buffer must be in [0, 1).")

    @property
    def daily_capacity(self) -> int:
        """Passengers one vehicle can carry per day (all trips full)."""
        return self.seats * self.trips_per_day

    def vehicles(self, forecast_passengers: float, min_vehicles: int = 1) -> int:
        """``ceil(forecast x (1 + buffer) / capacity)``, at least ``min_vehicles``.

        Always round *up*: a fraction of a vehicle cannot run, and rounding
        down would leave passengers stranded.
        """
        if forecast_passengers < 0 or not math.isfinite(forecast_passengers):
            raise ValueError("Forecast must be finite and non-negative.")
        need = forecast_passengers * (1 + self.buffer) / self.daily_capacity
        return max(min_vehicles, math.ceil(need - 1e-9))

    def load_factor(self, forecast_passengers: float, vehicles: int) -> float:
        """Share of seats filled if ``vehicles`` run all their trips."""
        if vehicles <= 0:
            raise ValueError("vehicles must be positive.")
        return forecast_passengers / (vehicles * self.daily_capacity)


# ---------------------------------------------------------------------------
# Simulation (extension)
# ---------------------------------------------------------------------------
DEFAULT_WEEKLY = {0: 1.00, 1: 0.95, 2: 0.97, 3: 1.02, 4: 1.30, 5: 1.10, 6: 0.65}  # Mon..Sun


def simulate_weekly_demand(base: float, n_days: int = 60, seed: int = 2026, start_weekday: int = 0,
                           weekly: Mapping[int, float] | None = None, noise_sd: float = 0.06,
                           trend_per_day: float = 0.0) -> np.ndarray:
    """Seeded daily passenger counts with a weekly pattern.

    ``y_t = round(base * (1 + trend * t) * weekly[weekday_t] * (1 + e_t))``,
    ``e_t ~ Normal(0, noise_sd)``; Fridays busiest and Sundays quietest by default.
    """
    if base <= 0 or n_days < 1:
        raise ValueError("base and n_days must be positive.")
    pattern = weekly or DEFAULT_WEEKLY
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    factors = np.array([pattern[(start_weekday + i) % 7] for i in t])
    y = base * (1 + trend_per_day * t) * factors * (1 + rng.normal(0, noise_sd, n_days))
    return np.clip(np.round(y), 0, None)
