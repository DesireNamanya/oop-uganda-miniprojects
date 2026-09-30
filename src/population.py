"""District population modelling and forecasting (Mini-Project 1).

This module contains everything the notebook ``project1_population.ipynb``
needs, so that the notebook only *uses* the logic and the tests can exercise
it directly.

Design overview
---------------
* :class:`DistrictPopulation` - a validated time series for one district
  (the "data" object). It knows how to describe itself (descriptive
  statistics, growth rates, CAGR) and how to split itself into train/test.
* :class:`Forecaster` - an abstract base class fixing the interface
  ``fit(years, values) -> self`` and ``predict(horizon) -> ndarray``.
  Three concrete subclasses implement the required models:
  :class:`LinearTrendForecaster`, :class:`ExponentialGrowthForecaster`
  and :class:`FibonacciRatioForecaster`.
* :class:`ModelSelector` - *composes* a set of forecasters and a metric to
  run the hold-out validation and pick the best model per district.
* :class:`BootstrapPredictionInterval` - residual bootstrap for prediction
  intervals; works with *any* :class:`Forecaster` (polymorphism).
* :class:`ClassroomPlanner` - turns a population forecast into a classroom
  requirement.

All populations are in **thousands of people** unless stated otherwise.
"""

from __future__ import annotations

import copy
import math
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np

__all__ = [
    "DistrictPopulation",
    "Forecaster",
    "LinearTrendForecaster",
    "ExponentialGrowthForecaster",
    "FibonacciRatioForecaster",
    "mae",
    "rmse",
    "mape",
    "ModelSelector",
    "BootstrapPredictionInterval",
    "ClassroomPlanner",
    "fibonacci",
]


# ---------------------------------------------------------------------------
# Data object
# ---------------------------------------------------------------------------
class DistrictPopulation:
    """A validated annual population series for one district.

    Parameters
    ----------
    name : str
        District name, e.g. ``"Kampala"``.
    years : Sequence[int]
        Calendar years, strictly increasing and consecutive.
    populations : Sequence[float]
        Population for each year (thousands). Must be non-negative.

    Raises
    ------
    ValueError
        If the name is blank, the series is empty, the lengths differ,
        any population is negative or non-finite, or the years are not
        strictly increasing by one.

    Examples
    --------
    >>> gulu = DistrictPopulation("Gulu", [2015, 2016], [320, 330])
    >>> len(gulu)
    2
    """

    def __init__(self, name: str, years: Sequence[int], populations: Sequence[float]) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("District name must be a non-empty string.")
        years_arr = np.asarray(years, dtype=int)
        pops_arr = np.asarray(populations, dtype=float)

        if years_arr.ndim != 1 or pops_arr.ndim != 1:
            raise ValueError("Years and populations must be one-dimensional.")
        if years_arr.size == 0:
            raise ValueError("A population series needs at least one observation.")
        if years_arr.size != pops_arr.size:
            raise ValueError(
                f"Length mismatch: {years_arr.size} years but {pops_arr.size} populations."
            )
        if not np.all(np.isfinite(pops_arr)):
            raise ValueError("Populations must be finite numbers (no NaN/inf).")
        if np.any(pops_arr < 0):
            raise ValueError("Populations cannot be negative.")
        if years_arr.size > 1 and not np.all(np.diff(years_arr) == 1):
            raise ValueError("Years must be consecutive and strictly increasing.")

        self._name = name.strip()
        self._years = years_arr
        self._populations = pops_arr
        # Freeze the arrays so a caller cannot silently corrupt validated data.
        self._years.setflags(write=False)
        self._populations.setflags(write=False)

    # -- dunder methods ----------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"DistrictPopulation(name={self._name!r}, "
            f"years={int(self._years[0])}-{int(self._years[-1])}, n={len(self)}, "
            f"last={self._populations[-1]:,.0f}k)"
        )

    def __len__(self) -> int:
        return int(self._years.size)

    def __iter__(self) -> Iterator[tuple[int, float]]:
        """Iterate over ``(year, population)`` pairs."""
        return iter(zip(self._years.tolist(), self._populations.tolist()))

    def __getitem__(self, year: int) -> float:
        """Return the population for a calendar ``year``."""
        idx = np.flatnonzero(self._years == year)
        if idx.size == 0:
            raise KeyError(f"{year} not in {self._name}'s series.")
        return float(self._populations[idx[0]])

    # -- read-only properties ----------------------------------------------
    @property
    def name(self) -> str:
        """District name."""
        return self._name

    @property
    def years(self) -> np.ndarray:
        """Years as a read-only integer array."""
        return self._years

    @property
    def populations(self) -> np.ndarray:
        """Populations (thousands) as a read-only float array."""
        return self._populations

    # -- descriptive statistics --------------------------------------------
    def stats_with_statistics(self) -> dict[str, float]:
        """Mean, median, *sample* variance and *sample* std via ``statistics``.

        ``statistics.variance`` divides by ``n - 1`` (Bessel's correction),
        treating the data as a sample from a larger process.
        """
        data = self._populations.tolist()
        if len(data) < 2:
            raise ValueError("Sample variance needs at least two observations.")
        return {
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "variance": statistics.variance(data),
            "std": statistics.stdev(data),
        }

    def stats_with_numpy(self, ddof: int = 0) -> dict[str, float]:
        """Mean, median, variance and std via NumPy.

        Parameters
        ----------
        ddof : int, default 0
            "Delta degrees of freedom": the divisor is ``n - ddof``.
            ``ddof=0`` (NumPy's default) gives the *population* variance;
            ``ddof=1`` reproduces ``statistics.variance``.
        """
        if ddof < 0 or ddof >= len(self):
            raise ValueError("ddof must satisfy 0 <= ddof < n.")
        x = self._populations
        return {
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "variance": float(np.var(x, ddof=ddof)),
            "std": float(np.std(x, ddof=ddof)),
        }

    # -- growth ------------------------------------------------------------
    def yoy_growth(self) -> np.ndarray:
        """Year-on-year growth rates ``(P_t - P_{t-1}) / P_{t-1}`` as fractions.

        Returns an array of length ``n - 1`` aligned with ``years[1:]``.
        """
        if len(self) < 2:
            raise ValueError("Growth needs at least two observations.")
        prev = self._populations[:-1]
        if np.any(prev == 0):
            raise ValueError("Cannot compute growth from a zero population.")
        return np.diff(self._populations) / prev

    def cagr(self) -> float:
        """Compound Annual Growth Rate between the first and last year.

        ``CAGR = (P_end / P_start) ** (1 / (n_years)) - 1`` where
        ``n_years = years[-1] - years[0]``.
        """
        if len(self) < 2:
            raise ValueError("CAGR needs at least two observations.")
        start, end = self._populations[0], self._populations[-1]
        if start <= 0:
            raise ValueError("CAGR is undefined for a non-positive starting value.")
        periods = int(self._years[-1] - self._years[0])
        return float((end / start) ** (1.0 / periods) - 1.0)

    # -- splitting ---------------------------------------------------------
    def split(self, last_train_year: int) -> tuple["DistrictPopulation", "DistrictPopulation"]:
        """Split into (train, test) series; ``last_train_year`` goes to train."""
        mask = self._years <= last_train_year
        if mask.all() or not mask.any():
            raise ValueError(f"{last_train_year} does not split the series into two parts.")
        train = DistrictPopulation(self._name, self._years[mask], self._populations[mask])
        test = DistrictPopulation(self._name, self._years[~mask], self._populations[~mask])
        return train, test


# ---------------------------------------------------------------------------
# Forecasters
# ---------------------------------------------------------------------------
class Forecaster(ABC):
    """Abstract base class for annual population forecasters.

    Subclasses implement :meth:`_fit` and :meth:`_predict`; the public
    :meth:`fit` / :meth:`predict` wrappers handle validation and state so
    that every model behaves the same way (template-method pattern).
    """

    #: Short human-readable label used in tables and legends.
    label: str = "Forecaster"

    def __init__(self) -> None:
        self._years: np.ndarray | None = None
        self._values: np.ndarray | None = None

    def __repr__(self) -> str:
        state = "fitted" if self.is_fitted else "unfitted"
        return f"{type(self).__name__}({state})"

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called successfully."""
        return self._values is not None

    def fit(self, years: Sequence[int], values: Sequence[float]) -> "Forecaster":
        """Fit the model to a training series and return ``self``."""
        y = np.asarray(years, dtype=float)
        v = np.asarray(values, dtype=float)
        if y.size != v.size:
            raise ValueError("years and values must have the same length.")
        if v.size < self.min_observations:
            raise ValueError(
                f"{type(self).__name__} needs at least {self.min_observations} observations."
            )
        self._years, self._values = y, v
        self._fit(y, v)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Forecast the next ``horizon`` years after the training data."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        return np.asarray(self._predict(int(horizon)), dtype=float)

    def fitted_values(self) -> np.ndarray:
        """In-sample fitted values aligned with the training years."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        return np.asarray(self._fitted(), dtype=float)

    def residuals(self) -> np.ndarray:
        """In-sample residuals ``actual - fitted``."""
        return self._values - self.fitted_values()  # type: ignore[operator]

    def fit_predict(self, series: DistrictPopulation, horizon: int) -> np.ndarray:
        """Convenience: fit on a :class:`DistrictPopulation` and forecast."""
        return self.fit(series.years, series.populations).predict(horizon)

    # -- hooks for subclasses ---------------------------------------------
    min_observations: int = 2

    @abstractmethod
    def _fit(self, years: np.ndarray, values: np.ndarray) -> None:
        """Estimate model parameters."""

    @abstractmethod
    def _predict(self, horizon: int) -> np.ndarray:
        """Return ``horizon`` out-of-sample forecasts."""

    @abstractmethod
    def _fitted(self) -> np.ndarray:
        """Return in-sample fitted values."""


class LinearTrendForecaster(Forecaster):
    """Straight-line trend ``P(t) = a + b·t`` estimated with ``np.polyfit``.

    Assumes a constant *absolute* increase (thousands of people per year).
    """

    label = "Linear trend"

    def _fit(self, years: np.ndarray, values: np.ndarray) -> None:
        # Centre the years to keep polyfit well-conditioned.
        self._t0 = years[0]
        self.slope_, self.intercept_ = np.polyfit(years - self._t0, values, deg=1)

    def _line(self, years: np.ndarray) -> np.ndarray:
        return self.intercept_ + self.slope_ * (years - self._t0)

    def _fitted(self) -> np.ndarray:
        return self._line(self._years)  # type: ignore[arg-type]

    def _predict(self, horizon: int) -> np.ndarray:
        future = self._years[-1] + np.arange(1, horizon + 1)  # type: ignore[index]
        return self._line(future)


class ExponentialGrowthForecaster(Forecaster):
    """Constant-percentage growth at the training-period CAGR.

    ``P(t) = P_0 · (1 + g)^(t - t_0)`` with ``g`` the CAGR of the training
    window. Because the CAGR is defined by the end points, the fitted curve
    passes exactly through the first and last training values, and forecasts
    continue from the last observation.
    """

    label = "Exponential (CAGR)"

    def _fit(self, years: np.ndarray, values: np.ndarray) -> None:
        if values[0] <= 0 or values[-1] <= 0:
            raise ValueError("Exponential growth needs positive start and end values.")
        periods = years[-1] - years[0]
        self.growth_rate_ = float((values[-1] / values[0]) ** (1.0 / periods) - 1.0)

    def _fitted(self) -> np.ndarray:
        steps = self._years - self._years[0]  # type: ignore[operator]
        return self._values[0] * (1.0 + self.growth_rate_) ** steps  # type: ignore[index]

    def _predict(self, horizon: int) -> np.ndarray:
        steps = np.arange(1, horizon + 1)
        return self._values[-1] * (1.0 + self.growth_rate_) ** steps  # type: ignore[index]


def fibonacci(n: int) -> list[int]:
    """Return the first ``n`` Fibonacci numbers ``[1, 1, 2, 3, 5, ...]``."""
    if n < 0:
        raise ValueError("n must be non-negative.")
    seq: list[int] = []
    a, b = 1, 1
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq


class FibonacciRatioForecaster(Forecaster):
    """The previous cohort's model: scale by successive Fibonacci ratios.

    Year ``k`` after the last observation is forecast as
    ``P_last · r_s · r_{s+1} · ... · r_{s+k-1}`` where
    ``r_i = F_{i+1} / F_i`` and ``s`` is ``start_index`` (1-based, so
    ``r_1 = F_2/F_1 = 1``, ``r_2 = 2``, ``r_3 = 1.5``, ... → φ ≈ 1.618).

    In-sample, the "fitted" value for year ``t`` is the one-step-ahead
    prediction ``P_{t-1} · r_{s+j}``; the first year has no prediction and is
    set to its actual value.

    The model has **no parameter estimated from data** - ``fit`` only stores
    the last value. That is the core of its weakness (see the notebook).

    Parameters
    ----------
    start_index : int, default 1
        Which Fibonacci ratio to start from.
    """

    label = "Fibonacci ratio"
    min_observations = 1

    def __init__(self, start_index: int = 1) -> None:
        super().__init__()
        if start_index < 1:
            raise ValueError("start_index must be >= 1.")
        self.start_index = start_index

    def ratios(self, count: int, offset: int = 0) -> np.ndarray:
        """Return ``count`` consecutive ratios ``F_{i+1}/F_i`` from ``start_index + offset``."""
        first = self.start_index + offset
        fib = fibonacci(first + count)
        # fib is 0-indexed: fib[i-1] = F_i
        return np.array([fib[i] / fib[i - 1] for i in range(first, first + count)], dtype=float)

    def _fit(self, years: np.ndarray, values: np.ndarray) -> None:
        self.last_value_ = float(values[-1])

    def _fitted(self) -> np.ndarray:
        v = self._values
        fitted = np.empty_like(v)  # type: ignore[arg-type]
        fitted[0] = v[0]  # type: ignore[index]
        if v.size > 1:  # type: ignore[union-attr]
            fitted[1:] = v[:-1] * self.ratios(v.size - 1)  # type: ignore[index, union-attr]
        return fitted

    def _predict(self, horizon: int) -> np.ndarray:
        return self.last_value_ * np.cumprod(self.ratios(horizon))


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------
def _check_pair(actual: Iterable[float], predicted: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(list(actual), dtype=float)
    p = np.asarray(list(predicted), dtype=float)
    if a.size == 0:
        raise ValueError("Cannot score an empty series.")
    if a.shape != p.shape:
        raise ValueError("actual and predicted must have the same length.")
    return a, p


def mae(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Mean Absolute Error (same units as the data)."""
    a, p = _check_pair(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Root Mean Squared Error (same units; penalises large misses)."""
    a, p = _check_pair(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Mean Absolute Percentage Error, in percent. Undefined if any actual is 0."""
    a, p = _check_pair(actual, predicted)
    if np.any(a == 0):
        raise ValueError("MAPE is undefined when an actual value is zero.")
    return float(np.mean(np.abs((a - p) / a)) * 100.0)


METRICS: dict[str, Callable[[Iterable[float], Iterable[float]], float]] = {
    "MAE": mae,
    "RMSE": rmse,
    "MAPE (%)": mape,
}


# ---------------------------------------------------------------------------
# Model selection (composition)
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """Hold-out scores for one model on one district."""

    model: str
    forecast: np.ndarray
    scores: dict[str, float]


class ModelSelector:
    """Hold-out validation and model choice for a set of forecasters.

    The selector *has* forecasters (composition) rather than *being* one.
    Each candidate is deep-copied before fitting, so the same prototype list
    can be reused safely across districts.

    Parameters
    ----------
    candidates : Sequence[Forecaster]
        Unfitted prototype forecasters.
    criterion : str, default "RMSE"
        Metric used to pick the winner (lower is better).
    """

    def __init__(self, candidates: Sequence[Forecaster], criterion: str = "RMSE") -> None:
        if not candidates:
            raise ValueError("Provide at least one candidate forecaster.")
        if criterion not in METRICS:
            raise ValueError(f"criterion must be one of {list(METRICS)}.")
        self.candidates = list(candidates)
        self.criterion = criterion

    def validate(self, series: DistrictPopulation, last_train_year: int) -> list[ValidationResult]:
        """Train on years ``<= last_train_year`` and score on the rest."""
        train, test = series.split(last_train_year)
        results: list[ValidationResult] = []
        for proto in self.candidates:
            model = copy.deepcopy(proto).fit(train.years, train.populations)
            forecast = model.predict(len(test))
            scores = {name: fn(test.populations, forecast) for name, fn in METRICS.items()}
            results.append(ValidationResult(model.label, forecast, scores))
        return results

    def best(self, results: Sequence[ValidationResult]) -> ValidationResult:
        """Return the result with the lowest ``criterion`` score."""
        return min(results, key=lambda r: r.scores[self.criterion])

    def prototype(self, label: str) -> Forecaster:
        """Return a fresh copy of the candidate with the given label."""
        for proto in self.candidates:
            if proto.label == label:
                return copy.deepcopy(proto)
        raise KeyError(label)


# ---------------------------------------------------------------------------
# Bootstrap prediction intervals (extension)
# ---------------------------------------------------------------------------
@dataclass
class PredictionInterval:
    """Point forecast with lower/upper bounds and the raw bootstrap paths."""

    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    paths: np.ndarray

    def width(self) -> np.ndarray:
        """Interval width for each horizon step."""
        return self.upper - self.lower


class BootstrapPredictionInterval:
    """Residual bootstrap prediction intervals for any :class:`Forecaster`.

    Algorithm (for ``b = 1..B``):

    1. Resample the in-sample residuals with replacement and add them to the
       fitted values to create a pseudo-history.
    2. Refit a fresh copy of the model on the pseudo-history
       (captures *parameter* uncertainty).
    3. Forecast and add freshly resampled residuals
       (captures *future noise*).

    The ``alpha/2`` and ``1 - alpha/2`` percentiles of the ``B`` paths give
    the interval.

    Parameters
    ----------
    n_resamples : int, default 1000
    alpha : float, default 0.05  (95 % interval)
    seed : int, default 42
    """

    def __init__(self, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 42) -> None:
        if n_resamples < 1:
            raise ValueError("n_resamples must be positive.")
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1).")
        self.n_resamples = n_resamples
        self.alpha = alpha
        self.seed = seed

    def run(self, model: Forecaster, series: DistrictPopulation, horizon: int) -> PredictionInterval:
        """Fit ``model`` on ``series`` and bootstrap ``horizon`` forecasts."""
        rng = np.random.default_rng(self.seed)
        base = copy.deepcopy(model).fit(series.years, series.populations)
        point = base.predict(horizon)
        fitted = base.fitted_values()
        resid = base.residuals()
        # Degrees-of-freedom-free centring keeps the resampled noise mean-zero.
        resid = resid - resid.mean()

        paths = np.empty((self.n_resamples, horizon))
        for b in range(self.n_resamples):
            pseudo = fitted + rng.choice(resid, size=resid.size, replace=True)
            pseudo = np.clip(pseudo, 1e-9, None)  # populations stay positive
            boot_model = copy.deepcopy(model).fit(series.years, pseudo)
            future_noise = rng.choice(resid, size=horizon, replace=True)
            paths[b] = boot_model.predict(horizon) + future_noise

        lo, hi = 100 * self.alpha / 2, 100 * (1 - self.alpha / 2)
        return PredictionInterval(
            point=point,
            lower=np.percentile(paths, lo, axis=0),
            upper=np.percentile(paths, hi, axis=0),
            paths=paths,
        )


# ---------------------------------------------------------------------------
# Planning output
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassroomPlanner:
    """Convert population (thousands) into primary classroom requirements.

    Parameters
    ----------
    school_age_share : float, default 0.18
        Share of the population aged roughly 6-12 (primary school age).
    pupils_per_classroom : int, default 53
        Classroom capacity.
    """

    school_age_share: float = 0.18
    pupils_per_classroom: int = 53

    def __post_init__(self) -> None:
        if not 0 < self.school_age_share <= 1:
            raise ValueError("school_age_share must be in (0, 1].")
        if self.pupils_per_classroom <= 0:
            raise ValueError("pupils_per_classroom must be positive.")

    def pupils(self, population_thousands: float) -> float:
        """Primary-school-age children for a population in thousands."""
        if population_thousands < 0:
            raise ValueError("Population cannot be negative.")
        return population_thousands * 1000.0 * self.school_age_share

    def classrooms(self, population_thousands: float) -> int:
        """Classrooms needed to seat every primary-age child (rounded up)."""
        return math.ceil(self.pupils(population_thousands) / self.pupils_per_classroom)

    def additional_classrooms(self, current_thousands: float, future_thousands: float) -> int:
        """Extra classrooms needed going from ``current`` to ``future`` population.

        Assumes today's stock exactly matches today's need, so only growth
        creates new demand. A shrinking population returns 0 (classrooms are
        not demolished).
        """
        return max(0, self.classrooms(future_thousands) - self.classrooms(current_thousands))
