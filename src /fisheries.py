"""Lake Victoria fish stock, price and revenue-risk model (Mini-Project 3).

Design overview
---------------
* :class:`HarvestPolicy` - the fraction of stock harvested in a given week.
  :class:`ClosedSeasonPolicy` *inherits* from it and returns zero during the
  closed weeks of each year (polymorphism: :class:`FishStock` never needs to
  know which policy it has).
* :class:`FishStock` - discrete logistic growth with proportional harvesting::

      N(t+1) = N(t) + r N(t) (1 - N(t)/K) - h_t N(t)

  It *has a* :class:`HarvestPolicy` (composition) and returns a
  :class:`StockTrajectory`. It also exposes the analytic equilibrium,
  maximum sustainable yield (MSY) and the harvest rate that achieves it.
* :class:`PriceModel` - a seeded, bounded random walk for the weekly price
  (UGX/kg), with reflecting (default) or clipping boundaries.
* :class:`RiskAssessor` - descriptive statistics with :mod:`statistics`,
  coefficient-of-variation risk classes, Value-at-Risk, and a combined
  financial + ecological rating.

Units: stock and harvest in **tonnes**, price in **UGX per kg**, revenue in
**UGX**. The rates ``r`` and ``h`` are **per week** (one simulation step).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "fibonacci_stock",
    "HarvestPolicy",
    "ClosedSeasonPolicy",
    "StockTrajectory",
    "FishStock",
    "PriceModel",
    "weekly_revenue",
    "RiskReport",
    "RiskAssessor",
]

KG_PER_TONNE = 1_000.0


def fibonacci_stock(n: int) -> list[int]:
    """The old baseline: the first ``n`` Fibonacci numbers ``1, 1, 2, 3, ...``."""
    if n < 0:
        raise ValueError("n must be non-negative.")
    seq: list[int] = []
    a, b = 1, 1
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq


# ---------------------------------------------------------------------------
# Harvest policies
# ---------------------------------------------------------------------------
class HarvestPolicy:
    """Constant proportional harvest rate ``h`` (fraction of stock per week).

    Parameters
    ----------
    rate : float
        Harvest fraction in ``[0, 1)``.
    """

    def __init__(self, rate: float) -> None:
        if not 0 <= rate < 1:
            raise ValueError("Harvest rate must be in [0, 1).")
        self.rate = float(rate)

    def rate_at(self, week: int) -> float:
        """Harvest rate applied in ``week`` (0-based)."""
        return self.rate

    def __repr__(self) -> str:
        return f"{type(self).__name__}(rate={self.rate})"


class ClosedSeasonPolicy(HarvestPolicy):
    """Proportional harvest, except for a closed season every year.

    Parameters
    ----------
    rate : float
        Harvest fraction during open weeks.
    closed_weeks : Iterable[int], default weeks 14-21 (8 weeks)
        Week-of-year indices (0-51) with no harvesting.
    weeks_per_year : int, default 52
    """

    def __init__(self, rate: float, closed_weeks: Iterable[int] = range(14, 22),
                 weeks_per_year: int = 52) -> None:
        super().__init__(rate)
        self.closed_weeks = frozenset(int(w) for w in closed_weeks)
        if any(not 0 <= w < weeks_per_year for w in self.closed_weeks):
            raise ValueError("closed_weeks must lie within one year.")
        self.weeks_per_year = weeks_per_year

    def rate_at(self, week: int) -> float:
        return 0.0 if week % self.weeks_per_year in self.closed_weeks else self.rate

    def __repr__(self) -> str:
        return f"ClosedSeasonPolicy(rate={self.rate}, closed={len(self.closed_weeks)} wks/yr)"


# ---------------------------------------------------------------------------
# Stock dynamics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StockTrajectory:
    """Output of :meth:`FishStock.simulate`.

    Attributes
    ----------
    stock : ndarray, shape (weeks + 1,)
        Stock at the start of each week, plus the final stock (tonnes).
    harvest : ndarray, shape (weeks,)
        Tonnes landed in each week.
    """

    stock: np.ndarray
    harvest: np.ndarray

    @property
    def final_stock(self) -> float:
        return float(self.stock[-1])

    @property
    def total_harvest(self) -> float:
        return float(self.harvest.sum())

    def __len__(self) -> int:
        return int(self.harvest.size)


class FishStock:
    """Discrete logistic fish stock with harvesting.

    Parameters
    ----------
    r : float, default 0.4
        Intrinsic growth rate per week (> 0).
    K : float, default 10,000
        Carrying capacity in tonnes (> 0).
    n0 : float, default 4,000
        Initial stock in tonnes (0 <= n0).
    policy : HarvestPolicy or float, default 0.1
        Harvest policy; a bare number is wrapped in :class:`HarvestPolicy`.
    """

    def __init__(self, r: float = 0.4, K: float = 10_000.0, n0: float = 4_000.0,
                 policy: HarvestPolicy | float = 0.1) -> None:
        if r <= 0:
            raise ValueError("r must be positive.")
        if K <= 0:
            raise ValueError("K must be positive.")
        if n0 < 0 or not math.isfinite(n0):
            raise ValueError("n0 must be finite and non-negative.")
        self.r, self.K, self.n0 = float(r), float(K), float(n0)
        self.policy = policy if isinstance(policy, HarvestPolicy) else HarvestPolicy(policy)

    def __repr__(self) -> str:
        return f"FishStock(r={self.r}, K={self.K:,.0f}, n0={self.n0:,.0f}, policy={self.policy!r})"

    # -- analytic benchmarks ---------------------------------------------------
    @property
    def msy(self) -> float:
        """Maximum sustainable yield per week, ``rK/4`` (tonnes)."""
        return self.r * self.K / 4

    @property
    def h_msy(self) -> float:
        """Harvest rate that delivers MSY, ``r/2``."""
        return self.r / 2

    def equilibrium(self, h: float | None = None) -> float:
        """Non-trivial steady state ``K(1 - h/r)`` (0 if ``h >= r``)."""
        h = self.policy.rate if h is None else h
        return max(0.0, self.K * (1 - h / self.r))

    def sustainable_yield(self, h: float | None = None) -> float:
        """Long-run weekly yield ``h * N*`` at a constant rate ``h``."""
        h = self.policy.rate if h is None else h
        return h * self.equilibrium(h)

    # -- simulation ---------------------------------------------------------
    def step(self, n: float, h: float) -> tuple[float, float]:
        """Advance one week. Returns ``(next_stock, harvest)``; stock never goes below 0."""
        harvest = h * n
        nxt = n + self.r * n * (1 - n / self.K) - harvest
        return max(nxt, 0.0), harvest

    def simulate(self, weeks: int = 52) -> StockTrajectory:
        """Simulate ``weeks`` steps from ``n0`` under the stock's policy."""
        if weeks < 1:
            raise ValueError("weeks must be a positive integer.")
        stock = np.empty(weeks + 1)
        harvest = np.empty(weeks)
        stock[0] = self.n0
        for t in range(weeks):
            stock[t + 1], harvest[t] = self.step(stock[t], self.policy.rate_at(t))
        return StockTrajectory(stock, harvest)


# ---------------------------------------------------------------------------
# Prices and revenue
# ---------------------------------------------------------------------------
class PriceModel:
    """Bounded Gaussian random walk for the weekly price (UGX/kg).

    ``P(0) = start``; ``P(t) = P(t-1) + e_t`` with ``e_t ~ Normal(0, step_sd)``,
    then kept inside ``[lower, upper]``.

    Parameters
    ----------
    start, lower, upper : float
        Starting price and bounds (``lower <= start <= upper``).
    step_sd : float, default 400
        Standard deviation of the weekly price change.
    boundary : {"reflect", "clip"}
        ``"reflect"`` mirrors an overshoot back inside the band; ``"clip"``
        pins it to the bound (which piles probability mass on the bounds).
    seed : int
        Seed for :func:`numpy.random.default_rng`.
    """

    def __init__(self, start: float = 12_000.0, lower: float = 9_000.0, upper: float = 16_000.0,
                 step_sd: float = 400.0, boundary: str = "reflect", seed: int = 2026) -> None:
        if not lower < upper:
            raise ValueError("lower must be below upper.")
        if not lower <= start <= upper:
            raise ValueError("start must lie within [lower, upper].")
        if step_sd < 0:
            raise ValueError("step_sd cannot be negative.")
        if boundary not in {"reflect", "clip"}:
            raise ValueError("boundary must be 'reflect' or 'clip'.")
        self.start, self.lower, self.upper = float(start), float(lower), float(upper)
        self.step_sd, self.boundary, self.seed = float(step_sd), boundary, seed
        self._rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        return (f"PriceModel(start={self.start:,.0f}, band=[{self.lower:,.0f}, {self.upper:,.0f}], "
                f"step_sd={self.step_sd:,.0f}, boundary={self.boundary!r}, seed={self.seed})")

    def reset(self) -> None:
        """Restart the random stream from the seed (for exact reproducibility)."""
        self._rng = np.random.default_rng(self.seed)

    def _bound(self, p: np.ndarray) -> np.ndarray:
        if self.boundary == "clip":
            return np.clip(p, self.lower, self.upper)
        # reflect repeatedly in case of a very large step
        width = self.upper - self.lower
        q = np.mod(p - self.lower, 2 * width)
        return self.lower + np.where(q > width, 2 * width - q, q)

    def simulate(self, weeks: int = 52, n_paths: int = 1) -> np.ndarray:
        """Return price paths of shape ``(n_paths, weeks)``; column 0 is the start price."""
        if weeks < 1 or n_paths < 1:
            raise ValueError("weeks and n_paths must be positive.")
        prices = np.empty((n_paths, weeks))
        prices[:, 0] = self.start
        shocks = self._rng.normal(0.0, self.step_sd, size=(n_paths, weeks - 1))
        for t in range(1, weeks):
            prices[:, t] = self._bound(prices[:, t - 1] + shocks[:, t - 1])
        return prices


def weekly_revenue(harvest_tonnes: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Revenue (UGX) = harvest (t) x 1,000 kg/t x price (UGX/kg).

    Broadcasts a ``(weeks,)`` harvest against ``(weeks,)`` or ``(n_paths, weeks)`` prices.
    """
    h = np.asarray(harvest_tonnes, dtype=float)
    p = np.asarray(prices, dtype=float)
    if h.shape[-1] != p.shape[-1]:
        raise ValueError("harvest and prices must cover the same number of weeks.")
    if np.any(h < 0) or np.any(p < 0):
        raise ValueError("harvest and prices must be non-negative.")
    return h * KG_PER_TONNE * p


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskReport:
    """Monte Carlo risk summary for annual revenue."""

    mean: float
    cv: float
    var_level: float      # 5th-percentile annual revenue (UGX)
    value_at_risk: float  # mean - var_level: shortfall not exceeded with 95% confidence
    financial_class: str
    stock_status: str
    overall_class: str


class RiskAssessor:
    """Classify revenue risk by coefficient of variation and compute VaR.

    Rule (see notebook for justification). With ``CV = std / mean``:

    * ``CV < low``  -> ``"Low"``
    * ``low <= CV < high`` -> ``"Moderate"``
    * ``CV >= high`` -> ``"High"``

    Defaults ``low = 0.10`` and ``high = 0.20``. If revenue were roughly
    normal, the 5 % worst year would fall about ``1.645 * CV`` below the
    mean, i.e. 16 % at the low threshold and 33 % at the high threshold.

    The *overall* class also considers the fish stock: a harvest rate above
    the MSY rate ``r/2`` (stock driven below ``K/2``) raises the class to at
    least ``"Moderate"``, and a collapsing stock (``h >= r``) to ``"High"``,
    whatever the price risk says.

    Parameters
    ----------
    low, high : float
        CV thresholds, ``0 < low < high``.
    alpha : float, default 0.05
        Tail probability for Value-at-Risk.
    """

    CLASSES = ("Low", "Moderate", "High")

    def __init__(self, low: float = 0.10, high: float = 0.20, alpha: float = 0.05) -> None:
        if not 0 < low < high:
            raise ValueError("Need 0 < low < high.")
        if not 0 < alpha < 0.5:
            raise ValueError("alpha must be in (0, 0.5).")
        self.low, self.high, self.alpha = low, high, alpha

    def __repr__(self) -> str:
        return f"RiskAssessor(low={self.low}, high={self.high}, alpha={self.alpha})"

    @staticmethod
    def describe(values: Sequence[float]) -> dict[str, float]:
        """Mean, median, sample variance, sample std and CV using :mod:`statistics`."""
        data = [float(v) for v in values]
        if len(data) < 2:
            raise ValueError("Need at least two values.")
        mean = statistics.mean(data)
        sd = statistics.stdev(data)
        return {
            "mean": mean,
            "median": statistics.median(data),
            "variance": statistics.variance(data),
            "std": sd,
            "cv": sd / mean if mean != 0 else math.inf,
        }

    def classify(self, cv: float) -> str:
        """Map a coefficient of variation to Low / Moderate / High."""
        if cv < 0 or math.isnan(cv):
            raise ValueError("CV must be a non-negative number.")
        if cv < self.low:
            return "Low"
        return "Moderate" if cv < self.high else "High"

    def value_at_risk(self, annual_revenues: Sequence[float]) -> tuple[float, float]:
        """Return ``(var_level, value_at_risk)``.

        ``var_level`` is the ``alpha``-quantile of annual revenue;
        ``value_at_risk = mean - var_level`` is the shortfall below the
        expected revenue that is exceeded only with probability ``alpha``.
        """
        a = np.asarray(annual_revenues, dtype=float)
        if a.size == 0:
            raise ValueError("No simulated revenues supplied.")
        level = float(np.percentile(a, 100 * self.alpha))
        return level, float(a.mean() - level)

    @staticmethod
    def stock_status(stock: FishStock) -> str:
        """Ecological status from the harvest rate relative to r/2 and r."""
        h = stock.policy.rate
        if h >= stock.r:
            return "Collapsing"
        return "Overfished" if h > stock.h_msy + 1e-12 else "Sustainable"

    def assess(self, annual_revenues: Sequence[float], stock: FishStock) -> RiskReport:
        """Combine financial (CV, VaR) and ecological risk into one report."""
        a = np.asarray(annual_revenues, dtype=float)
        stats = self.describe(a)
        level, var = self.value_at_risk(a)
        fin = self.classify(stats["cv"])
        status = self.stock_status(stock)
        floor = {"Sustainable": "Low", "Overfished": "Moderate", "Collapsing": "High"}[status]
        overall = max(fin, floor, key=self.CLASSES.index)
        return RiskReport(stats["mean"], stats["cv"], level, var, fin, status, overall)
