"""Rainfall regimes and crop suitability (Mini-Project 4).

Design overview
---------------
* :class:`Region` - twelve monthly rainfall totals (mm) for one place, with
  summary statistics, wettest/driest month, CV and automatic rainy-season
  detection (circular :func:`scipy.signal.find_peaks`).
* :class:`CropRule` - a crop's suitable monthly rainfall range. It classifies
  a month as ``"Good"``, ``"Drought risk"`` or ``"Waterlogging risk"``; the
  class method :meth:`CropRule.from_seasonal_need` derives the monthly range
  from a published seasonal water need and growing-period length.
* :class:`SuitabilityAnalyser` - applies a set of crop rules to a set of
  regions (composition) and returns tables and a numeric grid for heatmaps.
* :class:`RainfallRecord` - many years of monthly rainfall for one place
  (e.g. NASA POWER). It produces a climatological :class:`Region` and
  year-to-year variability statistics.
* Similarity helpers - a *correct* cosine similarity, Pearson correlation,
  Euclidean distance and a pairwise-matrix builder.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.signal import find_peaks

__all__ = [
    "MONTHS",
    "Region",
    "CropRule",
    "SuitabilityAnalyser",
    "RainfallRecord",
    "cosine_similarity",
    "flawed_cosine_similarity",
    "pearson_correlation",
    "euclidean_distance",
    "pairwise_matrix",
    "default_crop_rules",
]

MONTHS: tuple[str, ...] = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _validate_monthly(values: Sequence[float], what: str = "rainfall") -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (12,):
        raise ValueError(f"{what} must have exactly 12 monthly values, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what} contains missing or non-numeric values.")
    if np.any(arr < 0):
        raise ValueError(f"{what} cannot be negative.")
    return arr


# ---------------------------------------------------------------------------
# Region
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Season:
    """A detected rainy-season peak."""

    month: str
    rainfall: float
    prominence: float           # mm
    relative_prominence: float  # prominence / (max - min)


class Region:
    """Monthly rainfall regime for one place.

    Parameters
    ----------
    name : str
        Region name.
    rainfall : Sequence[float]
        Twelve monthly totals in mm, January to December.

    Raises
    ------
    ValueError
        If the name is blank, or the rainfall is not 12 finite, non-negative numbers.
    """

    #: Minimum relative prominence for a peak to count as a separate rainy season.
    SEASON_THRESHOLD: float = 0.30

    def __init__(self, name: str, rainfall: Sequence[float]) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Region name must be a non-empty string.")
        self.name = name.strip()
        self._rain = _validate_monthly(rainfall)
        self._rain.setflags(write=False)

    # -- dunder methods ----------------------------------------------------------
    def __repr__(self) -> str:
        return f"Region({self.name!r}, annual={self.annual_total():,.0f} mm, {self.modality()})"

    def __len__(self) -> int:
        return 12

    def __getitem__(self, month: int | str) -> float:
        """Rainfall for a month given as index 0-11 or name ('Jan')."""
        idx = MONTHS.index(month[:3].title()) if isinstance(month, str) else int(month)
        return float(self._rain[idx])

    def __iter__(self):
        return iter(zip(MONTHS, self._rain.tolist()))

    # -- statistics --------------------------------------------------------------
    @property
    def rainfall(self) -> np.ndarray:
        """Read-only monthly rainfall (mm)."""
        return self._rain

    def annual_total(self) -> float:
        """Sum of the 12 months (mm)."""
        return float(self._rain.sum())

    def mean(self) -> float:
        """Mean monthly rainfall (mm)."""
        return float(self._rain.mean())

    def wettest_month(self) -> str:
        """Name of the month with the most rain (first one if tied)."""
        return MONTHS[int(np.argmax(self._rain))]

    def driest_month(self) -> str:
        """Name of the month with the least rain (first one if tied)."""
        return MONTHS[int(np.argmin(self._rain))]

    def cv(self) -> float:
        """Coefficient of variation across months, ``s / mean`` (sample std, ddof=1).

        Measures how *seasonal* the rainfall is: 0 means equal rain every month.
        Returns ``nan`` for a region with no rain at all.
        """
        m = self._rain.mean()
        return float(self._rain.std(ddof=1) / m) if m > 0 else float("nan")

    # -- seasons -----------------------------------------------------------------
    def rainy_seasons(self, min_relative_prominence: float | None = None) -> list[Season]:
        """Detect rainy-season peaks with :func:`scipy.signal.find_peaks`.

        The year is circular (December is next to January), so the series is
        tiled three times and only peaks in the middle copy are kept. This
        lets a December or January peak be found, with its prominence
        measured correctly across the year boundary.

        A peak counts as a season if its prominence is at least
        ``min_relative_prominence`` x (annual max - annual min).
        """
        thr = self.SEASON_THRESHOLD if min_relative_prominence is None else min_relative_prominence
        if not 0 <= thr <= 1:
            raise ValueError("min_relative_prominence must be in [0, 1].")
        x = self._rain
        span = float(x.max() - x.min())
        if span == 0:
            return []  # perfectly flat: no seasons
        tiled = np.tile(x, 3)
        peaks, props = find_peaks(tiled, prominence=thr * span)
        seasons = [
            Season(MONTHS[p - 12], float(tiled[p]), float(prom), float(prom / span))
            for p, prom in zip(peaks, props["prominences"])
            if 12 <= p < 24
        ]
        return sorted(seasons, key=lambda s: MONTHS.index(s.month))

    def modality(self, min_relative_prominence: float | None = None) -> str:
        """``"unimodal"``, ``"bimodal"``, ``"multimodal"`` or ``"no clear season"``."""
        n = len(self.rainy_seasons(min_relative_prominence))
        return {0: "no clear season", 1: "unimodal", 2: "bimodal"}.get(n, "multimodal")


# ---------------------------------------------------------------------------
# Crop rules
# ---------------------------------------------------------------------------
class CropRule:
    """Suitable monthly rainfall range for a crop.

    Parameters
    ----------
    crop : str
    min_mm, max_mm : float
        Suitable monthly rainfall range, ``0 <= min_mm < max_mm``.
        Below ``min_mm`` -> drought risk; above ``max_mm`` -> waterlogging risk.
    source : str
        Citation for the thresholds.
    """

    GOOD, DROUGHT, WATERLOG = "Good", "Drought risk", "Waterlogging risk"

    def __init__(self, crop: str, min_mm: float, max_mm: float, source: str = "") -> None:
        if not crop or not crop.strip():
            raise ValueError("Crop name must be non-empty.")
        if not (0 <= min_mm < max_mm) or not math.isfinite(max_mm):
            raise ValueError("Need 0 <= min_mm < max_mm (finite).")
        self.crop = crop.strip()
        self.min_mm, self.max_mm, self.source = float(min_mm), float(max_mm), source

    def __repr__(self) -> str:
        return f"CropRule({self.crop!r}, {self.min_mm:.0f}-{self.max_mm:.0f} mm/month)"

    @classmethod
    def from_seasonal_need(cls, crop: str, need_mm: tuple[float, float],
                           season_days: tuple[float, float], source: str = "") -> "CropRule":
        """Build a monthly range from a seasonal water need and season length.

        The lower bound spreads the *smallest* need over the *longest* season;
        the upper bound spreads the *largest* need over the *shortest* season
        (1 month = 30 days). Values are rounded to the nearest 5 mm.
        """
        (need_lo, need_hi), (d_lo, d_hi) = need_mm, season_days
        if min(need_lo, d_lo) <= 0 or need_lo > need_hi or d_lo > d_hi:
            raise ValueError("Invalid seasonal need or season length.")
        lo = 5 * round(need_lo / (d_hi / 30) / 5)
        hi = 5 * round(need_hi / (d_lo / 30) / 5)
        return cls(crop, lo, hi, source)

    def classify(self, rainfall_mm: float) -> str:
        """Classify one month's rainfall."""
        if rainfall_mm < 0 or not math.isfinite(rainfall_mm):
            raise ValueError("Rainfall must be a finite, non-negative number.")
        if rainfall_mm < self.min_mm:
            return self.DROUGHT
        if rainfall_mm > self.max_mm:
            return self.WATERLOG
        return self.GOOD

    def score(self, rainfall_mm: float) -> int:
        """Numeric code for heatmaps: -1 drought, 0 good, +1 waterlogging."""
        return {self.DROUGHT: -1, self.GOOD: 0, self.WATERLOG: 1}[self.classify(rainfall_mm)]

    def good_months(self, region: Region) -> list[str]:
        """Months in which ``region`` falls inside the suitable range."""
        return [m for m, r in region if self.classify(r) == self.GOOD]


class SuitabilityAnalyser:
    """Apply several :class:`CropRule` objects to several :class:`Region` objects."""

    def __init__(self, regions: Iterable[Region], rules: Iterable[CropRule]) -> None:
        self.regions = list(regions)
        self.rules = list(rules)
        if not self.regions or not self.rules:
            raise ValueError("Need at least one region and one crop rule.")

    def month_label(self, region: Region, month: int) -> str:
        """Human-readable advice for one region-month, e.g. ``"Good for maize, beans"``."""
        rain = region.rainfall[month]
        good = [r.crop for r in self.rules if r.classify(rain) == CropRule.GOOD]
        if good:
            return "Good for " + ", ".join(good)
        if all(r.classify(rain) == CropRule.DROUGHT for r in self.rules):
            return CropRule.DROUGHT
        if all(r.classify(rain) == CropRule.WATERLOG for r in self.rules):
            return CropRule.WATERLOG
        return "Mixed: no crop in range"

    def labels(self) -> dict[str, list[str]]:
        """``{region: [label for Jan..Dec]}``."""
        return {reg.name: [self.month_label(reg, m) for m in range(12)] for reg in self.regions}

    def detail(self) -> dict[tuple[str, str], list[str]]:
        """``{(region, crop): [class for Jan..Dec]}``."""
        return {(reg.name, rule.crop): [rule.classify(v) for v in reg.rainfall]
                for reg in self.regions for rule in self.rules}

    def score_grid(self) -> tuple[np.ndarray, list[str]]:
        """Scores of shape ``(n_regions * n_crops, 12)`` and matching row labels."""
        rows, labels = [], []
        for reg in self.regions:
            for rule in self.rules:
                rows.append([rule.score(v) for v in reg.rainfall])
                labels.append(f"{reg.name} - {rule.crop}")
        return np.array(rows), labels


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
def _pair(a: Sequence[float], b: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size == 0:
        raise ValueError("Vectors must be non-empty, 1-D and of equal length.")
    return x, y


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of the angle between two vectors: ``a.b / (|a| |b|)``.

    Raises
    ------
    ValueError
        If either vector is all zeros (the angle is undefined).
    """
    x, y = _pair(a, b)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return float(np.dot(x, y) / (nx * ny))


def flawed_cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Last year's method, kept only to demonstrate the error.

    ``math.cos`` takes an *angle in radians*; feeding it a number derived from
    the data returns a meaningless value that oscillates between -1 and 1.
    """
    x, y = _pair(a, b)
    return math.cos(float(np.dot(x, y)))


def pearson_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson r = cosine similarity of the *mean-centred* vectors."""
    x, y = _pair(a, b)
    return cosine_similarity(x - x.mean(), y - y.mean())


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Straight-line distance ``|a - b|`` (mm)."""
    x, y = _pair(a, b)
    return float(np.linalg.norm(x - y))


def pairwise_matrix(regions: Sequence[Region],
                    metric: Callable[[Sequence[float], Sequence[float]], float]) -> np.ndarray:
    """Symmetric matrix ``M[i, j] = metric(region_i, region_j)``."""
    n = len(regions)
    out = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            out[i, j] = out[j, i] = metric(regions[i].rainfall, regions[j].rainfall)
    return out


# ---------------------------------------------------------------------------
# Multi-year record (extension)
# ---------------------------------------------------------------------------
class RainfallRecord:
    """Several years of monthly rainfall for one place.

    Parameters
    ----------
    name : str
    years : Sequence[int]
    monthly : array_like, shape (n_years, 12)
        Monthly totals in mm.
    """

    def __init__(self, name: str, years: Sequence[int], monthly: np.ndarray) -> None:
        data = np.asarray(monthly, dtype=float)
        if data.ndim != 2 or data.shape[1] != 12 or data.shape[0] != len(years) or data.shape[0] == 0:
            raise ValueError("monthly must have shape (n_years, 12) matching years.")
        if not np.all(np.isfinite(data)) or np.any(data < 0):
            raise ValueError("monthly contains missing or negative values.")
        self.name, self.years, self.monthly = name, list(years), data

    def __repr__(self) -> str:
        return f"RainfallRecord({self.name!r}, {self.years[0]}-{self.years[-1]}, n_years={len(self)})"

    def __len__(self) -> int:
        return len(self.years)

    def climatology(self) -> Region:
        """Long-term mean for each month, as a :class:`Region`."""
        return Region(self.name, self.monthly.mean(axis=0))

    def annual_totals(self) -> np.ndarray:
        """One total per year (mm)."""
        return self.monthly.sum(axis=1)

    def monthly_cv(self) -> np.ndarray:
        """Year-to-year CV for each calendar month (ddof=1)."""
        mean = self.monthly.mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.monthly.std(axis=0, ddof=1) / mean

    def probability_at_least(self, threshold_mm: float) -> np.ndarray:
        """Share of years in which each month reached ``threshold_mm``."""
        return (self.monthly >= threshold_mm).mean(axis=0)

    @classmethod
    def from_csv(cls, path: str | Path) -> dict[str, "RainfallRecord"]:
        """Read ``region, year, month, rainfall_mm`` rows into one record per region."""
        path = Path(path)
        store: dict[str, dict[int, np.ndarray]] = {}
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            need = {"region", "year", "month", "rainfall_mm"}
            if not need <= set(reader.fieldnames or []):
                raise ValueError(f"CSV must have columns {sorted(need)}.")
            for row in reader:
                year, month = int(row["year"]), int(row["month"])
                if not 1 <= month <= 12:
                    raise ValueError(f"Invalid month {month}.")
                value = float(row["rainfall_mm"])
                if value < 0:  # NASA POWER uses -999 for missing values
                    raise ValueError(f"Missing/negative value for {row['region']} {year}-{month:02d}.")
                store.setdefault(row["region"], {}).setdefault(year, np.full(12, np.nan))[month - 1] = value
        records = {}
        for name, by_year in store.items():
            years = sorted(by_year)
            records[name] = cls(name, years, np.vstack([by_year[y] for y in years]))
        return records


def default_crop_rules() -> list[CropRule]:
    """The three rules used in the notebook, with their sources.

    * Maize (grain) and dry beans: FAO (1986) *Irrigation Water Management:
      Irrigation Water Needs*, Training Manual No. 3, ch. 2, Table 4 (growing
      period) and Table 5 (seasonal water need): maize 500-800 mm over
      125-180 days; beans 300-500 mm over 95-110 days.
    * Robusta coffee: UCDA (2019) *Robusta Coffee Handbook*: 1,200-1,800 mm
      "well distributed over a period of 9 months", and about 25 mm every
      14 days to stimulate flowering (about 55 mm/month).
    """
    fao = "FAO (1986) Irrigation Water Needs, Training Manual 3, Tables 4-5"
    return [
        CropRule.from_seasonal_need("maize", (500, 800), (125, 180), fao),
        CropRule.from_seasonal_need("beans", (300, 500), (95, 110), fao),
        CropRule("coffee", 55, 200, "UCDA (2019) Robusta Coffee Handbook"),
    ]
