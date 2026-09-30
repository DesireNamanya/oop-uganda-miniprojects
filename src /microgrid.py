"""Solar micro-grid dispatch planning (Mini-Project 2).

A rural health centre in Kasese draws energy from solar panels (``x``) and
batteries (``y``) so that two demand constraints hold each day::

    3x + 2y = D1   (daytime load, kWh)
    4x +  y = D2   (critical-equipment load, kWh)

Design overview
---------------
* :class:`DemandSeries` - a validated set of daily demands (dates plus one
  array per constraint). It can be generated synthetically, written to CSV
  and read back from CSV.
* :func:`parse_demand` / :func:`prompt_demand` - robust interactive input.
  The input function is *injected*, so the re-prompt loop can be tested and
  demonstrated without a keyboard.
* :class:`MicroGrid` - holds an ``n x n`` coefficient matrix, the source
  names and unit costs. It checks well-posedness (determinant, condition
  number, rank), solves one day, solves many days (loop or vectorised),
  repairs infeasible days, costs a plan and runs a Monte Carlo sensitivity
  analysis.
* :class:`DispatchPlan` - the result of a month's dispatch: raw solution,
  adjusted (physically feasible) solution, feasibility flags and unmet
  demand.
* :class:`HybridMicroGrid` - a :class:`MicroGrid` subclass with a diesel
  generator and a third constraint. It adds rank-based diagnosis of
  linearly dependent constraints and a least-squares fallback.
"""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy import linalg
from scipy.optimize import linprog, nnls

__all__ = [
    "SingularSystemError",
    "parse_demand",
    "prompt_demand",
    "DemandSeries",
    "DispatchPlan",
    "SensitivityResult",
    "MicroGrid",
    "HybridMicroGrid",
]

#: Coefficient matrix from the brief.
KASESE_COEFFICIENTS: np.ndarray = np.array([[3.0, 2.0], [4.0, 1.0]])
#: Illustrative unit costs in UGX per kWh.
DEFAULT_COSTS: dict[str, float] = {"solar": 150.0, "battery": 450.0}


class SingularSystemError(ValueError):
    """Raised when the coefficient matrix has no unique solution."""


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
def parse_demand(text: str) -> float:
    """Convert user text to a non-negative, finite demand in kWh.

    Parameters
    ----------
    text : str
        Raw text as typed by the user.

    Returns
    -------
    float
        The parsed demand.

    Raises
    ------
    ValueError
        If the text is empty, not a number, NaN/inf, or negative.
    """
    if text is None or not str(text).strip():
        raise ValueError("Empty input - please type a number.")
    try:
        value = float(str(text).strip().replace(",", ""))
    except ValueError:
        raise ValueError(f"{text!r} is not a number.") from None
    if not np.isfinite(value):
        raise ValueError("Demand must be a finite number.")
    if value < 0:
        raise ValueError("Demand cannot be negative.")
    return value


def prompt_demand(
    prompt: str,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    max_attempts: int | None = None,
) -> float:
    """Ask for a demand value until a valid one is entered.

    Parameters
    ----------
    prompt : str
        Text shown to the user.
    input_fn : callable, default :func:`input`
        Source of the user's text. Injected so the loop can be tested and
        demonstrated with scripted answers.
    output_fn : callable, default :func:`print`
        Where error messages go.
    max_attempts : int or None
        Give up after this many invalid attempts (``None`` = never).

    Raises
    ------
    RuntimeError
        If ``max_attempts`` invalid values are entered.
    """
    attempts = 0
    while True:
        raw = input_fn(prompt)
        try:
            return parse_demand(raw)
        except ValueError as err:
            attempts += 1
            output_fn(f"  Invalid input: {err} Try again.")
            if max_attempts is not None and attempts >= max_attempts:
                raise RuntimeError(f"No valid value after {attempts} attempts.") from err


# ---------------------------------------------------------------------------
# Demand data
# ---------------------------------------------------------------------------
class DemandSeries:
    """Daily demands for each constraint of a micro-grid.

    Parameters
    ----------
    dates : Sequence[str | datetime.date]
        One date per day.
    demands : Mapping[str, Sequence[float]]
        Column name -> daily values (kWh), e.g. ``{"d1_kwh": [...], "d2_kwh": [...]}``.
        The column order defines the right-hand-side row order.

    Raises
    ------
    ValueError
        If there are no days, columns of unequal length, or negative/NaN values.
    """

    def __init__(self, dates: Sequence[str | dt.date], demands: Mapping[str, Sequence[float]]) -> None:
        if len(dates) == 0:
            raise ValueError("A demand series needs at least one day.")
        if not demands:
            raise ValueError("Provide at least one demand column.")
        self.dates = [dt.date.fromisoformat(str(d)) for d in dates]
        self.columns = list(demands)
        arrays = [np.asarray(v, dtype=float) for v in demands.values()]
        for name, arr in zip(self.columns, arrays):
            if arr.shape != (len(self.dates),):
                raise ValueError(f"Column {name!r} has {arr.size} values for {len(self.dates)} dates.")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Column {name!r} contains missing or non-numeric values.")
            if np.any(arr < 0):
                raise ValueError(f"Column {name!r} contains negative demand.")
        self._rhs = np.vstack(arrays)
        self._rhs.setflags(write=False)

    def __len__(self) -> int:
        return len(self.dates)

    def __repr__(self) -> str:
        return (f"DemandSeries({self.dates[0]} to {self.dates[-1]}, days={len(self)}, "
                f"columns={self.columns})")

    def __getitem__(self, column: str) -> np.ndarray:
        return self._rhs[self.columns.index(column)]

    @property
    def rhs(self) -> np.ndarray:
        """Right-hand side, shape ``(n_constraints, n_days)``."""
        return self._rhs

    @property
    def weekdays(self) -> list[str]:
        """Three-letter weekday name for each date."""
        return [d.strftime("%a") for d in self.dates]

    # -- I/O ---------------------------------------------------------------
    def to_csv(self, path: str | Path) -> Path:
        """Write ``date, weekday, <columns...>`` to ``path`` and return it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["date", "weekday", *self.columns])
            for i, day in enumerate(self.dates):
                writer.writerow([day.isoformat(), day.strftime("%a"),
                                 *[f"{self._rhs[j, i]:.2f}" for j in range(len(self.columns))]])
        return path

    @classmethod
    def from_csv(cls, path: str | Path, columns: Sequence[str] = ("d1_kwh", "d2_kwh")) -> "DemandSeries":
        """Load and validate a demand CSV.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If a required column is missing or any value is empty,
            non-numeric or negative (the row number is reported).
        """
        path = Path(path)
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            missing = {"date", *columns} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV is missing column(s): {sorted(missing)}")
            dates: list[str] = []
            values: dict[str, list[float]] = {c: [] for c in columns}
            for line_no, row in enumerate(reader, start=2):
                dates.append(row["date"])
                for c in columns:
                    try:
                        values[c].append(parse_demand(row[c]))
                    except ValueError as err:
                        raise ValueError(f"Line {line_no}, column {c!r}: {err}") from None
        return cls(dates, values)

    @classmethod
    def synthetic(
        cls,
        n_days: int = 30,
        start: dt.date = dt.date(2026, 9, 1),
        seed: int = 2026,
        base: tuple[float, float] = (95.0, 100.0),
        weekday_factor: Mapping[int, float] | None = None,
        noise_sd: tuple[float, float] = (0.04, 0.03),
    ) -> "DemandSeries":
        """Generate a seeded month of demand with a weekly pattern and noise.

        Model (for day *t*):
        ``D1_t = base1 * weekday_factor[weekday] * (1 + e1_t)`` - daytime load
        follows clinic activity (lower on Saturday, lowest on Sunday);
        ``D2_t = base2 * (1 + e2_t)`` - critical equipment (vaccine fridges,
        oxygen concentrators, theatre lights) runs every day.
        ``e ~ Normal(0, noise_sd)``.
        """
        if n_days < 1:
            raise ValueError("n_days must be positive.")
        rng = np.random.default_rng(seed)
        factor = weekday_factor or {0: 1.05, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.02, 5: 0.85, 6: 0.72}
        dates = [start + dt.timedelta(days=i) for i in range(n_days)]
        weekly = np.array([factor[d.weekday()] for d in dates])
        d1 = base[0] * weekly * (1 + rng.normal(0, noise_sd[0], n_days))
        d2 = base[1] * (1 + rng.normal(0, noise_sd[1], n_days))
        return cls(dates, {"d1_kwh": np.round(d1, 2), "d2_kwh": np.round(d2, 2)})


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class DispatchPlan:
    """Dispatch for a set of days.

    Attributes
    ----------
    sources : list[str]
        Source names, one per row of the solution arrays.
    raw : ndarray, shape (n_sources, n_days)
        Exact algebraic solution (may contain negatives).
    adjusted : ndarray, shape (n_sources, n_days)
        Physically feasible dispatch after applying ``strategy``.
    residual : ndarray, shape (n_constraints, n_days)
        ``demand - A @ adjusted``. Positive = unmet demand, negative = oversupply.
    strategy : str
        ``"none"``, ``"clip"``, ``"nnls"`` or ``"cover"``.
    """

    sources: list[str]
    raw: np.ndarray
    adjusted: np.ndarray
    residual: np.ndarray
    strategy: str
    tolerance: float = 1e-9

    @property
    def feasible(self) -> np.ndarray:
        """Boolean mask, True where the raw solution has no negative source."""
        return np.all(self.raw >= -self.tolerance, axis=0)

    @property
    def infeasible_days(self) -> np.ndarray:
        """Indices of days whose raw solution is infeasible."""
        return np.flatnonzero(~self.feasible)

    def __len__(self) -> int:
        return self.raw.shape[1]

    def __getitem__(self, source: str) -> np.ndarray:
        """Adjusted dispatch for one source, e.g. ``plan["solar"]``."""
        return self.adjusted[self.sources.index(source)]


@dataclass
class SensitivityResult:
    """Monte Carlo sensitivity of the solution to demand perturbations."""

    base_demand: np.ndarray
    base_solution: np.ndarray
    demands: np.ndarray    # (n_draws, n_constraints)
    solutions: np.ndarray  # (n_draws, n_sources)
    condition_number: float
    rel_input_change: np.ndarray = field(init=False)
    rel_output_change: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.rel_input_change = (np.linalg.norm(self.demands - self.base_demand, axis=1)
                                 / np.linalg.norm(self.base_demand))
        self.rel_output_change = (np.linalg.norm(self.solutions - self.base_solution, axis=1)
                                  / np.linalg.norm(self.base_solution))

    @property
    def amplification(self) -> np.ndarray:
        """Relative output change divided by relative input change, per draw."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.rel_output_change / self.rel_input_change


# ---------------------------------------------------------------------------
# Micro-grids
# ---------------------------------------------------------------------------
class MicroGrid:
    """A square linear dispatch model ``A @ s = d``.

    Parameters
    ----------
    coefficients : array_like, shape (n, n), optional
        Constraint coefficients; rows are constraints, columns are sources.
        Defaults to the Kasese matrix ``[[3, 2], [4, 1]]``.
    unit_costs : Mapping[str, float], optional
        UGX per kWh for each source, in column order. Its keys name the sources.
        Defaults to solar 150, battery 450.

    Raises
    ------
    ValueError
        If the matrix is not square, contains non-finite values, or does not
        match the number of sources, or if a cost is negative.
    """

    def __init__(self, coefficients: Sequence[Sequence[float]] | np.ndarray | None = None,
                 unit_costs: Mapping[str, float] | None = None) -> None:
        a = np.array(KASESE_COEFFICIENTS if coefficients is None else coefficients, dtype=float)
        costs = dict(DEFAULT_COSTS if unit_costs is None else unit_costs)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError(f"Coefficient matrix must be square, got shape {a.shape}.")
        if not np.all(np.isfinite(a)):
            raise ValueError("Coefficient matrix must be finite.")
        if len(costs) != a.shape[1]:
            raise ValueError(f"{len(costs)} unit costs for {a.shape[1]} sources.")
        if any(c < 0 for c in costs.values()):
            raise ValueError("Unit costs cannot be negative.")
        self._a = a
        self._a.setflags(write=False)
        self.sources = list(costs)
        self.unit_costs = np.array(list(costs.values()), dtype=float)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(sources={self.sources}, det={self.determinant:.3g}, "
                f"cond={self.condition_number:.3g})")

    # -- well-posedness ----------------------------------------------------
    @property
    def coefficients(self) -> np.ndarray:
        """Read-only coefficient matrix."""
        return self._a

    @property
    def n(self) -> int:
        """Number of sources (= number of constraints)."""
        return self._a.shape[0]

    @property
    def determinant(self) -> float:
        """Determinant of the coefficient matrix."""
        return float(np.linalg.det(self._a))

    @property
    def condition_number(self) -> float:
        """2-norm condition number ``sigma_max / sigma_min`` (inf if singular)."""
        return float(np.linalg.cond(self._a))

    @property
    def rank(self) -> int:
        """Numerical rank of the coefficient matrix."""
        return int(np.linalg.matrix_rank(self._a))

    def is_well_posed(self, max_cond: float = 1e12) -> bool:
        """True if the matrix is full rank and not badly conditioned."""
        return self.rank == self.n and self.condition_number < max_cond

    def well_posedness(self) -> dict[str, float | bool]:
        """Determinant, condition number, rank and verdict in one dict."""
        return {"determinant": self.determinant, "condition_number": self.condition_number,
                "rank": self.rank, "well_posed": self.is_well_posed()}

    def _require_well_posed(self) -> None:
        if not self.is_well_posed():
            raise SingularSystemError(
                f"Matrix rank {self.rank} < {self.n} (det={self.determinant:.3g}): "
                "the constraints are linearly dependent, so there is no unique dispatch.")

    def _check_rhs(self, rhs: np.ndarray) -> np.ndarray:
        b = np.asarray(rhs, dtype=float)
        if b.shape[0] != self.n:
            raise ValueError(f"Expected {self.n} demand values per day, got {b.shape[0]}.")
        if b.size == 0:
            raise ValueError("No demand data supplied.")
        if not np.all(np.isfinite(b)) or np.any(b < 0):
            raise ValueError("Demands must be finite and non-negative.")
        return b

    # -- solving -----------------------------------------------------------
    def solve_day(self, *demands: float) -> np.ndarray:
        """Solve one day exactly, e.g. ``grid.solve_day(d1, d2)``.

        Returns the raw solution (one value per source, may be negative).
        """
        b = self._check_rhs(np.array(demands, dtype=float))
        self._require_well_posed()
        return linalg.solve(self._a, b)

    def solve_loop(self, rhs: np.ndarray) -> np.ndarray:
        """Solve each column of ``rhs`` separately in a Python loop."""
        b = self._check_rhs(rhs)
        self._require_well_posed()
        out = np.empty_like(b)
        for j in range(b.shape[1]):
            out[:, j] = linalg.solve(self._a, b[:, j])
        return out

    def solve_vectorised(self, rhs: np.ndarray) -> np.ndarray:
        """Solve every column of ``rhs`` with one call (one LU factorisation)."""
        b = self._check_rhs(rhs)
        self._require_well_posed()
        return linalg.solve(self._a, b)

    def dispatch(self, rhs: np.ndarray, strategy: str = "cover") -> DispatchPlan:
        """Solve all days and repair infeasible (negative) ones.

        Parameters
        ----------
        rhs : ndarray, shape (n, n_days)
        strategy : {"cover", "nnls", "clip", "none"}
            Applied to infeasible days only; feasible days keep the exact solution.

            * ``"none"`` - keep the raw solution (for reporting only).
            * ``"clip"`` - set negative sources to zero and keep the others.
            * ``"nnls"`` - non-negative least squares: the non-negative
              dispatch that comes closest to meeting the loads *exactly*.
              It can leave some load unmet.
            * ``"cover"`` - linear programme: the *cheapest* non-negative
              dispatch with ``A @ s >= d``, so no load is ever left unmet and
              any surplus is curtailed. This is the default because a health
              centre cannot short its critical equipment.
        """
        if strategy not in {"cover", "nnls", "clip", "none"}:
            raise ValueError("strategy must be 'cover', 'nnls', 'clip' or 'none'.")
        b = self._check_rhs(rhs)
        raw = self.solve_vectorised(b)
        adjusted = raw.copy()
        bad = np.flatnonzero(np.any(raw < -1e-9, axis=0))
        if strategy == "clip":
            adjusted[:, bad] = np.clip(raw[:, bad], 0, None)
        elif strategy == "nnls":
            for j in bad:
                adjusted[:, j], _ = nnls(self._a, b[:, j])
        elif strategy == "cover":
            for j in bad:
                adjusted[:, j] = self.cheapest_cover(b[:, j])
        residual = b - self._a @ adjusted
        return DispatchPlan(self.sources, raw, adjusted, residual, strategy)

    def cheapest_cover(self, demand: Sequence[float]) -> np.ndarray:
        """Cheapest ``s >= 0`` with ``A @ s >= demand`` (``scipy.optimize.linprog``)."""
        d = self._check_rhs(np.asarray(demand, dtype=float))
        res = linprog(self.unit_costs, A_ub=-self._a, b_ub=-d,
                      bounds=[(0, None)] * self.n, method="highs")
        if not res.success:
            raise RuntimeError(f"No non-negative dispatch covers the demand: {res.message}")
        return res.x

    # -- cost --------------------------------------------------------------
    def daily_cost(self, solution: np.ndarray) -> np.ndarray:
        """UGX cost per day for a solution of shape ``(n,)`` or ``(n, n_days)``."""
        s = np.asarray(solution, dtype=float)
        if s.shape[0] != self.n:
            raise ValueError(f"Solution must have {self.n} rows.")
        return self.unit_costs @ s

    # -- sensitivity (extension) ------------------------------------------
    def sensitivity(self, demand: Sequence[float], rel_perturbation: float = 0.05,
                    n_draws: int = 1000, seed: int = 2026) -> SensitivityResult:
        """Monte Carlo: perturb each demand uniformly by ``+/- rel_perturbation``.

        Each draw multiplies every demand by an independent factor
        ``U(1 - p, 1 + p)`` and re-solves (vectorised).
        """
        if not 0 < rel_perturbation < 1:
            raise ValueError("rel_perturbation must be in (0, 1).")
        if n_draws < 1:
            raise ValueError("n_draws must be positive.")
        d0 = self._check_rhs(np.asarray(demand, dtype=float))
        rng = np.random.default_rng(seed)
        factors = rng.uniform(1 - rel_perturbation, 1 + rel_perturbation, size=(n_draws, self.n))
        demands = factors * d0
        solutions = self.solve_vectorised(demands.T).T
        return SensitivityResult(d0, self.solve_day(*d0), demands, solutions, self.condition_number)


class HybridMicroGrid(MicroGrid):
    """Solar + battery + diesel with a third (night-time) constraint.

    Default system (my design choice)::

        3x + 2y + 1z = D1   daytime load
        4x + 1y + 2z = D2   critical-equipment load
        0x + 1y + 1z = D3   night load (solar cannot contribute after dark)

    Parameters
    ----------
    coefficients : array_like, shape (3, 3), optional
    unit_costs : Mapping[str, float], optional
        Defaults to solar 150, battery 450, diesel 1,200 UGX/kWh (illustrative).
    """

    DEFAULT_COEFFICIENTS = np.array([[3.0, 2.0, 1.0], [4.0, 1.0, 2.0], [0.0, 1.0, 1.0]])
    DEFAULT_COSTS = {"solar": 150.0, "battery": 450.0, "diesel": 1200.0}

    def __init__(self, coefficients: Sequence[Sequence[float]] | np.ndarray | None = None,
                 unit_costs: Mapping[str, float] | None = None) -> None:
        super().__init__(self.DEFAULT_COEFFICIENTS if coefficients is None else coefficients,
                         self.DEFAULT_COSTS if unit_costs is None else unit_costs)
        if self.n != 3:
            raise ValueError("HybridMicroGrid needs a 3 x 3 system.")

    @classmethod
    def with_dependent_constraint(cls, weights: tuple[float, float] = (1.0, 1.0)) -> "HybridMicroGrid":
        """Build a grid whose third row is ``w1*row1 + w2*row2`` (singular by design)."""
        a = cls.DEFAULT_COEFFICIENTS.copy()
        a[2] = weights[0] * a[0] + weights[1] * a[1]
        return cls(a)

    def classify(self, *demands: float) -> str:
        """Classify the system for a given demand using ranks (Rouche-Capelli).

        Returns
        -------
        str
            ``"unique"`` if rank(A) = 3; ``"infinitely many"`` if
            rank(A) = rank([A|d]) < 3; ``"none"`` if rank(A) < rank([A|d]).
        """
        b = self._check_rhs(np.array(demands, dtype=float))
        rank_a = self.rank
        rank_ab = int(np.linalg.matrix_rank(np.column_stack([self._a, b])))
        if rank_a == self.n:
            return "unique"
        return "infinitely many" if rank_a == rank_ab else "none"

    def solve_least_squares(self, *demands: float) -> tuple[np.ndarray, float]:
        """Minimum-norm least-squares solution, usable even when singular.

        Returns
        -------
        (solution, residual_norm)
        """
        b = self._check_rhs(np.array(demands, dtype=float))
        sol, *_ = linalg.lstsq(self._a, b)
        return sol, float(np.linalg.norm(b - self._a @ sol))
