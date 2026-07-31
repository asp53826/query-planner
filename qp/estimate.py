"""Cardinality estimation: the part that is actually hard.

Join ordering is a search problem with a known-good algorithm. Cardinality
estimation is a modelling problem with no good algorithm, and it is where
optimizers go wrong: Leis et al. found in 2015 that the join-order search was
essentially solved and the estimates were off by orders of magnitude, and that
the estimates were what determined whether the plan was any good.

Everything here follows the classical System R lineage, because that is what is
in production systems and because its failure modes are the ones worth measuring:

  conjunction   selectivities multiply. True only if the predicates are
                independent, which for any two columns anyone would filter on
                together is exactly when it is false.
  join          |R JOIN S| = |R||S| / max(d(R.a), d(S.b)). Assumes containment
                of value domains and uniformity within them.
  transitive    a filter on one side of a join is not propagated to the other.

Each of those is a place the estimate can be wrong by a factor that compounds
with every join above it. bench/qerror.py measures by how much.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import ColumnStats, TableStats

MIN_ROWS = 1.0
"""Never estimate zero. A zero propagates up a plan tree and makes every cost
above it identical, so the optimizer stops being able to tell plans apart at
exactly the moment the estimate is most wrong."""


@dataclass(frozen=True)
class Predicate:
    table: str
    column: str
    op: str          # = | < | <= | > | >= | between
    value: float
    hi: float = 0.0

    def __repr__(self) -> str:
        if self.op == "between":
            return f"{self.table}.{self.column} between {self.value} and {self.hi}"
        return f"{self.table}.{self.column} {self.op} {self.value}"


@dataclass(frozen=True)
class JoinCond:
    left_table: str
    left_column: str
    right_table: str
    right_column: str

    def other(self, table: str) -> str:
        return self.right_table if table == self.left_table else self.left_table

    def touches(self, table: str) -> bool:
        return table in (self.left_table, self.right_table)

    def column_for(self, table: str) -> str:
        return self.left_column if table == self.left_table else self.right_column

    def __repr__(self) -> str:
        return (f"{self.left_table}.{self.left_column} = "
                f"{self.right_table}.{self.right_column}")


# ---- single-column selectivity -------------------------------------------

def _eq_selectivity(cs: ColumnStats, value: float) -> float:
    if len(cs.mcv_values):
        hit = np.flatnonzero(cs.mcv_values == value)
        if len(hit):
            return float(cs.mcv_freq[hit[0]])

    # Not a most-common value: distribute the remaining mass over the
    # remaining distinct values. This is the standard fallback and it is where
    # equality estimates on skewed columns go wrong, because "the rest" is not
    # uniform either.
    remaining_mass = max(0.0, 1.0 - cs.mcv_mass)
    remaining_distinct = max(1.0, cs.n_distinct - len(cs.mcv_values))
    return remaining_mass / remaining_distinct


def _hist_fraction_below(cs: ColumnStats, value: float) -> float:
    """Fraction of the non-MCV mass strictly below `value`, by linear
    interpolation inside the containing equi-depth bucket."""
    b = cs.bounds
    if len(b) < 2:
        return 0.0 if value <= cs.min else 1.0
    if value <= b[0]:
        return 0.0
    if value >= b[-1]:
        return 1.0

    idx = int(np.searchsorted(b, value, side="right")) - 1
    idx = max(0, min(idx, len(b) - 2))
    lo, hi = b[idx], b[idx + 1]
    within = 0.0 if hi <= lo else (value - lo) / (hi - lo)
    buckets = len(b) - 1
    return (idx + within) / buckets


def _range_selectivity(cs: ColumnStats, op: str, value: float,
                       hi: float = 0.0) -> float:
    non_mcv = max(0.0, 1.0 - cs.mcv_mass)

    def mcv_mass_where(pred) -> float:
        if not len(cs.mcv_values):
            return 0.0
        return float(cs.mcv_freq[pred(cs.mcv_values)].sum())

    if op in ("<", "<="):
        s = non_mcv * _hist_fraction_below(cs, value)
        s += mcv_mass_where(lambda v: v <= value if op == "<=" else v < value)
    elif op in (">", ">="):
        s = non_mcv * (1.0 - _hist_fraction_below(cs, value))
        s += mcv_mass_where(lambda v: v >= value if op == ">=" else v > value)
    elif op == "between":
        s = non_mcv * max(0.0, _hist_fraction_below(cs, hi)
                          - _hist_fraction_below(cs, value))
        s += mcv_mass_where(lambda v: (v >= value) & (v <= hi))
    else:
        raise ValueError(f"unknown range op {op}")
    return float(min(1.0, max(0.0, s)))


def selectivity(cs: ColumnStats, p: Predicate) -> float:
    if p.op == "=":
        return float(min(1.0, max(0.0, _eq_selectivity(cs, p.value))))
    return _range_selectivity(cs, p.op, p.value, p.hi)


def filter_selectivity(ts: TableStats, preds: list[Predicate]) -> float:
    """Conjunction of predicates on one table.

    Selectivities multiply. This is the independence assumption, and it is the
    single largest source of estimation error in this repository - see the
    correlation sweep in bench/qerror.py, where it costs two orders of
    magnitude on perfectly correlated columns.
    """
    s = 1.0
    for p in preds:
        cs = ts.columns.get(p.column)
        if cs is None:
            continue
        s *= selectivity(cs, p)
    return float(min(1.0, max(0.0, s)))


# ---- join cardinality -----------------------------------------------------

def join_cardinality(left_rows: float, right_rows: float,
                     left_distinct: float, right_distinct: float) -> float:
    """The System R formula.

    Every tuple on the smaller-domain side is assumed to find |other| /
    d(other) matches. It is exact for a uniform foreign key into a primary key
    and degrades from there; on a skewed key it under-estimates, because the
    heavy values produce far more matches than the average.
    """
    d = max(1.0, left_distinct, right_distinct)
    return max(MIN_ROWS, (left_rows * right_rows) / d)


def scaled_distinct(base_distinct: float, base_rows: float,
                    rows_now: float) -> float:
    """How many distinct values survive a filter.

    If a filter keeps a fraction f of the rows, the number of distinct values
    left is somewhere between 1 and min(d, f*n). This uses the standard
    approximation d * (1 - (1 - f)^(n/d)), which is the expected number of
    distinct values when f*n rows are drawn from d equally likely groups.
    """
    if base_rows <= 0 or base_distinct <= 0:
        return 1.0
    f = min(1.0, max(0.0, rows_now / base_rows))
    if f >= 1.0:
        return base_distinct
    per_group = base_rows / base_distinct
    survive = 1.0 - (1.0 - f) ** per_group
    return max(1.0, base_distinct * survive)
