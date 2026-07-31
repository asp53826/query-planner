"""Query generation.

Queries are drawn from a seed so a bad estimate can be replayed. Each query gets
filters on a random subset of the dimension tables with random selectivities,
which is what produces the spread of intermediate sizes that makes join order
matter at all - a workload where every filter keeps 50% of everything has one
good plan and no interesting failures.
"""

from __future__ import annotations

import numpy as np

from .enumerate import Query
from .estimate import JoinCond, Predicate


def star_query(rng: np.random.Generator, dims: int, n_filters: int = 2,
               correlated_pairs: bool = False) -> Query:
    tables = ["fact"] + [f"dim{d}" for d in range(dims)]
    joins = [JoinCond("fact", f"fk{d}", f"dim{d}", "id") for d in range(dims)]

    preds: list[Predicate] = []
    chosen = rng.choice(dims, size=min(n_filters, dims), replace=False)
    for d in chosen:
        t = f"dim{d}"
        hi = int(rng.integers(5, 95))
        preds.append(Predicate(t, "a", "<", float(hi)))
        if correlated_pairs:
            # A second predicate on the column correlated with `a`. This is the
            # pair that breaks the independence assumption, and the whole reason
            # the generator has the option.
            preds.append(Predicate(t, "b", "<", float(hi)))
    return Query(tables, joins, preds)


def mesh_query(rng: np.random.Generator, n_tables: int,
               n_filters: int = 1) -> Query:
    """m0.g = m1.g = m2.g ... - a chain of many-to-many joins."""
    tables = [f"m{i}" for i in range(n_tables)]
    joins = [JoinCond(f"m{i - 1}", "g", f"m{i}", "g") for i in range(1, n_tables)]
    preds = []
    for t in rng.choice(tables, size=min(n_filters, n_tables), replace=False):
        preds.append(Predicate(str(t), "a", "<", float(rng.integers(20, 90))))
    return Query(tables, joins, preds)


def chain_query(rng: np.random.Generator, n_tables: int,
                n_filters: int = 2) -> Query:
    tables = [f"t{i}" for i in range(n_tables)]
    joins = [JoinCond(f"t{i}", "prev_id", f"t{i - 1}", "id")
             for i in range(1, n_tables)]
    preds = []
    for t in rng.choice(tables, size=min(n_filters, n_tables), replace=False):
        preds.append(Predicate(str(t), "a", "<", float(rng.integers(5, 95))))
    return Query(tables, joins, preds)


def q_error(estimated_rows: float, actual_rows: float) -> float:
    """max(est/act, act/est), floored at 1.

    Both arguments are ROW COUNTS, not selectivities. Both are clamped up to 1
    before the ratio, so passing two fractions returns 1.0 regardless of how
    wrong the estimate was - which makes any assertion on the result pass
    silently. Three tests in this repository did exactly that before it was
    caught. Scale to rows at the call site.

    The standard metric, and the reason it is standard is that it is symmetric
    and unbounded in both directions. Relative error is not: under-estimating by
    a factor of a thousand and over-estimating by a factor of a thousand are
    equally bad for a plan, and relative error scores the first as 0.999 and the
    second as 999.

    Zero actuals are floored to 1. An empty intermediate is a real outcome but
    the ratio is undefined, and treating it as infinite would let one query
    dominate every aggregate.
    """
    e = max(1.0, float(estimated_rows))
    a = max(1.0, float(actual_rows))
    return max(e / a, a / e)


def summarize_q_errors(values) -> dict:
    """Geometric mean and tail.

    The arithmetic mean of a ratio distribution is meaningless - it is dominated
    by the largest over-estimate and says nothing about the typical case. The
    geometric mean is the right centre for a multiplicative quantity, and the
    tail percentiles are what actually determine whether a plan is bad, so both
    are reported and neither on its own.
    """
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    logs = np.log(arr)
    return {
        "n": int(arr.size),
        "geomean": float(np.exp(logs.mean())),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }
