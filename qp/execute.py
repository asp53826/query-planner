"""A vectorised executor, so plans can actually be run.

Without one, a cost model can only be compared against another cost model. With
one, three things become measurable that otherwise have to be asserted:

  the true cardinality at every node, which is the denominator of q-error;
  the wall-clock time of a plan, which is what the cost model is supposed to
  predict;
  the runtime of the plan the optimizer *should* have chosen, which is what
  turns "the estimate was wrong" into "the estimate was wrong and it cost 3.4x".

Everything is numpy over columnar arrays. Hash join uses a sort-merge of hashed
keys rather than a Python dict: a dict of a million numpy scalars spends all its
time in the interpreter, and the resulting timings would measure CPython rather
than the join.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .estimate import JoinCond, Predicate
from .plan import HashJoin, NestedLoop, Node, Scan


@dataclass
class Relation:
    """An intermediate result: named columns, all the same length."""
    columns: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        if not self.columns:
            return 0
        return len(next(iter(self.columns.values())))

    def take(self, idx: np.ndarray) -> "Relation":
        return Relation({k: v[idx] for k, v in self.columns.items()})

    def key(self, table: str, column: str) -> np.ndarray:
        return self.columns[(table, column)]


def apply_predicate(values: np.ndarray, p: Predicate) -> np.ndarray:
    if p.op == "=":
        return values == p.value
    if p.op == "<":
        return values < p.value
    if p.op == "<=":
        return values <= p.value
    if p.op == ">":
        return values > p.value
    if p.op == ">=":
        return values >= p.value
    if p.op == "between":
        return (values >= p.value) & (values <= p.hi)
    raise ValueError(f"unknown op {p.op}")


def scan_mask(table, preds: list[Predicate]) -> np.ndarray:
    mask = np.ones(table.n, dtype=bool)
    for p in preds:
        mask &= apply_predicate(table.col(p.column), p)
    return mask


def hash_join_indices(left_keys: np.ndarray,
                      right_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices of every matching pair, via sorting rather than a dict.

    Sort the right side once, then binary-search every left key for the start
    and end of its run. Both halves are numpy, so the cost is O(n log n) in C
    instead of O(n) in the interpreter, and for the sizes here that is a factor
    of thirty or so.
    """
    order = np.argsort(right_keys, kind="stable")
    sorted_right = right_keys[order]

    start = np.searchsorted(sorted_right, left_keys, side="left")
    end = np.searchsorted(sorted_right, left_keys, side="right")
    counts = end - start

    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    left_idx = np.repeat(np.arange(len(left_keys), dtype=np.int64), counts)
    # Offsets within each matching run, built without a Python loop.
    ends = np.cumsum(counts)
    starts = ends - counts
    ramp = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
    right_positions = np.repeat(start, counts) + ramp
    return left_idx, order[right_positions]


@dataclass
class ExecStats:
    seconds: float = 0.0
    rows_out: int = 0
    peak_rows: int = 0
    nodes: int = 0


class Executor:
    def __init__(self, db):
        self.db = db

    def run(self, node: Node, record: bool = True) -> tuple[Relation, ExecStats]:
        stats = ExecStats()
        started = time.perf_counter()
        rel = self._run(node, stats, record)
        stats.seconds = time.perf_counter() - started
        stats.rows_out = rel.n
        return rel, stats

    def _run(self, node: Node, stats: ExecStats, record: bool) -> Relation:
        stats.nodes += 1

        if isinstance(node, Scan):
            table = self.db[node.table]
            mask = scan_mask(table, node.preds)
            idx = np.flatnonzero(mask)
            rel = Relation({(node.table, name): col.values[idx]
                            for name, col in table.columns.items()})

        elif isinstance(node, (HashJoin, NestedLoop)):
            left = self._run(node.left, stats, record)
            right = self._run(node.right, stats, record)
            cond = node.cond

            if cond.left_table in node.left.relations():
                lk = left.key(cond.left_table, cond.left_column)
                rk = right.key(cond.right_table, cond.right_column)
            else:
                lk = left.key(cond.right_table, cond.right_column)
                rk = right.key(cond.left_table, cond.left_column)

            li, ri = hash_join_indices(lk, rk)
            merged = dict(left.take(li).columns)
            merged.update(right.take(ri).columns)
            rel = Relation(merged)
        else:
            raise TypeError(f"cannot execute {type(node).__name__}")

        if record:
            node.true_rows = float(rel.n)
        stats.peak_rows = max(stats.peak_rows, rel.n)
        return rel


class Truth:
    """Exact cardinalities, for costing a plan the way an oracle would.

    This is not something a real optimizer can call. It exists so that the
    search and the estimator can be scored separately: run the same enumeration
    with `Costing(truth=Truth(db))` and the plan that comes out is the best plan
    reachable by this search given perfect information. Anything worse that the
    real optimizer picks is attributable to estimation, not to the search.

    Join cardinalities are computed by actually executing the subplans, which is
    expensive and is why the optimality study is capped at a handful of
    relations.
    """

    def __init__(self, db):
        self.db = db
        self.exec = Executor(db)
        self._scan_cache: dict = {}
        self._join_cache: dict = {}

    def scan_rows(self, table: str, preds: list[Predicate]) -> int:
        key = (table, tuple(sorted(map(repr, preds))))
        if key not in self._scan_cache:
            self._scan_cache[key] = int(scan_mask(self.db[table], preds).sum())
        return self._scan_cache[key]

    def join_rows(self, left: Node, right: Node, cond: JoinCond) -> int:
        key = (self._sig(left), self._sig(right), repr(cond))
        if key in self._join_cache:
            return self._join_cache[key]
        node = HashJoin(left=left, right=right, cond=cond)
        rel, _ = self.exec.run(node, record=False)
        self._join_cache[key] = rel.n
        return rel.n

    @staticmethod
    def _sig(node: Node) -> str:
        if isinstance(node, Scan):
            return f"S({node.table};{sorted(map(repr, node.preds))})"
        return f"J({Truth._sig(node.left)},{Truth._sig(node.right)},{node.cond})"
