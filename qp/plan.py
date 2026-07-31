"""Plan trees and the cost model.

The cost model is deliberately simple and deliberately calibrated against the
executor in this repository rather than against a disk. Every constant below was
fitted by timing the corresponding operator in isolation, and the fit is
reported in bench/cost_vs_time.py along with how badly it does - which is the
point of having it. A cost model nobody has compared against a clock is a
scoring function, not a model.

Costs are in arbitrary units that happen to be calibrated to microseconds on the
machine the fit was run on. Absolute values do not matter; what matters is
whether the *ordering* the model induces over plans matches the ordering the
clock induces, and that is measured as a rank correlation, not as an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .estimate import (JoinCond, Predicate, filter_selectivity,
                       join_cardinality, scaled_distinct)
from .stats import TableStats

MIN_ROWS = 1.0


@dataclass
class CostConstants:
    seq_scan_per_row: float = 0.0042
    """Cost of touching one row in a base table scan."""
    filter_per_row: float = 0.0031
    """Cost of evaluating one predicate against one row."""
    hash_build_per_row: float = 0.0680
    """Cost of inserting one row into a hash table. An order of magnitude above
    a scan, which is why build-side choice matters."""
    hash_probe_per_row: float = 0.0410
    """Cost of probing once."""
    output_per_row: float = 0.0180
    """Cost of materialising one output row."""
    nested_loop_per_pair: float = 0.0009
    """Cost per candidate pair. Small, but multiplied by a product."""


@dataclass
class Node:
    rows: float = 0.0          # estimated
    cost: float = 0.0          # estimated, cumulative
    true_rows: float = -1.0    # filled in by the executor, -1 until then

    def relations(self) -> frozenset[str]:
        raise NotImplementedError

    def children(self) -> list["Node"]:
        return []

    def label(self) -> str:
        raise NotImplementedError

    def walk(self):
        yield self
        for c in self.children():
            yield from c.walk()

    def show(self, indent: int = 0) -> str:
        pad = "  " * indent
        actual = "" if self.true_rows < 0 else f" actual={self.true_rows:,.0f}"
        out = [f"{pad}{self.label()}  rows={self.rows:,.0f} "
               f"cost={self.cost:,.1f}{actual}"]
        for c in self.children():
            out.append(c.show(indent + 1))
        return "\n".join(out)


@dataclass
class Scan(Node):
    table: str = ""
    preds: list[Predicate] = field(default_factory=list)
    base_rows: float = 0.0
    selectivity: float = 1.0

    def relations(self) -> frozenset[str]:
        return frozenset([self.table])

    def label(self) -> str:
        if not self.preds:
            return f"Scan({self.table})"
        return f"Scan({self.table}, {' and '.join(map(str, self.preds))})"


@dataclass
class HashJoin(Node):
    left: Node = None
    right: Node = None
    cond: JoinCond = None

    def relations(self) -> frozenset[str]:
        return self.left.relations() | self.right.relations()

    def children(self) -> list[Node]:
        return [self.left, self.right]

    def label(self) -> str:
        return f"HashJoin({self.cond})"


@dataclass
class NestedLoop(Node):
    left: Node = None
    right: Node = None
    cond: JoinCond = None

    def relations(self) -> frozenset[str]:
        return self.left.relations() | self.right.relations()

    def children(self) -> list[Node]:
        return [self.left, self.right]

    def label(self) -> str:
        return f"NestedLoop({self.cond})"


# ---- construction with costing -------------------------------------------

class Costing:
    """Builds plan nodes with estimates attached.

    `truth` swaps the estimator for exact cardinalities measured from the data.
    That is not a mode a real optimizer can run in - it is the oracle. Costing
    the same plan space with true cardinalities gives the plan the optimizer
    *should* have picked, and the gap between that plan's runtime and the
    chosen plan's runtime is the only honest way to score an optimizer without
    confounding search quality with estimation quality.
    """

    def __init__(self, stats: dict[str, TableStats],
                 constants: CostConstants | None = None,
                 truth=None):
        self.stats = stats
        self.c = constants or CostConstants()
        self.truth = truth

    # -- scans --

    def scan(self, table: str, preds: list[Predicate]) -> Scan:
        ts = self.stats[table]
        n = float(ts.n_rows)
        if self.truth is not None:
            rows = float(self.truth.scan_rows(table, preds))
            sel = rows / n if n else 1.0
        else:
            sel = filter_selectivity(ts, preds)
            rows = max(MIN_ROWS, n * sel)

        cost = n * self.c.seq_scan_per_row
        cost += n * self.c.filter_per_row * len(preds)
        cost += rows * self.c.output_per_row
        return Scan(rows=rows, cost=cost, table=table, preds=list(preds),
                    base_rows=n, selectivity=sel)

    # -- distinct tracking --

    def _distinct(self, node: Node, table: str, column: str) -> float:
        """Distinct values of `table.column` in this subtree's output."""
        ts = self.stats[table]
        cs = ts.columns.get(column)
        if cs is None:
            return max(1.0, node.rows)
        base = cs.n_distinct
        # Rows for the table itself, before the joins above it, is not tracked
        # separately; using the subtree row count is the standard approximation
        # and it over-estimates distinctness after a fanout join.
        return min(scaled_distinct(base, float(ts.n_rows), node.rows),
                   max(1.0, node.rows))

    def _join_rows(self, left: Node, right: Node, cond: JoinCond) -> float:
        if self.truth is not None:
            return float(self.truth.join_rows(left, right, cond))
        lt, rt = cond.left_table, cond.right_table
        if lt in left.relations():
            ld = self._distinct(left, lt, cond.left_column)
            rd = self._distinct(right, rt, cond.right_column)
        else:
            ld = self._distinct(left, rt, cond.right_column)
            rd = self._distinct(right, lt, cond.left_column)
        return join_cardinality(left.rows, right.rows, ld, rd)

    def hash_join(self, left: Node, right: Node, cond: JoinCond) -> HashJoin:
        rows = self._join_rows(left, right, cond)
        # The smaller input is built; that is the executor's behaviour and the
        # model has to match it or the cost is of a plan nobody runs.
        build, probe = (right, left) if right.rows <= left.rows else (left, right)
        cost = left.cost + right.cost
        cost += build.rows * self.c.hash_build_per_row
        cost += probe.rows * self.c.hash_probe_per_row
        cost += rows * self.c.output_per_row
        return HashJoin(rows=rows, cost=cost, left=left, right=right, cond=cond)

    def nested_loop(self, left: Node, right: Node, cond: JoinCond) -> NestedLoop:
        rows = self._join_rows(left, right, cond)
        cost = left.cost + right.cost
        cost += left.rows * right.rows * self.c.nested_loop_per_pair
        cost += rows * self.c.output_per_row
        return NestedLoop(rows=rows, cost=cost, left=left, right=right, cond=cond)

    def best_join(self, left: Node, right: Node, cond: JoinCond) -> Node:
        h = self.hash_join(left, right, cond)
        # Nested loop is only ever competitive when one side is tiny. Costing it
        # always and taking the cheaper is what a real optimizer does, and it
        # keeps the physical-operator choice inside the same search.
        if min(left.rows, right.rows) <= 64:
            n = self.nested_loop(left, right, cond)
            if n.cost < h.cost:
                return n
        return h
