"""Synthetic tables with knobs on the two things that break estimators.

Real benchmark data is either uniform and independent, in which case textbook
estimation works and proves nothing, or it is real, in which case you cannot say
*which* property broke the estimate. Generated data lets skew and correlation be
dialled separately, so a q-error can be attributed to one of them.

Two knobs:

  zipf         marginal skew. 0.0 is uniform. Around 1.0 is what a real
               foreign key to a "customer" table looks like.
  correlation  how strongly two columns in the same table agree. The
               independence assumption in every classical estimator is exactly
               the assumption this violates, and it is where the tail of the
               q-error distribution comes from.

Columns are numpy arrays, not rows. Everything downstream is vectorised, so an
operator's cost tracks the number of tuples it touches rather than the number of
Python objects it allocates - which is the only way a cost model calibrated in
"tuples" can be compared against a wall clock at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Column:
    name: str
    values: np.ndarray

    @property
    def n(self) -> int:
        return len(self.values)


@dataclass
class Table:
    name: str
    columns: dict[str, Column] = field(default_factory=dict)

    @property
    def n(self) -> int:
        if not self.columns:
            return 0
        return next(iter(self.columns.values())).n

    def col(self, name: str) -> np.ndarray:
        return self.columns[name].values

    def add(self, name: str, values: np.ndarray) -> None:
        self.columns[name] = Column(name, np.asarray(values))

    def take(self, idx: np.ndarray) -> "Table":
        out = Table(self.name)
        for name, c in self.columns.items():
            out.add(name, c.values[idx])
        return out

    def __repr__(self) -> str:
        return f"Table({self.name}, n={self.n}, cols={sorted(self.columns)})"


def zipf_values(rng: np.random.Generator, n: int, domain: int,
                skew: float) -> np.ndarray:
    """Draw n values from [0, domain) with a Zipf-like weight.

    numpy's zipf() is unbounded and produces a handful of enormous values, which
    is the wrong shape for a key domain. This builds the weights explicitly over
    a fixed domain instead.
    """
    if skew <= 0:
        return rng.integers(0, domain, size=n)
    ranks = np.arange(1, domain + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, skew)
    weights /= weights.sum()
    return rng.choice(domain, size=n, p=weights)


def correlated_pair(rng: np.random.Generator, base: np.ndarray, domain: int,
                    correlation: float) -> np.ndarray:
    """A second column that agrees with `base` with probability `correlation`.

    At 1.0 the two columns are a deterministic function of each other, and any
    estimator multiplying their selectivities together is wrong by the full
    factor of the second predicate. At 0.0 they are independent and the textbook
    formula is exact. The interesting behaviour is in between.
    """
    n = len(base)
    independent = rng.integers(0, domain, size=n)
    keep = rng.random(n) < correlation
    scaled = (base.astype(np.int64) % domain)
    return np.where(keep, scaled, independent)


@dataclass
class Database:
    tables: dict[str, Table] = field(default_factory=dict)

    def add(self, table: Table) -> None:
        self.tables[table.name] = table

    def __getitem__(self, name: str) -> Table:
        return self.tables[name]

    def summary(self) -> str:
        rows = [f"  {t.name:<12} {t.n:>9,} rows  {len(t.columns)} cols"
                for t in self.tables.values()]
        return "\n".join(rows)


def star_schema(seed: int = 0, fact_rows: int = 200_000,
                dims: int = 4, dim_rows: int = 2_000,
                skew: float = 0.8, correlation: float = 0.0) -> Database:
    """One fact table with foreign keys into `dims` dimension tables.

    This shape is chosen because it is the one where join-order matters and
    where the estimate for a chain of joins compounds. Each dimension carries a
    filterable attribute and a second attribute correlated with the first by
    `correlation`.
    """
    rng = np.random.default_rng(seed)
    db = Database()

    for d in range(dims):
        t = Table(f"dim{d}")
        t.add("id", np.arange(dim_rows, dtype=np.int64))
        a = rng.integers(0, 100, size=dim_rows)
        t.add("a", a)
        t.add("b", correlated_pair(rng, a, 100, correlation))
        t.add("payload", rng.integers(0, 1_000_000, size=dim_rows))
        db.add(t)

    fact = Table("fact")
    fact.add("id", np.arange(fact_rows, dtype=np.int64))
    for d in range(dims):
        fact.add(f"fk{d}", zipf_values(rng, fact_rows, dim_rows, skew))
    fact.add("measure", rng.integers(0, 1000, size=fact_rows))
    db.add(fact)
    return db


def mesh_schema(seed: int = 0, tables: int = 5, rows: int = 900,
                groups: int = 100, skew: float = 0.0) -> Database:
    """Tables joined on a low-cardinality attribute, not on a key.

    Every join in a star schema lands on a dimension's primary key, so the
    result is bounded by the fact table and an estimation error at one level
    does not get multiplied at the next. That makes a star schema a poor place
    to look for compounding, which is worth saying rather than concluding that
    compounding does not happen.

    Here each join is many-to-many on a column with `groups` distinct values,
    so cardinality grows by roughly rows/groups per level and so does the error.
    This is the shape - a chain of non-key joins - where classical estimation is
    known to fall apart, and it is the shape the depth sweep uses.
    """
    rng = np.random.default_rng(seed)
    db = Database()
    for i in range(tables):
        t = Table(f"m{i}")
        t.add("id", np.arange(rows, dtype=np.int64))
        t.add("g", zipf_values(rng, rows, groups, skew))
        t.add("a", rng.integers(0, 100, size=rows))
        db.add(t)
    return db


def chain_schema(seed: int = 0, tables: int = 5, rows: int = 50_000,
                 fanout: float = 1.0, skew: float = 0.5) -> Database:
    """t0 - t1 - t2 - ... joined in a line.

    A chain has far fewer valid join orders than a clique, which makes it the
    shape where a greedy heuristic is most likely to match the optimum, and so
    the shape that flatters a bad optimizer. It is here as a control.
    """
    rng = np.random.default_rng(seed)
    db = Database()
    for i in range(tables):
        n = max(1000, int(rows * (fanout ** i)))
        t = Table(f"t{i}")
        t.add("id", np.arange(n, dtype=np.int64))
        t.add("a", rng.integers(0, 100, size=n))
        if i > 0:
            prev = db[f"t{i - 1}"].n
            t.add("prev_id", zipf_values(rng, n, prev, skew))
        db.add(t)
    return db
