"""Statistics the optimizer is allowed to look at.

The rule enforced here: an estimate may only use what a real system could
compute in a background ANALYZE pass over a sample. No peeking at the data at
estimation time, and no computing an exact answer and calling it an estimate.
That sounds obvious and it is the single easiest way to accidentally build an
optimizer that reports a q-error of 1.0 and means nothing.

Three structures, matching what PostgreSQL actually keeps:

  equi-depth histogram   range selectivity
  most-common values     the part a histogram smooths away, which for skewed
                         data is most of the mass
  distinct count         join cardinality, via HyperLogLog rather than an exact
                         count, because the exact count is not available on a
                         sample and pretending otherwise removes a real source
                         of error
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


# ---- HyperLogLog ----------------------------------------------------------

class HyperLogLog:
    """Distinct count from a fixed 2^p-byte sketch.

    Written out rather than imported because the error it introduces is part of
    what is being measured: join cardinality estimation divides by a distinct
    count, so a 2% error in the sketch becomes a 2% error in the join estimate
    before any modelling assumption has been made at all.
    """

    def __init__(self, p: int = 12):
        self.p = p
        self.m = 1 << p
        self.registers = np.zeros(self.m, dtype=np.uint8)

    @staticmethod
    def _hash(values: np.ndarray) -> np.ndarray:
        # A cheap 64-bit mix. Values are integers here; splitmix64 gives a
        # well-distributed hash without a per-element Python call.
        x = values.astype(np.uint64, copy=True)
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
        return x

    def add(self, values: np.ndarray) -> None:
        h = self._hash(np.asarray(values).ravel())
        idx = (h >> np.uint64(64 - self.p)).astype(np.int64)
        rest = (h << np.uint64(self.p)) | np.uint64(1 << (self.p - 1))
        # position of the leading 1 bit, 1-based
        rank = np.zeros(len(rest), dtype=np.uint8)
        probe = rest.copy()
        for bit in range(1, 65):
            mask = (probe & np.uint64(1 << 63)) != 0
            fresh = (rank == 0) & mask
            rank[fresh] = bit
            if rank.all():
                break
            probe = probe << np.uint64(1)
        np.maximum.at(self.registers, idx, rank)

    def count(self) -> float:
        m = float(self.m)
        alpha = 0.7213 / (1.0 + 1.079 / m)
        harmonic = np.sum(np.power(2.0, -self.registers.astype(np.float64)))
        raw = alpha * m * m / harmonic
        zeros = int(np.count_nonzero(self.registers == 0))
        if raw <= 2.5 * m and zeros > 0:
            return m * np.log(m / zeros)     # small-range correction
        return raw


# ---- per-column statistics ------------------------------------------------

@dataclass
class ColumnStats:
    name: str
    n_rows: int
    n_distinct: float
    min: float
    max: float
    bounds: np.ndarray = field(default_factory=lambda: np.array([]))
    """Equi-depth bucket boundaries. len(bounds) - 1 buckets, each holding
    roughly n_rows / buckets tuples."""
    mcv_values: np.ndarray = field(default_factory=lambda: np.array([]))
    mcv_freq: np.ndarray = field(default_factory=lambda: np.array([]))
    null_fraction: float = 0.0

    @property
    def buckets(self) -> int:
        return max(0, len(self.bounds) - 1)

    @property
    def mcv_mass(self) -> float:
        return float(self.mcv_freq.sum()) if len(self.mcv_freq) else 0.0


@dataclass
class TableStats:
    name: str
    n_rows: int
    columns: dict[str, ColumnStats] = field(default_factory=dict)


def analyze_column(name: str, values: np.ndarray, buckets: int = 32,
                   mcv: int = 16, sample: int = 0,
                   rng: np.random.Generator | None = None) -> ColumnStats:
    """Build statistics the way ANALYZE does: optionally from a sample.

    `sample=0` reads the whole column, which is what a small database does.
    Setting it to a few tens of thousands is what a large one does, and the
    extra error that introduces is measurable in bench/qerror.py.
    """
    values = np.asarray(values)
    n_rows = len(values)
    if n_rows == 0:
        return ColumnStats(name, 0, 0, 0, 0)

    if sample and sample < n_rows:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(n_rows, size=sample, replace=False)
        seen = values[idx]
        scale = n_rows / float(sample)
    else:
        seen = values
        scale = 1.0

    hll = HyperLogLog(12)
    hll.add(seen)
    # The sketch counts distinct values in the sample. Scaling that to the whole
    # column is Chao-style and deliberately crude - estimating distinct counts
    # from a sample is provably hard, and pretending otherwise would hide one of
    # the larger real error sources.
    distinct_seen = hll.count()
    n_distinct = min(float(n_rows), distinct_seen * (scale ** 0.5))

    uniq, counts = np.unique(seen, return_counts=True)
    order = np.argsort(counts)[::-1]
    top = order[:mcv]
    mcv_values = uniq[top]
    mcv_freq = counts[top] / float(len(seen))

    # Values covered by the MCV list are removed before the histogram is built,
    # which is what makes the histogram useful on skewed data instead of being
    # dominated by one value.
    heavy = np.isin(seen, mcv_values)
    rest = seen[~heavy]
    if len(rest) > buckets:
        qs = np.linspace(0, 100, buckets + 1)
        bounds = np.percentile(rest, qs)
        bounds = np.unique(bounds)
    elif len(rest):
        bounds = np.unique(rest)
    else:
        bounds = np.array([float(seen.min()), float(seen.max())])

    return ColumnStats(
        name=name,
        n_rows=n_rows,
        n_distinct=max(1.0, n_distinct),
        min=float(seen.min()),
        max=float(seen.max()),
        bounds=bounds.astype(np.float64),
        mcv_values=mcv_values,
        mcv_freq=mcv_freq,
    )


def analyze_table(table, buckets: int = 32, mcv: int = 16, sample: int = 0,
                  seed: int = 0) -> TableStats:
    rng = np.random.default_rng(seed)
    ts = TableStats(table.name, table.n)
    for name, col in table.columns.items():
        ts.columns[name] = analyze_column(name, col.values, buckets, mcv,
                                          sample, rng)
    return ts


def analyze_db(db, buckets: int = 32, mcv: int = 16, sample: int = 0,
               seed: int = 0) -> dict[str, TableStats]:
    return {name: analyze_table(t, buckets, mcv, sample, seed)
            for name, t in db.tables.items()}
