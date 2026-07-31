"""Estimator tests.

Two kinds. The first is against distributions whose answer can be written down:
on uniform independent data the textbook formulas are exact, so any error is a
bug rather than a modelling limitation. The second pins the *known* failure
modes, so that if someone later fixes the independence assumption these tests
fail loudly rather than silently continuing to assert the old behaviour.
"""

import numpy as np
import pytest

from qp.data import Table, star_schema
from qp.estimate import (Predicate, filter_selectivity, join_cardinality,
                         scaled_distinct, selectivity)
from qp.stats import HyperLogLog, analyze_column, analyze_table
from qp.workload import q_error, summarize_q_errors


def uniform_column(n=20_000, domain=100, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, domain, size=n)


# ---- HyperLogLog ----------------------------------------------------------

@pytest.mark.parametrize("n", [50, 500, 5_000, 50_000, 200_000])
def test_hll_is_within_a_few_percent(n):
    h = HyperLogLog(12)
    h.add(np.arange(n))
    est = h.count()
    assert abs(est - n) / n < 0.06, (n, est)


def test_hll_ignores_duplicates():
    h = HyperLogLog(12)
    h.add(np.repeat(np.arange(1000), 50))
    assert abs(h.count() - 1000) / 1000 < 0.08


def test_hll_is_deterministic():
    a, b = HyperLogLog(12), HyperLogLog(12)
    values = np.arange(5000)
    a.add(values)
    b.add(values)
    assert a.count() == b.count()


# ---- selectivity ----------------------------------------------------------

def test_range_selectivity_on_uniform_data_is_close_to_exact():
    values = uniform_column()
    cs = analyze_column("c", values, buckets=32)
    for threshold in (10, 25, 50, 75, 90):
        est = selectivity(cs, Predicate("t", "c", "<", float(threshold)))
        actual = float((values < threshold).mean())
        assert q_error(est * len(values), actual * len(values)) < 1.15, \
            (threshold, est, actual)


def test_equality_selectivity_on_uniform_data():
    values = uniform_column(domain=50)
    cs = analyze_column("c", values, buckets=32)
    est = selectivity(cs, Predicate("t", "c", "=", 7.0))
    assert 0.5 / 50 < est < 2.0 / 50


def test_equality_on_a_heavy_value_uses_the_mcv_list():
    """A value holding 40% of the table must not be estimated as 1/distinct."""
    rng = np.random.default_rng(1)
    values = np.concatenate([np.full(4000, 7), rng.integers(0, 100, size=6000)])
    cs = analyze_column("c", values, buckets=32, mcv=16)
    est = selectivity(cs, Predicate("t", "c", "=", 7.0))
    actual = float((values == 7).mean())
    # q_error takes row counts, not fractions: it floors both arguments at one
    # row, so handing it two selectivities returns 1.0 no matter how wrong the
    # estimate is. Scaling to rows first is the only correct way to call it.
    n = len(values)
    assert q_error(est * n, actual * n) < 1.2, (est, actual)


def test_selectivity_is_bounded():
    values = uniform_column()
    cs = analyze_column("c", values)
    for p in (Predicate("t", "c", "<", -1e9), Predicate("t", "c", ">", 1e9),
              Predicate("t", "c", "<", 1e9), Predicate("t", "c", ">", -1e9)):
        s = selectivity(cs, p)
        assert 0.0 <= s <= 1.0


def test_range_selectivity_is_monotonic():
    values = uniform_column()
    cs = analyze_column("c", values)
    last = -1.0
    for threshold in range(0, 101, 5):
        s = selectivity(cs, Predicate("t", "c", "<", float(threshold)))
        assert s >= last - 1e-9, threshold
        last = s


def test_independent_predicates_multiply_correctly():
    """On genuinely independent columns the product rule is right, so this is
    the control for the correlation test below."""
    rng = np.random.default_rng(3)
    t = Table("t")
    t.add("a", rng.integers(0, 100, size=20_000))
    t.add("b", rng.integers(0, 100, size=20_000))
    ts = analyze_table(t)
    est = filter_selectivity(ts, [Predicate("t", "a", "<", 50.0),
                                  Predicate("t", "b", "<", 50.0)])
    actual = float(((t.col("a") < 50) & (t.col("b") < 50)).mean())
    n = t.n
    assert q_error(est * n, actual * n) < 1.1, (est, actual)


def test_perfectly_correlated_predicates_are_underestimated():
    """The known failure. If someone implements multi-column statistics this
    test should fail, and that failure is the signal that it worked."""
    rng = np.random.default_rng(4)
    a = rng.integers(0, 100, size=20_000)
    t = Table("t")
    t.add("a", a)
    t.add("b", a.copy())
    ts = analyze_table(t)
    est = filter_selectivity(ts, [Predicate("t", "a", "<", 50.0),
                                  Predicate("t", "b", "<", 50.0)])
    actual = float(((t.col("a") < 50) & (t.col("b") < 50)).mean())
    n = t.n
    assert actual > 0.4
    assert est < 0.3
    assert q_error(est * n, actual * n) > 1.7


# ---- join cardinality -----------------------------------------------------

def test_pk_fk_join_is_exact_when_uniform():
    n_fact, n_dim = 50_000, 500
    est = join_cardinality(n_fact, n_dim, n_dim, n_dim)
    assert est == pytest.approx(n_fact, rel=1e-9)


def test_join_cardinality_uses_the_larger_domain():
    assert join_cardinality(1000, 1000, 10, 100) == pytest.approx(10_000)
    assert join_cardinality(1000, 1000, 100, 10) == pytest.approx(10_000)


def test_join_cardinality_never_returns_zero():
    assert join_cardinality(0, 0, 1, 1) >= 1.0


def test_scaled_distinct_shrinks_with_the_filter():
    full = scaled_distinct(1000.0, 10_000.0, 10_000.0)
    half = scaled_distinct(1000.0, 10_000.0, 5_000.0)
    tiny = scaled_distinct(1000.0, 10_000.0, 10.0)
    assert full == pytest.approx(1000.0)
    assert 1.0 <= tiny < half < full


# ---- the metric itself ----------------------------------------------------

def test_q_error_is_symmetric():
    assert q_error(100, 10) == pytest.approx(q_error(10, 100))
    assert q_error(50, 50) == pytest.approx(1.0)


def test_q_error_floors_at_one():
    assert q_error(0, 0) == pytest.approx(1.0)
    assert q_error(0.2, 0.9) == pytest.approx(1.0)


def test_summary_uses_a_geometric_mean():
    """The arithmetic mean of 100x and 0.01x-equivalent errors is dominated by
    the larger one; the geometric mean is not, which is the whole reason the
    metric is reported this way."""
    s = summarize_q_errors([100.0, 100.0, 1.0, 1.0])
    assert s["geomean"] == pytest.approx(10.0)
    assert s["max"] == pytest.approx(100.0)


def test_statistics_do_not_look_at_the_query():
    """Guards the rule that makes the whole measurement meaningful: stats are
    built from the column alone, so an estimate cannot cheat by peeking."""
    db = star_schema(seed=9, fact_rows=5000, dims=1, dim_rows=100)
    ts = analyze_table(db["dim0"])
    cs = ts.columns["a"]
    assert cs.n_rows == db["dim0"].n
    assert len(cs.mcv_values) <= 16
    assert cs.buckets <= 32
