"""The executor has to be right or nothing else measured here means anything.

Every q-error in this repository is estimated-over-actual, and "actual" comes
from these code paths. A join that silently drops tuples would make the
estimator look bad; one that duplicates them would make it look good. So the
join is checked against a brute-force nested loop on data with duplicate keys,
missing keys, and empty inputs.
"""

import numpy as np
import pytest

from qp.data import Table, mesh_schema, star_schema
from qp.enumerate import Query, dp_bushy
from qp.estimate import JoinCond, Predicate
from qp.execute import Executor, apply_predicate, hash_join_indices, scan_mask
from qp.plan import Costing
from qp.stats import analyze_db


def brute_force_pairs(left, right):
    out = []
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if a == b:
                out.append((i, j))
    return sorted(out)


@pytest.mark.parametrize("seed", range(12))
def test_hash_join_matches_a_nested_loop(seed):
    rng = np.random.default_rng(seed)
    # A small domain forces duplicates on both sides, which is the case the
    # index arithmetic gets wrong if the run offsets are computed sloppily.
    left = rng.integers(0, 6, size=int(rng.integers(0, 40)))
    right = rng.integers(0, 6, size=int(rng.integers(0, 40)))

    li, ri = hash_join_indices(left, right)
    got = sorted(zip(li.tolist(), ri.tolist()))
    assert got == brute_force_pairs(left, right)


def test_hash_join_on_empty_inputs():
    empty = np.array([], dtype=np.int64)
    for a, b in ((empty, np.arange(5)), (np.arange(5), empty), (empty, empty)):
        li, ri = hash_join_indices(a, b)
        assert len(li) == 0 and len(ri) == 0


def test_hash_join_with_no_overlap():
    li, ri = hash_join_indices(np.arange(0, 5), np.arange(100, 105))
    assert len(li) == 0


def test_hash_join_cross_product_on_one_key():
    li, ri = hash_join_indices(np.zeros(4, dtype=np.int64),
                               np.zeros(3, dtype=np.int64))
    assert len(li) == 12
    assert sorted(zip(li.tolist(), ri.tolist())) == \
        sorted((i, j) for i in range(4) for j in range(3))


@pytest.mark.parametrize("op,value,expect", [
    ("=", 3, lambda v: v == 3),
    ("<", 3, lambda v: v < 3),
    ("<=", 3, lambda v: v <= 3),
    (">", 3, lambda v: v > 3),
    (">=", 3, lambda v: v >= 3),
])
def test_predicates(op, value, expect):
    values = np.arange(10)
    got = apply_predicate(values, Predicate("t", "c", op, value))
    assert (got == expect(values)).all()


def test_between_is_inclusive():
    values = np.arange(10)
    got = apply_predicate(values, Predicate("t", "c", "between", 3, 5))
    assert values[got].tolist() == [3, 4, 5]


def test_conjunction_of_filters():
    t = Table("t")
    t.add("a", np.arange(20))
    t.add("b", np.arange(20) % 3)
    mask = scan_mask(t, [Predicate("t", "a", "<", 10),
                         Predicate("t", "b", "=", 0)])
    assert np.flatnonzero(mask).tolist() == [0, 3, 6, 9]


def test_plan_output_matches_a_manual_join():
    db = star_schema(seed=2, fact_rows=3000, dims=2, dim_rows=100, skew=0.5)
    stats = analyze_db(db)
    q = Query(["fact", "dim0"],
              [JoinCond("fact", "fk0", "dim0", "id")],
              [Predicate("dim0", "a", "<", 50.0)])
    plan = dp_bushy(q, Costing(stats)).plan
    rel, _ = Executor(db).run(plan)

    keep = db["dim0"].col("a") < 50
    allowed = set(db["dim0"].col("id")[keep].tolist())
    expected = sum(1 for k in db["fact"].col("fk0").tolist() if k in allowed)
    assert rel.n == expected


def test_true_rows_recorded_at_every_node():
    db = star_schema(seed=4, fact_rows=5000, dims=3, dim_rows=200, skew=0.5)
    stats = analyze_db(db)
    q = Query(["fact", "dim0", "dim1", "dim2"],
              [JoinCond("fact", f"fk{d}", f"dim{d}", "id") for d in range(3)],
              [Predicate("dim1", "a", "<", 40.0)])
    plan = dp_bushy(q, Costing(stats)).plan
    Executor(db).run(plan)
    for node in plan.walk():
        assert node.true_rows >= 0


def test_join_order_does_not_change_the_result():
    """Every plan for a query returns the same number of rows. If this fails,
    comparing plans by runtime is comparing different computations."""
    db = mesh_schema(seed=6, tables=4, rows=300, groups=40, skew=0.3)
    stats = analyze_db(db)
    q = Query([f"m{i}" for i in range(4)],
              [JoinCond(f"m{i-1}", "g", f"m{i}", "g") for i in range(1, 4)],
              [Predicate("m0", "a", "<", 60.0)])

    from qp.enumerate import all_plans, dp_linear, greedy
    ex = Executor(db)
    counts = set()
    for plan in all_plans(q, Costing(stats))[:24]:
        rel, _ = ex.run(plan, record=False)
        counts.add(rel.n)
    for algo in (dp_bushy, dp_linear, greedy):
        rel, _ = ex.run(algo(q, Costing(stats)).plan, record=False)
        counts.add(rel.n)
    assert len(counts) == 1, counts
