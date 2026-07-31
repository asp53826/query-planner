"""Search tests.

The load-bearing one is `test_dp_finds_the_cheapest_plan_in_the_space`. The
whole optimality study rests on the claim that DP with true cardinalities finds
the best plan the search can reach; if DP is not actually finding the minimum,
the "oracle" is not an oracle and every ratio measured against it is wrong.
"""

import numpy as np
import pytest

from qp.data import mesh_schema, star_schema
from qp.enumerate import Query, all_plans, dp_bushy, dp_linear, greedy
from qp.estimate import JoinCond, Predicate
from qp.plan import Costing, HashJoin, NestedLoop, Scan
from qp.stats import analyze_db
from qp.workload import mesh_query, star_query


def star_setup(dims=3, seed=1):
    db = star_schema(seed=seed, fact_rows=20_000, dims=dims, dim_rows=400,
                     skew=0.6)
    return db, analyze_db(db)


def star_q(dims=3):
    return Query(["fact"] + [f"dim{d}" for d in range(dims)],
                 [JoinCond("fact", f"fk{d}", f"dim{d}", "id")
                  for d in range(dims)],
                 [Predicate("dim0", "a", "<", 30.0)])


@pytest.mark.parametrize("algo", [dp_bushy, dp_linear, greedy])
def test_every_algorithm_covers_every_table(algo):
    db, stats = star_setup()
    q = star_q()
    plan = algo(q, Costing(stats)).plan
    assert plan.relations() == frozenset(q.tables)


@pytest.mark.parametrize("algo", [dp_bushy, dp_linear, greedy])
def test_no_relation_appears_twice(algo):
    db, stats = star_setup(dims=4)
    q = star_q(4)
    plan = algo(q, Costing(stats)).plan
    scans = [n.table for n in plan.walk() if isinstance(n, Scan)]
    assert sorted(scans) == sorted(q.tables)


def test_dp_finds_the_cheapest_plan_in_the_space():
    db, stats = star_setup(dims=4)
    q = star_q(4)
    costing = Costing(stats)
    best = dp_bushy(q, costing).plan
    every = all_plans(q, costing)
    assert every
    assert best.cost <= min(p.cost for p in every) + 1e-6


def test_dp_is_at_least_as_good_as_greedy():
    for seed in range(6):
        db, stats = star_setup(dims=4, seed=seed)
        rng = np.random.default_rng(seed)
        q = star_query(rng, 4, n_filters=2)
        costing = Costing(stats)
        assert dp_bushy(q, costing).plan.cost <= greedy(q, costing).plan.cost + 1e-6


def test_bushy_is_at_least_as_good_as_left_deep():
    """Left-deep is a subset of the bushy space, so this can only fail if the
    DP is broken."""
    for seed in range(6):
        db, stats = star_setup(dims=4, seed=seed)
        rng = np.random.default_rng(seed + 50)
        q = star_query(rng, 4, n_filters=2)
        costing = Costing(stats)
        assert dp_bushy(q, costing).plan.cost <= \
            dp_linear(q, costing).plan.cost + 1e-6


def test_left_deep_plans_are_actually_left_deep():
    db, stats = star_setup(dims=4)
    plan = dp_linear(star_q(4), Costing(stats)).plan
    for node in plan.walk():
        if isinstance(node, (HashJoin, NestedLoop)):
            assert isinstance(node.right, Scan), "right input is not a base table"


def test_search_effort_grows_with_the_query():
    db, stats = star_setup(dims=5)
    costing = Costing(stats)
    small = dp_bushy(star_q(3), costing).considered
    large = dp_bushy(star_q(5), costing).considered
    assert large > small


def test_cross_products_are_not_enumerated():
    """Two tables with no edge between them must never be joined directly."""
    db, stats = star_setup(dims=3)
    q = star_q(3)
    plan = dp_bushy(q, Costing(stats)).plan
    for node in plan.walk():
        if isinstance(node, (HashJoin, NestedLoop)):
            assert q.edges_between(node.left.relations(), node.right.relations())


def test_connectivity_is_computed_correctly():
    q = star_q(3)
    assert q.connected(frozenset(["fact", "dim0"]))
    assert q.connected(frozenset(["fact", "dim0", "dim1"]))
    assert not q.connected(frozenset(["dim0", "dim1"]))
    assert q.connected(frozenset(["fact"]))


def test_chain_connectivity():
    q = mesh_query(np.random.default_rng(0), 4)
    assert q.connected(frozenset(["m0", "m1", "m2"]))
    assert not q.connected(frozenset(["m0", "m2"]))


def test_estimates_are_never_zero():
    """A zero row estimate makes every plan above it cost the same, which is
    the worst possible time for the optimizer to stop being able to choose."""
    db = mesh_schema(seed=8, tables=4, rows=200, groups=50)
    stats = analyze_db(db)
    rng = np.random.default_rng(3)
    for _ in range(10):
        q = mesh_query(rng, 4, n_filters=2)
        plan = dp_bushy(q, Costing(stats)).plan
        for node in plan.walk():
            assert node.rows >= 1.0


def test_costs_increase_up_the_tree():
    db, stats = star_setup(dims=4)
    plan = dp_bushy(star_q(4), Costing(stats)).plan
    for node in plan.walk():
        for child in node.children():
            assert node.cost >= child.cost
