"""How wrong the estimates are, and what makes them wrong.

q-error is max(est/act, act/est). It is the standard metric because it is
symmetric and unbounded in both directions: under-estimating by 1000x and
over-estimating by 1000x are equally destructive to a plan, and relative error
scores one of them as 0.999 and the other as 999.

Two sweeps.

Correlation, which attacks the independence assumption directly. Each dimension
table has a column `b` that agrees with `a` with a controlled probability, and
the query filters on both. At correlation 0 the textbook formula is right; at
correlation 1 the second predicate removes nothing and the estimator still
multiplies by its selectivity.

Join depth, which is the compounding result. An estimate at depth k is built
from an estimate at depth k-1, so the errors multiply rather than average out.
This is why the geometric mean is reported per level and not just overall.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qp.data import mesh_schema, star_schema                     # noqa: E402
from qp.enumerate import dp_bushy                                # noqa: E402
from qp.execute import Executor                                  # noqa: E402
from qp.plan import Costing, Scan                                # noqa: E402
from qp.stats import analyze_db                                  # noqa: E402
from qp.workload import (mesh_query, q_error, star_query,        # noqa: E402
                         summarize_q_errors)


def depth_of(node, root_depth=0):
    """Number of joins below a node. Scans are depth 0."""
    if isinstance(node, Scan):
        return 0
    return 1 + max(depth_of(c) for c in node.children())


def collect(plan):
    """(depth, estimated, actual) for every node that was executed."""
    out = []
    for n in plan.walk():
        if n.true_rows >= 0:
            out.append((depth_of(n), n.rows, n.true_rows))
    return out


def sweep_correlation(args):
    print("Independence assumption under correlated predicates")
    print("Each query filters dim.a and dim.b; b agrees with a at the given rate.\n")
    header = (f"{'correlation':>12} {'queries':>8} {'nodes':>7} {'geomean':>9} "
              f"{'median':>8} {'p95':>10} {'max':>12}")
    print(header)
    print("-" * len(header))

    rows = []
    for corr in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]:
        db = star_schema(seed=3, fact_rows=args.fact_rows, dims=args.dims,
                         dim_rows=args.dim_rows, skew=args.skew,
                         correlation=corr)
        stats = analyze_db(db)
        costing = Costing(stats)
        ex = Executor(db)

        errors = []
        rng = np.random.default_rng(100)
        for _ in range(args.queries):
            q = star_query(rng, args.dims, n_filters=2, correlated_pairs=True)
            plan = dp_bushy(q, costing).plan
            ex.run(plan)
            errors += [q_error(e, a) for _, e, a in collect(plan)]

        s = summarize_q_errors(errors)
        rows.append({"correlation": corr, **s})
        print(f"{corr:>12.2f} {args.queries:>8} {s['n']:>7} {s['geomean']:>9.2f} "
              f"{s['median']:>8.2f} {s['p95']:>10.2f} {s['max']:>12.1f}")
    return rows


def _depth_table(title, note, db, make_query, n_tables, queries, seed):
    print(f"\n\n{title}")
    print(note + "\n")
    header = (f"{'join depth':>11} {'nodes':>7} {'est rows':>12} "
              f"{'act rows':>12} {'geomean':>9} {'p95':>10} {'max':>12}")
    print(header)
    print("-" * len(header))

    stats = analyze_db(db)
    costing = Costing(stats)
    ex = Executor(db)

    by_depth: dict[int, list] = {}
    rng = np.random.default_rng(seed)
    for _ in range(queries):
        q = make_query(rng, n_tables)
        plan = dp_bushy(q, costing).plan
        ex.run(plan)
        for d, e, a in collect(plan):
            by_depth.setdefault(d, []).append((q_error(e, a), e, a))

    rows = []
    for d in sorted(by_depth):
        errs = [x[0] for x in by_depth[d]]
        est = float(np.mean([x[1] for x in by_depth[d]]))
        act = float(np.mean([x[2] for x in by_depth[d]]))
        s = summarize_q_errors(errs)
        rows.append({"depth": d, "mean_est": est, "mean_act": act, **s})
        print(f"{d:>11} {s['n']:>7} {est:>12,.0f} {act:>12,.0f} "
              f"{s['geomean']:>9.2f} {s['p95']:>10.2f} {s['max']:>12.1f}")
    return rows


def sweep_depth(args):
    star = _depth_table(
        "Error compounding with join depth: star schema (key joins)",
        "Every join lands on a dimension's primary key, so the fact table\n"
        "bounds the result and an error at one level is not multiplied at the\n"
        "next. This is the control, and it barely compounds.",
        star_schema(seed=5, fact_rows=args.fact_rows, dims=args.dims,
                    dim_rows=args.dim_rows, skew=args.skew, correlation=0.0),
        lambda rng, n: star_query(rng, n, n_filters=2), args.dims,
        args.queries * 2, 200)

    mesh = _depth_table(
        "Error compounding with join depth: chain of many-to-many joins",
        "m0.g = m1.g = m2.g ... on a column with 100 distinct values. Each\n"
        "level multiplies the row count and the error along with it. This is\n"
        "the shape classical estimation is known to fail on.",
        mesh_schema(seed=11, tables=args.mesh_tables, rows=args.mesh_rows,
                    groups=args.mesh_groups, skew=0.4),
        lambda rng, n: mesh_query(rng, n, n_filters=1), args.mesh_tables,
        args.queries, 201)

    return {"star": star, "mesh": mesh}


def sweep_skew(args):
    print("\n\nSkew in the join key")
    print("Zipf exponent on the fact table's foreign keys. The System R join "
          "formula\nassumes uniformity within the key domain.\n")
    header = (f"{'zipf skew':>10} {'nodes':>7} {'geomean':>9} {'p95':>10} "
              f"{'max':>12}")
    print(header)
    print("-" * len(header))

    rows = []
    for skew in [0.0, 0.4, 0.8, 1.2]:
        db = star_schema(seed=7, fact_rows=args.fact_rows, dims=args.dims,
                         dim_rows=args.dim_rows, skew=skew, correlation=0.0)
        stats = analyze_db(db)
        costing = Costing(stats)
        ex = Executor(db)
        errors = []
        rng = np.random.default_rng(300)
        for _ in range(args.queries):
            q = star_query(rng, args.dims, n_filters=2)
            plan = dp_bushy(q, costing).plan
            ex.run(plan)
            errors += [q_error(e, a) for _, e, a in collect(plan)]
        s = summarize_q_errors(errors)
        rows.append({"skew": skew, **s})
        print(f"{skew:>10.1f} {s['n']:>7} {s['geomean']:>9.2f} "
              f"{s['p95']:>10.2f} {s['max']:>12.1f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--dims", type=int, default=4)
    ap.add_argument("--fact-rows", type=int, default=200_000)
    ap.add_argument("--dim-rows", type=int, default=2_000)
    ap.add_argument("--skew", type=float, default=0.8)
    ap.add_argument("--mesh-tables", type=int, default=5)
    ap.add_argument("--mesh-rows", type=int, default=900)
    ap.add_argument("--mesh-groups", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if args.quick:
        args.queries = 8
        args.fact_rows = 50_000
        args.mesh_tables = 4
        args.mesh_rows = 500

    out = {
        "correlation": sweep_correlation(args),
        "depth": sweep_depth(args),
        "skew": sweep_skew(args),
    }
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
