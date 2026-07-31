"""How much the bad estimates actually cost, in seconds.

A q-error is a number about a number. This is the study that turns it into a
statement about runtime, and it works by separating the two things an optimizer
can get wrong:

  search    given perfect cardinalities, does the enumeration find the cheapest
            plan? Run the same DP with `Costing(truth=Truth(db))`. That plan is
            the best this search can do with perfect information.
  estimates given the search is fixed, how much worse is the plan chosen from
            estimated cardinalities?

The headline number is `runtime(chosen) / runtime(oracle)`, over many queries.
A ratio of 1.0 means the estimates were wrong but the plan came out the same,
which happens more often than the q-errors suggest and is the single most
under-reported fact about query optimization: plans are robust to a lot of
estimation error, right up until they are not.

Both plans are executed, alternating and repeated, because the first run of
anything in numpy pays for page faults the second does not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qp.data import mesh_schema, star_schema                     # noqa: E402
from qp.enumerate import all_plans, dp_bushy, dp_linear, greedy  # noqa: E402
from qp.execute import Executor, Truth                           # noqa: E402
from qp.plan import Costing                                      # noqa: E402
from qp.stats import analyze_db                                  # noqa: E402
from qp.workload import mesh_query, star_query                   # noqa: E402


def timed(ex: Executor, plan, reps: int = 3) -> float:
    best = float("inf")
    for _ in range(reps):
        _, st = ex.run(plan, record=False)
        best = min(best, st.seconds)
    return best


def plan_signature(node) -> str:
    from qp.plan import Scan
    if isinstance(node, Scan):
        return node.table
    return f"({plan_signature(node.left)}*{plan_signature(node.right)})"


def study(name, db, make_query, n_tables, queries, seed, exhaustive_upto=0):
    stats = analyze_db(db)
    est_costing = Costing(stats)
    ex = Executor(db)
    rng = np.random.default_rng(seed)

    ratios = []
    identical = 0
    greedy_ratios = []
    linear_ratios = []
    noise = []
    oracle_is_global = 0
    oracle_checked = 0

    for _ in range(queries):
        q = make_query(rng, n_tables)

        truth_costing = Costing(stats, truth=Truth(db))
        oracle = dp_bushy(q, truth_costing).plan
        chosen = dp_bushy(q, est_costing).plan
        g = greedy(q, est_costing).plan
        lin = dp_linear(q, est_costing).plan

        t_oracle = timed(ex, oracle)
        t_chosen = timed(ex, chosen)
        t_greedy = timed(ex, g)
        t_linear = timed(ex, lin)

        # The noise floor: the same plan, timed again. Any ratio inside this
        # band is the clock, not the optimizer. Without it a geomean of 0.94
        # reads as "the estimates helped", which is nonsense.
        noise.append(timed(ex, oracle) / max(t_oracle, 1e-9))

        ratios.append(t_chosen / max(t_oracle, 1e-9))
        greedy_ratios.append(t_greedy / max(t_oracle, 1e-9))
        linear_ratios.append(t_linear / max(t_oracle, 1e-9))
        if plan_signature(chosen) == plan_signature(oracle):
            identical += 1

        if exhaustive_upto and n_tables <= exhaustive_upto:
            # Confirm the DP really does find the minimum of the space it
            # searches. If this ever fails the search is broken and every other
            # number in this file is measuring the wrong thing.
            every = all_plans(q, truth_costing)
            if every:
                oracle_checked += 1
                if abs(min(p.cost for p in every) - oracle.cost) < 1e-6:
                    oracle_is_global += 1

    def summarise(vals):
        a = np.asarray(vals)
        return {
            "geomean": float(np.exp(np.log(np.maximum(a, 1e-9)).mean())),
            "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)),
            "max": float(a.max()),
        }

    out = {
        "name": name,
        "queries": queries,
        "same_plan_as_oracle": identical,
        "dp_vs_oracle": summarise(ratios),
        "greedy_vs_oracle": summarise(greedy_ratios),
        "left_deep_vs_oracle": summarise(linear_ratios),
        "noise_floor": summarise(noise),
    }
    if oracle_checked:
        out["dp_matched_exhaustive"] = f"{oracle_is_global}/{oracle_checked}"

    print(f"\n{name}  ({queries} queries, {n_tables} tables)")
    header = (f"  {'plan chosen by':<34} {'geomean':>9} {'median':>8} "
              f"{'p90':>8} {'worst':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, key in (("DP + estimated cardinalities", "dp_vs_oracle"),
                       ("left-deep DP + estimates", "left_deep_vs_oracle"),
                       ("greedy + estimates", "greedy_vs_oracle"),
                       ("[noise floor: oracle vs itself]", "noise_floor")):
        s = out[key]
        print(f"  {label:<34} {s['geomean']:>9.2f} {s['median']:>8.2f} "
              f"{s['p90']:>8.2f} {s['max']:>8.2f}")
    print(f"\n  identical to the oracle plan:   {identical}/{queries}")
    if oracle_checked:
        print(f"  DP found the global optimum:    {oracle_is_global}/{oracle_checked}"
              f"  (checked against exhaustive enumeration)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=25)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    queries = 6 if args.quick else args.queries
    fact = 40_000 if args.quick else 120_000

    print("Runtime relative to the plan chosen with perfect cardinalities.")
    print("1.00 means the estimation error cost nothing. Higher is what it cost.")

    results = []
    results.append(study(
        "star schema, 4 dimensions, uncorrelated",
        star_schema(seed=21, fact_rows=fact, dims=4, dim_rows=2000,
                    skew=0.8, correlation=0.0),
        lambda rng, n: star_query(rng, n, n_filters=3), 4, queries, 500,
        exhaustive_upto=5))

    results.append(study(
        "star schema, 4 dimensions, correlated predicates",
        star_schema(seed=22, fact_rows=fact, dims=4, dim_rows=2000,
                    skew=0.8, correlation=1.0),
        lambda rng, n: star_query(rng, n, n_filters=3, correlated_pairs=True),
        4, queries, 501, exhaustive_upto=5))

    results.append(study(
        "chain of many-to-many joins",
        mesh_schema(seed=23, tables=4, rows=700, groups=90, skew=0.4),
        lambda rng, n: mesh_query(rng, n, n_filters=1), 4, queries, 502,
        exhaustive_upto=5))

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
