"""Does the cost model predict runtime?

This is the question cost models are rarely asked. A cost model is fitted once,
against a machine that no longer exists, and thereafter treated as ground truth
because it is the thing that decides which plan runs. Comparing it to a clock is
usually embarrassing, which is a good reason to do it.

The right metric is a rank correlation, not an error. Nobody cares whether the
model says 21,152 and the truth is 31ms; what matters is whether the model
orders plans the same way the clock does, because ordering is all the optimizer
uses the number for. So: Spearman's rho over every plan for a query, plus the
two questions that actually decide plan quality:

  where does the model's favourite plan rank by actual runtime?
  how much slower is it than the genuinely fastest plan?

A model can have a mediocre rho and still be fine if its top pick is good, and
a model can have a decent rho and still be useless if its top pick is terrible.
Both are reported.

Costs here are computed from *true* cardinalities, so this measures the cost
model alone. Feeding it estimates as well would confound two different errors,
and bench/optimality.py already measures the combination.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qp.data import mesh_schema, star_schema           # noqa: E402
from qp.enumerate import all_plans                     # noqa: E402
from qp.execute import Executor, Truth                 # noqa: E402
from qp.plan import Costing                            # noqa: E402
from qp.stats import analyze_db                        # noqa: E402
from qp.workload import mesh_query, star_query         # noqa: E402


def spearman(a, b) -> float:
    """Rank correlation, written out to keep the dependency list at numpy.

    Ties get average ranks, which matters here because several plans in a
    family can cost exactly the same.
    """
    def rank(x):
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x, kind="stable")
        ranks = np.empty(len(x), dtype=np.float64)
        ranks[order] = np.arange(len(x), dtype=np.float64)
        # average the ranks of tied values
        _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inverse, ranks)
        return (sums / counts)[inverse]

    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def timed(ex, plan, reps=3) -> float:
    best = float("inf")
    for _ in range(reps):
        _, st = ex.run(plan, record=False)
        best = min(best, st.seconds)
    return best


def study(name, db, make_query, n_tables, queries, seed, max_plans=60):
    stats = analyze_db(db)
    ex = Executor(db)
    rng = np.random.default_rng(seed)

    rhos = []
    pick_ranks = []
    pick_penalty = []
    total_plans = 0
    dropped = 0

    for _ in range(queries):
        q = make_query(rng, n_tables)
        costing = Costing(stats, truth=Truth(db))
        plans = all_plans(q, costing)
        if len(plans) < 4:
            continue
        if len(plans) > max_plans:
            dropped += len(plans) - max_plans
            idx = rng.choice(len(plans), size=max_plans, replace=False)
            plans = [plans[i] for i in idx]
        total_plans += len(plans)

        costs = np.array([p.cost for p in plans])
        times = np.array([timed(ex, p) for p in plans])

        rhos.append(spearman(costs, times))

        cheapest = int(np.argmin(costs))
        fastest_time = float(times.min())
        # 0.0 means the model's pick is the fastest plan; 1.0 the slowest.
        rank_of_pick = float((times < times[cheapest]).sum()) / (len(times) - 1)
        pick_ranks.append(rank_of_pick)
        pick_penalty.append(times[cheapest] / max(fastest_time, 1e-9))

    out = {
        "name": name,
        "queries": len(rhos),
        "plans_scored": total_plans,
        "plans_sampled_away": dropped,
        "spearman_mean": float(np.mean(rhos)),
        "spearman_min": float(np.min(rhos)),
        "pick_percentile_mean": float(np.mean(pick_ranks)),
        "pick_penalty_geomean": float(np.exp(np.log(pick_penalty).mean())),
        "pick_penalty_max": float(np.max(pick_penalty)),
        "pick_was_fastest": int(sum(1 for r in pick_ranks if r == 0.0)),
    }

    print(f"\n{name}")
    print(f"  queries                              {out['queries']}")
    print(f"  plans costed and timed               {out['plans_scored']}")
    if dropped:
        print(f"  plans sampled away (cap {max_plans})        {dropped}")
    print(f"  Spearman rho, cost vs runtime        {out['spearman_mean']:.3f}"
          f"   (worst query {out['spearman_min']:.3f})")
    print(f"  model's pick was the fastest plan    "
          f"{out['pick_was_fastest']}/{out['queries']}")
    print(f"  model's pick, mean percentile        "
          f"{out['pick_percentile_mean'] * 100:.0f}th")
    print(f"  slowdown vs the fastest plan         "
          f"{out['pick_penalty_geomean']:.2f}x geomean, "
          f"{out['pick_penalty_max']:.2f}x worst")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=15)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    queries = 4 if args.quick else args.queries
    fact = 40_000 if args.quick else 120_000

    print("Cost model fidelity: predicted cost against measured runtime.")
    print("Costs use TRUE cardinalities, so this isolates the model from the "
          "estimator.")

    results = [
        study("star schema, 4 dimensions",
              star_schema(seed=31, fact_rows=fact, dims=4, dim_rows=2000,
                          skew=0.8, correlation=0.0),
              lambda rng, n: star_query(rng, n, n_filters=3), 4, queries, 900),
        study("chain of many-to-many joins",
              mesh_schema(seed=32, tables=4, rows=700, groups=90, skew=0.4),
              lambda rng, n: mesh_query(rng, n, n_filters=1), 4, queries, 901),
    ]

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
