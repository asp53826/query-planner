# Design

## The question this repository is built to answer

Leis et al., *How Good Are Query Optimizers, Really?* (VLDB 2015), took the join
ordering apart and found that the search was essentially solved and the
cardinality estimates were off by orders of magnitude - and that it was the
estimates, not the search, that decided whether a plan was any good.

That result is only reproducible if the two can be measured separately, and
separating them needs three things a normal optimizer does not have:

1. **an executor**, so a plan has a runtime and not just a score;
2. **true cardinalities**, so the estimate has a denominator;
3. **an oracle plan** — the plan the same search would pick given perfect
   information — so "the estimate was wrong" can be converted into "the estimate
   was wrong and it cost 1.8x".

`Costing(stats, truth=Truth(db))` is the third. It runs the same dynamic program
with exact cardinalities measured by actually executing subplans. No real
optimizer can do that; it is expensive and it is the point.

## The rule that makes the numbers mean anything

**An estimate may only use what `ANALYZE` could have computed.** Per-column
histograms, a most-common-value list, an approximate distinct count. Nothing
else. No peeking at the data at estimation time, no computing the exact answer
and calling it an estimate.

That sounds too obvious to state, and it is the easiest way to accidentally
build an optimizer that reports a q-error of 1.0 and has measured nothing. The
distinct counts come from a HyperLogLog sketch rather than an exact count for
the same reason: the sketch's 1-2% error is a real error a real system has, and
removing it would flatter every join estimate downstream of it.

## What is estimated, and the assumption each one makes

| | formula | assumes |
|---|---|---|
| range predicate | equi-depth histogram, linear inside the bucket | uniformity within a bucket |
| equality predicate | MCV frequency, else remaining mass / remaining distinct | the non-heavy tail is uniform |
| conjunction | selectivities multiply | the columns are independent |
| join | `\|R\|\|S\| / max(d(R.a), d(S.b))` | containment and uniformity of the key domains |
| distinct after filter | `d * (1 - (1-f)^(n/d))` | values are spread evenly across groups |

Each row is a place the estimate can be wrong. `bench/qerror.py` attacks the
third and fourth directly by dialling correlation and skew, which is why the
data generator has those knobs instead of using a fixed benchmark dataset: on
real data you can see that the estimate is wrong but not *which* assumption
broke.

## Why q-error and not relative error

q-error is `max(est/act, act/est)`. It is symmetric and unbounded in both
directions.

Relative error is neither. Under-estimating a 1,000,000-row intermediate as 1,000
and over-estimating a 1,000-row intermediate as 1,000,000 are equally
destructive to a plan — one causes a hash table to spill, the other causes the
optimizer to avoid a join that would have been cheap — and relative error scores
the first as 0.999 and the second as 999. An optimizer tuned to minimise mean
relative error would systematically prefer under-estimating, which is the
failure mode that actually kills queries.

Aggregates are geometric, because the quantity is multiplicative. The arithmetic
mean of a ratio distribution is whatever the largest over-estimate was.

**One trap, recorded because it bit:** `q_error` clamps both arguments up to 1,
because an intermediate of zero rows is a real outcome but an undefined ratio.
That means passing it two *selectivities* — both less than 1 — returns exactly
1.0 no matter how wrong the estimate is. Three tests asserted on selectivities
and passed vacuously until one of them was expected to fail and did not.

## The cost model

Six constants, each fitted by timing the corresponding operator in isolation
against the executor in this repository. Absolute values are meaningless; what
matters is whether the ordering the model induces over plans matches the
ordering the clock induces, which is why `bench/cost_vs_time.py` reports a
Spearman rank correlation and not an error.

Building the smaller input is not an optimization the model is free to assume:
the executor does it, so the model has to cost it that way or it is costing a
plan nobody runs.

## The search

Three algorithms, so search quality and estimate quality can be varied
independently:

- **`dp_bushy`** — Selinger-style DP over connected subsets. Finds the minimum
  of the space it searches, which `test_dp_finds_the_cheapest_plan_in_the_space`
  confirms against exhaustive enumeration. With `truth=` it is the oracle.
- **`dp_linear`** — the same DP restricted to left-deep plans, which is what a
  lot of production optimizers actually search.
- **`greedy`** — repeatedly join the pair with the smallest estimated result.

Cross products are never enumerated. Allowing disconnected pairs multiplies the
search space and produces plans no executor wants; the standard restriction is
to skip them unless the query graph forces one, and these workloads never do.

## Why the executor is vectorised

Hash join is a sort-merge over hashed keys, not a Python dict. A dict keyed on a
million numpy scalars spends essentially all of its time in the interpreter, and
the resulting timings would be a measurement of CPython rather than of the join.
With numpy doing the sort and the binary search, an operator's runtime tracks the
number of tuples it touches — which is the only way a cost model calibrated in
tuples can be compared against a wall clock at all.

That is also the honest caveat: this is a columnar in-memory executor with no
disk, no buffer pool, no spilling and no parallelism. A cost model that
correlates well here would not necessarily correlate well against a system that
can run out of memory.

## The noise floor

`bench/optimality.py` times the oracle plan against *itself* and reports the
resulting ratio distribution alongside the real comparisons.

This is not decoration. The first run of the study produced a geomean of 0.87
for "DP with estimates versus the oracle" on a workload where the two picked
identical plans 6 times out of 6 — a plan cannot be 13% faster than itself, so
that number was entirely clock noise. Without the floor printed next to it, it
reads as "the estimates helped".

## What is not modelled

- **No disk, no buffer pool, no spilling.** Everything is in memory. The cost
  model has no I/O term because there is no I/O.
- **No parallelism.** One thread, so no exchange operators and no
  partitioning decisions.
- **No sort-merge join, no index nested loop, no indexes at all.** Physical
  operator choice is hash join versus nested loop and nothing else.
- **No subqueries, no aggregation, no outer joins.** Select-project-join only.
- **No multi-column statistics.** Deliberately: the independence assumption is
  the thing being measured, and fixing it would remove the measurement. The
  test that pins it is written so that implementing multi-column stats makes it
  fail, which is the correct signal.
- **No sampling-based or learned estimation.** Both are the obvious next step
  and both would need this harness to evaluate.
