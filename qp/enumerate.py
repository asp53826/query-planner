"""Join order search.

Three algorithms, because the point of the repository is to separate search
quality from estimate quality and that needs a spread:

  dp_bushy     Selinger-style dynamic programming over connected subsets. Finds
               the cheapest plan *according to the cost model it is given*. With
               true cardinalities this is the oracle.
  dp_linear    the same DP restricted to left-deep plans, which is what many
               production optimizers actually search.
  greedy       repeatedly join the pair with the smallest estimated result.
               Cheap, and the thing DP has to beat.

Only connected subsets are considered. Allowing a disconnected pair means
enumerating cross products, which multiplies the search space and produces plans
no sane executor wants; the standard restriction is to skip them unless the
query graph forces one, and these workloads never do.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .estimate import JoinCond, Predicate
from .plan import Costing, Node


@dataclass
class Query:
    tables: list[str]
    joins: list[JoinCond]
    preds: list[Predicate]

    def preds_for(self, table: str) -> list[Predicate]:
        return [p for p in self.preds if p.table == table]

    def edges_between(self, a: frozenset[str],
                      b: frozenset[str]) -> list[JoinCond]:
        out = []
        for j in self.joins:
            if (j.left_table in a and j.right_table in b) or \
               (j.left_table in b and j.right_table in a):
                out.append(j)
        return out

    def neighbours(self, s: frozenset[str]) -> set[str]:
        out = set()
        for j in self.joins:
            if j.left_table in s and j.right_table not in s:
                out.add(j.right_table)
            elif j.right_table in s and j.left_table not in s:
                out.add(j.left_table)
        return out

    def connected(self, s: frozenset[str]) -> bool:
        if not s:
            return False
        seen = {next(iter(s))}
        frontier = [next(iter(s))]
        while frontier:
            cur = frontier.pop()
            for j in self.joins:
                nxt = None
                if j.left_table == cur and j.right_table in s:
                    nxt = j.right_table
                elif j.right_table == cur and j.left_table in s:
                    nxt = j.left_table
                if nxt and nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        return seen == set(s)


@dataclass
class SearchResult:
    plan: Node
    considered: int
    """Number of (subset, split) pairs costed. This is the search-effort number
    and it is what grows factorially, so it is reported next to plan quality
    rather than hidden."""


def _scans(q: Query, costing: Costing) -> dict[frozenset[str], Node]:
    return {frozenset([t]): costing.scan(t, q.preds_for(t)) for t in q.tables}


def dp_bushy(q: Query, costing: Costing) -> SearchResult:
    best: dict[frozenset[str], Node] = _scans(q, costing)
    considered = 0
    tables = list(q.tables)

    for size in range(2, len(tables) + 1):
        for combo in itertools.combinations(tables, size):
            s = frozenset(combo)
            if not q.connected(s):
                continue
            winner = None
            # Split into two non-empty halves. Iterating over proper subsets of
            # s that contain a fixed element visits each unordered split once.
            fixed = next(iter(s))
            rest = sorted(s - {fixed})
            for k in range(len(rest) + 1):
                for sub in itertools.combinations(rest, k):
                    a = frozenset(sub) | {fixed}
                    b = s - a
                    if not b:
                        continue
                    if a not in best or b not in best:
                        continue
                    edges = q.edges_between(a, b)
                    if not edges:
                        continue
                    considered += 1
                    cand = costing.best_join(best[a], best[b], edges[0])
                    if winner is None or cand.cost < winner.cost:
                        winner = cand
            if winner is not None:
                best[s] = winner

    full = frozenset(q.tables)
    return SearchResult(best[full], considered)


def dp_linear(q: Query, costing: Costing) -> SearchResult:
    """Left-deep only: the right input of every join is a base table."""
    best: dict[frozenset[str], Node] = _scans(q, costing)
    considered = 0
    tables = list(q.tables)

    for size in range(2, len(tables) + 1):
        for combo in itertools.combinations(tables, size):
            s = frozenset(combo)
            if not q.connected(s):
                continue
            winner = None
            for t in s:
                a = s - {t}
                if a not in best or not q.connected(a):
                    continue
                edges = q.edges_between(a, frozenset([t]))
                if not edges:
                    continue
                considered += 1
                cand = costing.best_join(best[a], best[frozenset([t])], edges[0])
                if winner is None or cand.cost < winner.cost:
                    winner = cand
            if winner is not None:
                best[s] = winner

    return SearchResult(best[frozenset(q.tables)], considered)


def greedy(q: Query, costing: Costing) -> SearchResult:
    """Join the pair producing the smallest estimated result, repeatedly.

    The classic cheap heuristic. It is not obviously bad - minimising the
    intermediate result is usually the right instinct - and how often it lands
    on the DP plan is one of the numbers this repository reports.
    """
    parts: list[Node] = list(_scans(q, costing).values())
    considered = 0

    while len(parts) > 1:
        best = None
        best_pair = None
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                edges = q.edges_between(parts[i].relations(), parts[j].relations())
                if not edges:
                    continue
                considered += 1
                cand = costing.best_join(parts[i], parts[j], edges[0])
                if best is None or cand.rows < best.rows:
                    best, best_pair = cand, (i, j)
        if best is None:
            # disconnected remainder; take any pair so the loop terminates
            best = costing.best_join(parts[0], parts[1], q.joins[0])
            best_pair = (0, 1)
        i, j = best_pair
        parts = [p for k, p in enumerate(parts) if k not in (i, j)] + [best]

    return SearchResult(parts[0], considered)


def all_plans(q: Query, costing: Costing, limit: int = 20_000) -> list[Node]:
    """Every connected bushy plan, for small queries.

    Grows faster than factorially, so it is capped and the cap is reported
    rather than silently truncating. Used by bench/cost_vs_time.py, which needs
    a population of plans to correlate predicted cost against measured runtime,
    and by the tests, which need to confirm that DP actually finds the minimum
    of the space it claims to search.
    """
    scans = _scans(q, costing)
    memo: dict[frozenset[str], list[Node]] = {k: [v] for k, v in scans.items()}
    tables = list(q.tables)

    for size in range(2, len(tables) + 1):
        for combo in itertools.combinations(tables, size):
            s = frozenset(combo)
            if not q.connected(s):
                continue
            options: list[Node] = []
            fixed = next(iter(s))
            rest = sorted(s - {fixed})
            for k in range(len(rest) + 1):
                for sub in itertools.combinations(rest, k):
                    a = frozenset(sub) | {fixed}
                    b = s - a
                    if not b or a not in memo or b not in memo:
                        continue
                    edges = q.edges_between(a, b)
                    if not edges:
                        continue
                    for left in memo[a]:
                        for right in memo[b]:
                            if len(options) >= limit:
                                return options
                            options.append(
                                costing.hash_join(left, right, edges[0]))
            if options:
                memo[s] = options

    return memo.get(frozenset(tables), [])


ALGORITHMS = {
    "dp_bushy": dp_bushy,
    "dp_linear": dp_linear,
    "greedy": greedy,
}
