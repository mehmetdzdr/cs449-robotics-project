"""Minimal set of blockers to remove (#100): which boxes to take away, in which order, so that the target box has a
free grasp axis.

A removal is only possible if the box itself has a free grasp axis at that moment (`grasp_access`) and can be
reached; removing it may first need other removals. The plan with the lowest total cost (by default: the fewest
removals) is found by a cheapest-first search over the sets of removed boxes (at most 2^n states, n = number of
neighbours: 8 in a 3 x 3 cluster). Ties go to the plan found first, with removals tried in the order of `boxes`.
"""
import heapq
from dataclasses import dataclass
from typing import Callable

import numpy as np

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.planning.access import grasp_access


@dataclass
class Removal:
    box: int            # index into `boxes`
    axis: str           # grasp axis free for it at that moment ("along" / "across")


@dataclass
class ClearingPlan:
    removals: list[Removal]   # in this order, then the target
    target_axis: str          # grasp axis free for the target after the removals


def plan_clearing(target: np.ndarray, boxes: list[tuple[BoxColor, np.ndarray]],
                  reachable: Callable[[int], bool] = lambda i: True,
                  cost: Callable[[int], float] = lambda i: 1.0) -> ClearingPlan | None:
    """The cheapest ordered removals after which the box at `target` has a free grasp axis (no removal if it has
    one already), or None if there is none: every box that would have to go is blocked itself or unreachable.
    `boxes` are the other boxes ((color, xy)); `reachable(i)` / `cost(i)` say whether box i can be removed and
    what that costs (e.g. a gray box into the bin is cheaper than a colored box into its region)."""
    def free_axes(position: np.ndarray, others: list[int]) -> list[str]:
        return grasp_access(position, [boxes[j] for j in others]).free

    every = frozenset(range(len(boxes)))
    queue = [(0.0, 0, frozenset(), [])]           # (cost, tie-break, removed, removals so far)
    seen, counter = set(), 0
    while queue:
        total, _, removed, removals = heapq.heappop(queue)
        if removed in seen:
            continue
        seen.add(removed)
        remaining = sorted(every - removed)
        axes = free_axes(target, remaining)
        if axes:
            return ClearingPlan(removals, axes[0])
        for i in remaining:
            if not reachable(i):
                continue
            # the target stays where it is, so it can block a neighbour too
            neighbours = [boxes[j] for j in remaining if j != i] + [(BoxColor.GRAY, target)]
            axes_i = grasp_access(boxes[i][1], neighbours).free
            if axes_i and removed | {i} not in seen:
                counter += 1
                heapq.heappush(queue, (total + cost(i), counter, removed | {i}, removals + [Removal(i, axes_i[0])]))
    return None
