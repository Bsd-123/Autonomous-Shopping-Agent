"""Exact enumeration of store subsets (architecture section 5.5).

This module is the engine behind ``decide()`` and is not imported from
anywhere else. For every subset S of one, two or three stores it computes:

    fulfilled(S) = number of items with at least one non-empty cost in S
    basket(S)    = sum over fulfilled items of the lowest cost in S
    travel(S)    = c x shortest tour home -> stores in S -> home
    total(S)     = basket(S) + F x (|S| - 1) + travel(S)

A subset in which some store supplies no item is marked: the same basket is
already covered by the subset without that store, so the penalty and travel of
an idle store are never charged. ``decide()`` leaves the marked subsets out.

All subsets of one size are evaluated together as NumPy arrays.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from shopping_agent.decision.geo import MAX_TOUR_STORES, haversine_km

# The value of an empty cell of the cost matrix: the item is not listed at the store.
NOT_LISTED = -1
MAX_STORES = 40
NO_STORE = -1

Int = NDArray[np.int64]
Float = NDArray[np.float64]
_FAR = np.iinfo(np.int64).max


@dataclass(frozen=True, slots=True, eq=False)
class SubsetTable:
    """Every evaluated subset, one row each, ordered by size and then by store positions.

    ``stores`` holds the positions of a subset's stores in the cost matrix,
    padded with ``NO_STORE`` for subsets of fewer than three. Distances stay
    floating-point kilometres; ``travel_agorot`` is the travel cost rounded once.
    ``has_idle_store`` marks a subset in which some store supplies no item.
    """

    stores: Int
    size: Int
    fulfilled: Int
    basket_agorot: Int
    travel_km: Float
    travel_agorot: Int
    total_agorot: Int
    has_idle_store: NDArray[np.bool_]

    def __len__(self) -> int:
        return len(self.size)

    def where(self, keep: NDArray[np.bool_]) -> "SubsetTable":
        """The rows for which ``keep`` is true."""
        return SubsetTable(
            self.stores[keep],
            self.size[keep],
            self.fulfilled[keep],
            self.basket_agorot[keep],
            self.travel_km[keep],
            self.travel_agorot[keep],
            self.total_agorot[keep],
            self.has_idle_store[keep],
        )


def _distances(
    home: tuple[float, float], stores: Sequence[tuple[float, float]]
) -> tuple[Float, Float]:
    from_home = np.array([haversine_km(home, store) for store in stores], dtype=np.float64)
    between = np.zeros((len(stores), len(stores)), dtype=np.float64)
    for (a, first), (b, second) in combinations(enumerate(stores), 2):
        between[a, b] = between[b, a] = haversine_km(first, second)
    return from_home, between


def _tour_km(subsets: Int, from_home: Float, between: Float) -> Float:
    """The shortest tour of each subset; ``subsets`` has one column per store."""
    size = subsets.shape[1]
    if size == 1:
        there_and_back: Float = 2 * from_home[subsets[:, 0]]
        return there_and_back
    if size == 2:
        a, b = subsets.T
        triangle: Float = from_home[a] + between[a, b] + from_home[b]
        return triangle
    # Three stores: a tour is fixed by the store visited in the middle, and its
    # reverse is equally long, so three of the six orders are enough.
    a, b, c = subsets.T
    tours = [
        from_home[first] + between[first, middle] + between[middle, last] + from_home[last]
        for first, middle, last in ((a, b, c), (a, c, b), (b, a, c))
    ]
    shortest: Float = np.minimum.reduce(tours)
    return shortest


def _round_half_up(values: Float) -> Int:
    return np.floor(values + 0.5).astype(np.int64)


def _has_idle_store(priced: Int, subsets: Int, cheapest: Int) -> NDArray[np.bool_]:
    """Whether some store of each subset supplies no item.

    A store is idle when the subset without it gives every item the same lowest
    cost, so nothing is bought there. A store that only matches another store's
    price is idle too: the basket is the same without it.
    """
    size = subsets.shape[1]
    if size == 1:
        alone: NDArray[np.bool_] = ~(cheapest != _FAR).any(axis=0)
        return alone
    idle = np.zeros(len(subsets), dtype=np.bool_)
    for position in range(size):
        others = np.delete(subsets, position, axis=1)
        without = priced[:, others].min(axis=2)
        idle |= (without == cheapest).all(axis=0)
    return idle


def evaluate_subsets(
    cost: Int,
    home: tuple[float, float],
    stores: Sequence[tuple[float, float]],
    *,
    penalty_agorot: int,
    travel_agorot_per_km: float,
) -> SubsetTable:
    """Evaluate every subset of one, two and three stores.

    ``cost`` has one row per item and one column per store, in whole agorot,
    with ``NOT_LISTED`` in the empty cells.
    """
    store_count = len(stores)
    from_home, between = _distances(home, stores)
    listed = cost != NOT_LISTED
    priced = np.where(listed, cost, _FAR)

    parts: list[tuple[Int, ...]] = []
    travel_parts: list[Float] = []
    idle_parts: list[NDArray[np.bool_]] = []
    for size in range(1, min(MAX_TOUR_STORES, store_count) + 1):
        subsets = np.array(list(combinations(range(store_count), size)), dtype=np.int64)
        # cheapest[i, n] is the lowest cost of item i among the stores of subset n.
        cheapest = priced[:, subsets].min(axis=2)
        covered = cheapest != _FAR
        basket = np.where(covered, cheapest, 0).sum(axis=0)
        travel_km = _tour_km(subsets, from_home, between)
        travel = _round_half_up(travel_agorot_per_km * travel_km)
        padded = np.full((len(subsets), MAX_TOUR_STORES), NO_STORE, dtype=np.int64)
        padded[:, :size] = subsets
        parts.append(
            (
                padded,
                np.full(len(subsets), size, dtype=np.int64),
                covered.sum(axis=0).astype(np.int64),
                basket,
                travel,
                basket + penalty_agorot * (size - 1) + travel,
            )
        )
        travel_parts.append(travel_km)
        idle_parts.append(_has_idle_store(priced, subsets, cheapest))

    if not parts:
        empty = np.zeros(0, dtype=np.int64)
        return SubsetTable(
            np.zeros((0, MAX_TOUR_STORES), dtype=np.int64),
            empty,
            empty,
            empty,
            np.zeros(0, dtype=np.float64),
            empty,
            empty,
            np.zeros(0, dtype=np.bool_),
        )
    padded_all, sizes, fulfilled, baskets, travels, totals = (
        np.concatenate(column) for column in zip(*parts, strict=True)
    )
    return SubsetTable(
        stores=padded_all,
        size=sizes,
        fulfilled=fulfilled,
        basket_agorot=baskets,
        travel_km=np.concatenate(travel_parts),
        travel_agorot=travels,
        total_agorot=totals,
        has_idle_store=np.concatenate(idle_parts),
    )
