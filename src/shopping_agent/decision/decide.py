"""The ``decide()`` interface: the only way into the decision engine (architecture section 2).

The engine behind it can be replaced, for example by a solver in a later
phase, without a change for its callers.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike

from shopping_agent.decision._enumeration import (
    MAX_STORES,
    NOT_LISTED,
    SubsetTable,
    evaluate_subsets,
)

# Defaults of architecture section 1; both are tuned against the end-to-end baskets.
DEFAULT_PENALTY_AGOROT = 1_000  # F, per store beyond the first
DEFAULT_TRAVEL_AGOROT_PER_KM = 100.0  # c


@dataclass(frozen=True, slots=True, eq=False)
class Decision:
    """The outcome of ``decide()``.

    ``subsets`` holds every subset of one to three stores in which each store
    supplies at least one item. ``evaluated`` counts all the subsets that were
    looked at, including those left out for an idle store.
    """

    subsets: SubsetTable
    evaluated: int


def _checked_cost(cost: ArrayLike, store_count: int) -> np.ndarray:
    matrix = np.asarray(cost)
    if matrix.ndim != 2:
        raise ValueError("cost must have one row per item and one column per store")
    if matrix.shape[1] != store_count:
        raise ValueError(
            f"cost has {matrix.shape[1]} store columns but {store_count} stores were given"
        )
    if not np.issubdtype(matrix.dtype, np.integer):
        raise TypeError("cost must hold whole agorot")
    if matrix.size and (invalid := matrix[(matrix <= 0) & (matrix != NOT_LISTED)]).size:
        raise ValueError(
            f"a cost must be positive, or {NOT_LISTED} when not listed; got {int(invalid[0])}"
        )
    return matrix.astype(np.int64)


def decide(
    cost: ArrayLike,
    home: tuple[float, float],
    stores: Sequence[tuple[float, float]],
    *,
    penalty_agorot: int = DEFAULT_PENALTY_AGOROT,
    travel_agorot_per_km: float = DEFAULT_TRAVEL_AGOROT_PER_KM,
) -> Decision:
    """Evaluate every way of buying the basket from one, two or three stores.

    ``cost[i, s]`` is the cost in whole agorot of item ``i`` at store ``s``, or
    ``NOT_LISTED`` when the store does not list it. ``home`` and ``stores`` are
    ``(lat, lon)`` pairs; ``stores`` follows the columns of ``cost``.
    """
    if len(stores) > MAX_STORES:
        raise ValueError(f"at most {MAX_STORES} candidate stores, got {len(stores)}")
    if isinstance(penalty_agorot, bool) or not isinstance(penalty_agorot, int):
        raise TypeError("penalty_agorot must be a whole number of agorot")
    if penalty_agorot < 0:
        raise ValueError("penalty_agorot cannot be negative")
    if not isfinite(travel_agorot_per_km) or travel_agorot_per_km < 0:
        raise ValueError("travel_agorot_per_km must be a finite number, zero or more")
    matrix = _checked_cost(cost, len(stores))
    table = evaluate_subsets(
        matrix,
        home,
        stores,
        penalty_agorot=penalty_agorot,
        travel_agorot_per_km=travel_agorot_per_km,
    )
    return Decision(subsets=table.where(~table.has_idle_store), evaluated=len(table))
