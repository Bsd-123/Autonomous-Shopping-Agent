"""The ``decide()`` interface: the only way into the decision engine (architecture section 2).

The engine behind it can be replaced, for example by a solver in a later
phase, without a change for its callers.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from shopping_agent.decision._enumeration import (
    MAX_STORES,
    NOT_LISTED,
    SubsetTable,
    evaluate_subsets,
)
from shopping_agent.decision.config import DecisionConfig


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
    config: DecisionConfig | None = None,
) -> Decision:
    """Evaluate every way of buying the basket from one, two or three stores.

    ``cost[i, s]`` is the cost in whole agorot of item ``i`` at store ``s``, or
    ``NOT_LISTED`` when the store does not list it. ``home`` and ``stores`` are
    ``(lat, lon)`` pairs; ``stores`` follows the columns of ``cost``. ``config``
    holds the store penalty and the travel cost; without it the defaults apply.
    """
    config = config or DecisionConfig()
    if len(stores) > MAX_STORES:
        raise ValueError(f"at most {MAX_STORES} candidate stores, got {len(stores)}")
    matrix = _checked_cost(cost, len(stores))
    table = evaluate_subsets(
        matrix,
        home,
        stores,
        penalty_agorot=config.penalty_agorot,
        travel_agorot_per_km=config.travel_agorot_per_km,
    )
    return Decision(subsets=table.where(~table.has_idle_store), evaluated=len(table))
