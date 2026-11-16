"""From the evaluated subsets to the options shown to the user (architecture section 5.5).

    for each store count k in {1, 2, 3}:
        option(k) = the subset of size k with the highest fulfilled(S), then the lowest total(S)

An option is dropped when one with fewer stores fulfils at least as many items
for no higher total. The recommended option is the one that fulfils the most
items and, among those, has the lowest total.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from shopping_agent.decision._enumeration import NO_STORE, NOT_LISTED, SubsetTable
from shopping_agent.decision.config import DecisionConfig
from shopping_agent.decision.geo import shortest_tour


@dataclass(frozen=True, slots=True)
class Option:
    """The best basket for one store count.

    ``stores`` are positions in the cost matrix, in the order of the shortest
    tour from home. ``supplier`` gives, for each item, the position of the store
    it is bought from, or ``NO_STORE`` when no store of the option lists it.
    """

    stores: tuple[int, ...]
    supplier: tuple[int, ...]
    fulfilled: int
    basket_agorot: int
    penalty_agorot: int
    travel_km: float
    travel_agorot: int
    total_agorot: int

    @property
    def unfulfilled(self) -> tuple[int, ...]:
        """The items no store of the option lists."""
        return tuple(item for item, store in enumerate(self.supplier) if store == NO_STORE)


def _best_row(table: SubsetTable, size: int) -> int | None:
    """The row of the best subset of a size: most items fulfilled, then the lowest total.

    Subsets equal in both are told apart by the shorter tour and then by their
    position in the table, so the choice never depends on chance.
    """
    rows = np.flatnonzero(table.size == size)
    if not len(rows):
        return None
    order = np.lexsort(
        (rows, table.travel_km[rows], table.total_agorot[rows], -table.fulfilled[rows])
    )
    return int(rows[order[0]])


def _build(
    table: SubsetTable,
    row: int,
    cost: NDArray[np.int64],
    home: tuple[float, float],
    stores: Sequence[tuple[float, float]],
    config: DecisionConfig,
) -> Option:
    members = [int(store) for store in table.stores[row] if store != NO_STORE]
    supplier = []
    for item_costs in cost[:, members].tolist():
        listed = [(price, store) for price, store in zip(item_costs, members, strict=True)]
        listed = [pair for pair in listed if pair[0] != NOT_LISTED]
        supplier.append(min(listed)[1] if listed else NO_STORE)
    visit_order = shortest_tour(home, [stores[store] for store in members]).order
    return Option(
        stores=tuple(members[position] for position in visit_order),
        supplier=tuple(supplier),
        fulfilled=int(table.fulfilled[row]),
        basket_agorot=int(table.basket_agorot[row]),
        penalty_agorot=config.penalty(len(members)),
        travel_km=float(table.travel_km[row]),
        travel_agorot=int(table.travel_agorot[row]),
        total_agorot=int(table.total_agorot[row]),
    )


def _is_dominated(option: Option, smaller: Sequence[Option]) -> bool:
    return any(
        other.fulfilled >= option.fulfilled and other.total_agorot <= option.total_agorot
        for other in smaller
    )


def choose_options(
    table: SubsetTable,
    cost: NDArray[np.int64],
    home: tuple[float, float],
    stores: Sequence[tuple[float, float]],
    config: DecisionConfig,
) -> tuple[tuple[Option, ...], int | None]:
    """Return up to three options, by store count, and the index of the recommended one."""
    best: list[Option] = []
    for size in (1, 2, 3):
        row = _best_row(table, size)
        if row is not None:
            best.append(_build(table, row, cost, home, stores, config))

    # Each option is compared with every best option of fewer stores, kept or not.
    options = tuple(
        option for position, option in enumerate(best) if not _is_dominated(option, best[:position])
    )
    if not options:
        return (), None
    recommended = min(
        range(len(options)),
        key=lambda index: (-options[index].fulfilled, options[index].total_agorot, index),
    )
    return options, recommended
