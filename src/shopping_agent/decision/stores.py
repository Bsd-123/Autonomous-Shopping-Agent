"""The stores considered for a request (architecture section 5.5).

All stores within the radius are candidates, capped at the nearest 10 per
chain and 40 overall. The overall cap is filled chain by chain in turn: each
chain's nearest store, then each chain's second nearest, and so on, so a
slightly farther discount chain is not cut out by a chain with many branches
close by. A store without coordinates is never selected.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

from shopping_agent.decision.geo import haversine_km

# Defaults of architecture section 1.
DEFAULT_RADIUS_KM = 5.0
MAX_STORES_PER_CHAIN = 10
MAX_STORES_OVERALL = 40


@dataclass(frozen=True, slots=True)
class Store:
    """A store as the dataset knows it; ``lat`` and ``lon`` are ``None`` when not geocoded."""

    store_id: str
    chain: str
    lat: float | None
    lon: float | None


@dataclass(frozen=True, slots=True)
class StoreInRange:
    """A selected store and its distance from the user."""

    store_id: str
    chain: str
    lat: float
    lon: float
    distance_km: float


def _coordinates(store: Store) -> tuple[float, float] | None:
    """The store's position, or ``None`` when it has no usable coordinates."""
    if store.lat is None or store.lon is None:
        return None
    if not (isfinite(store.lat) and isfinite(store.lon)):
        return None
    return store.lat, store.lon


def _nearest_first(store: StoreInRange) -> tuple[float, str]:
    return (store.distance_km, store.store_id)


def _group_in_range(
    home: tuple[float, float], stores: Iterable[Store], radius_km: float, per_chain: int
) -> dict[str, list[StoreInRange]]:
    """The stores within the radius, grouped by chain, each chain's nearest ones first."""
    by_chain: dict[str, list[StoreInRange]] = {}
    seen: set[str] = set()
    for store in stores:
        if store.store_id in seen:
            raise ValueError(f"store {store.store_id} was given twice")
        seen.add(store.store_id)
        position = _coordinates(store)
        if position is None:
            continue
        distance = haversine_km(home, position)
        if distance <= radius_km:
            by_chain.setdefault(store.chain, []).append(
                StoreInRange(store.store_id, store.chain, *position, distance)
            )
    return {
        chain: sorted(candidates, key=_nearest_first)[:per_chain]
        for chain, candidates in by_chain.items()
    }


def stores_in_range(
    home: tuple[float, float],
    stores: Iterable[Store],
    *,
    radius_km: float = DEFAULT_RADIUS_KM,
    per_chain: int = MAX_STORES_PER_CHAIN,
    overall: int = MAX_STORES_OVERALL,
) -> tuple[StoreInRange, ...]:
    """Select the candidate stores for a request, nearest first.

    A store exactly on the radius is in range. Stores at the same distance are
    ordered by ``store_id``, so the selection is the same on every run.
    """
    if not isfinite(radius_km) or radius_km <= 0:
        raise ValueError("radius_km must be a positive number")
    if per_chain < 1 or overall < 1:
        raise ValueError("the per-chain and overall caps must be at least 1")

    by_chain = _group_in_range(home, stores, radius_km, per_chain)
    # One round takes the next-nearest store of every chain that still has one.
    selected: list[StoreInRange] = []
    for rank in range(per_chain):
        round_ = [candidates[rank] for candidates in by_chain.values() if rank < len(candidates)]
        selected.extend(sorted(round_, key=_nearest_first)[: overall - len(selected)])
        if len(selected) >= overall:
            break
    return tuple(sorted(selected, key=_nearest_first))
