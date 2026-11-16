"""Distances and the shopping tour (architecture section 5.5).

The travel part of a basket's cost is the length of the tour
home -> stores -> home. Distances are great-circle (Haversine) and stay
floating-point kilometres; turning them into agorot, with a single rounding,
is the caller's job.
"""

from collections.abc import Sequence
from itertools import permutations
from math import asin, cos, isfinite, radians, sin, sqrt
from typing import NamedTuple

EARTH_RADIUS_KM = 6371.0088  # mean Earth radius
# A basket uses at most three stores, so at most six visit orders are tried.
MAX_TOUR_STORES = 3


class Point(NamedTuple):
    """A place on the map, in decimal degrees."""

    lat: float
    lon: float


class Tour(NamedTuple):
    """The shortest round trip from home through the stores.

    ``order`` holds the positions of the stores, as they were given, in the
    order they are visited.
    """

    distance_km: float
    order: tuple[int, ...]


def _checked(point: tuple[float, float]) -> Point:
    lat, lon = point
    if not (isfinite(lat) and isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"not a valid latitude and longitude: {lat!r}, {lon!r}")
    return Point(lat, lon)


def haversine_km(origin: tuple[float, float], destination: tuple[float, float]) -> float:
    """Great-circle distance between two points, in kilometres."""
    lat1, lon1 = map(radians, _checked(origin))
    lat2, lon2 = map(radians, _checked(destination))
    half_chord = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    # Rounding can push the value a hair above 1 for opposite points.
    return 2 * EARTH_RADIUS_KM * asin(sqrt(min(1.0, half_chord)))


def shortest_tour(home: tuple[float, float], stores: Sequence[tuple[float, float]]) -> Tour:
    """The shortest tour home -> stores -> home over every visit order.

    With no stores the tour is empty. A tour and its reverse are equally long;
    the one found first is returned, so the result does not depend on chance.
    """
    if len(stores) > MAX_TOUR_STORES:
        raise ValueError(f"a tour visits at most {MAX_TOUR_STORES} stores, got {len(stores)}")
    if not stores:
        _checked(home)
        return Tour(0.0, ())

    from_home = [haversine_km(home, store) for store in stores]
    between = [[haversine_km(first, second) for second in stores] for first in stores]

    def length(order: tuple[int, ...]) -> float:
        legs = sum(between[a][b] for a, b in zip(order, order[1:], strict=False))
        return from_home[order[0]] + legs + from_home[order[-1]]

    best = min(permutations(range(len(stores))), key=length)
    return Tour(length(best), best)
