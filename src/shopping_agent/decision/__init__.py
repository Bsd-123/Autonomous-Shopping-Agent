"""The decision engine: which stores to buy the basket from.

The engine is reached through ``decide()`` only.
"""

from shopping_agent.decision._enumeration import MAX_STORES, NO_STORE, NOT_LISTED, SubsetTable
from shopping_agent.decision._options import Option
from shopping_agent.decision.config import (
    DEFAULT_PENALTY_AGOROT,
    DEFAULT_TRAVEL_AGOROT_PER_KM,
    PENALTY_VARIABLE,
    TRAVEL_VARIABLE,
    ConfigError,
    DecisionConfig,
)
from shopping_agent.decision.decide import Decision, decide
from shopping_agent.decision.geo import (
    EARTH_RADIUS_KM,
    MAX_TOUR_STORES,
    Point,
    Tour,
    haversine_km,
    shortest_tour,
)

__all__ = [
    "DEFAULT_PENALTY_AGOROT",
    "DEFAULT_TRAVEL_AGOROT_PER_KM",
    "EARTH_RADIUS_KM",
    "MAX_STORES",
    "MAX_TOUR_STORES",
    "NOT_LISTED",
    "NO_STORE",
    "PENALTY_VARIABLE",
    "TRAVEL_VARIABLE",
    "ConfigError",
    "Decision",
    "DecisionConfig",
    "Option",
    "Point",
    "SubsetTable",
    "Tour",
    "decide",
    "haversine_km",
    "shortest_tour",
]
