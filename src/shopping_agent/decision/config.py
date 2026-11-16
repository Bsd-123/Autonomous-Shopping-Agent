"""The two tunable costs of the decision (architecture sections 1 and 5.5).

``F`` is the penalty for each store beyond the first, and ``c`` is the cost of
a kilometre of travel. Both are tuned against the end-to-end baskets, so they
are configuration, not constants: the service reads them from the environment
once at start-up and passes them to ``decide()``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from os import environ
from typing import Self

DEFAULT_PENALTY_AGOROT = 1_000  # F
DEFAULT_TRAVEL_AGOROT_PER_KM = 100.0  # c
PENALTY_VARIABLE = "SHOPPING_AGENT_STORE_PENALTY_AGOROT"
TRAVEL_VARIABLE = "SHOPPING_AGENT_TRAVEL_AGOROT_PER_KM"


class ConfigError(ValueError):
    """A configuration value is missing its meaning: not a number, or out of range."""


@dataclass(frozen=True, slots=True)
class DecisionConfig:
    """``F`` and ``c``.

    A basket bought from ``n`` stores carries a penalty of ``F x (n - 1)``. Its
    travel cost is ``c`` times the tour in kilometres, rounded once to whole
    agorot.
    """

    penalty_agorot: int = DEFAULT_PENALTY_AGOROT
    travel_agorot_per_km: float = DEFAULT_TRAVEL_AGOROT_PER_KM

    def __post_init__(self) -> None:
        if isinstance(self.penalty_agorot, bool) or not isinstance(self.penalty_agorot, int):
            raise ConfigError("the store penalty must be a whole number of agorot")
        if self.penalty_agorot < 0:
            raise ConfigError("the store penalty cannot be negative")
        per_km = self.travel_agorot_per_km
        if isinstance(per_km, bool) or not isinstance(per_km, int | float):
            raise ConfigError("the travel cost per kilometre must be a number")
        if not isfinite(per_km) or per_km < 0:
            raise ConfigError("the travel cost per kilometre must be finite and not negative")

    def penalty(self, store_count: int) -> int:
        """The penalty of a basket bought from ``store_count`` stores: ``F x (n - 1)``."""
        if store_count < 1:
            raise ValueError("a basket is bought from at least one store")
        return self.penalty_agorot * (store_count - 1)

    def travel_cost(self, distance_km: float) -> int:
        """The cost of a tour, rounded once to whole agorot; a half rounds up."""
        if not isfinite(distance_km) or distance_km < 0:
            raise ValueError("a distance must be finite and not negative")
        return int(self.travel_agorot_per_km * distance_km + 0.5)

    @classmethod
    def from_env(cls, env: Mapping[str, str] = environ) -> Self:
        """Read ``F`` and ``c`` from the environment; an unset variable keeps its default."""
        penalty: int = DEFAULT_PENALTY_AGOROT
        per_km: float = DEFAULT_TRAVEL_AGOROT_PER_KM
        if (raw := env.get(PENALTY_VARIABLE, "").strip()) != "":
            try:
                penalty = int(raw)
            except ValueError:
                raise ConfigError(
                    f"{PENALTY_VARIABLE} must be a whole number of agorot, got {raw!r}"
                ) from None
        if (raw := env.get(TRAVEL_VARIABLE, "").strip()) != "":
            try:
                per_km = float(raw)
            except ValueError:
                raise ConfigError(f"{TRAVEL_VARIABLE} must be a number, got {raw!r}") from None
        try:
            return cls(penalty_agorot=penalty, travel_agorot_per_km=per_km)
        except ConfigError as error:
            raise ConfigError(f"{PENALTY_VARIABLE} / {TRAVEL_VARIABLE}: {error}") from None
