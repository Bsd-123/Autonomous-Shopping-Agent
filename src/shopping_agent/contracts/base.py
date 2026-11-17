"""Building blocks shared by the request, node and state contracts."""

from collections.abc import Hashable, Iterable
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from shopping_agent.contracts.parsed_item import MAX_TEXT_LENGTH

# Limits from the architecture's defaults (section 1) and request flow (section 5).
MAX_ITEMS = 100
MAX_LIST_TEXT_LENGTH = 5_000
DEFAULT_RADIUS_KM = 5.0
MAX_RADIUS_KM = 50.0
MAX_STORES_PER_BASKET = 3
MAX_STORES_IN_RANGE = 40
MAX_CANDIDATES = 200
MAX_EXPLANATION_LENGTH = 2_000


class Contract(BaseModel):
    """Base of every contract: unknown fields are rejected and values are read-only."""

    model_config = ConfigDict(extra="forbid", frozen=True)


Label = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_LENGTH)
]
ProductKey = Label
# ``<chain>:<store number>``, as written by the ingestion adapters.
StoreId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]+:[0-9]+$", max_length=64)]
DatasetVersion = Label
ItemIndex = Annotated[int, Field(ge=0, lt=MAX_ITEMS, strict=True)]
Agorot = Annotated[int, Field(ge=0, strict=True)]
Latitude = Annotated[float, Field(ge=-90, le=90, allow_inf_nan=False, strict=True)]
Longitude = Annotated[float, Field(ge=-180, le=180, allow_inf_nan=False, strict=True)]
Kilometres = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
RadiusKm = Annotated[float, Field(gt=0, le=MAX_RADIUS_KM, allow_inf_nan=False, strict=True)]


class Location(Contract):
    """A point on the map, in decimal degrees."""

    lat: Latitude
    lon: Longitude


def require_unique(values: Iterable[Hashable], what: str) -> None:
    """Raise when a value appears twice."""
    seen: set[Hashable] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"{what} repeat: {value!r}")
        seen.add(value)


def require_item_order(indices: Iterable[int], what: str) -> None:
    """Raise unless the indices are 0, 1, 2, ... in order, one per list item."""
    found = list(indices)
    if found != list(range(len(found))):
        raise ValueError(f"{what} must follow the order of the list items: 0, 1, 2, ...")
