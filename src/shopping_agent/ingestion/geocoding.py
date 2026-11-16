"""Store geocoding with a cache, a rate limit and manual overrides (architecture section 4.7).

Every store gets ``lat``, ``lon`` and a ``geocode_status``:

* ``override``: the coordinates come from ``geocode_overrides.csv``, which
  always wins.
* ``geocoded``: the geocoder found the address, now or on an earlier run.
* ``not_found``: the geocoder answered and does not know the address. The
  answer is kept, so the address is not asked again until it changes.
* ``error``: the request failed. The store is asked again on the next run.

Only new or changed stores are sent to the geocoder: a store whose address and
city are the same as in the previous run keeps its earlier result. Nothing is
geocoded at request time.

The geocoder is Nominatim. Its usage policy asks for an identifying
``User-Agent`` and at most one request per second; both are enforced here.
"""

import csv
import json
import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NamedTuple, Protocol
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import polars as pl

logger = logging.getLogger(__name__)

STATUS_OVERRIDE = "override"
STATUS_GEOCODED = "geocoded"
STATUS_NOT_FOUND = "not_found"
STATUS_ERROR = "error"
# Results that are reused while the store's address stays the same.
_CACHED_STATUSES = (STATUS_GEOCODED, STATUS_NOT_FOUND)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
MIN_SECONDS_BETWEEN_REQUESTS = 1.0
REQUEST_TIMEOUT_SECONDS = 10.0
COUNTRY = "ישראל"
# The chains write the city as a locality code; the geocoder needs its name.
LOCALITY_NAMES: Mapping[str, str] = {"6100": "בני ברק"}

OVERRIDE_COLUMNS = ("store_id", "lat", "lon")
GEOCODE_COLUMNS = {"lat": pl.Float64(), "lon": pl.Float64(), "geocode_status": pl.String()}
REPORT_SCHEMA = pl.Schema(
    {
        "chain": pl.String(),
        "stores": pl.UInt32(),
        STATUS_OVERRIDE: pl.UInt32(),
        STATUS_GEOCODED: pl.UInt32(),
        STATUS_NOT_FOUND: pl.UInt32(),
        STATUS_ERROR: pl.UInt32(),
        "without_coordinates": pl.UInt32(),
    }
)


class GeocodeError(Exception):
    """A geocoding request failed, or the overrides file is not valid."""


class Coordinates(NamedTuple):
    lat: float
    lon: float


class Geocoder(Protocol):
    def __call__(self, query: str) -> Coordinates | None:
        """Return the coordinates of an address, or ``None`` when it is not known.

        Raises ``GeocodeError`` when the request itself fails.
        """


def _valid(lat: float, lon: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lon <= 180


def _fetch(url: str, headers: Mapping[str, str]) -> bytes:
    request = Request(url, headers=dict(headers))  # noqa: S310 - the URL is the constant above
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
        body: bytes = response.read()
    return body


class NominatimGeocoder:
    """Calls Nominatim, never more often than once per second.

    ``user_agent`` identifies the application to the service, as its usage
    policy requires; there is no default, so it is always a deliberate value.
    ``fetch``, ``clock`` and ``sleep`` exist so that tests need no network and
    no waiting.
    """

    def __init__(
        self,
        user_agent: str,
        *,
        min_interval: float = MIN_SECONDS_BETWEEN_REQUESTS,
        fetch: Callable[[str, Mapping[str, str]], bytes] = _fetch,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("Nominatim requires an identifying User-Agent")
        if min_interval < MIN_SECONDS_BETWEEN_REQUESTS:
            raise ValueError(
                f"requests must be at least {MIN_SECONDS_BETWEEN_REQUESTS:g} second apart"
            )
        self._headers = {"User-Agent": user_agent.strip()}
        self._min_interval = min_interval
        self._fetch = fetch
        self._clock = clock
        self._sleep = sleep
        self._last_request: float | None = None

    def _wait_for_turn(self) -> None:
        if self._last_request is not None:
            remaining = self._min_interval - (self._clock() - self._last_request)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request = self._clock()

    def __call__(self, query: str) -> Coordinates | None:
        parameters = {"q": query, "format": "jsonv2", "limit": 1, "countrycodes": "il"}
        self._wait_for_turn()
        try:
            body = self._fetch(f"{NOMINATIM_URL}?{urlencode(parameters)}", self._headers)
            places = json.loads(body)
        except (URLError, TimeoutError, OSError, ValueError) as error:
            raise GeocodeError(f"the geocoding request failed: {error}") from error
        if not isinstance(places, list):
            raise GeocodeError("the geocoder returned an unexpected answer")
        if not places:
            return None
        try:
            found = Coordinates(float(places[0]["lat"]), float(places[0]["lon"]))
        except (KeyError, TypeError, ValueError) as error:
            raise GeocodeError("the geocoder returned an unexpected answer") from error
        if not _valid(*found):
            raise GeocodeError("the geocoder returned coordinates outside the map")
        return found


def read_overrides(path: Path) -> dict[str, Coordinates]:
    """Read ``geocode_overrides.csv``: one row per store with ``store_id``, ``lat``, ``lon``.

    Further columns, such as a note on where the coordinates came from, are
    ignored. A missing file means there are no overrides.
    """
    if not Path(path).exists():
        return {}
    overrides: dict[str, Coordinates] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in OVERRIDE_COLUMNS if column not in (reader.fieldnames or ())]
        if missing:
            raise GeocodeError(f"{path} is missing columns: {', '.join(missing)}")
        for line, row in enumerate(reader, start=2):
            store_id = (row["store_id"] or "").strip()
            try:
                found = Coordinates(float(row["lat"]), float(row["lon"]))
            except (TypeError, ValueError):
                raise GeocodeError(f"{path} line {line}: lat and lon must be numbers") from None
            if not store_id or not _valid(*found):
                raise GeocodeError(f"{path} line {line}: not a store with valid coordinates")
            if store_id in overrides:
                raise GeocodeError(f"{path} line {line}: {store_id} appears twice")
            overrides[store_id] = found
    return overrides


def address_query(address: str | None, city: str | None) -> str | None:
    """The text sent to the geocoder, or ``None`` when the store has no address."""
    if address is None or not address.strip():
        return None
    parts = [address.strip()]
    if city is not None and city.strip():
        parts.append(LOCALITY_NAMES.get(city.strip(), city.strip()))
    return ", ".join([*parts, COUNTRY])


def _cached(previous: pl.DataFrame | None) -> dict[str, dict[str, object]]:
    if previous is None:
        return {}
    usable = previous.filter(pl.col("geocode_status").is_in(_CACHED_STATUSES))
    return {row["store_id"]: row for row in usable.iter_rows(named=True)}


def geocode_stores(
    stores: pl.DataFrame,
    geocoder: Geocoder,
    *,
    previous: pl.DataFrame | None = None,
    overrides: Mapping[str, Coordinates] | None = None,
) -> pl.DataFrame:
    """Add ``lat``, ``lon`` and ``geocode_status`` to the stores.

    ``previous`` is the stores table of the last run, with those three columns;
    it is the cache. ``overrides`` comes from ``read_overrides``.
    """
    overrides = overrides or {}
    cache = _cached(previous)
    lats: list[float | None] = []
    lons: list[float | None] = []
    statuses: list[str] = []
    for store in stores.iter_rows(named=True):
        store_id, address, city = store["store_id"], store["address"], store["city"]
        found: Coordinates | None = None
        earlier = cache.get(store_id)
        if store_id in overrides:
            found, status = overrides[store_id], STATUS_OVERRIDE
        elif earlier is not None and (earlier["address"], earlier["city"]) == (address, city):
            status = str(earlier["geocode_status"])
            if status == STATUS_GEOCODED:
                found = Coordinates(earlier["lat"], earlier["lon"])  # type: ignore[arg-type]
        elif (query := address_query(address, city)) is None:
            status = STATUS_NOT_FOUND
            logger.warning("store %s has no address to geocode", store_id)
        else:
            try:
                found = geocoder(query)
            except GeocodeError as error:
                status = STATUS_ERROR
                logger.warning(
                    "geocoding failed for store %s, address %r: %s", store_id, query, error
                )
            else:
                status = STATUS_GEOCODED if found is not None else STATUS_NOT_FOUND
                if found is None:
                    logger.warning("no coordinates for store %s, address %r", store_id, query)
        lats.append(found.lat if found else None)
        lons.append(found.lon if found else None)
        statuses.append(status)
    return stores.with_columns(
        pl.Series("lat", lats, dtype=pl.Float64),
        pl.Series("lon", lons, dtype=pl.Float64),
        pl.Series("geocode_status", statuses, dtype=pl.String),
    )


def geocode_report(stores: pl.DataFrame) -> pl.DataFrame:
    """One row per chain: its stores by status, and how many have no coordinates."""
    statuses = (STATUS_OVERRIDE, STATUS_GEOCODED, STATUS_NOT_FOUND, STATUS_ERROR)
    return (
        stores.group_by("chain")
        .agg(
            stores=pl.len(),
            **{status: (pl.col("geocode_status") == status).sum() for status in statuses},
            without_coordinates=(pl.col("lat").is_null() | pl.col("lon").is_null()).sum(),
        )
        .sort("chain")
        .cast(REPORT_SCHEMA)
    )


def with_coordinates(stores: pl.DataFrame) -> pl.DataFrame:
    """Leave out the stores that have no coordinates; ``geocode_report`` counts them."""
    return stores.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null())
