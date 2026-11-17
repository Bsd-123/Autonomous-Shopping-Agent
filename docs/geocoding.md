# Store geocoding

Stores get their coordinates during ingestion (architecture section 4.7). Nothing is geocoded at request time. The code is in `src/shopping_agent/ingestion/geocoding.py`.

```python
from shopping_agent.ingestion.geocoding import (
    NominatimGeocoder,
    geocode_stores,
    geocode_report,
    read_overrides,
    with_coordinates,
)

geocoder = NominatimGeocoder(user_agent)  # an identifying User-Agent, set by the operator
overrides = read_overrides(Path("config/geocode_overrides.csv"))
stores = geocode_stores(stores, geocoder, previous=last_run_stores, overrides=overrides)
report = geocode_report(stores)  # per chain, for the data-quality report
stores = with_coordinates(stores)  # stores without coordinates are left out
```

## What each store gets

`geocode_stores` adds `lat`, `lon` and `geocode_status`:

| Status | Meaning | Asked again? |
|---|---|---|
| `override` | Coordinates from `geocode_overrides.csv`. Always wins. | Never sent |
| `geocoded` | The geocoder found the address, now or on an earlier run. | Only when the address or city changes |
| `not_found` | The geocoder answered and does not know the address, or the store has no address. | Only when the address or city changes |
| `error` | The request failed. | On the next run |

Every `not_found` and `error` is logged as a warning with the store and the address that was sent.

## The cache

`previous` is the stores table of the last run. A store whose `address` and `city` are unchanged keeps its `geocoded` or `not_found` result and is not sent again, so a normal daily run sends nothing.

## Nominatim

- Every request carries the `User-Agent` given to `NominatimGeocoder`. There is no default: the operator sets a value that identifies the deployment, as the service's usage policy asks.
- Requests are at least one second apart. The interval can be raised but not lowered.
- The query is `<address>, <city name>, ישראל`. The chains write the city as a locality code, so `LOCALITY_NAMES` maps the code to the name (`6100` is Bnei Brak).
- Attribution to OpenStreetMap is shown in the app.

## Overrides

`config/geocode_overrides.csv` holds hand-checked coordinates:

```csv
store_id,lat,lon,note
osher_ad:11,32.0911,34.8259,checked on the map
```

`store_id`, `lat` and `lon` are required; other columns are ignored. A row with a bad number, coordinates outside the map or a repeated store stops the run with the line number.

## The report

`geocode_report` gives one row per chain: `stores`, the count of each status, and `without_coordinates`. `with_coordinates` then leaves those stores out of the dataset.
