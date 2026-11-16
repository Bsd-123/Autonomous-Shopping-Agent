# Canonical schema

The final schema of the four canonical tables (architecture section 4.1), adjusted to the real Kaggle files. The code is in `src/shopping_agent/ingestion/canonical.py`; the adapter columns it reads are described in `data-sources.md`.

## Output layout

```
<out>/v=<dataset_version>/
    products.parquet
    prices.parquet
    stores.parquet
    ingestion_runs.parquet
```

Build it with:

```bash
uv run python -m shopping_agent.ingestion --raw data/raw --out data/canonical
```

`dataset_version` is the run date in UTC (`2026-11-15`) unless `--version` is given.

## Status of columns

Every section 4.1 column exists from the start. A column owned by a later pipeline step holds null or an interim value until that step is added, so the tables never change shape.

| Status | Meaning |
|---|---|
| Final | Filled from the source files as it will stay. |
| Interim | Filled with a simple rule that a later step replaces. |
| Empty | Null until a later step fills it. |

## `products`

One row per `product_key`, described by the product's most recently published listing.

| Column | Type | Source | Status |
|---|---|---|---|
| `product_key` | string | `<chain>:<itemcode>` | Interim: shared keys for valid barcodes come with the product-key step |
| `barcode` | string | | Empty: product-key step |
| `name` | string | `itemname` | Final |
| `name_normalized` | string | | Empty: text normalization |
| `brand` | string | | Empty: attribute extraction |
| `category` | string | | Empty: category assignment |
| `size_value` | float | `quantity` | Final |
| `size_unit` | string | `unitqty`, as written by the chain | Final |
| `base_quantity` | float | | Empty: unit normalization |
| `base_unit` | string | | Empty: unit normalization |
| `is_weighed` | boolean | `bisweighted` | Final |
| `attributes` | string (JSON object) | | Empty: attribute extraction |

## `prices`

One row per listing found in the price files. The same product can appear more than once per store until the cleaning step removes duplicates.

| Column | Type | Source | Status |
|---|---|---|---|
| `product_key` | string | as in `products` | Interim |
| `store_id` | string | `<chain>:<store number>` | Final |
| `price_agorot` | integer | `itemprice` × 100, rounded | Interim: the per-kilogram rule for weighed items comes with the agorot step |
| `price_flag` | string | `ok` | Interim: cleaning sets `carried_forward` |
| `last_seen_date` | date | Israeli date of `source_timestamp` | Final |
| `source_timestamp` | timestamp (Israel time) | publish time in the price file's name | Final |
| `ingestion_timestamp` | timestamp (UTC) | start of the ingestion run | Final |
| `dataset_version` | string | the run's version | Final |

**Freshness:** `source_timestamp` is when the chain published the price file, taken from the date and time at the end of the file name (`Price…-20261114-090012.xml`), in Israel local time (`Asia/Jerusalem`), which the column keeps. The per-item `priceupdatetime` in the source is the last price change and is often years old, so it is not used for freshness. `last_seen_date` is the Israeli date the listing was last seen in a published file (a file published at 00:30 in Israel belongs to that day, although it is still the day before in UTC), and the staleness cutoff applies to it.

## `stores`

| Column | Type | Source | Status |
|---|---|---|---|
| `store_id` | string | `<chain>:<store number>`, leading zeros removed | Final |
| `chain` | string | `yesh_hesed`, `osher_ad`, `maayan_2000` | Final |
| `name` | string | `storename` | Final |
| `address` | string | `address` | Final |
| `city` | string | `city`, usually a locality code (Bnei Brak is `6100`) | Final |
| `lat`, `lon` | float | | Empty: geocoding |
| `geocode_status` | string | `pending` | Interim: geocoding |

## `ingestion_runs`

One row per chain per run.

| Column | Type | Source | Status |
|---|---|---|---|
| `dataset_version` | string | the run's version | Final |
| `chain` | string | | Final |
| `validation_status` | string | `not_validated` | Interim: schema validation |
| `rows_in` | integer | price rows read from the chain's adapter | Final |
| `rows_rejected` | integer | `0` | Interim: cleaning and validation |
| `ingestion_timestamp` | timestamp (UTC) | start of the ingestion run | Final |
| `newest_source_timestamp` | timestamp (Israel time) | newest `source_timestamp` of the chain | Final |

Further quality metrics are added with the data-quality report.

## Differences from the section 4.1 proposal

- `prices` is a single file for now; partitioning by chain through the folder layout is a separate step.
- `attributes` is stored as a JSON object in a string column, so every product can carry a different set of attributes.
- `ingestion_runs` gains `ingestion_timestamp` and `newest_source_timestamp`.
