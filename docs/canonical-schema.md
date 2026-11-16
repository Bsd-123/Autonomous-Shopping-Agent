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

`dataset_version` is the run date in UTC (`2026-11-15`) unless `--dataset-version` is given. It becomes a folder name and an image tag, so it may only contain letters, digits, `.`, `_` and `-`, and must start with a letter or a digit.

## Guarantees

A run either writes a complete dataset or writes nothing. A problem in the data or the settings is reported as a one-line error with exit code 1:

- **The folder appears whole or not at all.** The files are written to a hidden folder of the run's own (`.v=<dataset_version>.<random>.tmp`) next to `v=<dataset_version>` and moved into place once all four exist. A failed run leaves nothing behind, and a failed re-run of the same version keeps the previous dataset. Two runs of the same version cannot mix their files; the one that finishes last wins.
- **Replacing a dataset is two renames, not one atomic step.** For a moment `v=<dataset_version>` does not exist, and a crash right then leaves the previous dataset in a hidden `.v=<dataset_version>.<random>.old` folder. A reader that needs a name that is always valid should follow a version pointer, not the folder name. Hidden `.tmp` and `.old` folders left by a killed run can be deleted.
- **Keys exist and are unique:** `products.product_key` and `stores.store_id` have no nulls and no repeats. A listing without an item code fails the run.
- **Every price points at a row:** `prices.product_key` is in `products` and `prices.store_id` is in `stores`. A price row for a store that is missing from the store file fails the run for Osher Ad and Maayan 2000. Yesh Hesed shares Shufersal's files, so it leaves out the rows of stores it does not list, because they belong to other sub-chains.
- **`ingestion_timestamp` must carry a time zone.**

**Decision to revisit with the cleaning step (#7):** stopping the run for an unknown store is deliberate, because a failed run deploys nothing. If a closed store's price file ever outlives its entry in the store file, a single shop would block the daily run. Then those rows should be counted in `ingestion_runs.rows_rejected` (always 0 today) instead of failing the run. The real data of 2026-11-14/15 has no such rows.

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
