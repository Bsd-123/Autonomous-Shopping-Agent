# Data sources and chain adapters

## Source

Kaggle dataset `erlichsefi/israeli-supermarkets-2024`, refreshed daily. Licence: CC BY-NC-SA 3.0 IGO (attribution, non-commercial, share-alike).

The dataset holds one folder per chain. Each folder has CSV files converted from the chain's XML price files:

| File | Content |
|---|---|
| `store_file_<chain>.csv` | The chain's stores; may hold several daily snapshots |
| `price_full_file_<chain>.csv` | Full price list per store |
| `price_file_<chain>.csv` | Price updates published since the full list |
| `promo_*.csv` | Promotions; not used in Phase 1 |

## Phase 1 chains

| Chain | Adapter | Dataset folder | Notes |
|---|---|---|---|
| Yesh Hesed | `YeshHesedAdapter` | `shufersal` | A sub-chain inside Shufersal's files (`subchainname` = `יש חסד`). The adapter keeps only its stores and their prices. |
| Osher Ad | `OsherAdAdapter` | `osherad` | Writes `לא ידוע` where a value is unknown. |
| Maayan 2000 | `Maayan2000Adapter` | `maayan2000` | Writes `לא ידוע` or `XXXXX` where a value is unknown. |

Maayan 2000 replaces Netiv HaHesed as the third Phase 1 chain: the `netivhased` folder in the dataset contains no price or store files.

## Raw file encoding

The CSVs are compressed from row to row:

- An empty cell means "same value as the row above".
- A cell holding two single quotes (`''`) means the value is missing.

`read_expanded` in `src/shopping_agent/ingestion/adapters/base.py` undoes this before any column is interpreted.

## Adapter output

Every adapter returns two frames with the same columns and types for all chains, defined in `src/shopping_agent/ingestion/adapters/layout.py`.

**Stores:** `chain`, `store_id`, `name`, `address`, `city`, `zipcode`, `source_timestamp`, `source_file`.

**Prices:** `chain`, `store_id`, `item_code`, `name`, `manufacturer`, `is_weighed`, `quantity`, `unit`, `unit_of_measure`, `qty_in_package`, `price`, `source_timestamp`, `source_file`.

- `store_id` is `<chain>:<store number>`, with leading zeros removed, so it is unique across chains.
- `city` is the raw value from the file, usually a Central Bureau of Statistics locality code (Bnei Brak is `6100`).
- `price` is the shelf price in shekels as published. Conversion to integer agorot, cleaning, unit normalization and the canonical tables are later pipeline steps.
- Adapters return every row found in the price files, including repeated listings of the same item; duplicates are resolved in the cleaning step.

## Sample files

`tests/fixtures/raw/` holds small samples taken from the dataset on 2026-11-15, used by the adapter tests. Rows are copied as published. The Shufersal price sample joins two slices of the full file (the first 40 rows, and the first 160 rows of Yesh Hesed store 611); the first row of the second slice has its "same as above" cells written out so it does not depend on rows that were left out. They are redistributed under the dataset's licence.
