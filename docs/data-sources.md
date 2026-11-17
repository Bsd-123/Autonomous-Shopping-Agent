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
| Yesh Hesed | `YeshHesedAdapter` | `shufersal` | A sub-chain inside Shufersal's files (`subchainname` = `יש חסד`). The adapter keeps only its stores and their prices. Rows of other sub-chains are not fully validated: in the price files only their store numbers are checked (they are dropped before parsing), and in the store file only their snapshot times (needed to pick the newest snapshot). |
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
- `source_timestamp` is Israel local time (`Asia/Jerusalem`). The files carry no UTC offset, so this is an assumption about how the chains publish; it keeps the timestamps comparable with timezone-aware "now" values in the data-quality report. A time that does not exist (the hour skipped when clocks go forward) raises `RawFormatError`; a repeated hour is read as the earlier one.
- The store files can hold several daily snapshots. The adapters keep the newest one, chosen by parsed time. Every snapshot row is validated, including older ones.
- Adapters return every row found in the price files, including repeated listings of the same item; duplicates are resolved in the cleaning step.
- A value that is present but does not parse (a store number, price, quantity, weighed flag or timestamp), or a store row without an update time, raises `RawFormatError`. It never becomes a silent null. Prices must have at most two decimal places, and quantities must be finite numbers (not `NaN` or infinity).
- A change in the shape of the data also raises `RawFormatError`: a store or price file without rows, a Yesh Hesed sub-chain with no stores or no prices (for example after a rename), and two stores that share a `store_id` in the newest snapshot (such as `011` and `11`). An empty chain is never returned as if it were valid data.
- `read_prices(raw_dir, stores)` takes the frame from `read_stores` for every adapter, so callers can loop over `ADAPTERS` the same way. When it is given, a price row for a store the frame does not list raises `RawFormatError` (Osher Ad, Maayan 2000), or is left out because it belongs to another sub-chain (Yesh Hesed, which also avoids reading the store file twice). Without it, Osher Ad and Maayan 2000 return their rows unchecked.

## Product keys

`src/shopping_agent/ingestion/product_key.py` turns a chain's item code into a `product_key`.

| Code | Key | Example |
|---|---|---|
| A real barcode | The barcode as 13 digits, shared by every chain | `811146031465` becomes `0811146031465` |
| Anything else | `<chain>:<code>`, never joined across chains | `116` at Maayan 2000 becomes `maayan_2000:116` |

A code is a real barcode when it has only digits, has 8, 11, 12 or 13 of them, ends with the correct GS1 check digit, and is not in a range GS1 reserves for numbering inside one company (prefixes 020-029 and 200-299).

- **Eleven digits** are accepted because the chains publish UPC-A codes with the leading zero dropped. All 991 such codes in the files of 2026-11-14/15 pass the check digit; chance would pass one in ten.
- **Thirteen digits with leading zeros** is the single written form, so a product gets the same key whether or not a chain dropped the zeros.
- **Shorter codes** are chain-internal numbers. About one in ten passes the check digit by chance, which is why length is checked too.

On the files of 2026-11-14/15 the three chains list 31,630 codes, of which 31,142 are barcodes. They form 22,362 distinct keys; 6,884 are listed by at least two chains and 2,384 by all three.

## Sample files

`tests/fixtures/raw/` holds small samples taken from the dataset on 2026-11-15, used by the adapter tests. Rows are copied as published. The Shufersal price sample joins two slices of the full file (the first 40 rows, and the first 160 rows of Yesh Hesed store 611); the first row of the second slice has its "same as above" cells written out so it does not depend on rows that were left out. They are redistributed under the dataset's licence.
