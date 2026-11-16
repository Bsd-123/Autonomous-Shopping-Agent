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

## Name normalization

`normalize_name` in `src/shopping_agent/ingestion/text_normalization.py` produces `name_normalized`. It is also meant for the user's list items at request time, so both sides of a search are written the same way.

| Step | Example |
|---|---|
| Remove niqqud and invisible direction marks | `חָלָב` becomes `חלב` |
| Write every geresh and gershayim variant as `'` and `"` | `צ׳יפס`, ``צ`יפס`` become `צ'יפס`; `תפו״א`, `תפו''א` become `תפו"א` |
| Replace punctuation noise with a space | `גבי.נפולאון24%+זיתים225ג` becomes `גבי נפולאון 24% זיתים 225 גרם` |
| Lower-case Latin letters; write multiplication between numbers as `*` | `400X3` becomes `400*3` |
| Separate Hebrew letters from digits | `מ400` becomes `מ 400` |
| Write a unit abbreviation after a number in one form | `500גר'`, `500 ג` become `500 גרם` |

Canonical units and the spellings mapped to them:

| Canonical | Spellings after a number |
|---|---|
| `גרם` | `גרמים`, `גר'`, `גר`, `ג'`, `ג` |
| `ק"ג` | `קילוגרם`, `קילו`, `קג`, `קג'`, `ק'ג`, `ק` |
| `מ"ל` | `מיליליטר`, `מל`, `מל'`, `מ'ל` |
| `ליטר` | `ליטרים`, `ליט`, `ליט'`, `לי`, `ל'`, `ל` |
| `יחידות` | `יחידה`, `יחיד`, `יחי`, `יחי'`, `יח`, `יח'` |
| `ס"מ` | `סנטימטר`, `סמ`, `סמ'` |
| `מ"מ` | |
| `מטר` | `מטרים`, `מ'` |

- A unit is rewritten only directly after a number, so `ג'ל`, `לשיער` and other words are untouched.
- A lone `מ` after a number is left as it is: it stands for millilitres on some products and metres on others.
- A geresh or gershayim is kept only where it belongs to a word; decimal points, `%` after a number and `&` inside a name are kept.
- Prefixes are never stripped, so `לחם`, `מלח`, `בשר` and `שמן` stay whole.
- Normalizing an already normalized name changes nothing.

## Unit normalization

`base_size` in `src/shopping_agent/ingestion/unit_normalization.py` gives every product a `base_quantity` in a `base_unit`: `ml`, `g` or `unit`. Litres become millilitres and kilograms become grams.

A product's size is stated in two places that do not always agree: the chain's fields (`quantity` with `unit`, or `unit_of_measure` when the unit is missing) and the product name. The rules, in order:

| # | Rule | Example |
|---|---|---|
| 1 | A weighed product is priced per kilogram: 1,000 g | `חזה עוף` |
| 2 | A multi-pack in the name, in grams or millilitres, is expanded | `6 * 1.5 ליטר` gives 9,000 ml |
| 3 | A weight or volume in the chain's fields is used, unless the name states a size for the same pack | fields 1,750 ml, name `1.95 ליטר`: 1,950 ml |
| 4 | A single weight or volume in the name | `חלה 500 גר'`: 500 g |
| 5 | A count in the name | `36 יח`: 36 units; `4*64 יח`: 256 units |
| 6 | A count in the chain's fields | 80 units |
| 7 | Otherwise one unit | `טלק` |

Details of rule 3:

- The name replaces the fields when both are the same kind of measure and neither is more than twice the other, or when they differ in kind but are within 15% (a liquid recorded in grams: fields 250 g, name `250 מ"ל` gives 250 ml).
- When the fields hold a whole multiple of the size in the name, they already are the expanded multi-pack and are kept (fields 9 L, name `1.5 ליטר`).
- A field outside 5 to 20,000 g or ml gives way to a plausible size in the name (fields 115,000 g, name `115 גר`).
- Fifty or more litres or kilograms in the fields are read as millilitres or grams.

Other points:

- A weight or volume always wins over a count, because it is what makes products comparable.
- A range in the name (`12-23 ק"ג` on nappies) is not a size.
- Two different sizes in one name are ambiguous and the name is not used.
- Lengths (foil, in metres) have no base unit; such a product is one unit.
- `base_source` records which rule applied: `weighed`, `name_multipack`, `name`, `fields` or `default`.

On the files of 2026-11-14/15 all 31,630 products get a size: 19,060 in grams, 6,616 in millilitres and 5,954 in units. The source is the name for 15,116, the chain's fields for 11,806, the weighed rule for 2,395, a multi-pack in the name for 768 and the one-unit default for 1,545 (1,349 of them at Osher Ad, whose names are cut to 20 characters and whose fields often hold zero).

## Sample files

`tests/fixtures/raw/` holds small samples taken from the dataset on 2026-11-15, used by the adapter tests. Rows are copied as published. The Shufersal price sample joins two slices of the full file (the first 40 rows, and the first 160 rows of Yesh Hesed store 611); the first row of the second slice has its "same as above" cells written out so it does not depend on rows that were left out. They are redistributed under the dataset's licence.
