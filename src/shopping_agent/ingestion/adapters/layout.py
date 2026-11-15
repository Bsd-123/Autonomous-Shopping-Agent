"""The common row layout every chain adapter emits.

Adapters absorb each chain's raw format and return frames with exactly these
columns and types, so the later pipeline steps see a single layout.
"""

import polars as pl

PRICE_SCHEMA: dict[str, pl.DataType] = {
    "chain": pl.String(),
    "store_id": pl.String(),
    "item_code": pl.String(),
    "name": pl.String(),
    "manufacturer": pl.String(),
    "is_weighed": pl.Boolean(),
    "quantity": pl.Float64(),
    "unit": pl.String(),
    "unit_of_measure": pl.String(),
    "qty_in_package": pl.Float64(),
    "price": pl.Decimal(12, 2),
    "price_updated_at": pl.Datetime("ms"),
    "source_timestamp": pl.Datetime("ms"),
    "source_file": pl.String(),
}

STORE_SCHEMA: dict[str, pl.DataType] = {
    "chain": pl.String(),
    "store_id": pl.String(),
    "name": pl.String(),
    "address": pl.String(),
    "city": pl.String(),
    "zipcode": pl.String(),
    "source_timestamp": pl.Datetime("ms"),
    "source_file": pl.String(),
}
