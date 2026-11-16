"""The four canonical tables and the step that writes them as Parquet.

The schema follows section 4.1 of the architecture. Columns that a later
pipeline step owns are present from the start, holding null or an interim
value, so the tables keep one shape while those steps are added.
See ``docs/canonical-schema.md`` for the column-by-column description.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from shopping_agent.ingestion.adapters import ADAPTERS, ChainAdapter
from shopping_agent.ingestion.adapters.layout import SOURCE_TIME

PRICE_FLAG_OK = "ok"
PRICE_FLAG_CARRIED_FORWARD = "carried_forward"
GEOCODE_PENDING = "pending"
VALIDATION_NOT_RUN = "not_validated"

PRODUCTS_SCHEMA = pl.Schema(
    {
        "product_key": pl.String(),
        "barcode": pl.String(),
        "name": pl.String(),
        "name_normalized": pl.String(),
        "brand": pl.String(),
        "category": pl.String(),
        "size_value": pl.Float64(),
        "size_unit": pl.String(),
        "base_quantity": pl.Float64(),
        "base_unit": pl.String(),
        "is_weighed": pl.Boolean(),
        "attributes": pl.String(),
    }
)

PRICES_SCHEMA = pl.Schema(
    {
        "product_key": pl.String(),
        "store_id": pl.String(),
        "price_agorot": pl.Int64(),
        "price_flag": pl.String(),
        "last_seen_date": pl.Date(),
        "source_timestamp": SOURCE_TIME,
        "ingestion_timestamp": pl.Datetime("ms", "UTC"),
        "dataset_version": pl.String(),
    }
)

STORES_SCHEMA = pl.Schema(
    {
        "store_id": pl.String(),
        "chain": pl.String(),
        "name": pl.String(),
        "address": pl.String(),
        "city": pl.String(),
        "lat": pl.Float64(),
        "lon": pl.Float64(),
        "geocode_status": pl.String(),
    }
)

INGESTION_RUNS_SCHEMA = pl.Schema(
    {
        "dataset_version": pl.String(),
        "chain": pl.String(),
        "validation_status": pl.String(),
        "rows_in": pl.Int64(),
        "rows_rejected": pl.Int64(),
        "ingestion_timestamp": pl.Datetime("ms", "UTC"),
        "newest_source_timestamp": SOURCE_TIME,
    }
)

SCHEMAS: dict[str, pl.Schema] = {
    "products": PRODUCTS_SCHEMA,
    "prices": PRICES_SCHEMA,
    "stores": STORES_SCHEMA,
    "ingestion_runs": INGESTION_RUNS_SCHEMA,
}


def _product_key() -> pl.Expr:
    # Interim key: every code is namespaced by its chain. Sharing a key across
    # chains for codes with a valid barcode is added with the product-key step.
    return pl.concat_str("chain", pl.lit(":"), "item_code")


def _null(name: str, schema: pl.Schema) -> pl.Expr:
    return pl.lit(None, dtype=schema[name]).alias(name)


def build_products(adapter_prices: pl.DataFrame) -> pl.DataFrame:
    """One row per product, described by its most recently published listing."""
    return (
        adapter_prices.with_columns(product_key=_product_key())
        .sort("source_timestamp", nulls_last=False, maintain_order=True)
        .unique("product_key", keep="last", maintain_order=True)
        .select(
            "product_key",
            _null("barcode", PRODUCTS_SCHEMA),
            "name",
            _null("name_normalized", PRODUCTS_SCHEMA),
            _null("brand", PRODUCTS_SCHEMA),
            _null("category", PRODUCTS_SCHEMA),
            pl.col("quantity").alias("size_value"),
            pl.col("unit").alias("size_unit"),
            _null("base_quantity", PRODUCTS_SCHEMA),
            _null("base_unit", PRODUCTS_SCHEMA),
            "is_weighed",
            _null("attributes", PRODUCTS_SCHEMA),
        )
        .cast(PRODUCTS_SCHEMA)
        .sort("product_key")
    )


def build_prices(
    adapter_prices: pl.DataFrame, dataset_version: str, ingestion_timestamp: datetime
) -> pl.DataFrame:
    return adapter_prices.select(
        product_key=_product_key(),
        store_id="store_id",
        price_agorot=(pl.col("price").cast(pl.Float64) * 100).round().cast(pl.Int64),
        price_flag=pl.lit(PRICE_FLAG_OK),
        last_seen_date=pl.col("source_timestamp").dt.date(),
        source_timestamp="source_timestamp",
        ingestion_timestamp=pl.lit(ingestion_timestamp),
        dataset_version=pl.lit(dataset_version),
    ).cast(PRICES_SCHEMA)


def build_stores(adapter_stores: pl.DataFrame) -> pl.DataFrame:
    return (
        adapter_stores.select(
            "store_id",
            "chain",
            "name",
            "address",
            "city",
            _null("lat", STORES_SCHEMA),
            _null("lon", STORES_SCHEMA),
            geocode_status=pl.lit(GEOCODE_PENDING),
        )
        .cast(STORES_SCHEMA)
        .sort("store_id")
    )


def build_ingestion_runs(
    adapter_prices: pl.DataFrame, dataset_version: str, ingestion_timestamp: datetime
) -> pl.DataFrame:
    return (
        adapter_prices.group_by("chain")
        .agg(
            rows_in=pl.len(),
            newest_source_timestamp=pl.col("source_timestamp").max(),
        )
        .select(
            dataset_version=pl.lit(dataset_version),
            chain="chain",
            validation_status=pl.lit(VALIDATION_NOT_RUN),
            rows_in="rows_in",
            rows_rejected=pl.lit(0),
            ingestion_timestamp=pl.lit(ingestion_timestamp),
            newest_source_timestamp="newest_source_timestamp",
        )
        .cast(INGESTION_RUNS_SCHEMA)
        .sort("chain")
    )


def build_canonical(
    raw_dir: Path,
    dataset_version: str,
    ingestion_timestamp: datetime,
    adapters: Sequence[ChainAdapter] = ADAPTERS,
) -> dict[str, pl.DataFrame]:
    """Read every chain through its adapter and build the four tables."""
    adapter_prices = pl.concat([adapter.read_prices(raw_dir) for adapter in adapters])
    adapter_stores = pl.concat([adapter.read_stores(raw_dir) for adapter in adapters])
    return {
        "products": build_products(adapter_prices),
        "prices": build_prices(adapter_prices, dataset_version, ingestion_timestamp),
        "stores": build_stores(adapter_stores),
        "ingestion_runs": build_ingestion_runs(
            adapter_prices, dataset_version, ingestion_timestamp
        ),
    }


def write_canonical(
    raw_dir: Path,
    out_dir: Path,
    dataset_version: str | None = None,
    ingestion_timestamp: datetime | None = None,
    adapters: Sequence[ChainAdapter] = ADAPTERS,
) -> Path:
    """Write the four tables to ``<out_dir>/v=<dataset_version>/`` and return that folder."""
    ingestion_timestamp = ingestion_timestamp or datetime.now(UTC)
    dataset_version = dataset_version or ingestion_timestamp.date().isoformat()
    tables = build_canonical(raw_dir, dataset_version, ingestion_timestamp, adapters)
    folder = Path(out_dir) / f"v={dataset_version}"
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.write_parquet(folder / f"{name}.parquet")
    return folder
