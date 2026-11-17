"""The four canonical tables and the step that writes them as Parquet.

The schema follows section 4.1 of the architecture. Columns that a later
pipeline step owns are present from the start, holding null or an interim
value, so the tables keep one shape while those steps are added.
See ``docs/canonical-schema.md`` for the column-by-column description.
"""

import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from shopping_agent.ingestion.adapters import ADAPTERS, ChainAdapter
from shopping_agent.ingestion.adapters.layout import SOURCE_TIME
from shopping_agent.ingestion.money import require_integer_money

PRICE_FLAG_OK = "ok"
PRICE_FLAG_CARRIED_FORWARD = "carried_forward"
GEOCODE_PENDING = "pending"
VALIDATION_NOT_RUN = "not_validated"

# The version becomes a folder name and an image tag, so it has to be a plain token.
DATASET_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# Columns that identify a row, and the columns that must point at one.
KEYS = (("products", "product_key"), ("stores", "store_id"))
REFERENCES = (("prices", "product_key", "products"), ("prices", "store_id", "stores"))


class CanonicalError(ValueError):
    """The settings or the input of a run cannot produce a valid canonical dataset."""


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


def _require_aware(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        raise CanonicalError("ingestion_timestamp must be timezone-aware")
    return moment


def _require_valid_version(dataset_version: str) -> str:
    if not DATASET_VERSION_PATTERN.fullmatch(dataset_version):
        raise CanonicalError(
            f"invalid dataset version {dataset_version!r}: use letters, digits, '.', '_' or '-'"
        )
    return dataset_version


def require_integrity(tables: Mapping[str, pl.DataFrame]) -> None:
    """Fail when a key is missing or repeated, or a price points at nothing.

    Without these checks a listing without an item code, or a price for a store
    that is not in the store file, would be written as if it were valid data.
    """
    for table, column in (*KEYS, *((table, column) for table, column, _ in REFERENCES)):
        nulls = tables[table][column].null_count()
        if nulls:
            raise CanonicalError(f"{table}.{column} has {nulls} null value(s)")
    for table, column in KEYS:
        if not tables[table][column].is_unique().all():
            raise CanonicalError(f"{table}.{column} is not unique")
    for table, column, parent in REFERENCES:
        values = tables[table][column]
        missing = values.filter(~values.is_in(tables[parent][column].implode()))
        if not missing.is_empty():
            listed = ", ".join(missing.unique().sort().head(5))
            raise CanonicalError(
                f"{table}.{column} refers to values missing from {parent}: {listed}"
            )


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
        .sort("source_timestamp", "store_id", nulls_last=False, maintain_order=True)
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
        price_agorot=(pl.col("price") * 100).cast(pl.Int64),
        price_flag=pl.lit(PRICE_FLAG_OK),
        last_seen_date=pl.col("source_timestamp").dt.date(),
        source_timestamp="source_timestamp",
        ingestion_timestamp=pl.lit(_require_aware(ingestion_timestamp)),
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
            ingestion_timestamp=pl.lit(_require_aware(ingestion_timestamp)),
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
    """Read every chain through its adapter and build the four tables.

    Raises ``CanonicalError`` for an invalid version, a timestamp without a time
    zone, or tables that break the key and reference rules.
    """
    _require_valid_version(dataset_version)
    _require_aware(ingestion_timestamp)
    if not adapters:
        raise CanonicalError("no chain adapters were given")
    chain_stores = [adapter.read_stores(raw_dir) for adapter in adapters]
    chain_prices = [
        adapter.read_prices(raw_dir, stores)
        for adapter, stores in zip(adapters, chain_stores, strict=True)
    ]
    adapter_stores, adapter_prices = pl.concat(chain_stores), pl.concat(chain_prices)
    tables = {
        "products": build_products(adapter_prices),
        "prices": build_prices(adapter_prices, dataset_version, ingestion_timestamp),
        "stores": build_stores(adapter_stores),
        "ingestion_runs": build_ingestion_runs(
            adapter_prices, dataset_version, ingestion_timestamp
        ),
    }
    require_integrity(tables)
    require_integer_money(tables)
    return tables


def write_canonical(
    raw_dir: Path,
    out_dir: Path,
    dataset_version: str | None = None,
    ingestion_timestamp: datetime | None = None,
    adapters: Sequence[ChainAdapter] = ADAPTERS,
) -> Path:
    """Write the four tables to ``<out_dir>/v=<dataset_version>/`` and return that folder.

    The files are written to a folder of their own next to it and moved into place
    once all four exist. A failed run leaves no partial dataset and a failed re-run
    keeps the previous one. Overlapping runs of one version cannot mix their files;
    the last one to finish wins. Two renames are not one atomic step, so while a
    previous dataset is being replaced ``v=<dataset_version>`` briefly does not
    exist, and a crash right then leaves the old one in a hidden ``.v=*.old`` folder.
    """
    ingestion_timestamp = _require_aware(ingestion_timestamp or datetime.now(UTC))
    if dataset_version is None:
        dataset_version = ingestion_timestamp.astimezone(UTC).date().isoformat()
    tables = build_canonical(raw_dir, dataset_version, ingestion_timestamp, adapters)
    folder = Path(out_dir) / f"v={dataset_version}"
    folder.parent.mkdir(parents=True, exist_ok=True)
    # Dot-prefixed, so a reader looking for ``v=*`` never meets a half-written folder,
    # and unique, so a second run of the same version cannot write into this one.
    staging = folder.with_name(f".{folder.name}.{uuid.uuid4().hex}.tmp")
    staging.mkdir()  # not mkdtemp, which would make the published folder owner-only
    try:
        for name, table in tables.items():
            table.write_parquet(staging / f"{name}.parquet")
        _publish(staging, folder)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return folder


def _publish(staging: Path, folder: Path) -> None:
    """Move the finished ``staging`` folder to ``folder``, replacing a previous run whole."""
    previous = folder.with_name(f".{folder.name}.{uuid.uuid4().hex}.old")
    if folder.exists():
        folder.rename(previous)
    try:
        staging.rename(folder)
    except OSError:
        if previous.exists():
            previous.rename(folder)
        raise
    shutil.rmtree(previous, ignore_errors=True)
