"""Tests for the canonical tables and the Parquet writer."""

from datetime import UTC, date, datetime

import polars as pl
import pytest

from shopping_agent.ingestion.canonical import (
    INGESTION_RUNS_SCHEMA,
    PRICES_SCHEMA,
    PRODUCTS_SCHEMA,
    SCHEMAS,
    STORES_SCHEMA,
    build_canonical,
    write_canonical,
)

VERSION = "2026-11-15"
RUN_AT = datetime(2026, 11, 15, 3, 0, tzinfo=UTC)
CHAINS = ["maayan_2000", "osher_ad", "yesh_hesed"]


@pytest.fixture
def tables(raw_dir) -> dict[str, pl.DataFrame]:
    return build_canonical(raw_dir, VERSION, RUN_AT)


def test_tables_have_the_section_4_1_columns(tables):
    assert list(tables) == ["products", "prices", "stores", "ingestion_runs"]
    assert list(PRODUCTS_SCHEMA) == [
        "product_key",
        "barcode",
        "name",
        "name_normalized",
        "brand",
        "category",
        "size_value",
        "size_unit",
        "base_quantity",
        "base_unit",
        "is_weighed",
        "attributes",
    ]
    assert list(PRICES_SCHEMA) == [
        "product_key",
        "store_id",
        "price_agorot",
        "price_flag",
        "last_seen_date",
        "source_timestamp",
        "ingestion_timestamp",
        "dataset_version",
    ]
    assert list(STORES_SCHEMA) == [
        "store_id",
        "chain",
        "name",
        "address",
        "city",
        "lat",
        "lon",
        "geocode_status",
    ]
    assert list(INGESTION_RUNS_SCHEMA)[:5] == [
        "dataset_version",
        "chain",
        "validation_status",
        "rows_in",
        "rows_rejected",
    ]
    for name, table in tables.items():
        assert dict(table.schema) == SCHEMAS[name]


def test_freshness_columns_are_populated(tables):
    prices = tables["prices"]
    freshness = ["source_timestamp", "ingestion_timestamp", "dataset_version", "last_seen_date"]

    assert sum(prices.select(pl.col(freshness).null_count()).row(0)) == 0
    assert prices["dataset_version"].unique().to_list() == [VERSION]
    assert prices["ingestion_timestamp"].unique().to_list() == [RUN_AT]
    assert (prices["last_seen_date"] == prices["source_timestamp"].dt.date()).all()
    assert prices["last_seen_date"].unique().to_list() == [date(2026, 11, 14)]


def test_prices_are_integer_agorot_with_a_flag(tables):
    prices = tables["prices"]

    assert prices.height == 218 + 300 + 160
    assert prices["price_flag"].unique().to_list() == ["ok"]
    assert (prices["price_agorot"] > 0).all()
    muffin = prices.filter(product_key="osher_ad:7290006775405").row(0, named=True)
    assert muffin["price_agorot"] == 150
    assert muffin["store_id"] == "osher_ad:6"


def test_every_price_refers_to_a_product(tables):
    products, prices = tables["products"], tables["prices"]

    assert products["product_key"].is_unique().all()
    assert prices.join(products, on="product_key", how="anti").height == 0
    assert products.height == prices["product_key"].n_unique()


def test_products_carry_source_fields_and_leave_later_columns_empty(tables):
    products = tables["products"]

    potatoes = products.filter(product_key="maayan_2000:116").row(0, named=True)
    assert potatoes["name"] == 'תפו"א לבן ארוז'
    assert potatoes["is_weighed"] is True
    assert potatoes["size_unit"] == "קילוגרם"
    later = ["barcode", "name_normalized", "brand", "category", "base_quantity", "attributes"]
    assert products.select(pl.col(later).null_count()).row(0) == (products.height,) * len(later)


def test_stores_wait_for_geocoding(tables):
    stores = tables["stores"]

    assert stores.height == 24 + 38 + 22
    assert stores["store_id"].is_unique().all()
    assert sorted(stores["chain"].unique()) == CHAINS
    assert stores["geocode_status"].unique().to_list() == ["pending"]
    assert stores.select(pl.col("lat", "lon").null_count()).row(0) == (stores.height,) * 2


def test_ingestion_runs_has_one_row_per_chain(tables):
    runs = tables["ingestion_runs"]

    assert runs["chain"].to_list() == CHAINS
    assert runs["rows_in"].to_list() == [300, 218, 160]
    assert runs["rows_rejected"].to_list() == [0, 0, 0]
    assert runs["validation_status"].unique().to_list() == ["not_validated"]
    assert runs["dataset_version"].unique().to_list() == [VERSION]
    assert runs["newest_source_timestamp"].null_count() == 0


def test_write_canonical_writes_four_parquet_files_to_a_dated_folder(raw_dir, tmp_path):
    folder = write_canonical(raw_dir, tmp_path / "canonical", VERSION, RUN_AT)

    assert folder == tmp_path / "canonical" / "v=2026-11-15"
    assert sorted(path.name for path in folder.iterdir()) == [
        "ingestion_runs.parquet",
        "prices.parquet",
        "products.parquet",
        "stores.parquet",
    ]
    for name, schema in SCHEMAS.items():
        table = pl.read_parquet(folder / f"{name}.parquet")
        assert dict(table.schema) == schema
        assert table.height > 0


def test_dataset_version_defaults_to_the_run_date(raw_dir, tmp_path):
    folder = write_canonical(raw_dir, tmp_path / "canonical", ingestion_timestamp=RUN_AT)

    assert folder.name == "v=2026-11-15"
