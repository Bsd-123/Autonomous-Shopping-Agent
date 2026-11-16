"""Tests for the canonical tables and the Parquet writer."""

import shutil
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from shopping_agent.ingestion.adapters import OsherAdAdapter, RawFormatError
from shopping_agent.ingestion.adapters.layout import SOURCE_TIME
from shopping_agent.ingestion.canonical import (
    INGESTION_RUNS_SCHEMA,
    PRICES_SCHEMA,
    PRODUCTS_SCHEMA,
    SCHEMAS,
    STORES_SCHEMA,
    CanonicalError,
    build_canonical,
    build_prices,
    build_products,
    require_integrity,
    write_canonical,
)
from support import OSHER_PRICES, OSHER_STORES, RAW, stage_fixture

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


def test_published_times_keep_israel_time(tables):
    assert PRICES_SCHEMA["source_timestamp"] == SOURCE_TIME
    assert INGESTION_RUNS_SCHEMA["newest_source_timestamp"] == SOURCE_TIME
    assert tables["prices"]["source_timestamp"][0].utcoffset().total_seconds() == 2 * 3600


def test_last_seen_date_is_the_israeli_date_of_the_publish_time(raw_dir, tmp_path):
    # 00:30 in Israel is still the day before in UTC.
    stage_fixture(
        tmp_path, OSHER_PRICES, file_name="Price7290103152017-001-006-20261114-003000.xml"
    )

    prices = build_prices(OsherAdAdapter().read_prices(tmp_path), VERSION, RUN_AT)

    assert prices["last_seen_date"].unique().to_list() == [date(2026, 11, 14)]
    assert prices["source_timestamp"][0].isoformat() == "2026-11-14T00:30:00+02:00"


def osher_folder(raw_dir_: Path, **first_row: str) -> Path:
    """Osher Ad's sample files, with cells of the first price row replaced."""
    stage_fixture(raw_dir_, OSHER_PRICES, **first_row)
    shutil.copy(OSHER_STORES, raw_dir_ / "osherad" / OSHER_STORES.name)
    return raw_dir_


class TestIntegrity:
    def test_a_listing_without_an_item_code_is_rejected(self, tmp_path):
        folder = osher_folder(tmp_path, itemcode="''")

        with pytest.raises(CanonicalError, match=r"products\.product_key has 1 null value"):
            build_canonical(folder, VERSION, RUN_AT, [OsherAdAdapter()])

    def test_prices_of_a_store_missing_from_the_store_file_are_rejected(self, tmp_path):
        folder = osher_folder(tmp_path, storeid="999")

        with pytest.raises(RawFormatError, match="prices for unknown stores: osher_ad:999"):
            build_canonical(folder, VERSION, RUN_AT, [OsherAdAdapter()])

    def test_the_tables_of_a_run_pass(self, tables):
        require_integrity(tables)

    def test_a_price_for_an_unlisted_store_is_rejected(self, tables):
        stores = tables["stores"].filter(pl.col("store_id") != "osher_ad:6")

        with pytest.raises(CanonicalError, match=r"prices\.store_id refers to values missing"):
            require_integrity({**tables, "stores": stores})

    def test_a_price_for_an_unknown_product_is_rejected(self, tables):
        products = tables["products"].filter(pl.col("product_key") != "osher_ad:7290006775405")

        with pytest.raises(CanonicalError, match=r"prices\.product_key refers to .*7290006775405"):
            require_integrity({**tables, "products": products})

    @pytest.mark.parametrize(
        ("table", "column"), [("products", "product_key"), ("stores", "store_id")]
    )
    def test_a_repeated_key_is_rejected(self, tables, table, column):
        repeated = pl.concat([tables[table], tables[table].head(1)])

        with pytest.raises(CanonicalError, match=f"{table}.{column} is not unique"):
            require_integrity({**tables, table: repeated})

    @pytest.mark.parametrize(
        ("table", "column"),
        [
            ("products", "product_key"),
            ("stores", "store_id"),
            ("prices", "product_key"),
            ("prices", "store_id"),
        ],
    )
    def test_a_missing_key_is_rejected(self, tables, table, column):
        first = pl.int_range(pl.len()) == 0
        broken = tables[table].with_columns(
            pl.when(first).then(None).otherwise(pl.col(column)).alias(column)
        )

        with pytest.raises(CanonicalError, match=f"{table}.{column} has 1 null value"):
            require_integrity({**tables, table: broken})


class TestWriteIsAtomic:
    @staticmethod
    def fail_on_the_third_write(monkeypatch):
        writes = []
        original = pl.DataFrame.write_parquet

        def flaky(self, *args, **kwargs):
            writes.append(1)
            if len(writes) == 3:
                raise OSError("disk full")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(pl.DataFrame, "write_parquet", flaky)

    def test_a_failed_new_version_leaves_nothing_behind(self, raw_dir, tmp_path, monkeypatch):
        self.fail_on_the_third_write(monkeypatch)

        with pytest.raises(OSError, match="disk full"):
            write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)

        assert list(tmp_path.iterdir()) == []

    def test_a_failed_rerun_keeps_the_previous_dataset_whole(self, raw_dir, tmp_path, monkeypatch):
        write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)
        self.fail_on_the_third_write(monkeypatch)

        with pytest.raises(OSError, match="disk full"):
            write_canonical(raw_dir, tmp_path, VERSION, RUN_AT + timedelta(hours=6))

        assert [path.name for path in tmp_path.iterdir()] == ["v=2026-11-15"]
        for name in ("prices", "ingestion_runs"):
            table = pl.read_parquet(tmp_path / "v=2026-11-15" / f"{name}.parquet")
            assert table["ingestion_timestamp"].unique().to_list() == [RUN_AT]

    def test_a_rerun_replaces_every_file_of_the_dataset(self, raw_dir, tmp_path):
        later = RUN_AT + timedelta(hours=6)
        write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)

        write_canonical(raw_dir, tmp_path, VERSION, later)

        assert [path.name for path in tmp_path.iterdir()] == ["v=2026-11-15"]
        for name in ("prices", "ingestion_runs"):
            table = pl.read_parquet(tmp_path / "v=2026-11-15" / f"{name}.parquet")
            assert table["ingestion_timestamp"].unique().to_list() == [later]

    def test_a_failed_swap_restores_the_previous_dataset(self, raw_dir, tmp_path, monkeypatch):
        write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)
        original = Path.rename

        def refuse_staging(self, target):
            if self.name.endswith(".tmp"):
                raise OSError("rename failed")
            return original(self, target)

        monkeypatch.setattr(Path, "rename", refuse_staging)

        with pytest.raises(OSError, match="rename failed"):
            write_canonical(raw_dir, tmp_path, VERSION, RUN_AT + timedelta(hours=6))

        assert [path.name for path in tmp_path.iterdir()] == ["v=2026-11-15"]
        prices = pl.read_parquet(tmp_path / "v=2026-11-15" / "prices.parquet")
        assert prices["ingestion_timestamp"].unique().to_list() == [RUN_AT]

    def test_a_failed_swap_of_a_new_version_leaves_nothing_behind(
        self, raw_dir, tmp_path, monkeypatch
    ):
        original = Path.rename

        def refuse_staging(self, target):
            if self.name.endswith(".tmp"):
                raise OSError("rename failed")
            return original(self, target)

        monkeypatch.setattr(Path, "rename", refuse_staging)

        with pytest.raises(OSError, match="rename failed"):
            write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)

        assert list(tmp_path.iterdir()) == []

    def test_no_version_folder_is_visible_while_the_files_are_written(
        self, raw_dir, tmp_path, monkeypatch
    ):
        seen = []
        original = pl.DataFrame.write_parquet

        def spy(self, file, *args, **kwargs):
            seen.append(sorted(path.name for path in tmp_path.iterdir()))
            return original(self, file, *args, **kwargs)

        monkeypatch.setattr(pl.DataFrame, "write_parquet", spy)

        write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)

        assert seen and all(not name.startswith("v=") for names in seen for name in names)


class TestDatasetVersion:
    @pytest.mark.parametrize("version", ["2026-11-15", "2026-11-15.2", "rc_1", "7"])
    def test_a_plain_token_is_accepted(self, raw_dir, version):
        assert build_canonical(raw_dir, version, RUN_AT)["prices"]["dataset_version"][0] == version

    @pytest.mark.parametrize("version", ["", " ", "a b", "../x", "a/b", "a\\b", ".hidden", "-x"])
    def test_anything_else_is_rejected_before_a_file_is_written(self, raw_dir, tmp_path, version):
        with pytest.raises(CanonicalError, match="invalid dataset version"):
            write_canonical(raw_dir, tmp_path / "out", version, RUN_AT)

        assert not (tmp_path / "out").exists()

    def test_the_default_is_the_utc_date_of_the_run(self, raw_dir, tmp_path):
        # 00:30 in Israel is still the day before in UTC.
        israel = timezone(timedelta(hours=2))

        folder = write_canonical(
            raw_dir, tmp_path, ingestion_timestamp=datetime(2026, 11, 15, 0, 30, tzinfo=israel)
        )

        assert folder.name == "v=2026-11-14"


class TestIngestionTimestamp:
    def test_a_timestamp_without_a_time_zone_is_rejected(self, raw_dir, tmp_path):
        naive = datetime(2026, 11, 15, 3, 0)

        with pytest.raises(CanonicalError, match="timezone-aware"):
            build_canonical(raw_dir, VERSION, naive)
        with pytest.raises(CanonicalError, match="timezone-aware"):
            build_prices(OsherAdAdapter().read_prices(RAW), VERSION, naive)
        with pytest.raises(CanonicalError, match="timezone-aware"):
            write_canonical(raw_dir, tmp_path, ingestion_timestamp=naive)


def test_products_choose_between_listings_published_together_by_store():
    base = OsherAdAdapter().read_prices(RAW).head(1)
    listings = [
        base.with_columns(store_id=pl.lit("osher_ad:1"), name=pl.lit("from store 1")),
        base.with_columns(store_id=pl.lit("osher_ad:2"), name=pl.lit("from store 2")),
    ]

    forward = build_products(pl.concat(listings))
    backward = build_products(pl.concat(listings[::-1]))

    assert forward["name"].to_list() == backward["name"].to_list() == ["from store 2"]


def test_prices_become_exact_agorot():
    shekels = ["0.29", "1.15", "19.99", "0.07", "8.20", "99999999.99"]
    prices = OsherAdAdapter().read_prices(RAW).head(len(shekels))
    prices = prices.with_columns(price=pl.Series(shekels).cast(pl.Decimal(12, 2)))

    result = build_prices(prices, VERSION, RUN_AT)

    assert result["price_agorot"].to_list() == [29, 115, 1999, 7, 820, 9_999_999_999]
