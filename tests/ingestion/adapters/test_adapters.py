"""Adapter tests run against sample files cut from each chain's real data."""

from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from shopping_agent.ingestion.adapters import (
    PRICE_SCHEMA,
    STORE_SCHEMA,
    Maayan2000Adapter,
    OsherAdAdapter,
    RawFormatError,
    YeshHesedAdapter,
)
from shopping_agent.ingestion.adapters.base import read_expanded

RAW = Path(__file__).parents[2] / "fixtures" / "raw"
PLACEHOLDERS = ["''", "", "לא ידוע", "XXXXX"]


def assert_no_placeholders(frame: pl.DataFrame) -> None:
    text = frame.select(pl.col(pl.String))
    leaked = text.select(pl.all().is_in(PLACEHOLDERS).sum()).row(0)
    assert sum(leaked) == 0


def assert_store_layout(stores: pl.DataFrame, chain: str) -> None:
    assert dict(stores.schema) == STORE_SCHEMA
    assert stores["chain"].unique().to_list() == [chain]
    assert stores["store_id"].is_unique().all()
    assert stores["store_id"].str.starts_with(f"{chain}:").all()
    assert stores.select(pl.col("store_id", "name", "source_timestamp").null_count().sum()).row(
        0
    ) == (0, 0, 0)
    assert_no_placeholders(stores)


def assert_price_layout(prices: pl.DataFrame, chain: str) -> None:
    assert dict(prices.schema) == PRICE_SCHEMA
    assert prices["chain"].unique().to_list() == [chain]
    required = ["store_id", "item_code", "name", "is_weighed", "price", "source_file"]
    assert sum(prices.select(pl.col(required).null_count()).row(0)) == 0
    assert (prices["price"] > 0).all()
    assert_no_placeholders(prices)


def row(frame: pl.DataFrame, **where: str) -> dict:
    matches = frame.filter(**where)
    assert matches.height >= 1, where
    return matches.row(0, named=True)


class TestReadExpanded:
    def test_empty_cell_repeats_the_row_above_and_quotes_mean_missing(self, tmp_path):
        path = tmp_path / "sample.csv"
        path.write_text("a,b,c\nx,1,''\n,2,\n''  ,,z\n", encoding="utf-8")

        frame = read_expanded(path, ("a", "b", "c"))

        assert frame.rows() == [("x", "1", None), ("x", "2", None), (None, "2", "z")]

    def test_missing_column_fails_loudly(self, tmp_path):
        path = tmp_path / "sample.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")

        with pytest.raises(RawFormatError, match="missing columns: c"):
            read_expanded(path, ("a", "c"))

    def test_missing_file_fails_loudly(self, tmp_path):
        with pytest.raises(RawFormatError, match="not found"):
            OsherAdAdapter().read_stores(tmp_path)
        with pytest.raises(RawFormatError, match="no price files"):
            OsherAdAdapter().read_prices(tmp_path)


class TestOsherAd:
    def test_stores(self):
        stores = OsherAdAdapter().read_stores(RAW)

        assert_store_layout(stores, "osher_ad")
        assert stores.height == 24
        bnei_brak = row(stores, store_id="osher_ad:11")
        assert bnei_brak["name"] == "בני ברק"
        assert bnei_brak["address"] == "הקישון 11"
        assert bnei_brak["city"] == "6100"

    def test_prices(self):
        prices = OsherAdAdapter().read_prices(RAW)

        assert_price_layout(prices, "osher_ad")
        assert prices.height == 218
        assert set(prices["store_id"]) == {"osher_ad:6", "osher_ad:22"}
        # Second row of the file: most cells are empty and repeat the first row.
        muffin = row(prices, item_code="7290006775405")
        assert muffin["store_id"] == "osher_ad:6"
        assert muffin["price"] == Decimal("1.50")
        assert muffin["quantity"] == 45.0
        assert muffin["unit"] == "גרם"
        assert muffin["is_weighed"] is False
        assert muffin["source_file"].startswith("Price7290103152017-001-006-")
        # The chain's "unknown" placeholder becomes null.
        assert prices["qty_in_package"].null_count() == prices.height
        assert prices["unit"].null_count() == 76
        assert not prices["unit_of_measure"].str.ends_with(" ").any()


class TestMaayan2000:
    def test_stores(self):
        stores = Maayan2000Adapter().read_stores(RAW)

        assert_store_layout(stores, "maayan_2000")
        assert stores.height == 38
        bnei_brak = row(stores, store_id="maayan_2000:38")
        assert bnei_brak["address"] == "רח רבי עקיבא 94"
        assert bnei_brak["city"] == "6100"

    def test_prices(self):
        prices = Maayan2000Adapter().read_prices(RAW)

        assert_price_layout(prices, "maayan_2000")
        assert prices.height == 300
        potatoes = row(prices, item_code="116", store_id="maayan_2000:18")
        assert potatoes["price"] == Decimal("5.90")
        assert potatoes["quantity"] == 0.15
        assert potatoes["unit"] == "קילוגרם"
        assert potatoes["is_weighed"] is True
        assert potatoes["manufacturer"] is None
        assert potatoes["source_timestamp"] is not None


class TestYeshHesed:
    def test_stores_keep_only_the_sub_chain_from_the_latest_snapshot(self):
        stores = YeshHesedAdapter().read_stores(RAW)

        assert_store_layout(stores, "yesh_hesed")
        assert stores.height == 22
        assert stores["name"].str.starts_with("יש חסד").all()
        assert stores["source_file"].unique().to_list() == [
            "Stores7290027600007-000-20261115-020.xml"
        ]
        bnei_brak = stores.filter(city="6100")["store_id"].to_list()
        assert sorted(bnei_brak) == ["yesh_hesed:219", "yesh_hesed:295", "yesh_hesed:611"]

    def test_prices_keep_only_rows_of_yesh_hesed_stores(self):
        # The sample holds 40 rows of store 11, which belongs to another
        # Shufersal sub-chain, followed by 160 rows of Yesh Hesed store 611.
        prices = YeshHesedAdapter().read_prices(RAW)

        assert_price_layout(prices, "yesh_hesed")
        assert prices.height == 160
        assert prices["store_id"].unique().to_list() == ["yesh_hesed:611"]
        assert prices["source_file"].unique().to_list() == [
            "PriceFull7290027600007-006-611-20261114-030000.xml"
        ]
        beans = row(prices, item_code="11182700954")
        assert beans["name"] == "שעועית שחורה 439 גרם"
        assert beans["price"] == Decimal("9.90")
        assert beans["quantity"] == 439.0
        assert beans["unit"] == "גרם"
        assert beans["qty_in_package"] == 1.0
        assert beans["is_weighed"] is False
        # Shufersal timestamps carry no milliseconds.
        assert prices["source_timestamp"].null_count() == 0


def copy_fixture(source: Path, target: Path, **first_row: str) -> None:
    """Copy a fixture CSV, replacing cells of its first row (which holds every value)."""
    frame = pl.read_csv(source, infer_schema=False)
    first = pl.int_range(pl.len()) == 0
    frame = frame.with_columns(
        pl.when(first).then(pl.lit(value)).otherwise(pl.col(column)).alias(column)
        for column, value in first_row.items()
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(target)


class TestFailsLoudlyOnUnparseableValues:
    PRICES = RAW / "osherad" / "price_file_osher_ad.csv"
    STORES = RAW / "osherad" / "store_file_osher_ad.csv"

    @pytest.mark.parametrize(
        ("column", "value"),
        [
            ("itemprice", "12,90"),
            ("itemprice", "1.505"),
            ("quantity", "abc"),
            ("qtyinpackage", "1/2"),
            ("bisweighted", "yes"),
            ("storeid", "A6"),
            ("priceupdatetime", "yesterday"),
        ],
    )
    def test_prices(self, tmp_path, column, value):
        copy_fixture(self.PRICES, tmp_path / "osherad" / self.PRICES.name, **{column: value})

        with pytest.raises(RawFormatError, match=f"{column} value"):
            OsherAdAdapter().read_prices(tmp_path)

    def test_store_number(self, tmp_path):
        copy_fixture(self.STORES, tmp_path / "osherad" / self.STORES.name, storeid="A1")

        with pytest.raises(RawFormatError, match="storeid value"):
            OsherAdAdapter().read_stores(tmp_path)

    def test_store_rows_without_update_time(self, tmp_path):
        copy_fixture(self.STORES, tmp_path / "osherad" / self.STORES.name, lastupdatedate="''")

        with pytest.raises(RawFormatError, match="without an update time"):
            OsherAdAdapter().read_stores(tmp_path)


def test_full_and_update_price_files_are_both_read(tmp_path):
    source = RAW / "osherad" / "price_file_osher_ad.csv"
    folder = tmp_path / "osherad"
    copy_fixture(source, folder / "price_file_osher_ad.csv")
    copy_fixture(source, folder / "price_full_file_osher_ad.csv", file_name="PriceFull.xml")

    prices = OsherAdAdapter().read_prices(tmp_path)

    assert_price_layout(prices, "osher_ad")
    # The full list comes first, then the updates; rows are kept as published.
    full, update = prices.head(218), prices.tail(218)
    assert full["source_file"][0] == "PriceFull.xml"
    assert full.drop("source_file").equals(update.drop("source_file"))
