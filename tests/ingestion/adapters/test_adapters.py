"""Adapter tests run against sample files cut from each chain's real data."""

import shutil
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

    def test_prices_keep_only_rows_of_yesh_hesed_stores(self, tmp_path):
        # Constructed price file in Shufersal's layout, paired with the real
        # store file. Store 681 belongs to another Shufersal sub-chain; store
        # 611 is a Yesh Hesed branch. Shufersal timestamps carry no milliseconds.
        folder = tmp_path / "shufersal"
        folder.mkdir()
        shutil.copy(RAW / "shufersal" / "store_file_shufersal.csv", folder)
        header = (
            "found_folder,file_name,chainid,subchainid,storeid,bikoretno,priceupdatetime,"
            "itemcode,lastsaledatetime,itemtype,itemname,manufacturename,manufacturecountry,"
            "manufactureitemdescription,unitqty,quantity,unitofmeasure,bisweighted,"
            "qtyinpackage,itemprice,unitofmeasureprice,allowdiscount,itemstatus"
        )
        rows = [
            "app_data/dumps/Shufersal,Price7290027600007-005-681-20261114-010000.xml,"
            "7290027600007,005,681,9,2026-11-14T00:10:00,3574661098531,2026-11-13T11:29:55,1,"
            "תח.גוף ניוטרוגינה מ400,ש.שסטוביץ,GR,תח.גוף ניוטרוגינה מ400,מיליליטר,400.00,"
            "100 מיליליטר,0,1,37.90,9.48,1,1",
            ",Price7290027600007-006-611-20261114-010000.xml,,006,611,,,3838824438696,"
            "2026-11-13T11:00:09,,קרם סופרם-צבע לשיער-6-0,סלובניה,SI,קרם סופרם-צבע לשיער-6-0,"
            "יחידות,1.00,יחידות,,,49.90,49.90,,",
            ",,,,,,,7290000000015,2026-11-13T12:00:00,,פריט לדוגמה,'',IL,פריט לדוגמה,"
            ",,,,,12.50,12.50,,",
        ]
        (folder / "price_file_shufersal.csv").write_text(
            "\n".join([header, *rows]) + "\n", encoding="utf-8"
        )

        prices = YeshHesedAdapter().read_prices(tmp_path)

        assert_price_layout(prices, "yesh_hesed")
        assert prices["store_id"].to_list() == ["yesh_hesed:611", "yesh_hesed:611"]
        assert prices["item_code"].to_list() == ["3838824438696", "7290000000015"]
        assert prices["price"].to_list() == [Decimal("49.90"), Decimal("12.50")]
        assert prices["manufacturer"].to_list() == ["סלובניה", None]
        assert prices["qty_in_package"].to_list() == [1.0, 1.0]
        assert prices["source_timestamp"].null_count() == 0
