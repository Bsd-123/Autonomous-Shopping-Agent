"""Adapter tests run against sample files cut from each chain's real data."""

from decimal import Decimal

import pytest

from shopping_agent.ingestion.adapters import (
    ADAPTERS,
    Maayan2000Adapter,
    OsherAdAdapter,
    RawFormatError,
    YeshHesedAdapter,
)
from shopping_agent.ingestion.adapters.base import read_expanded
from support import (
    OSHER_PRICES,
    RAW,
    assert_price_layout,
    assert_store_layout,
    row,
    stage_fixture,
    stage_shufersal,
    write_store_snapshots,
)


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

    def test_full_and_update_price_files_are_both_read(self, tmp_path):
        full = stage_fixture(tmp_path, OSHER_PRICES, file_name="PriceFull.xml")
        full.rename(full.with_name("price_full_file_osher_ad.csv"))
        stage_fixture(tmp_path, OSHER_PRICES)

        prices = OsherAdAdapter().read_prices(tmp_path)

        assert_price_layout(prices, "osher_ad")
        # The full list comes first, then the updates; rows are kept as published.
        full_rows, update_rows = prices.head(218), prices.tail(218)
        assert full_rows["source_file"][0] == "PriceFull.xml"
        assert full_rows.drop("source_file").equals(update_rows.drop("source_file"))


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

    def test_stores_can_be_passed_in_so_the_store_file_is_not_read_again(self, tmp_path):
        stores = YeshHesedAdapter().read_stores(RAW)
        stage_shufersal(tmp_path, stores=False)

        prices = YeshHesedAdapter().read_prices(tmp_path, stores)

        assert prices.equals(YeshHesedAdapter().read_prices(RAW))


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda adapter: adapter.chain)
def test_passing_the_stores_never_changes_the_prices(adapter):
    stores = adapter.read_stores(RAW)

    assert adapter.read_prices(RAW, stores).equals(adapter.read_prices(RAW))


class TestIsraelLocalTime:
    @pytest.mark.parametrize(
        ("published", "expected"),
        [
            ("2026-01-15T07:46:57", "2026-01-15T07:46:57+02:00"),
            ("2026-07-01T12:00:00", "2026-07-01T12:00:00+03:00"),
            # The repeated hour of the clocks going back is read as the earlier one.
            ("2026-10-25T01:30:00", "2026-10-25T01:30:00+03:00"),
        ],
    )
    def test_published_time_is_read_as_israel_local_time(self, tmp_path, published, expected):
        stage_fixture(tmp_path, OSHER_PRICES, priceupdatetime=published)

        timestamp = OsherAdAdapter().read_prices(tmp_path)["source_timestamp"][0]

        assert timestamp.isoformat() == expected

    def test_store_snapshot_time_is_israel_local_time(self):
        stores = OsherAdAdapter().read_stores(RAW)

        assert stores["source_timestamp"][0].isoformat() == "2026-11-14T08:05:00.900000+02:00"


class TestStoreSnapshots:
    def test_newest_snapshot_is_chosen_by_time_not_by_text(self, tmp_path):
        # As text "9:05:00" sorts after "10:01:00", but it is the earlier time.
        write_store_snapshots(
            tmp_path, ("2026-11-14", "9:05:00", "1"), ("2026-11-14", "10:01:00", "2")
        )

        stores = OsherAdAdapter().read_stores(tmp_path)

        assert stores["store_id"].to_list() == ["osher_ad:2"]

    def test_a_store_listed_in_two_snapshots_is_not_a_duplicate(self, tmp_path):
        write_store_snapshots(
            tmp_path, ("2026-11-13", "08:00:00", "11"), ("2026-11-14", "08:05:00", "11")
        )

        stores = OsherAdAdapter().read_stores(tmp_path)

        assert stores["store_id"].to_list() == ["osher_ad:11"]
