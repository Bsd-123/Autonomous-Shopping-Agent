"""The adapters fail loudly when the raw files stop looking the way they were read."""

import polars as pl
import pytest
from support import (
    OSHER_PRICES,
    OSHER_STORES,
    RAW,
    stage_fixture,
    stage_shufersal,
    write_header_only,
    write_store_snapshots,
)

from shopping_agent.ingestion.adapters import (
    ChainAdapter,
    OsherAdAdapter,
    RawFormatError,
    YeshHesedAdapter,
)
from shopping_agent.ingestion.adapters.base import (
    RAW_PRICE_COLUMNS,
    RAW_STORE_COLUMNS,
    Parsed,
    build_frame,
)


class TestUnparseableValues:
    @pytest.mark.parametrize(
        ("column", "value"),
        [
            ("itemprice", "12,90"),
            ("itemprice", "1.505"),
            ("quantity", "abc"),
            ("quantity", "NaN"),
            ("quantity", "inf"),
            ("quantity", "1e999"),
            ("qtyinpackage", "1/2"),
            ("qtyinpackage", "NaN"),
            ("bisweighted", "yes"),
            ("storeid", "A6"),
            ("priceupdatetime", "yesterday"),
            # The hour skipped when the clocks go forward does not exist in Israel.
            ("priceupdatetime", "2026-03-27T02:30:00"),
        ],
    )
    def test_prices(self, tmp_path, column, value):
        stage_fixture(tmp_path, OSHER_PRICES, **{column: value})

        with pytest.raises(RawFormatError, match=f"{column} value"):
            OsherAdAdapter().read_prices(tmp_path)

    def test_store_number(self, tmp_path):
        stage_fixture(tmp_path, OSHER_STORES, storeid="A1")

        with pytest.raises(RawFormatError, match="storeid value"):
            OsherAdAdapter().read_stores(tmp_path)

    def test_store_rows_without_update_time(self, tmp_path):
        stage_fixture(tmp_path, OSHER_STORES, lastupdatedate="''")

        with pytest.raises(RawFormatError, match="without an update time"):
            OsherAdAdapter().read_stores(tmp_path)

    def test_an_unreadable_time_in_an_older_snapshot(self, tmp_path):
        write_store_snapshots(
            tmp_path, ("2026-11-13", "not a time", "1"), ("2026-11-14", "10:01:00", "2")
        )

        with pytest.raises(RawFormatError, match="updated value"):
            OsherAdAdapter().read_stores(tmp_path)

    def test_every_parsed_column_is_checked_for_lost_values(self):
        raw = pl.DataFrame({"file_name": ["f.xml"], "amount": ["x"]})
        columns = {"amount": Parsed("amount", pl.col("amount").cast(pl.Int64, strict=False))}

        with pytest.raises(RawFormatError, match=r"f\.xml: 1 amount value"):
            build_frame(raw, columns, pl.Schema({"amount": pl.Int64()}))


class TestEmptyResults:
    def test_a_price_file_without_rows(self, tmp_path):
        write_header_only(tmp_path, OSHER_PRICES, RAW_PRICE_COLUMNS)

        with pytest.raises(RawFormatError, match="no osher_ad price rows"):
            OsherAdAdapter().read_prices(tmp_path)

    def test_a_store_file_without_rows(self, tmp_path):
        write_header_only(tmp_path, OSHER_STORES, RAW_STORE_COLUMNS)

        with pytest.raises(RawFormatError, match="no osher_ad stores"):
            OsherAdAdapter().read_stores(tmp_path)

    def test_a_renamed_sub_chain_is_not_returned_as_an_empty_chain(self, monkeypatch):
        monkeypatch.setattr(
            "shopping_agent.ingestion.adapters.yesh_hesed.SUB_CHAIN_NAME", "renamed"
        )

        with pytest.raises(RawFormatError, match="no yesh_hesed stores"):
            YeshHesedAdapter().read_stores(RAW)

    def test_a_price_file_without_rows_of_the_chain_stores(self):
        # The sample holds rows of store 11 (another sub-chain) and of Yesh Hesed store 611.
        stores = YeshHesedAdapter().read_stores(RAW)
        without_611 = stores.filter(pl.col("store_id") != "yesh_hesed:611")

        with pytest.raises(RawFormatError, match="no yesh_hesed price rows"):
            YeshHesedAdapter().read_prices(RAW, without_611)


class TestStoreKey:
    def test_the_same_store_under_two_paddings_is_rejected(self, tmp_path):
        write_store_snapshots(
            tmp_path, ("2026-11-14", "08:05:00", "011"), ("2026-11-14", "08:05:00", "11")
        )

        with pytest.raises(RawFormatError, match="duplicate store_id values: osher_ad:11"):
            OsherAdAdapter().read_stores(tmp_path)


class TestYeshHesedValidationScope:
    def test_values_of_other_sub_chains_are_not_validated(self, tmp_path):
        # The first rows of the sample belong to store 11 of another sub-chain.
        stage_shufersal(tmp_path, itemprice="not a price")

        assert YeshHesedAdapter().read_prices(tmp_path).height == 160

    def test_values_of_yesh_hesed_stores_are_validated(self, tmp_path):
        stage_shufersal(tmp_path, storeid="611", itemprice="not a price")

        with pytest.raises(RawFormatError, match="itemprice value"):
            YeshHesedAdapter().read_prices(tmp_path)

    def test_an_unreadable_store_number_fails_even_in_rows_of_other_sub_chains(self, tmp_path):
        stage_shufersal(tmp_path, storeid="A6")

        with pytest.raises(RawFormatError, match="storeid value"):
            YeshHesedAdapter().read_prices(tmp_path)


def test_a_chain_adapter_without_its_settings_fails_when_defined():
    with pytest.raises(TypeError, match="source_folder, source_name"):

        class Incomplete(ChainAdapter):
            chain = "incomplete"
