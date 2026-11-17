"""Tests for product keys and barcode validation."""

from pathlib import Path

import polars as pl
import pytest

from shopping_agent.ingestion.adapters import ADAPTERS
from shopping_agent.ingestion.product_key import (
    barcode_expr,
    normalize_barcode,
    product_key,
    product_key_expr,
)

RAW = Path(__file__).parents[1] / "fixtures" / "raw"
CHAINS = ["yesh_hesed", "osher_ad", "maayan_2000"]

# Real codes from the chains' files.
EAN_13 = "7290006775405"
UPC_A = "811146031465"
UPC_A_WITHOUT_LEADING_ZERO = "11182700954"
EAN_8 = "72940952"

NOT_BARCODES = {
    "invalid check digit": "7290006775406",
    "short internal code": "116",
    "short code that passes the check digit": "505",
    "seven digits": "1234565",
    "nine digits": "123456782",
    "ten digits": "1234567895",
    "fourteen digits": "17290006775402",
    "letters": "72900067754A5",
    "inner space": "7290006 775405",
    "padding": " 7290006775405",
    "non-ASCII digits": "٧٢٩٠٠٠٦٧٧٥٤٠٥",
    "empty": "",
    "in-store EAN-13": "2000000000015",
    "in-store UPC-A": "200000000008",
    "in-store EAN-8": "21610660",
}


class TestBarcode:
    @pytest.mark.parametrize(
        ("code", "barcode"),
        [
            (EAN_13, "7290006775405"),
            (UPC_A, "0811146031465"),
            (UPC_A_WITHOUT_LEADING_ZERO, "0011182700954"),
            (EAN_8, "0000072940952"),
        ],
        ids=["EAN-13", "UPC-A", "UPC-A without leading zero", "EAN-8"],
    )
    def test_valid_barcode_is_written_as_13_digits(self, code, barcode):
        assert normalize_barcode(code) == barcode

    @pytest.mark.parametrize("code", NOT_BARCODES.values(), ids=NOT_BARCODES.keys())
    def test_other_codes_are_not_barcodes(self, code):
        assert normalize_barcode(code) is None

    def test_missing_code_is_not_a_barcode(self):
        assert normalize_barcode(None) is None

    def test_dropped_leading_zeros_do_not_change_the_barcode(self):
        assert normalize_barcode("0" + UPC_A) == normalize_barcode(UPC_A)
        assert normalize_barcode("0" + UPC_A_WITHOUT_LEADING_ZERO) == normalize_barcode(
            UPC_A_WITHOUT_LEADING_ZERO
        )


class TestProductKey:
    @pytest.mark.parametrize("code", [EAN_13, UPC_A, UPC_A_WITHOUT_LEADING_ZERO, EAN_8])
    def test_valid_barcode_gives_the_same_key_in_every_chain(self, code):
        keys = {product_key(chain, code) for chain in CHAINS}

        assert keys == {normalize_barcode(code)}

    @pytest.mark.parametrize("code", NOT_BARCODES.values(), ids=NOT_BARCODES.keys())
    def test_other_codes_are_namespaced_by_chain(self, code):
        keys = [product_key(chain, code) for chain in CHAINS]

        assert keys == [f"{chain}:{code}" for chain in CHAINS]
        assert len(set(keys)) == len(CHAINS)

    def test_the_three_cases_named_in_the_task(self):
        assert product_key("osher_ad", "7290006775405") == "7290006775405"
        assert product_key("osher_ad", "7290006775406") == "osher_ad:7290006775406"
        assert product_key("maayan_2000", "116") == "maayan_2000:116"

    def test_a_namespaced_key_never_equals_a_barcode_key(self):
        assert ":" in product_key("osher_ad", "116")
        assert ":" not in product_key("osher_ad", EAN_13)


class TestColumnForm:
    def test_matches_the_plain_functions_on_every_case(self):
        codes = [EAN_13, UPC_A, UPC_A_WITHOUT_LEADING_ZERO, EAN_8, *NOT_BARCODES.values(), None]
        frame = pl.DataFrame({"chain": ["osher_ad"] * len(codes), "item_code": codes})

        result = frame.select(barcode=barcode_expr(), key=product_key_expr())

        assert result["barcode"].to_list() == [normalize_barcode(code) for code in codes]
        expected_keys = [None if code is None else product_key("osher_ad", code) for code in codes]
        assert result["key"].to_list() == expected_keys

    @pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda adapter: adapter.chain)
    def test_matches_the_plain_functions_on_the_chain_samples(self, adapter):
        prices = adapter.read_prices(RAW).select("chain", "item_code").unique()

        keys = prices.with_columns(key=product_key_expr())

        expected = [product_key(chain, code) for chain, code in prices.iter_rows()]
        assert keys["key"].to_list() == expected
        # Every chain's sample holds real barcodes.
        assert keys["key"].str.contains(":").not_().any()

    def test_chain_internal_codes_of_a_real_sample_stay_inside_the_chain(self):
        maayan = next(adapter for adapter in ADAPTERS if adapter.chain == "maayan_2000")
        prices = maayan.read_prices(RAW).select("chain", "item_code").unique()

        keys = prices.with_columns(key=product_key_expr())

        potatoes = keys.filter(item_code="116")["key"].to_list()
        assert potatoes == ["maayan_2000:116"]
        internal = keys.filter(pl.col("item_code").str.len_chars() < 8)
        assert internal.height > 0
        assert internal["key"].str.starts_with("maayan_2000:").all()
