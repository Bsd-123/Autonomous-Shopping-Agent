"""Tests for unit normalization and multi-pack expansion.

The cases are real products from the chains' files: name, quantity, unit,
unit_of_measure and the weighed flag as published.
"""

from pathlib import Path

import polars as pl
import pytest

from shopping_agent.ingestion.adapters import ADAPTERS
from shopping_agent.ingestion.unit_normalization import (
    BASE_SIZE_DTYPE,
    BaseSize,
    base_size,
    base_size_expr,
)

RAW = Path(__file__).parents[1] / "fixtures" / "raw"
FIELDS = ["name", "quantity", "unit", "unit_of_measure", "is_weighed"]


def size(name, quantity=None, unit=None, unit_of_measure=None, is_weighed=False) -> BaseSize:
    return base_size(name, quantity, unit, unit_of_measure, is_weighed)


class TestVolume:
    def test_millilitres(self):
        assert size('רוטב להקפצה 700 מ"ל', 700.0, "מיליליטר", "100 מיליליטר") == (700, "ml", "name")

    def test_litres_become_millilitres(self):
        assert size("ג'אמפ אשכוליות 1.5 ליטר", 1.5, "ליטר", "100 ליטר") == (1500, "ml", "name")

    def test_fields_alone(self):
        assert size("קצפת צמחית", 250.0, "מיליליטר", "100 מיליליטר") == (250, "ml", "fields")

    def test_unit_is_read_from_unit_of_measure_when_missing(self):
        assert size("מרכך כביסה לבן", 4.0, None, "ליטר") == (4000, "ml", "fields")
        assert size("חומץ בלסמי אורגני", 500.0, None, '100 מ"ל') == (500, "ml", "fields")

    def test_name_alone(self):
        assert size("אגקס לרצפה לבנדר 1 ליטר", 0.0, "ליטר", "1ליטר") == (1000, "ml", "name")


class TestWeight:
    def test_grams(self):
        assert size("לחם רחיים קל 500 גרם", 500.0, "גרם", "100 גרם") == (500, "g", "name")

    def test_kilograms_become_grams(self):
        assert size('קורנפלקס קלוגס 1 ק"ג', 1.0, "קילוגרם", "1 קילוגרם") == (1000, "g", "name")
        assert size("חומוס הנמל", 1.2, None, 'ק"ג') == (1200, "g", "fields")

    def test_fields_alone(self):
        assert size("מעדן סויה עם תות", 150.0, "גרם", "100 גרם") == (150, "g", "fields")

    def test_name_alone(self):
        assert size("חלה ברמן 500 גר'", 1.0, None, None) == (500, "g", "name")


class TestUnitCount:
    def test_count_in_the_name(self):
        assert size("ליפתנית קרם 36 יח", 1.0, "יחידות", "יחידות") == (36, "unit", "name")

    def test_count_in_the_fields(self):
        assert size("כפפות ניטריל שחורות", 80.0, "יחידות", "יחידות") == (80, "unit", "fields")

    def test_single_item(self):
        assert size("מטאטא מגב מסיליקון", 1.0, "יחידות", "יחידות") == (1, "unit", "fields")

    def test_no_size_anywhere_is_one_unit(self):
        assert size("טלק", 1.0, None, None) == (1, "unit", "default")
        assert size("נענע בוסתן", 0.0, "יחידות", "יחידה") == (1, "unit", "default")
        assert size(None) == (1, "unit", "default")

    def test_lengths_have_no_base_unit(self):
        assert size("רדיד אלומיניום 30X7.5מ'", 60.0, "מטרים", "מטרים") == (1, "unit", "default")

    def test_weight_wins_over_count(self):
        assert size("פיתות המכבים 5 יח", 500.0, None, "100 גרם") == (500, "g", "fields")
        assert size("טילון וניל 75 גרם", 1.0, "יחידות", "יחידות") == (75, "g", "name")


class TestMultiPack:
    def test_six_bottles_of_one_and_a_half_litres(self):
        name = "נביעות מים מינרליים, שישיה 6 * 1.5 ליטר פקדון"

        assert size(name, 1.5, "ליטר", "1 ליטר") == (9000, "ml", "name_multipack")

    @pytest.mark.parametrize(
        ("name", "quantity", "unit", "unit_of_measure", "expected"),
        [
            ("מים נביעות 6*1.5 ליט", 9.0, None, "ליטר", (9000, "ml")),
            ('בירה קרומבאכר מארז 330X6 מ"ל', 1900.0, "מיליליטר", "1 ליטר", (1980, "ml")),
            ("עגבניות חתוכות  מארז 400X3ג'", 1200.0, "גרם", "100 גרם", (1200, "g")),
            ("יופלה לבן 1.5% 8*150 גרם", 1.2, "קילוגרם", "1קילוגרם", (1200, "g")),
            ('שוקולד חלב לל"ס 2*100ג', 100.0, "גרם", "100 גרם", (200, "g")),
            ('סודה טמפו 500*12 מ"ל', 0.0, "ליטר", "1ליטר", (6000, "ml")),
            ("מאגדת פאף ברביקיו8*15גרם", 1.0, "יחידות", "יחידות", (120, "g")),
        ],
    )
    def test_weight_and_volume_packs_are_expanded(
        self, name, quantity, unit, unit_of_measure, expected
    ):
        assert size(name, quantity, unit, unit_of_measure) == (*expected, "name_multipack")

    def test_count_packs_are_expanded(self):
        result = size("מגבוני קמיל בלו 4*64 יח", 1.0, "יחידות", "יחידות")

        assert result == (256, "unit", "name_multipack")

    def test_fields_that_already_hold_the_total_are_kept(self):
        assert size("מארז 6 בקבוקים 1.5 ליטר", 9.0, "ליטר", "1 ליטר") == (9000, "ml", "fields")

    def test_a_weight_in_the_fields_wins_over_a_count_pack_in_the_name(self):
        result = size("קפסולות סטארבקס פמפקין 57*12 יח'", 684.0, "גרם", "100 גרם")

        assert result == (684, "g", "fields")

    def test_dimensions_are_not_a_multi_pack(self):
        assert size("וילון אמבטיה צבעוני 120*180סמ", 1.0, "יחידות", "יחידות") == (
            1,
            "unit",
            "fields",
        )


class TestWeighed:
    @pytest.mark.parametrize(
        ("name", "quantity", "unit", "unit_of_measure"),
        [
            ("חזה עוף קהילות מחיר לפי", 1.0, "קילוגרם", "1קילוגרם"),
            ("שניצל עוף טרי", 1.0, "יחידות", "יחידה"),
            ("רגל עוף", 0.26, "קילוגרם", "1 קילוגרם"),
            ('תמר ברהי מובחר 5 ק"ג', 1.0, "קילוגרם", "1קילוגרם"),
        ],
    )
    def test_a_weighed_product_is_one_kilogram(self, name, quantity, unit, unit_of_measure):
        assert size(name, quantity, unit, unit_of_measure, is_weighed=True) == (
            1000,
            "g",
            "weighed",
        )


class TestWhenNameAndFieldsDisagree:
    def test_the_label_wins_when_both_describe_the_same_pack(self):
        assert size("אריאל ג'ל 1.95 ליטר קולור", 1750.0, "מיליליטר", "1 ליטר")[:2] == (1950, "ml")
        assert size("גבישי שום מיה 250 גר", 300.0, "גרם", "100 גרם")[:2] == (250, "g")

    def test_a_liquid_recorded_in_grams_takes_the_label_unit(self):
        assert size('חומץ בלסמי 250 מ"ל', 250.0, "גרם", "100 גרם") == (250, "ml", "name")
        assert size('קארו סירופ תירס בהיר 470 מ"ל', 473.0, "גרם", "100 גרם") == (470, "ml", "name")

    def test_fields_win_when_the_name_states_something_else(self):
        assert size("קפסולות עלית מס 11 ל", 50.0, "גרם", "100 גרם") == (50, "g", "fields")
        assert size("מגנום שוקולד לבן 120מל", 85.0, "גרם", "100 גרם") == (85, "g", "fields")
        assert size("תה ירוק נענע 25 שק*1.5גר", 37.0, "גרם", "100 גרם") == (37, "g", "fields")

    def test_fields_win_over_an_impossible_size_in_the_name(self):
        assert size("פיירי לימון 1190 ליטר", 1190.0, "מיליליטר", "1 ליטר") == (1190, "ml", "fields")
        assert size('פטריות שמפניון ארוז 300 ק"ג', 300.0, "גרם", "100 גרם") == (300, "g", "fields")

    def test_the_name_wins_over_an_impossible_size_in_the_fields(self):
        assert size("מקלות חמצוצים פירות יער 115 גר", 115000.0, "גרם", "100 גרם") == (
            115,
            "g",
            "name",
        )
        assert size('כוהל 95% 500 מ"ל', 1.0, "מיליליטר", "100 מיליליטר") == (500, "ml", "name")

    def test_litres_entered_as_millilitres_are_corrected(self):
        assert size("נוזל רצפות", 1500.0, "ליטר", "100 ליטר") == (1500, "ml", "fields")

    def test_a_range_in_the_name_is_not_a_size(self):
        result = size("האגיס חיתול לילה מידה 5 12-23 קג 26 יח'", 26.0, "יחידות", "יחידות")

        assert result == (26, "unit", "name")

    def test_an_implausible_size_in_the_name_alone_is_ignored(self):
        assert size("מילוי גז לבלון 60 ליטר", 1.0, "יחידות", "יחידות") == (1, "unit", "fields")

    def test_two_different_sizes_in_the_name_are_ambiguous(self):
        result = size('סט 3 קופסאות 400 מ"ל 800 מ"ל', 3.0, "יחידות", "יחידות")

        assert result == (3, "unit", "fields")


@pytest.fixture(scope="module")
def products() -> pl.DataFrame:
    frames = [
        adapter.read_prices(RAW).select("chain", "item_code", *FIELDS) for adapter in ADAPTERS
    ]
    return pl.concat(frames).unique(["chain", "item_code"]).sort("chain", "item_code")


@pytest.fixture(scope="module")
def sized(products) -> pl.DataFrame:
    return products.with_columns(size=base_size_expr()).unnest("size")


class TestOnRealSamples:
    def test_every_product_has_a_base_quantity_and_unit(self, sized):
        assert sized.height > 400
        assert sized.select(pl.col("base_quantity", "base_unit").null_count()).row(0) == (0, 0)
        assert (sized["base_quantity"] > 0).all()
        assert set(sized["base_unit"]) <= {"ml", "g", "unit"}

    def test_every_weighed_product_is_one_kilogram(self, sized):
        weighed = sized.filter("is_weighed")

        assert weighed.height > 0
        assert weighed.select("base_quantity", "base_unit").unique().rows() == [(1000.0, "g")]

    def test_column_form_matches_the_plain_function(self, products, sized):
        expected = [base_size(*row) for row in products.select(FIELDS).iter_rows()]

        assert sized["base_quantity"].to_list() == [size.quantity for size in expected]
        assert sized["base_unit"].to_list() == [size.unit for size in expected]
        assert sized["base_source"].to_list() == [size.source for size in expected]

    def test_column_form_has_a_fixed_type(self, products):
        assert products.select(size=base_size_expr()).schema["size"] == BASE_SIZE_DTYPE
