"""Tests for Hebrew product-name normalization."""

import re
from pathlib import Path

import polars as pl
import pytest

from shopping_agent.ingestion.adapters import ADAPTERS
from shopping_agent.ingestion.text_normalization import (
    CANONICAL_UNITS,
    UNIT_VARIANTS,
    normalize_name,
    normalize_name_expr,
)

RAW = Path(__file__).parents[1] / "fixtures" / "raw"


def cases(table: dict[str, str]):
    return pytest.mark.parametrize(("name", "normalized"), table.items(), ids=list(table))


class TestNiqqud:
    @cases(
        {
            "חָלָב": "חלב",
            "שָׁלוֹם": "שלום",
            "לֶחֶם מָלֵא": "לחם מלא",
            "קֻבִּיּוֹת": "קביות",
        }
    )
    def test_niqqud_is_removed(self, name, normalized):
        assert normalize_name(name) == normalized

    def test_invisible_direction_marks_and_odd_spaces_are_removed(self):
        assert normalize_name("מים‏ 1.5 ליטר") == "מים 1.5 ליטר"

    def test_maqaf_separates_words(self):
        assert normalize_name("תל־אביב") == "תל אביב"


class TestGereshAndGershayim:
    @cases(
        {
            "צ׳יפס": "צ'יפס",
            "צ`יפס": "צ'יפס",
            "צ’יפס": "צ'יפס",
            "צ'יפס": "צ'יפס",
            "קוטג'": "קוטג'",
        }
    )
    def test_geresh_variants_become_an_apostrophe(self, name, normalized):
        assert normalize_name(name) == normalized

    @cases(
        {
            'תפו"א': 'תפו"א',
            "תפו״א": 'תפו"א',
            "תפו”א": 'תפו"א',
            "תפו''א": 'תפו"א',
            'בד"ץ': 'בד"ץ',
        }
    )
    def test_gershayim_variants_become_a_quotation_mark(self, name, normalized):
        assert normalize_name(name) == normalized

    @cases(
        {
            '"ליברס"': "ליברס",
            'מלח הארץ 1 ק"': 'מלח הארץ 1 ק"ג',
            '12" פיצה': "12 פיצה",
        }
    )
    def test_quotation_marks_that_are_not_part_of_a_word_are_dropped(self, name, normalized):
        assert normalize_name(name) == normalized


class TestUnits:
    @pytest.mark.parametrize(
        ("unit", "variant"),
        [(unit, variant) for unit, variants in UNIT_VARIANTS.items() for variant in variants],
    )
    def test_every_spelling_after_a_number_becomes_the_canonical_unit(self, unit, variant):
        assert normalize_name(f"מוצר 5{variant}") == f"מוצר 5 {unit}"
        assert normalize_name(f"מוצר 5 {variant} טעים") == f"מוצר 5 {unit} טעים"

    def test_canonical_units(self):
        assert CANONICAL_UNITS == ("גרם", 'ק"ג', 'מ"ל', "ליטר", "יחידות", 'ס"מ', 'מ"מ', "מטר")

    @cases(
        {
            "קרקר 110 גר": "קרקר 110 גרם",
            "ארטישוק  400ג'": "ארטישוק 400 גרם",
            'לל"ס21ג': 'לל"ס 21 גרם',
            'קצפת 170מ"ל': 'קצפת 170 מ"ל',
            "קצפת 170 מל": 'קצפת 170 מ"ל',
            "חלב טרי 1 ל": "חלב טרי 1 ליטר",
            "מים 1.5 ל'": "מים 1.5 ליטר",
            "מים 1,5 ל'": "מים 1,5 ליטר",
            "וודקה 1ליטר": "וודקה 1 ליטר",
            "אבקת כביסה 2.5ק FLOW": 'אבקת כביסה 2.5 ק"ג flow',
            "ביצים 30 יח' עטוף": "ביצים 30 יחידות עטוף",
            "ניילון נצמד 150 מ` רוח": "ניילון נצמד 150 מטר רוח",
            "מארז 400X3ג'": "מארז 400*3 גרם",
        }
    )
    def test_real_spellings(self, name, normalized):
        assert normalize_name(name) == normalized

    @cases(
        {
            "10 ג'ל": "10 ג'ל",
            "3 לשיער": "3 לשיער",
            "5 דגנים מ": "5 דגנים מ",
            "שמפו 750מ": "שמפו 750 מ",
            "גר' טעים": "גר' טעים",
            "ויטמין B12 ג": "ויטמין b12 ג",
            "לי ולך": "לי ולך",
        }
    )
    def test_text_that_only_looks_like_a_unit_is_left_alone(self, name, normalized):
        assert normalize_name(name) == normalized


class TestPunctuation:
    @cases(
        {
            "גבי.נפולאון24%+זיתים225ג": "גבי נפולאון 24% זיתים 225 גרם",
            "קרם סופרם-צבע לשיער-6-0": "קרם סופרם צבע לשיער 6 0",
            "מנקה רצפות בייבי, 2 ליטר": "מנקה רצפות בייבי 2 ליטר",
            "אורז פרסי (מובחר) 1.": "אורז פרסי מובחר 1",
            "* מבצע * 100% טבעי!": "מבצע 100% טבעי",
            "צנונית -יח": "צנונית יח",
        }
    )
    def test_noise_becomes_a_space(self, name, normalized):
        assert normalize_name(name) == normalized

    @cases(
        {
            "רוטב 1.89 ליטר": "רוטב 1.89 ליטר",
            "חלב 3 % שומן": "חלב 3% שומן",
            "שמנת 10%שומן": "שמנת 10% שומן",
            "טופי 8*15": "טופי 8*15",
            'קולה 6 * 330 מ"ל': 'קולה 6*330 מ"ל',
            "קולה 330X6": "קולה 330*6",
            "קולה 6×330": "קולה 6*330",
            "ג'ל כביסה 2 x צבעוני": "ג'ל כביסה 2 x צבעוני",
            "M&M 45 גרם": "m&m 45 גרם",
            "Kellogg's": "kellogg's",
        }
    )
    def test_meaningful_marks_are_kept_in_one_form(self, name, normalized):
        assert normalize_name(name) == normalized

    def test_letters_and_digits_are_separated(self):
        assert normalize_name("ניוטרוגינה מ400") == "ניוטרוגינה מ 400"
        assert normalize_name("מולר3% קיווי") == "מולר 3% קיווי"

    def test_latin_letters_are_lower_cased_and_whitespace_collapsed(self):
        assert normalize_name("  ביצים   גודל XL \n") == "ביצים גודל xl"

    @pytest.mark.parametrize("name", ["", "   ", "***", "‏"])
    def test_a_name_with_no_content_becomes_empty(self, name):
        assert normalize_name(name) == ""

    def test_missing_name_stays_missing(self):
        assert normalize_name(None) is None


class TestPrefixesAreKept:
    @pytest.mark.parametrize("word", ["לחם", "מלח", "בשר", "שמן"])
    def test_words_that_start_with_a_prefix_letter_are_unchanged(self, word):
        assert normalize_name(word) == word

    @pytest.mark.parametrize(
        "name",
        ["לחם מלא פרוס", "מלח גס", "בשר טחון", "שמן זית", "טונה בשמן", "ללא סוכר", "מהשדה", "וניל"],
    )
    def test_names_with_prefixes_are_unchanged(self, name):
        assert normalize_name(name) == name


@pytest.fixture(scope="module")
def names() -> list[str]:
    """Every distinct product name in the chains' sample files."""
    frames = [adapter.read_prices(RAW).select("name") for adapter in ADAPTERS]
    return pl.concat(frames)["name"].unique().sort().to_list()


class TestOnRealSamples:
    def test_normalizing_twice_changes_nothing(self, names):
        once = [normalize_name(name) for name in names]

        assert [normalize_name(name) for name in once] == once

    def test_no_name_is_emptied(self, names):
        assert all(normalize_name(name) for name in names)

    def test_only_meaningful_marks_remain(self, names):
        leftovers = {
            character
            for name in names
            for character in normalize_name(name)
            if not (character.isalnum() or character == " ")
        }

        assert leftovers <= set("'\"%*.,&")

    def test_no_unit_abbreviation_is_left_after_a_number(self, names):
        abbreviations = [
            variant
            for variants in UNIT_VARIANTS.values()
            for variant in variants
            if variant not in CANONICAL_UNITS
        ]
        pattern = re.compile(
            r"\d\s?(?:"
            + "|".join(re.escape(variant) for variant in abbreviations)
            + r")(?![א-ת'\"])"
        )

        assert [name for name in names if pattern.search(normalize_name(name))] == []

    def test_column_form_matches_the_plain_function(self, names):
        frame = pl.DataFrame({"name": [*names, None]})

        result = frame.select(normalized=normalize_name_expr())["normalized"].to_list()

        assert result == [normalize_name(name) for name in names] + [None]
