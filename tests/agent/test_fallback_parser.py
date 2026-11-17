"""Tests for the rule-based fallback parser."""

import pytest

from shopping_agent.agent.fallback_parser import (
    MAX_LINES,
    FallbackParse,
    parse_line,
    parse_list,
)
from shopping_agent.contracts import MAX_COUNT, Attribute, ParsedItem, Quantity
from shopping_agent.ingestion.text_normalization import UNIT_VARIANTS


def ml(value: float) -> Quantity:
    return Quantity(value=value, unit="ml")


def g(value: float) -> Quantity:
    return Quantity(value=value, unit="g")


class TestSize:
    """A number followed by a unit is a size."""

    @pytest.mark.parametrize(
        ("line", "item", "quantity"),
        [
            ("חלב 3 ליטר", "חלב", ml(3000)),
            ("3 ליטר חלב", "חלב", ml(3000)),
            ("חלב 1.5 ל'", "חלב", ml(1500)),
            ('שמנת 250 מ"ל', "שמנת", ml(250)),
            ("שמנת 250מל", "שמנת", ml(250)),
            ('קמח 1 ק"ג', "קמח", g(1000)),
            ("עגבניות 1.5 קילו", "עגבניות", g(1500)),
            ("2 קילו תפוחי אדמה", "תפוחי אדמה", g(2000)),
            ("גבינה צהובה 200 גרם", "גבינה צהובה", g(200)),
            ("גבינה צהובה 200גר'", "גבינה צהובה", g(200)),
            ('אורז 1,5 ק"ג', "אורז", g(1500)),
        ],
    )
    def test_a_size_alone_is_a_total_quantity(self, line, item, quantity):
        assert parse_line(line) == ParsedItem(item=item, total_quantity=quantity)

    def test_the_unit_table_is_the_one_ingestion_uses(self):
        for unit, expected in (("גרם", g(500)), ('מ"ל', ml(500))):
            for spelling in UNIT_VARIANTS[unit]:
                parsed = parse_line(f"מוצר 500 {spelling}")
                assert parsed.total_quantity == expected, spelling
                assert parsed.item == "מוצר", spelling
        for unit, expected in (('ק"ג', g(2000)), ("ליטר", ml(2000))):
            for spelling in UNIT_VARIANTS[unit]:
                assert parse_line(f"מוצר 2 {spelling}").total_quantity == expected, spelling

    def test_a_length_is_not_a_size(self):
        parsed = parse_line("נייר אפייה 20 מטר")

        assert parsed.total_quantity is None
        assert parsed.pack_size is None
        assert parsed.item == "נייר אפייה 20 מטר"


class TestCount:
    """A bare number is a count."""

    @pytest.mark.parametrize(
        ("line", "item", "count"),
        [
            ("חלב 2", "חלב", 2),
            ("2 חלב", "חלב", 2),
            ("6 עגבניות", "עגבניות", 6),
            ("חלב x2", "חלב", 2),
            ("חלב X 2", "חלב", 2),
            ("חלב 2x", "חלב", 2),
            ("חלב *2", "חלב", 2),
            ("לחם ×3", "לחם", 3),
            ("ביצים 12 יחידות", "ביצים", 12),
            ("ביצים 12 יח'", "ביצים", 12),
        ],
    )
    def test_count_only(self, line, item, count):
        assert parse_line(line) == ParsedItem(item=item, count=count)

    def test_without_a_number_the_count_is_one(self):
        assert parse_line("חלב") == ParsedItem(item="חלב")
        assert parse_line("גבינה לבנה") == ParsedItem(item="גבינה לבנה", count=1)

    @pytest.mark.parametrize(
        ("line", "item", "pack", "count"),
        [
            ("חלב 1 ליטר x2", "חלב", ml(1000), 2),
            ("חלב 1 ליטר 2", "חלב", ml(1000), 2),
            ("2 חלב 1 ליטר", "חלב", ml(1000), 2),
            ("קוטג' 250 גרם x3", "קוטג'", g(250), 3),
            ("מים 1.5 ליטר 6 יחידות", "מים", ml(1500), 6),
        ],
    )
    def test_a_size_with_a_count_is_a_pack_size(self, line, item, pack, count):
        assert parse_line(line) == ParsedItem(item=item, count=count, pack_size=pack)


class TestPercent:
    """A number followed by % is an attribute."""

    @pytest.mark.parametrize(
        ("line", "item", "value"),
        [
            ("חלב 3%", "חלב", 3),
            ("חלב 3 %", "חלב", 3),
            ("3% חלב", "חלב", 3),
            ("קוטג' 5%", "קוטג'", 5),
            ("שמנת 15%", "שמנת", 15),
            ("יוגורט 1.5%", "יוגורט", 1.5),
        ],
    )
    def test_percent_is_the_fat_percentage(self, line, item, value):
        parsed = parse_line(line)

        assert parsed.item == item
        assert dict(parsed.attributes) == {"fat_pct": Attribute(value=value, strict=True)}
        assert type(parsed.attributes["fat_pct"].value) is type(value)
        assert parsed.count == 1

    def test_percent_size_and_count_together(self):
        parsed = parse_line("חלב תנובה 3% 1 ליטר x2")

        assert parsed == ParsedItem(
            item="חלב תנובה",
            count=2,
            pack_size=ml(1000),
            attributes={"fat_pct": {"value": 3}},
        )

    def test_a_percentage_is_not_a_count(self):
        assert parse_line("חלב 3%").count == 1


class TestLines:
    def test_each_line_is_one_item(self):
        result = parse_list('חלב 3% 2\nלחם\n\n  עגבניות 1 ק"ג  \n')

        assert result.items == (
            ParsedItem(item="חלב", count=2, attributes={"fat_pct": {"value": 3}}),
            ParsedItem(item="לחם"),
            ParsedItem(item="עגבניות", total_quantity=g(1000)),
        )
        assert result.skipped_lines == ()

    def test_the_result_is_flagged_as_simplified(self):
        result = parse_list("חלב")

        assert isinstance(result, FallbackParse)
        assert result.simplified_parsing is True
        assert parse_list("").simplified_parsing is True

    def test_windows_line_endings(self):
        assert [item.item for item in parse_list("חלב\r\nלחם\r\n").items] == ["חלב", "לחם"]

    @pytest.mark.parametrize("marker", ["- ", "* ", "• ", "1. ", "2) ", "12. "])
    def test_list_markers_are_not_part_of_the_item(self, marker):
        assert parse_line(f"{marker}חלב 2") == ParsedItem(item="חלב", count=2)

    def test_commas_do_not_split_a_line(self):
        assert len(parse_list("חלב, לחם").items) == 1

    @pytest.mark.parametrize("text", ["", "   ", "\n\n", "\t\n  \n"])
    def test_an_empty_list_has_no_items(self, text):
        assert parse_list(text) == FallbackParse(items=())

    def test_a_line_without_words_is_skipped(self):
        result = parse_list("חלב\n3\n---\n500 גרם\n!!!\nלחם")

        assert [item.item for item in result.items] == ["חלב", "לחם"]
        assert result.skipped_lines == ("3", "---", "500 גרם", "!!!")

    def test_at_most_a_hundred_lines_are_read(self):
        text = "\n".join(f"מוצר {chr(1488 + n % 22)}" for n in range(MAX_LINES + 5))

        result = parse_list(text)

        assert len(result.items) == MAX_LINES == 100
        assert len(result.skipped_lines) == 5


class TestRobustness:
    """The parser never fails on what a user typed."""

    def test_a_count_out_of_range_keeps_the_text(self):
        parsed = parse_line(f"חלב {MAX_COUNT + 1}")

        assert parsed == ParsedItem(item=f"חלב {MAX_COUNT + 1}")

    def test_a_zero_count_keeps_the_text(self):
        assert parse_line("קולה 0") == ParsedItem(item="קולה 0")

    def test_a_very_long_line_is_cut(self):
        parsed = parse_line("חלב " * 200)

        assert len(parsed.item) <= 200

    def test_a_second_size_stays_in_the_text(self):
        parsed = parse_line("חלב 1 ליטר 2 ליטר")

        assert parsed.total_quantity == ml(1000)
        assert parsed.item == "חלב 2 ליטר"

    def test_a_number_inside_a_word_is_not_a_count(self):
        assert parse_line("7up") == ParsedItem(item="7up")
        assert parse_line("ויטמין b12") == ParsedItem(item="ויטמין b12")

    def test_a_decimal_number_without_a_unit_is_left_alone(self):
        assert parse_line("קולה 1.5") == ParsedItem(item="קולה 1.5")

    def test_latin_items(self):
        assert parse_line("Milk 2") == ParsedItem(item="milk", count=2)

    def test_niqqud_and_quote_variants_are_normalized(self):
        assert parse_line("חָלָב 1 ק״ג") == ParsedItem(item="חלב", total_quantity=g(1000))

    @pytest.mark.parametrize(
        "line",
        ["%%%", "x", "5%", "1 ליטר", "‏", "🙂", "ליטר", "חלב 99999999999999999999 ליטר"],
    )
    def test_odd_lines_do_not_raise(self, line):
        result = parse_line(line)

        assert result is None or isinstance(result, ParsedItem)

    def test_every_item_is_a_valid_parsed_item(self):
        text = "חלב 3% 1 ליטר x2\n6 עגבניות\nקמח 1 ק\"ג\nביצים 12 יח'\nלחם"

        for item in parse_list(text).items:
            assert ParsedItem.model_validate(item.model_dump()) == item
            assert item.pack_size is None or item.total_quantity is None

    @pytest.mark.parametrize(
        "line", ["חלב {digits}", "חלב x{digits}", "חלב {digits}x", "חלב {digits} יחידות"]
    )
    def test_a_number_beyond_the_integer_conversion_limit_does_not_raise(self, line):
        # A list may hold 5,000 characters, and Python refuses to read 4,300+ digits as an int.
        parsed = parse_line(line.format(digits="9" * 4_400))

        assert parsed is not None
        assert parsed.count == 1
