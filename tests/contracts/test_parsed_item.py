"""Tests for the parsed item schema."""

import pytest
from pydantic import ValidationError

from shopping_agent.contracts import Attribute, ParsedItem, Quantity

ARCHITECTURE_EXAMPLE = {
    "item": "milk",
    "count": 2,
    "pack_size": {"value": 1000, "unit": "ml"},
    "total_quantity": None,
    "attributes": {
        "brand": {"value": "Tnuva", "strict": True},
        "fat_pct": {"value": 3, "strict": True},
    },
}


def test_the_architecture_example_round_trips():
    item = ParsedItem.model_validate(ARCHITECTURE_EXAMPLE)

    assert item.count == 2
    assert item.pack_size == Quantity(value=1000, unit="ml")
    assert item.total_quantity is None
    assert item.attributes["brand"] == Attribute(value="Tnuva")
    assert item.attributes["fat_pct"].value == 3
    assert ParsedItem.model_validate_json(item.model_dump_json()) == item
    assert item.model_dump() == ARCHITECTURE_EXAMPLE


class TestSize:
    def test_pack_size_and_total_quantity_together_are_rejected(self):
        size = {"value": 1000, "unit": "ml"}

        with pytest.raises(ValidationError, match="at most one of pack_size and total_quantity"):
            ParsedItem(item="milk", pack_size=size, total_quantity=size)

    def test_total_quantity_alone(self):
        item = ParsedItem(item="milk", total_quantity={"value": 3000, "unit": "ml"})

        assert item.total_quantity == Quantity(value=3000, unit="ml")
        assert item.pack_size is None
        assert item.count == 1

    def test_pack_size_with_count(self):
        item = ParsedItem(item="milk", count=2, pack_size={"value": 1000, "unit": "ml"})

        assert (item.count, item.pack_size.value) == (2, 1000)

    def test_count_only(self):
        item = ParsedItem(item="tomatoes", count=6)

        assert item.count == 6
        assert item.pack_size is None
        assert item.total_quantity is None

    @pytest.mark.parametrize(
        "size",
        [{"value": 0, "unit": "ml"}, {"value": -1, "unit": "g"}, {"value": 1, "unit": "litre"}],
    )
    def test_invalid_quantity_is_rejected(self, size):
        with pytest.raises(ValidationError):
            ParsedItem(item="milk", pack_size=size)


class TestAttributes:
    def test_strict_defaults_to_true(self):
        item = ParsedItem.model_validate({"item": "milk", "attributes": {"fat_pct": {"value": 3}}})

        assert item.attributes["fat_pct"].strict is True
        assert item.model_dump()["attributes"]["fat_pct"] == {"value": 3, "strict": True}

    def test_strict_false_is_kept(self):
        attribute = Attribute(value="Tnuva", strict=False)

        assert attribute.strict is False

    def test_attributes_default_to_empty(self):
        assert ParsedItem(item="milk").attributes == {}

    def test_attribute_without_value_is_rejected(self):
        with pytest.raises(ValidationError, match="value"):
            ParsedItem.model_validate({"item": "milk", "attributes": {"brand": {"strict": True}}})

    def test_bare_attribute_value_is_rejected(self):
        with pytest.raises(ValidationError):
            ParsedItem.model_validate({"item": "milk", "attributes": {"brand": "Tnuva"}})

    @pytest.mark.parametrize("value", ["Tnuva", 3, 1.5, True])
    def test_value_types_are_preserved(self, value):
        kept = Attribute(value=value).value

        assert kept == value
        assert type(kept) is type(value)


class TestItem:
    @pytest.mark.parametrize("count", [0, -1, 1.5])
    def test_count_must_be_a_positive_integer(self, count):
        with pytest.raises(ValidationError):
            ParsedItem(item="milk", count=count)

    def test_empty_item_is_rejected(self):
        with pytest.raises(ValidationError):
            ParsedItem(item="")

    def test_unknown_fields_are_rejected(self):
        with pytest.raises(ValidationError):
            ParsedItem.model_validate({"item": "milk", "quantity": 2})
