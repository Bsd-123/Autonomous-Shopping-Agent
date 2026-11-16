"""Tests for the parsed item schema."""

import copy
import pickle

import pytest
from pydantic import ValidationError

from shopping_agent import contracts
from shopping_agent.contracts import (
    MAX_ATTRIBUTES,
    MAX_COUNT,
    MAX_QUANTITY,
    MAX_TEXT_LENGTH,
    Attribute,
    ParsedItem,
    Quantity,
)

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


class TestBlankText:
    @pytest.mark.parametrize("text", ["", "   ", "\t\n"])
    def test_blank_item_is_rejected(self, text):
        with pytest.raises(ValidationError):
            ParsedItem(item=text)

    def test_padding_is_stripped(self):
        assert ParsedItem(item=" milk ").item == "milk"
        assert Attribute(value=" Tnuva ").value == "Tnuva"

    def test_blank_attribute_key_is_rejected(self):
        with pytest.raises(ValidationError):
            ParsedItem(item="milk", attributes={" ": {"value": 3}})

    @pytest.mark.parametrize("value", ["", "  "])
    def test_blank_attribute_value_is_rejected(self, value):
        with pytest.raises(ValidationError):
            Attribute(value=value)


class TestFiniteNumbers:
    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
    def test_quantity_must_be_finite(self, value):
        with pytest.raises(ValidationError):
            Quantity(value=value, unit="g")

    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
    def test_attribute_value_must_be_finite(self, value):
        with pytest.raises(ValidationError):
            Attribute(value=value)

    def test_count_together_with_total_quantity_is_accepted(self):
        # The architecture does not rule it out; the Parse rules decide what is produced.
        item = ParsedItem(item="milk", count=2, total_quantity={"value": 3000, "unit": "ml"})

        assert (item.count, item.total_quantity.value) == (2, 3000)


class TestNoCoercion:
    @pytest.mark.parametrize(
        "payload",
        [
            '{"item": "milk", "count": "2"}',
            '{"item": "milk", "count": 2.0}',
            '{"item": "milk", "count": true}',
            '{"item": "milk", "pack_size": {"value": "1000", "unit": "ml"}}',
            '{"item": "milk", "attributes": {"brand": {"value": "Tnuva", "strict": "false"}}}',
            '{"item": "milk", "attributes": {"brand": {"value": "Tnuva", "strict": 0}}}',
        ],
    )
    def test_values_of_the_wrong_type_are_rejected_not_converted(self, payload):
        with pytest.raises(ValidationError):
            ParsedItem.model_validate_json(payload)

    def test_an_integer_quantity_is_accepted_as_a_float(self):
        assert Quantity(value=1000, unit="ml").value == 1000.0


class TestLimits:
    def test_values_at_the_limit_are_accepted(self):
        ParsedItem(
            item="a" * MAX_TEXT_LENGTH,
            count=MAX_COUNT,
            pack_size={"value": MAX_QUANTITY, "unit": "g"},
            attributes={f"a{index}": {"value": index} for index in range(MAX_ATTRIBUTES)},
        )

    @pytest.mark.parametrize(
        "fields",
        [
            {"item": "a" * (MAX_TEXT_LENGTH + 1)},
            {"item": "milk", "count": MAX_COUNT + 1},
            {"item": "milk", "pack_size": {"value": MAX_QUANTITY * 2, "unit": "g"}},
            {
                "item": "milk",
                "attributes": {
                    f"a{index}": {"value": index} for index in range(MAX_ATTRIBUTES + 1)
                },
            },
        ],
        ids=["item length", "count", "quantity", "attribute count"],
    )
    def test_values_beyond_the_limit_are_rejected(self, fields):
        with pytest.raises(ValidationError):
            ParsedItem.model_validate(fields)


class TestAttributesAreReadOnly:
    def test_attributes_cannot_be_changed_in_place(self):
        item = ParsedItem(item="milk", attributes={"brand": {"value": "Tnuva"}})

        with pytest.raises(TypeError):
            item.attributes["fat_pct"] = Attribute(value=3)

        assert list(item.attributes) == ["brand"]

    def test_the_item_is_still_serialized_as_plain_data(self):
        item = ParsedItem(item="milk", attributes={"brand": {"value": "Tnuva"}})

        assert type(item.model_dump()["attributes"]) is dict
        assert ParsedItem.model_validate_json(item.model_dump_json()) == item

    def test_a_copy_with_other_fields_keeps_the_attributes(self):
        item = ParsedItem(item="milk", attributes={"brand": {"value": "Tnuva"}})

        assert dict(item.model_copy(update={"count": 2}).attributes) == dict(item.attributes)

    def test_the_default_attributes_are_read_only_too(self):
        item = ParsedItem(item="milk")

        with pytest.raises(TypeError):
            item.attributes["fat_pct"] = Attribute(value=3)

    def test_the_item_can_be_copied_pickled_and_hashed(self):
        item = ParsedItem(item="milk", attributes={"brand": {"value": "Tnuva"}})
        same = ParsedItem.model_validate_json(item.model_dump_json())

        assert copy.deepcopy(item) == item
        assert item.model_copy(deep=True) == item
        assert pickle.loads(pickle.dumps(item)) == item
        assert hash(item) == hash(same)
        assert len({item, same}) == 1

    @pytest.mark.parametrize("mode", ["validation", "serialization"])
    def test_the_json_schema_still_describes_a_plain_object(self, mode):
        schema = ParsedItem.model_json_schema(mode=mode)

        assert schema["properties"]["attributes"]["type"] == "object"

    def test_the_attributes_have_a_readable_repr(self):
        item = ParsedItem(item="milk", attributes={"brand": {"value": "Tnuva"}})

        assert repr(item.attributes) == (
            "ReadOnlyAttributes({'brand': Attribute(value='Tnuva', strict=True)})"
        )


class TestAttributeNames:
    @pytest.mark.parametrize(
        "names", [("brand", " brand"), ("brand", "brand "), (" brand ", "brand")]
    )
    def test_names_that_differ_only_by_whitespace_are_rejected(self, names):
        attributes = {name: {"value": "Tnuva"} for name in names}

        with pytest.raises(ValidationError, match="attribute names repeat"):
            ParsedItem(item="milk", attributes=attributes)

    def test_the_rule_applies_to_json_too(self):
        payload = (
            '{"item": "milk", "attributes": {"brand": {"value": "a"}, "brand ": {"value": "b"}}}'
        )

        with pytest.raises(ValidationError, match="attribute names repeat"):
            ParsedItem.model_validate_json(payload)

    def test_a_padded_name_is_stripped(self):
        item = ParsedItem(item="milk", attributes={" brand ": {"value": "Tnuva"}})

        assert list(item.attributes) == ["brand"]

    def test_different_names_are_all_kept(self):
        attributes = {"brand": {"value": "Tnuva"}, "fat_pct": {"value": 3}}

        assert list(ParsedItem(item="milk", attributes=attributes).attributes) == [
            "brand",
            "fat_pct",
        ]


class TestAttributeEquality:
    def test_the_type_is_kept_but_equality_follows_python(self):
        flag, number = Attribute(value=True), Attribute(value=1)

        assert (type(flag.value), type(number.value)) == (bool, int)
        assert flag == number
        assert hash(flag) == hash(number)
        assert Attribute(value=3) == Attribute(value=3.0)


def test_the_package_exports_what_other_steps_need():
    assert set(contracts.__all__) == {
        "MAX_ATTRIBUTES",
        "MAX_COUNT",
        "MAX_QUANTITY",
        "MAX_TEXT_LENGTH",
        "Attribute",
        "ParsedItem",
        "Quantity",
        "ReadOnlyAttributes",
        "Unit",
    }
    assert all(hasattr(contracts, name) for name in contracts.__all__)
    assert isinstance(ParsedItem(item="milk").attributes, contracts.ReadOnlyAttributes)


@pytest.mark.parametrize("attributes", [["brand"], "brand", 3, None])
def test_attributes_must_be_an_object(attributes):
    with pytest.raises(ValidationError):
        ParsedItem.model_validate({"item": "milk", "attributes": attributes})
