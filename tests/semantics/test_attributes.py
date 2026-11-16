"""Tests for default application and the two missing-value rules.

The examples use the milk profile of architecture section 3.1.
"""

import pytest

from shopping_agent.contracts import Attribute, ParsedItem
from shopping_agent.semantics.attributes import (
    AttributeFailure,
    Failure,
    attribute_failures,
    matches_attributes,
    same_value,
)

MILK_DEFAULTS = {"animal": "cow", "flavor": "plain", "storage": "fresh"}
MILK_FREE = ["fat_pct", "brand"]
NOTHING_STATED: dict[str, Attribute] = {}


def matches(product: dict, requested=NOTHING_STATED, defaults=MILK_DEFAULTS, free=MILK_FREE):
    return matches_attributes(product, requested, defaults=defaults, free=free)


def failures(product: dict, requested=NOTHING_STATED, defaults=MILK_DEFAULTS, free=MILK_FREE):
    return attribute_failures(product, requested, defaults=defaults, free=free)


def stated(**values) -> dict[str, Attribute]:
    return {name: Attribute(value=value) for name, value in values.items()}


class TestDefaults:
    """A default passes on a missing or equal value and fails only on a different one."""

    def test_a_missing_value_passes(self):
        # Ordinary milk never says "cow" on the label.
        assert matches({})
        assert matches({"animal": None, "flavor": None, "storage": None})

    def test_an_equal_value_passes(self):
        assert matches({"animal": "cow", "flavor": "plain", "storage": "fresh"})

    def test_an_explicit_different_value_fails(self):
        assert not matches({"animal": "goat"})
        assert failures({"animal": "goat"}) == (
            AttributeFailure("animal", Failure.DEFAULT_DIFFERENT, expected="cow", found="goat"),
        )

    def test_each_default_is_checked(self):
        found = failures({"animal": "goat", "flavor": "chocolate", "storage": "fresh"})

        assert [failure.attribute for failure in found] == ["animal", "flavor"]
        assert {failure.reason for failure in found} == {Failure.DEFAULT_DIFFERENT}

    def test_one_different_value_is_enough_to_fail(self):
        assert not matches({"animal": "cow", "flavor": "plain", "storage": "long_life"})

    def test_an_empty_text_counts_as_missing(self):
        assert matches({"animal": "", "flavor": "  "})

    def test_without_a_profile_nothing_is_filtered(self):
        assert matches({"animal": "goat", "flavor": "chocolate"}, defaults=None, free=())


class TestUserRequests:
    """A user-stated attribute must be present on the product and equal."""

    def test_an_equal_value_passes(self):
        assert matches({"brand": "Tnuva", "fat_pct": 3}, stated(brand="Tnuva", fat_pct=3))

    def test_a_missing_value_fails(self):
        requested = stated(fat_pct=3)

        assert not matches({}, requested)
        assert not matches({"fat_pct": None}, requested)
        assert failures({}, requested) == (
            AttributeFailure("fat_pct", Failure.USER_VALUE_MISSING, expected=3, found=None),
        )

    def test_a_different_value_fails(self):
        assert failures({"fat_pct": 1}, stated(fat_pct=3)) == (
            AttributeFailure("fat_pct", Failure.USER_VALUE_DIFFERENT, expected=3, found=1),
        )

    def test_an_empty_text_counts_as_missing(self):
        found = failures({"brand": " "}, stated(brand="Tnuva"))

        assert [failure.reason for failure in found] == [Failure.USER_VALUE_MISSING]

    def test_the_two_missing_value_rules_side_by_side(self):
        product_without_values: dict = {}

        # The default "cow" passes on the missing value; the stated "cow" does not.
        assert matches(product_without_values)
        assert not matches(product_without_values, stated(animal="cow"))

    def test_a_stated_value_replaces_the_default(self):
        goat_milk = {"animal": "goat"}

        assert not matches(goat_milk)
        assert matches(goat_milk, stated(animal="goat"))
        assert failures({"animal": "cow"}, stated(animal="goat")) == (
            AttributeFailure("animal", Failure.USER_VALUE_DIFFERENT, expected="goat", found="cow"),
        )

    def test_the_other_defaults_still_apply(self):
        chocolate_goat_milk = {"animal": "goat", "flavor": "chocolate"}

        found = failures(chocolate_goat_milk, stated(animal="goat"))

        assert [(failure.attribute, failure.reason) for failure in found] == [
            ("flavor", Failure.DEFAULT_DIFFERENT)
        ]

    def test_an_attribute_outside_the_profile_can_be_stated(self):
        requested = stated(organic=True)

        assert matches({"organic": True}, requested)
        assert not matches({"organic": False}, requested)
        assert not matches({}, requested)

    def test_a_preference_that_is_not_strict_is_not_a_filter(self):
        soft = {"brand": Attribute(value="Tnuva", strict=False)}

        assert matches({"brand": "Tara"}, soft)
        assert matches({}, soft)

    def test_a_soft_preference_still_replaces_the_default(self):
        soft = {"animal": Attribute(value="goat", strict=False)}

        assert matches({"animal": "goat"}, soft)
        assert matches({"animal": "cow"}, soft)

    def test_the_attributes_of_a_parsed_item_are_accepted(self):
        item = ParsedItem(
            item="milk", attributes={"brand": {"value": "Tnuva"}, "fat_pct": {"value": 3}}
        )

        assert matches({"brand": "Tnuva", "fat_pct": 3}, item.attributes)
        assert not matches({"brand": "Tnuva"}, item.attributes)


class TestFreeAttributes:
    """A free attribute accepts any value when the user did not state it."""

    @pytest.mark.parametrize("product", [{}, {"fat_pct": 1}, {"fat_pct": 3.8, "brand": "Tara"}])
    def test_any_value_is_accepted(self, product):
        assert matches(product)

    def test_a_stated_free_attribute_is_checked(self):
        assert matches({"fat_pct": 3}, stated(fat_pct=3))
        assert not matches({"fat_pct": 1}, stated(fat_pct=3))

    def test_an_attribute_the_profile_does_not_mention_accepts_any_value(self):
        assert matches({"kashrut": "mehadrin", "package": "carton"})

    def test_an_attribute_cannot_be_both_a_default_and_free(self):
        with pytest.raises(ValueError, match="both a default and free: flavor"):
            matches({}, defaults={"flavor": "plain"}, free=["flavor"])


class TestSameValue:
    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ("Tnuva", "tnuva"),
            (" תנובה ", "תנובה"),
            (3, 3.0),
            (1.5, 1.5),
            (True, True),
            (False, False),
        ],
    )
    def test_equal_values(self, first, second):
        assert same_value(first, second)
        assert same_value(second, first)

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ("Tnuva", "Tara"),
            (3, 1),
            (3, "3"),
            (True, 1),
            (False, 0),
            (True, "true"),
            (True, False),
        ],
    )
    def test_different_values(self, first, second):
        assert not same_value(first, second)
        assert not same_value(second, first)

    def test_text_case_does_not_fail_a_request(self):
        assert matches({"brand": "TNUVA"}, stated(brand="tnuva"))

    def test_a_whole_number_matches_its_decimal_form(self):
        assert matches({"fat_pct": 3.0}, stated(fat_pct=3))
