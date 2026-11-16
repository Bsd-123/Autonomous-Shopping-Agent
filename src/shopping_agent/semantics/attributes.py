"""Attribute matching: user requests, category defaults and the two missing-value rules.

A product is checked against an item attribute by attribute (architecture
section 3.1):

* **The user stated the attribute.** The product must carry that value. A
  product with no extracted value for it does not match: missing values fail
  user requests.
* **The user did not state it and the category has a default.** The default is
  a hard filter, but it fails only on an explicit different value. A product
  with no value passes: ordinary milk never says "cow" on the label, so
  missing values pass defaults.
* **The attribute is free in the category**, or the category says nothing
  about it. Any value is accepted, so the cheapest product wins.

A user-stated attribute always takes the place of the category's default for
it, and applies even when the category lists the attribute as free.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum

from shopping_agent.contracts import Attribute

AttributeValue = bool | int | float | str


class Failure(StrEnum):
    """Why a product does not match on one attribute."""

    USER_VALUE_MISSING = "user_value_missing"
    USER_VALUE_DIFFERENT = "user_value_different"
    DEFAULT_DIFFERENT = "default_different"


@dataclass(frozen=True, slots=True)
class AttributeFailure:
    """One attribute on which a product fails, with the value that was expected."""

    attribute: str
    reason: Failure
    expected: AttributeValue
    found: AttributeValue | None


def same_value(first: AttributeValue, second: AttributeValue) -> bool:
    """Whether two attribute values are the same.

    Text is compared without regard to case or surrounding spaces. Numbers are
    compared as numbers, so 3 equals 3.0. A boolean equals only a boolean.
    """
    if isinstance(first, bool) or isinstance(second, bool):
        return isinstance(first, bool) and isinstance(second, bool) and first == second
    if isinstance(first, str) or isinstance(second, str):
        return (
            isinstance(first, str)
            and isinstance(second, str)
            and first.strip().casefold() == second.strip().casefold()
        )
    return first == second


def _value_of(product: Mapping[str, AttributeValue | None], name: str) -> AttributeValue | None:
    """The product's value, where an empty text counts as no value."""
    value = product.get(name)
    if isinstance(value, str) and not value.strip():
        return None
    return value


def attribute_failures(
    product: Mapping[str, AttributeValue | None],
    requested: Mapping[str, Attribute],
    *,
    defaults: Mapping[str, AttributeValue] | None = None,
    free: Collection[str] = (),
) -> tuple[AttributeFailure, ...]:
    """Return every attribute on which the product fails; none means it matches.

    ``product`` holds the attribute values extracted for the product, where a
    missing key or ``None`` means no value was extracted. ``requested`` holds
    the attributes the user stated. ``defaults`` and ``free`` come from the
    item's category profile and are empty when it has none.

    A stated attribute that is not strict is a preference, not a filter, and
    is not checked here.
    """
    defaults = defaults or {}
    both = sorted(set(defaults) & set(free))
    if both:
        raise ValueError(f"an attribute cannot be both a default and free: {', '.join(both)}")

    failures: list[AttributeFailure] = []
    for name, stated in requested.items():
        if not stated.strict:
            continue
        found = _value_of(product, name)
        if found is None:
            failures.append(AttributeFailure(name, Failure.USER_VALUE_MISSING, stated.value, None))
        elif not same_value(found, stated.value):
            failures.append(
                AttributeFailure(name, Failure.USER_VALUE_DIFFERENT, stated.value, found)
            )
    for name, default in defaults.items():
        if name in requested:
            # The user's own choice replaces the default, strict or not.
            continue
        found = _value_of(product, name)
        if found is not None and not same_value(found, default):
            failures.append(AttributeFailure(name, Failure.DEFAULT_DIFFERENT, default, found))
    return tuple(failures)


def matches_attributes(
    product: Mapping[str, AttributeValue | None],
    requested: Mapping[str, Attribute],
    *,
    defaults: Mapping[str, AttributeValue] | None = None,
    free: Collection[str] = (),
) -> bool:
    """Whether the product passes the user's attributes and the category defaults."""
    return not attribute_failures(product, requested, defaults=defaults, free=free)
