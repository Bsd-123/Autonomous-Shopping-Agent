"""Unit normalization with multi-pack expansion (architecture section 4.3).

Every product gets a ``base_quantity`` in a base unit: millilitres, grams or
units. That is what makes package sizes comparable across products and chains.

A product's size is stated in two places that do not always agree:

* the chain's own fields: ``quantity`` with ``unit``, or ``unit_of_measure``
  when the unit is missing;
* the product name, as a single size ("500 גרם") or a multi-pack ("6*1.5 ליטר").

``base_size`` picks between them with fixed rules, in this order:

1. A weighed product is priced per kilogram, so its base is 1,000 g.
2. A multi-pack written in the name in grams or millilitres is expanded:
   6 x 1.5 L gives 9,000 ml.
3. A weight or volume from the chain's fields is used, except that a single
   size in the name replaces it when the two plausibly describe the same
   pack: the same kind of measure and neither more than twice the other, or
   different kinds within 15% of each other (a liquid recorded in grams).
   When the fields hold a whole multiple of the size in the name, the fields
   already are the expanded multi-pack and are kept. A field that is not a
   plausible pack size (115,000 g) gives way to a plausible one in the name.
4. A single weight or volume in the name, when it is a plausible pack size.
5. A count from the name ("36 יח", "4*64 יח").
6. A count from the chain's fields.
7. Otherwise one unit.

A weight or volume always wins over a count, because it is what makes a
product comparable with another.

A size written as a range ("12-23 ק"ג" on nappies) describes who the product
is for, not how much it holds, and is ignored.
"""

import re
from typing import Any, Literal, NamedTuple

import polars as pl

from shopping_agent.ingestion.text_normalization import UNIT_VARIANTS, normalize_name

BaseUnit = Literal["ml", "g", "unit"]
Source = Literal["weighed", "name_multipack", "name", "fields", "default"]

WEIGHED_BASE_GRAMS = 1000.0
# A liquid is often recorded by the chain in grams, or the reverse, with
# nearly the same number as on the label.
SAME_SIZE_TOLERANCE = 0.15
# The label and the chain's fields describe the same pack when neither is more
# than this many times the other. Beyond it, the name holds something else,
# such as a price or a cut-off unit.
SAME_PACK_RATIO = 2.0
# Grams or millilitres a pack plausibly holds. A size read from the name alone
# is trusted only inside this range, and it replaces a chain field outside it.
PLAUSIBLE_PACK = (5.0, 20_000.0)
MAX_PACKS = 100
# No grocery product holds this many litres or kilograms; a larger number in
# those units is millilitres or grams entered under the wrong unit.
MAX_LARGE_UNITS = 50

# Canonical unit to base unit and factor. Lengths have no base unit.
_BASE: dict[str, tuple[BaseUnit, float]] = {
    "גרם": ("g", 1.0),
    'ק"ג': ("g", 1000.0),
    'מ"ל': ("ml", 1.0),
    "ליטר": ("ml", 1000.0),
    "יחידות": ("unit", 1.0),
}
_CANONICAL_BY_SPELLING = {
    spelling: unit for unit, spellings in UNIT_VARIANTS.items() for spelling in spellings
}

_NUMBER = r"(\d+(?:\.\d+)?)"
_UNITS = "|".join(re.escape(unit) for unit in _BASE)
_UNIT_END = r"(?![א-ת\"'])"
_MULTIPACK = re.compile(rf"(?<![\w.,*]){_NUMBER}\*{_NUMBER} ({_UNITS}){_UNIT_END}")
_SINGLE = re.compile(rf"(?<![\w.,*]){_NUMBER} ({_UNITS}){_UNIT_END}")
_LEADING_AMOUNT = re.compile(r"^[\d.,\s]+")
# A range and the unit word that follows it.
_RANGE = re.compile(r"\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*[א-ת\"']+")


class BaseSize(NamedTuple):
    """A product's size in a base unit, and where it was read from."""

    quantity: float
    unit: BaseUnit
    source: Source


class _Size(NamedTuple):
    quantity: float
    unit: BaseUnit


def _to_base(quantity: float, canonical_unit: str | None) -> _Size | None:
    if canonical_unit not in _BASE or not quantity > 0:
        return None
    unit, factor = _BASE[canonical_unit]
    return _Size(quantity * factor, unit)


def _canonical_unit(text: str | None) -> str | None:
    """The canonical unit named by a chain field such as "קילוגרם" or "100 מ"ל"."""
    if text is None:
        return None
    spelling = _LEADING_AMOUNT.sub("", normalize_name(text) or "")
    return _CANONICAL_BY_SPELLING.get(spelling)


def _from_fields(
    quantity: float | None, unit: str | None, unit_of_measure: str | None
) -> _Size | None:
    canonical = _canonical_unit(unit) or _canonical_unit(unit_of_measure)
    if quantity is None or canonical is None:
        return None
    if canonical in ("ליטר", 'ק"ג') and quantity >= MAX_LARGE_UNITS:
        canonical = 'מ"ל' if canonical == "ליטר" else "גרם"
    return _to_base(quantity, canonical)


class _NameSizes(NamedTuple):
    measure: _Size | None
    counted: _Size | None
    measure_is_multipack: bool
    count_is_multipack: bool


def _only(sizes: list[_Size]) -> _Size | None:
    """The size, when the name states exactly one."""
    distinct = set(sizes)
    return distinct.pop() if len(distinct) == 1 else None


def _from_name(name: str | None) -> _NameSizes:
    text = normalize_name(_RANGE.sub(" ", name or "")) or ""
    multipacks: list[_Size] = []
    for first, second, unit in _MULTIPACK.findall(text):
        size = _to_base(float(first) * float(second), unit)
        if size is not None:
            multipacks.append(size)
    singles = [
        size
        for amount, unit in _SINGLE.findall(text)
        if (size := _to_base(float(amount), unit)) is not None
    ]

    def pick(wanted_count: bool) -> tuple[_Size | None, bool]:
        of_kind = [size for size in multipacks if (size.unit == "unit") == wanted_count]
        if of_kind:
            return _only(of_kind), True
        return _only([size for size in singles if (size.unit == "unit") == wanted_count]), False

    measure, measure_is_multipack = pick(wanted_count=False)
    count, count_is_multipack = pick(wanted_count=True)
    return _NameSizes(measure, count, measure_is_multipack, count_is_multipack)


def _is_whole_multiple(total: float, part: float) -> bool:
    packs = round(total / part)
    return 2 <= packs <= MAX_PACKS and abs(total - packs * part) <= 0.01 * total


def _is_plausible_pack(size: _Size) -> bool:
    return PLAUSIBLE_PACK[0] <= size.quantity <= PLAUSIBLE_PACK[1]


def _name_replaces_fields(named: _Size, fields: _Size) -> bool:
    if not _is_plausible_pack(fields):
        # The chain entered the amount under the wrong unit, for example
        # 115,000 g for a 115 g pack.
        return _is_plausible_pack(named)
    smaller, larger = sorted((named.quantity, fields.quantity))
    if named.unit == fields.unit:
        same_pack = larger <= SAME_PACK_RATIO * smaller
        return same_pack and not _is_whole_multiple(fields.quantity, named.quantity)
    return larger - smaller <= SAME_SIZE_TOLERANCE * larger


def base_size(
    name: str | None,
    quantity: float | None,
    unit: str | None,
    unit_of_measure: str | None,
    is_weighed: bool | None,
) -> BaseSize:
    """Return the product's size in millilitres, grams or units."""
    if is_weighed:
        return BaseSize(WEIGHED_BASE_GRAMS, "g", "weighed")

    named = _from_name(name)
    fields = _from_fields(quantity, unit, unit_of_measure)

    if named.measure is not None and named.measure_is_multipack:
        return BaseSize(*named.measure, "name_multipack")
    if fields is not None and fields.unit != "unit":
        if named.measure is not None and _name_replaces_fields(named.measure, fields):
            return BaseSize(*named.measure, "name")
        return BaseSize(*fields, "fields")
    if named.measure is not None and _is_plausible_pack(named.measure):
        return BaseSize(*named.measure, "name")
    if named.counted is not None:
        return BaseSize(*named.counted, "name_multipack" if named.count_is_multipack else "name")
    if fields is not None:
        return BaseSize(*fields, "fields")
    return BaseSize(1.0, "unit", "default")


BASE_SIZE_DTYPE = pl.Struct(
    {"base_quantity": pl.Float64(), "base_unit": pl.String(), "base_source": pl.String()}
)


def base_size_expr(
    name: str = "name",
    quantity: str = "quantity",
    unit: str = "unit",
    unit_of_measure: str = "unit_of_measure",
    is_weighed: str = "is_weighed",
) -> pl.Expr:
    """Column form of ``base_size``: a struct of quantity, unit and source."""

    def convert(row: dict[str, Any]) -> dict[str, Any]:
        size = base_size(row[name], row[quantity], row[unit], row[unit_of_measure], row[is_weighed])
        return {
            "base_quantity": size.quantity,
            "base_unit": size.unit,
            "base_source": size.source,
        }

    return pl.struct(name, quantity, unit, unit_of_measure, is_weighed).map_elements(
        convert, return_dtype=BASE_SIZE_DTYPE
    )
