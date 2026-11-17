"""The rule-based parser used when the LLM is unavailable (architecture section 5.2).

Each line of the list is one item. Within a line:

* a number followed by a unit is a size, read with the same unit table as
  ingestion ("1.5 ליטר", "500 גר'", "2 קילו");
* a number followed by ``%`` is an attribute, the fat percentage;
* a bare number is a count ("2", "x2").

A size with a count is a pack size ("milk 1 L x2"); a size alone is a total
quantity ("3 L milk"), as in the Parse rules.

The parser never fails on what a user typed: a line it cannot read fully
becomes an item with the text alone. Its results are always flagged as
simplified, because a wrong parse that still finds valid products would not
reach Clarify, and the UI asks the user to check them.
"""

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from shopping_agent.contracts import MAX_TEXT_LENGTH, ParsedItem
from shopping_agent.ingestion.text_normalization import CANONICAL_UNITS, normalize_name

MAX_LINES = 100
PERCENT_ATTRIBUTE = "fat_pct"

# Canonical units of the ingestion unit table that measure a size, with their
# base unit. Lengths are not sizes of groceries and stay in the item's text.
_SIZE_UNITS: dict[str, tuple[Literal["ml", "g"], float]] = {
    "גרם": ("g", 1.0),
    'ק"ג': ("g", 1000.0),
    'מ"ל': ("ml", 1.0),
    "ליטר": ("ml", 1000.0),
}
_COUNT_UNIT = "יחידות"

_NUMBER = r"(\d+(?:[.,]\d+)?)"
# A count has at most this many digits. A longer number stays in the name, and
# ``int`` is never asked to read a digit string beyond Python's conversion limit.
_COUNT = r"(\d{1,6})"
_NOT_IN_A_WORD = r"(?<![\w.,%*])"
_UNIT_END = r"(?![א-ת\"'])"
_PERCENT = re.compile(rf"{_NOT_IN_A_WORD}{_NUMBER}%")
_SIZE = re.compile(
    rf"{_NOT_IN_A_WORD}{_NUMBER} ({'|'.join(map(re.escape, _SIZE_UNITS))}){_UNIT_END}"
)
_UNITS = re.compile(rf"{_NOT_IN_A_WORD}{_COUNT} {_COUNT_UNIT}{_UNIT_END}")
_TIMES = re.compile(rf"(?<![^\W\d_])x ?{_COUNT}(?![\w.,%])|(?<![\w.,%]){_COUNT} ?x(?![^\W\d_])")
# A number that is followed by a unit, even one that is not a size, is not a count.
_BARE = re.compile(
    rf"(?<![\w.,%*]){_COUNT}(?![\w.,%*])"
    rf"(?! (?:{'|'.join(map(re.escape, CANONICAL_UNITS))}){_UNIT_END})"
)
_LETTER = re.compile(r"[^\W\d_]")
_LIST_MARKER = re.compile(r"^\s*(?:[-*•·–]+|\d{1,3}[.)])\s+")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class FallbackParse:
    """The items read from the list, and the lines that held no item.

    ``simplified_parsing`` is always true: it tells the UI to show the
    "simplified parsing, please check" label.
    """

    items: tuple[ParsedItem, ...]
    skipped_lines: tuple[str, ...] = ()
    simplified_parsing: Literal[True] = True


def _number(text: str) -> float:
    return float(text.replace(",", "."))


def _take(pattern: re.Pattern[str], text: str) -> tuple[re.Match[str] | None, str]:
    """Find the first match and return it with the text that is left without it."""
    match = pattern.search(text)
    if match is None:
        return None, text
    return match, f"{text[: match.start()]} {text[match.end() :]}"


def parse_line(line: str) -> ParsedItem | None:
    """Read one line of the list, or return ``None`` when it names no item."""
    text = normalize_name(_LIST_MARKER.sub("", line)) or ""
    if not text:
        return None
    plain_text = text

    fields: dict[str, object] = {}
    percent, text = _take(_PERCENT, text)
    if percent is not None:
        value = _number(percent.group(1))
        fields["attributes"] = {
            PERCENT_ATTRIBUTE: {"value": int(value) if value.is_integer() else value}
        }

    size, text = _take(_SIZE, text)
    count: int | None = None
    for pattern in (_UNITS, _TIMES, _BARE):
        found, text = _take(pattern, text)
        if found is not None:
            count = int(next(group for group in found.groups() if group is not None))
            break

    if size is not None:
        unit, factor = _SIZE_UNITS[size.group(2)]
        quantity = {"value": _number(size.group(1)) * factor, "unit": unit}
        fields["pack_size" if count is not None else "total_quantity"] = quantity
    if count is not None:
        fields["count"] = count

    name = _WHITESPACE.sub(" ", text).strip()
    if not _LETTER.search(name):
        # Only numbers and sizes: the line names no item.
        return None
    try:
        return ParsedItem.model_validate({"item": name, **fields})
    except ValidationError:
        # A number out of range: keep the line as typed, without reading numbers.
        try:
            return ParsedItem(item=plain_text[:MAX_TEXT_LENGTH])
        except ValidationError:
            return None


def parse_list(list_text: str) -> FallbackParse:
    """Parse a shopping list, one item per line, without an LLM.

    Empty lines are ignored. At most ``MAX_LINES`` lines are read; the rest,
    and any line that names no item, are returned in ``skipped_lines``.
    """
    lines = [line.strip() for line in list_text.splitlines() if line.strip()]
    items: list[ParsedItem] = []
    skipped: list[str] = list(lines[MAX_LINES:])
    for line in lines[:MAX_LINES]:
        item = parse_line(line)
        if item is None:
            skipped.append(line)
        else:
            items.append(item)
    return FallbackParse(items=tuple(items), skipped_lines=tuple(skipped))
