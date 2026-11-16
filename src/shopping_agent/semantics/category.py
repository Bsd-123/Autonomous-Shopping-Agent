"""Category assignment, for products at ingestion and for list items at request time.

A name gets a category when a profile keyword is its first content word, after
the brand is removed, and no exclude keyword of that profile appears anywhere
in it (architecture section 3.1). This keeps "milk chocolate", "soy milk" and
"tomato paste" out of the milk and tomato categories.

The same function serves both sides, so a product and the list item that asks
for it always land in the same category.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

# Words are separated by spaces and by punctuation that the chains use between
# words. The Hebrew geresh and gershayim stay inside a word: קוטג', תפו"א.
_SEPARATORS = re.compile(r"[\s,;:+/()\[\]*|!?]+")
_EDGES = ".-–_=`"
# Where a number is glued to the end of a word, as in "וניל2%" or "עיזים4%שומן".
_LETTER_THEN_DIGIT = re.compile(r"(?<=[^\W\d_])(?=\d)")


class CategoryKeywords(Protocol):
    """The part of a category profile that assignment needs."""

    @property
    def keywords(self) -> Sequence[str]: ...

    @property
    def exclude_keywords(self) -> Sequence[str]: ...


@dataclass(frozen=True, slots=True)
class Keywords:
    """Keywords for one category, for callers that have no full profile at hand."""

    keywords: tuple[str, ...]
    exclude_keywords: tuple[str, ...] = ()


def words(text: str | None) -> tuple[str, ...]:
    """Split a name into its words, lower-cased, in order.

    A number glued to the end of a word starts a word of its own, so the word
    before it is still read: "וניל2%" gives "וניל" and "2%".
    """
    if not text:
        return ()
    spaced = _LETTER_THEN_DIGIT.sub(" ", text)
    tokens = (token.strip(_EDGES).casefold() for token in _SEPARATORS.split(spaced))
    return tuple(token for token in tokens if token)


def _is_content(word: str) -> bool:
    """A content word has letters; a number, a size or a percentage does not start one."""
    return word[0].isalpha()


def _without(sequence: tuple[str, ...], part: tuple[str, ...]) -> tuple[str, ...]:
    """Remove every occurrence of ``part`` from ``sequence``."""
    if not part:
        return sequence
    kept: list[str] = []
    position = 0
    while position < len(sequence):
        if sequence[position : position + len(part)] == part:
            position += len(part)
        else:
            kept.append(sequence[position])
            position += 1
    return tuple(kept)


def _contains(sequence: tuple[str, ...], part: tuple[str, ...]) -> bool:
    return any(
        sequence[start : start + len(part)] == part
        for start in range(len(sequence) - len(part) + 1)
    )


def assign_category(
    name: str | None,
    profiles: Mapping[str, CategoryKeywords],
    *,
    brand: str | None = None,
) -> str | None:
    """Return the category of a product name or list item, or ``None`` when no profile fits.

    ``name`` is the normalized name. ``brand``, when known, is removed from it
    first, so "תנובה חלב 3%" is read as "חלב 3%". When keywords of two profiles
    fit, the longer keyword wins: "שמן זית" is olive oil, not oil.
    """
    name_words = _without(words(name), words(brand))
    start = next((i for i, word in enumerate(name_words) if _is_content(word)), None)
    if start is None:
        return None
    leading = name_words[start:]

    best: tuple[int, str] | None = None
    for category, profile in profiles.items():
        for keyword in profile.keywords:
            keyword_words = words(keyword)
            if not keyword_words or leading[: len(keyword_words)] != keyword_words:
                continue
            if best is None or len(keyword_words) > best[0]:
                best = (len(keyword_words), category)
    if best is None:
        return None

    category = best[1]
    for exclude in profiles[category].exclude_keywords:
        if (exclude_words := words(exclude)) and _contains(name_words, exclude_words):
            return None
    return category
