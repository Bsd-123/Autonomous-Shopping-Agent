"""Hebrew product-name normalization (architecture section 4.4).

``normalize_name`` produces ``name_normalized``. The same function is meant
for the user's list items at request time, so both sides of a search are
written the same way.

What it does, in order:

1. Removes niqqud, cantillation marks and invisible direction marks.
2. Writes every geresh and gershayim variant as ``'`` and ``"``.
3. Replaces punctuation noise with spaces. A geresh or gershayim that belongs
   to a word is kept, as are decimal points and ``%`` after a number.
4. Lower-cases Latin letters and writes multiplication between numbers as
   ``*`` ("400X3" becomes "400*3").
5. Separates Hebrew letters from digits ("זיתים225" becomes "זיתים 225").
6. Writes a unit abbreviation that follows a number in one canonical form
   ("500גר'" and "500 ג" both become "500 גרם").
7. Collapses whitespace.

Prefixes are never stripped: doing so would damage real product words such
as לחם, מלח, בשר and שמן.
"""

import re
import unicodedata

import polars as pl

GERESH = "'"
GERSHAYIM = '"'

# Canonical unit and the spellings found after a number in the chains' files.
# A lone "מ" is left alone: it stands for millilitres on some products and
# metres on others.
UNIT_VARIANTS: dict[str, tuple[str, ...]] = {
    "גרם": ("גרמים", "גרם", "גר'", "גר", "ג'", "ג"),
    'ק"ג': ("קילוגרם", "קילו", "ק\"ג'", 'ק"ג', "ק'ג", "קג'", "קג", "ק"),
    'מ"ל': ("מיליליטר", 'מ"ל', "מ'ל", "מל'", "מל"),
    "ליטר": ("ליטרים", "ליטר", "ליט'", "ליט", "לי", "ל'", "ל"),
    "יחידות": ("יחידות", "יחידה", "יחיד", "יחי'", "יחי", "יח'", "יח"),
    'ס"מ': ("סנטימטר", 'ס"מ', "סמ'", "סמ"),
    'מ"מ': ('מ"מ',),
    "מטר": ("מטרים", "מטר", "מ'"),
}
CANONICAL_UNITS = tuple(UNIT_VARIANTS)

_UNIT_BY_VARIANT = {
    variant: unit for unit, variants in UNIT_VARIANTS.items() for variant in variants
}

_HEBREW = "א-ת"
_GERESH_VARIANTS = "׳‘’‚‛ʹʼ´`"
_GERSHAYIM_VARIANTS = "״“”„‟ʺ"
_TRANSLATION = str.maketrans(
    {
        **dict.fromkeys(_GERESH_VARIANTS, GERESH),
        **dict.fromkeys(_GERSHAYIM_VARIANTS, GERSHAYIM),
        # The maqaf joins words like a hyphen.
        "־": " ",
    }
)

# Niqqud and cantillation marks, plus the points that are not letters.
_MARKS = re.compile("[֑-ֽֿ-ׇ]")
_INVISIBLE = re.compile("[​-‏‪-‮⁦-⁩﻿]")
_TWO_GERESH = re.compile(rf"(?<=[{_HEBREW}])''(?=[{_HEBREW}])")
_WHITESPACE = re.compile(r"\s+")
_AFTER_NUMBER = r"(?:(?<=\d)|(?<=\d\s))"
_KEEP = re.compile(
    "|".join(
        (
            r"[^\W_]",  # letters and digits
            r"\s",
            r"(?<=\d)[.,](?=\d)",  # decimal point or comma
            rf"{_AFTER_NUMBER}%",  # percent after a number
            rf"{_AFTER_NUMBER}[*×](?=\s?\d)",  # multiplication between numbers
            rf"(?<=[{_HEBREW}])\"(?=[{_HEBREW}])",  # gershayim inside a word
            r"(?<=[^\W\d_])'",  # geresh after a letter
            r"(?<=\S)&(?=\S)",  # ampersand inside a name, as in M&M
        )
    )
)
_SPACE_BEFORE_PERCENT = re.compile(r"(?<=\d)\s%")
_PERCENT_THEN_TEXT = re.compile(r"%(?=\S)")
_TIMES = re.compile(r"(?<=\d)\s?[*x×]\s?(?=\d)")
_LETTER_THEN_DIGIT = re.compile(rf"(?<=[{_HEBREW}'\"])(?=\d)")
_DIGIT_THEN_LETTER = re.compile(rf"(?<=\d)(?=[{_HEBREW}])")
_UNIT_AFTER_NUMBER = re.compile(
    r"(?<![\w.,])(\d+(?:[.,]\d+)?)\s?("
    + "|".join(re.escape(variant) for variant in sorted(_UNIT_BY_VARIANT, key=len, reverse=True))
    + rf")(?![{_HEBREW}])(?!['\"][{_HEBREW}])"
)


def _strip_noise(text: str) -> str:
    """Replace every character that is not kept by ``_KEEP`` with a space."""
    kept: list[str] = []
    position = 0
    for match in _KEEP.finditer(text):
        if match.start() > position:
            kept.append(" ")
        kept.append(match.group())
        position = match.end()
    if position < len(text):
        kept.append(" ")
    return "".join(kept)


def normalize_name(name: str | None) -> str | None:
    """Return the normalized form of a product name or list item."""
    if name is None:
        return None
    text = unicodedata.normalize("NFKC", name)
    text = _INVISIBLE.sub("", _MARKS.sub("", text))
    text = text.translate(_TRANSLATION)
    text = _TWO_GERESH.sub(GERSHAYIM, text)
    text = _WHITESPACE.sub(" ", text).lower()
    text = _WHITESPACE.sub(" ", _strip_noise(text))
    text = _PERCENT_THEN_TEXT.sub("% ", _SPACE_BEFORE_PERCENT.sub("%", text))
    text = _TIMES.sub("*", text)
    text = _DIGIT_THEN_LETTER.sub(" ", _LETTER_THEN_DIGIT.sub(" ", text))
    text = _UNIT_AFTER_NUMBER.sub(
        lambda match: f"{match.group(1)} {_UNIT_BY_VARIANT[match.group(2)]}", text
    )
    return _WHITESPACE.sub(" ", text).strip()


def normalize_name_expr(name: str | pl.Expr = "name") -> pl.Expr:
    """Column form of ``normalize_name``."""
    name = pl.col(name) if isinstance(name, str) else name
    return name.map_elements(normalize_name, return_dtype=pl.String)
