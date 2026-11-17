"""Product keys (architecture section 4.1).

A code that is a real barcode identifies the same product in every chain, so
it becomes the key on its own. Any other code is only meaningful inside the
chain that issued it, so it is namespaced as ``chain:code`` and never joins
across chains.

A code counts as a barcode when all of the following hold:

* it has only the digits 0-9;
* its length is 8 (EAN-8), 12 (UPC-A), 13 (EAN-13) or 11;
* its last digit is the correct GS1 check digit;
* it is not in a range GS1 reserves for numbering inside one company.

Eleven digits is not a barcode length, but the chains publish UPC-A codes
with the leading zero dropped. The check digit is unaffected by leading
zeros, and every 11-digit code in the three chains' files passes it, which
chance would allow for one code in ten.

Barcodes are written as 13 digits with leading zeros, so the same product
gets the same key whether or not a chain dropped the zeros. Shorter codes
are chain-internal numbers; one in ten of them passes the check digit by
chance, which is why the length is checked as well.

Each rule exists twice: as a plain function for single values, and as a
Polars expression for whole columns. The tests hold the two to the same
results.
"""

import re

import polars as pl

BARCODE_LENGTHS = (8, 11, 12, 13)
BARCODE_DIGITS = 13
# GS1 prefixes 020-029 and 200-299 are for numbering inside one company, such
# as price-embedded labels for weighed goods. Two chains can use the same
# number for different products. An 8-digit code is padded with zeros, so its own
# leading 2 has to be checked before padding.
RESTRICTED_PREFIXES = ("02", "2")
EAN8_LENGTH = 8

_DIGITS = re.compile(r"[0-9]+")


def _check_digit(body: str) -> int:
    """The GS1 check digit for the digits that precede it."""
    total = sum(int(digit) * (3 if index % 2 == 0 else 1) for index, digit in enumerate(body[::-1]))
    return (10 - total % 10) % 10


def normalize_barcode(code: str | None) -> str | None:
    """Return the code as a 13-digit barcode, or None when it is not a barcode."""
    if code is None or _DIGITS.fullmatch(code) is None or len(code) not in BARCODE_LENGTHS:
        return None
    if _check_digit(code[:-1]) != int(code[-1]):
        return None
    barcode = code.zfill(BARCODE_DIGITS)
    if barcode.startswith(RESTRICTED_PREFIXES):
        return None
    if len(code) == EAN8_LENGTH and code.startswith(RESTRICTED_PREFIXES[1]):
        return None
    return barcode


def product_key(chain: str, code: str) -> str:
    """The barcode when the code is one, otherwise ``chain:code``."""
    return normalize_barcode(code) or f"{chain}:{code}"


def barcode_expr(code: str | pl.Expr = "item_code") -> pl.Expr:
    """Column form of ``normalize_barcode``."""
    code = pl.col(code) if isinstance(code, str) else code
    has_barcode_shape = code.str.contains(r"^[0-9]+$") & code.str.len_chars().is_in(BARCODE_LENGTHS)
    # Only codes with a barcode shape are padded, so every slice below is a digit.
    padded = pl.when(has_barcode_shape).then(code.str.zfill(BARCODE_DIGITS))
    digits = [padded.str.slice(index, 1).cast(pl.Int32) for index in range(BARCODE_DIGITS)]
    # Weights run 3, 1, 3, ... leftwards from the digit before the check digit.
    weighted = pl.sum_horizontal(
        digit * (3 if index % 2 == 1 else 1) for index, digit in enumerate(digits[:-1])
    )
    check_passes = (10 - weighted % 10) % 10 == digits[-1]
    restricted = pl.any_horizontal(padded.str.starts_with(prefix) for prefix in RESTRICTED_PREFIXES)
    ean8_in_store = code.str.len_chars().eq(EAN8_LENGTH) & code.str.starts_with("2")
    restricted = restricted | ean8_in_store
    return pl.when(check_passes & ~restricted).then(padded)


def product_key_expr(chain: str | pl.Expr = "chain", code: str | pl.Expr = "item_code") -> pl.Expr:
    """Column form of ``product_key``."""
    chain = pl.col(chain) if isinstance(chain, str) else chain
    code = pl.col(code) if isinstance(code, str) else code
    return pl.coalesce(barcode_expr(code), pl.concat_str(chain, pl.lit(":"), code))
