"""Money in the canonical tables (architecture section 4.1).

* Every price is a whole number of agorot. No money column is a float.
* For a weighed product the price is per kilogram, and ``is_weighed`` is set
  on the product. This is how the chains publish weighed products, so the
  published price is stored as it is.
* A unit price is never stored. It is computed at query time and used only as
  a tie-breaker.

``require_integer_money`` enforces the first and third rules on the tables
before they are written. ``weighed_not_per_kilogram`` lists the few weighed
listings whose own unit of measure names another basis, for a person to check.
"""

from collections.abc import Mapping

import polars as pl

MONEY_SUFFIX = "_agorot"
# Columns that hold money without the suffix would be listed here; there are none.
NOT_MONEY = frozenset({"price_flag"})
FORBIDDEN_COLUMNS = ("unit_price",)

_KILOGRAM = r'קילו|ק"ג|קג'
_GRAM = "גרם|גר'|גר"


class MoneyError(ValueError):
    """A table breaks a money rule."""


def is_money_column(name: str) -> bool:
    """Whether a column holds an amount of money, judged by its name."""
    lowered = name.lower()
    if lowered in NOT_MONEY:
        return False
    return lowered.endswith(MONEY_SUFFIX) or "price" in lowered or "cost" in lowered


def require_integer_money(tables: Mapping[str, pl.DataFrame]) -> None:
    """Fail when a money column is not a whole number of agorot, or a unit price is stored."""
    for table, frame in tables.items():
        for name, dtype in frame.schema.items():
            lowered = name.lower()
            if any(forbidden in lowered for forbidden in FORBIDDEN_COLUMNS):
                raise MoneyError(
                    f"{table}.{name}: a unit price is not stored; it is computed at query time"
                )
            if not is_money_column(name):
                continue
            if not dtype.is_integer():
                raise MoneyError(f"{table}.{name} must be whole agorot, not {dtype}")
            if not lowered.endswith(MONEY_SUFFIX):
                raise MoneyError(f"{table}.{name}: a money column is named with '{MONEY_SUFFIX}'")


def weighed_not_per_kilogram(adapter_prices: pl.DataFrame) -> pl.DataFrame:
    """Weighed products whose unit of measure names grams and not a kilogram.

    Nearly every weighed product is published per kilogram. A handful state
    "100 גרם"; whether their price is per 100 grams or per kilogram cannot be
    told from the file, so they are stored as published and listed here.
    """
    measure = pl.col("unit_of_measure").fill_null("")
    return (
        adapter_prices.filter(
            pl.col("is_weighed") & measure.str.contains(_GRAM) & ~measure.str.contains(_KILOGRAM)
        )
        .select("chain", "item_code", "name", "quantity", "unit", "unit_of_measure", "price")
        .unique(["chain", "item_code"], keep="first", maintain_order=True)
        .sort("chain", "item_code")
    )
