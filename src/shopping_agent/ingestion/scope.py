"""The Phase 1 scope: the Bnei Brak branches of the three chains (architecture section 1).

The adapters read every store and price a chain publishes. This step keeps
only the branches in the Phase 1 cities and the prices that belong to them,
and counts the branches per chain for the data-quality report.
"""

from collections.abc import Collection
from typing import NamedTuple

import polars as pl

# The chains write the city as a locality code of the Central Bureau of Statistics.
BNEI_BRAK = "6100"
PHASE_1_CITIES: frozenset[str] = frozenset({BNEI_BRAK})

SCOPE_REPORT_SCHEMA = pl.Schema(
    {
        "chain": pl.String(),
        "branches": pl.UInt32(),
        "branches_with_prices": pl.UInt32(),
        "price_rows": pl.UInt32(),
    }
)


class ScopeError(ValueError):
    """The data cannot be restricted to the Phase 1 scope."""


class Scoped(NamedTuple):
    """The stores and prices in scope, and the branch count per chain."""

    stores: pl.DataFrame
    prices: pl.DataFrame
    report: pl.DataFrame


def stores_in_scope(stores: pl.DataFrame, cities: Collection[str] = PHASE_1_CITIES) -> pl.DataFrame:
    """Keep the stores of the given cities.

    Every chain in ``stores`` must have at least one branch there: a chain that
    drops out means its city codes changed, and must not pass silently.
    """
    if not cities:
        raise ScopeError("no cities were given for the scope")
    kept = stores.filter(pl.col("city").is_in(list(cities)))
    missing = sorted(set(stores["chain"].unique()) - set(kept["chain"].unique()))
    if missing:
        raise ScopeError(f"no branches in {', '.join(sorted(cities))} for: {', '.join(missing)}")
    return kept


def prices_in_scope(prices: pl.DataFrame, stores: pl.DataFrame) -> pl.DataFrame:
    """Keep the price rows that belong to the given stores, in their original order."""
    return prices.filter(pl.col("store_id").is_in(stores["store_id"].unique().implode()))


def scope_report(stores: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """One row per chain: its branches, how many of them have prices, and the price rows."""
    per_store = prices.group_by("store_id").agg(price_rows=pl.len())
    return (
        stores.select("chain", "store_id")
        .unique()
        .join(per_store, on="store_id", how="left")
        .group_by("chain")
        .agg(
            branches=pl.len(),
            branches_with_prices=pl.col("price_rows").is_not_null().sum(),
            price_rows=pl.col("price_rows").fill_null(0).sum(),
        )
        .sort("chain")
        .cast(SCOPE_REPORT_SCHEMA)
    )


def restrict_to_scope(
    stores: pl.DataFrame, prices: pl.DataFrame, cities: Collection[str] = PHASE_1_CITIES
) -> Scoped:
    """Restrict adapter stores and prices to the Phase 1 scope."""
    kept_stores = stores_in_scope(stores, cities)
    kept_prices = prices_in_scope(prices, kept_stores)
    return Scoped(kept_stores, kept_prices, scope_report(kept_stores, kept_prices))
