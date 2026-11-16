"""The cleaning rules for the prices table (architecture section 4.2).

``clean_prices`` applies five rules, in this order:

1. **Zero or negative price**: the row is dropped and counted.
2. **Duplicate listing**: of several rows for one product in one store, the
   latest published one is kept; the rest are counted.
3. **Implausible price**: a price below the minimum floor, or one that changed
   tenfold against the previous dataset version, is not trusted. The previous
   version's row is carried forward with its original ``last_seen_date``, and
   ``price_flag`` is set to ``carried_forward``. With no previous row there is
   nothing to carry, and the row is dropped.
4. **Large but plausible change**: accepted, and listed in the report.
5. **Staleness**: a listing whose ``last_seen_date`` is older than the cutoff is
   excluded. A row that keeps being carried forward ages, so this rule removes
   it when the problem persists.

The thresholds are proposals to be tuned on real data, so they are
configuration.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import NamedTuple

import polars as pl

from shopping_agent.ingestion.canonical import (
    PRICE_FLAG_CARRIED_FORWARD,
    PRICE_FLAG_OK,
    PRICES_SCHEMA,
)

KEY = ["product_key", "store_id"]

REPORT_SCHEMA = pl.Schema(
    {
        "chain": pl.String(),
        "rows_in": pl.UInt32(),
        "non_positive_dropped": pl.UInt32(),
        "duplicates_dropped": pl.UInt32(),
        "implausible_dropped": pl.UInt32(),
        "carried_forward": pl.UInt32(),
        "large_changes": pl.UInt32(),
        "stale_excluded": pl.UInt32(),
        "rows_out": pl.UInt32(),
    }
)
CHANGES_SCHEMA = pl.Schema(
    {
        "product_key": pl.String(),
        "store_id": pl.String(),
        "previous_price_agorot": pl.Int64(),
        "published_price_agorot": pl.Int64(),
        "reason": pl.String(),
    }
)
REASON_BELOW_FLOOR = "below_floor"
REASON_IMPLAUSIBLE_CHANGE = "implausible_change"
REASON_LARGE_CHANGE = "large_change"


@dataclass(frozen=True, slots=True)
class CleaningConfig:
    """The thresholds of the cleaning rules.

    A price changed ``implausible_ratio`` times or more, up or down, is not
    trusted. One changed ``large_change_ratio`` times or more is accepted and
    reported.
    """

    min_price_agorot: int = 10
    implausible_ratio: float = 10.0
    large_change_ratio: float = 2.0
    staleness_days: int = 7

    def __post_init__(self) -> None:
        if self.min_price_agorot < 1:
            raise ValueError("the minimum price must be at least 1 agora")
        if not 1 < self.large_change_ratio < self.implausible_ratio:
            raise ValueError("ratios must satisfy 1 < large change < implausible")
        if self.staleness_days < 0:
            raise ValueError("the staleness cutoff cannot be negative")


class Cleaned(NamedTuple):
    """The cleaned prices, the counts per chain, and the price changes worth a look.

    ``changes`` lists every carried-forward row and every large accepted
    change, with the previous and the published price.
    """

    prices: pl.DataFrame
    report: pl.DataFrame
    changes: pl.DataFrame


def _chain() -> pl.Expr:
    return pl.col("store_id").str.split(":").list.first().alias("chain")


def _count(frame: pl.DataFrame, name: str) -> pl.DataFrame:
    return frame.group_by(_chain()).agg(pl.len().alias(name))


def _latest_per_listing(prices: pl.DataFrame) -> pl.DataFrame:
    """Keep the latest published row per product and store; the last row wins a tie."""
    return (
        prices.with_row_index("_row")
        .sort("source_timestamp", "_row")
        .unique(KEY, keep="last", maintain_order=True)
        .sort("_row")
        .drop("_row")
    )


def clean_prices(
    prices: pl.DataFrame,
    *,
    as_of: date,
    previous: pl.DataFrame | None = None,
    config: CleaningConfig | None = None,
) -> Cleaned:
    """Apply the cleaning rules to a prices table.

    ``previous`` is the cleaned prices table of the previous dataset version;
    without it nothing can be carried forward. ``as_of`` is the date the
    staleness cutoff is counted back from, normally the run's date in Israel.
    """
    config = config or CleaningConfig()
    # A missing price is treated like a zero price.
    positive = prices.filter(pl.col("price_agorot") > 0)
    latest = _latest_per_listing(positive)
    counts = [_count(prices, "rows_in"), _count(positive, "_positive"), _count(latest, "_latest")]

    earlier = (pl.DataFrame(schema=PRICES_SCHEMA) if previous is None else previous).select(
        *KEY,
        previous_price_agorot="price_agorot",
        previous_last_seen_date="last_seen_date",
        previous_source_timestamp="source_timestamp",
    )
    compared = latest.join(earlier, on=KEY, how="left").with_columns(
        ratio=pl.max_horizontal(
            pl.col("price_agorot") / pl.col("previous_price_agorot"),
            pl.col("previous_price_agorot") / pl.col("price_agorot"),
        )
    )
    below_floor = pl.col("price_agorot") < config.min_price_agorot
    tenfold = pl.col("ratio") >= config.implausible_ratio
    implausible = below_floor | tenfold.fill_null(False)
    has_previous = pl.col("previous_price_agorot").is_not_null()
    large = (~implausible) & (pl.col("ratio") >= config.large_change_ratio).fill_null(False)
    compared = compared.with_columns(
        _implausible=implausible,
        _large=large,
        _reason=pl.when(below_floor)
        .then(pl.lit(REASON_BELOW_FLOOR))
        .when(tenfold.fill_null(False))
        .then(pl.lit(REASON_IMPLAUSIBLE_CHANGE))
        .when(large)
        .then(pl.lit(REASON_LARGE_CHANGE)),
    )

    dropped = compared.filter(pl.col("_implausible") & ~has_previous)
    carried = compared.filter(pl.col("_implausible") & has_previous)
    counts.append(_count(dropped, "implausible_dropped"))
    counts.append(_count(carried, "carried_forward"))
    counts.append(_count(compared.filter("_large"), "large_changes"))

    changes = (
        compared.filter((pl.col("_implausible") & has_previous) | pl.col("_large"))
        .select(
            *KEY,
            "previous_price_agorot",
            published_price_agorot="price_agorot",
            reason="_reason",
        )
        .cast(CHANGES_SCHEMA)
    )

    carry = pl.col("_implausible") & has_previous
    resolved = (
        compared.filter(~(pl.col("_implausible") & ~has_previous))
        .with_columns(
            price_agorot=pl.when(carry).then("previous_price_agorot").otherwise("price_agorot"),
            last_seen_date=pl.when(carry)
            .then("previous_last_seen_date")
            .otherwise("last_seen_date"),
            source_timestamp=pl.when(carry)
            .then("previous_source_timestamp")
            .otherwise("source_timestamp"),
            price_flag=pl.when(carry)
            .then(pl.lit(PRICE_FLAG_CARRIED_FORWARD))
            .otherwise(pl.lit(PRICE_FLAG_OK)),
        )
        .select(prices.columns)
    )

    cutoff = as_of - timedelta(days=config.staleness_days)
    fresh = resolved.filter(pl.col("last_seen_date") >= cutoff)
    counts.append(_count(resolved.filter(pl.col("last_seen_date") < cutoff), "stale_excluded"))
    counts.append(_count(fresh, "rows_out"))

    report = counts[0]
    for part in counts[1:]:
        report = report.join(part, on="chain", how="full", coalesce=True)
    report = (
        report.fill_null(0)
        .with_columns(
            non_positive_dropped=pl.col("rows_in") - pl.col("_positive"),
            duplicates_dropped=pl.col("_positive") - pl.col("_latest"),
        )
        .select(REPORT_SCHEMA.names())
        .sort("chain")
        .cast(REPORT_SCHEMA)
    )
    return Cleaned(fresh.cast(PRICES_SCHEMA), report, changes)
