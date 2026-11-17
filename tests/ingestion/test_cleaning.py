"""Tests for the cleaning rules, one class per rule."""

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from shopping_agent.ingestion.canonical import PRICES_SCHEMA, build_canonical
from shopping_agent.ingestion.cleaning import (
    CHANGES_SCHEMA,
    REPORT_SCHEMA,
    CleaningConfig,
    clean_prices,
)

ISRAEL = ZoneInfo("Asia/Jerusalem")
TODAY = date(2026, 11, 15)
YESTERDAY = TODAY - timedelta(days=1)
RUN_AT = datetime(2026, 11, 15, 3, 0, tzinfo=UTC)
MILK = "7290000000015"
BREAD = "7290000000022"
STORE = "osher_ad:11"


def listing(
    product_key: str = MILK,
    price_agorot: int | None = 590,
    store_id: str = STORE,
    seen: date = TODAY,
    hour: int = 9,
    flag: str = "ok",
    version: str = "2026-11-15",
) -> dict:
    return {
        "product_key": product_key,
        "store_id": store_id,
        "price_agorot": price_agorot,
        "price_flag": flag,
        "last_seen_date": seen,
        "source_timestamp": datetime(seen.year, seen.month, seen.day, hour, tzinfo=ISRAEL),
        "ingestion_timestamp": RUN_AT,
        "dataset_version": version,
    }


def prices(*rows: dict) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=PRICES_SCHEMA)


def earlier(price_agorot: int, seen: date = YESTERDAY, **fields) -> dict:
    return listing(price_agorot=price_agorot, seen=seen, version="2026-11-14", **fields)


def report_row(cleaned, chain: str = "osher_ad") -> dict:
    return cleaned.report.filter(chain=chain).row(0, named=True)


def clean(current: pl.DataFrame, previous: pl.DataFrame | None = None, **settings):
    return clean_prices(current, as_of=TODAY, previous=previous, **settings)


class TestZeroAndNegativePrices:
    @pytest.mark.parametrize("price", [0, -590, None])
    def test_the_row_is_dropped_and_counted(self, price):
        cleaned = clean(prices(listing(price_agorot=price), listing(BREAD, 790)))

        assert cleaned.prices["product_key"].to_list() == [BREAD]
        assert report_row(cleaned)["non_positive_dropped"] == 1
        assert report_row(cleaned)["rows_out"] == 1

    def test_a_zero_price_is_not_carried_forward(self):
        cleaned = clean(prices(listing(price_agorot=0)), prices(earlier(590)))

        assert cleaned.prices.height == 0
        assert report_row(cleaned)["carried_forward"] == 0


class TestDuplicates:
    def test_the_latest_published_row_is_kept_and_the_rest_are_counted(self):
        current = prices(
            listing(price_agorot=590, hour=9),
            listing(price_agorot=610, hour=19),
            listing(price_agorot=600, hour=12),
        )

        cleaned = clean(current)

        assert cleaned.prices["price_agorot"].to_list() == [610]
        assert report_row(cleaned)["duplicates_dropped"] == 2

    def test_identical_rows_count_as_duplicates(self):
        cleaned = clean(prices(listing(), listing(), listing()))

        assert cleaned.prices.height == 1
        assert report_row(cleaned)["duplicates_dropped"] == 2

    def test_rows_published_at_the_same_time_keep_the_last_one(self):
        cleaned = clean(prices(listing(price_agorot=590), listing(price_agorot=610)))

        assert cleaned.prices["price_agorot"].to_list() == [610]

    def test_the_same_product_in_another_store_or_chain_is_not_a_duplicate(self):
        current = prices(
            listing(),
            listing(store_id="osher_ad:22"),
            listing(store_id="yesh_hesed:611"),
            listing(BREAD),
        )

        cleaned = clean(current)

        assert cleaned.prices.height == 4
        assert cleaned.report["duplicates_dropped"].sum() == 0

    def test_a_zero_price_never_wins_over_a_real_one(self):
        current = prices(listing(price_agorot=590, hour=9), listing(price_agorot=0, hour=19))

        cleaned = clean(current)

        assert cleaned.prices["price_agorot"].to_list() == [590]
        assert report_row(cleaned)["non_positive_dropped"] == 1
        assert report_row(cleaned)["duplicates_dropped"] == 0

    def test_rows_keep_their_original_order(self):
        current = prices(
            listing(BREAD, 790), listing(MILK, 590, hour=9), listing(MILK, 600, hour=19)
        )

        assert clean(current).prices["product_key"].to_list() == [BREAD, MILK]


class TestImplausiblePrices:
    @pytest.mark.parametrize("published", [5900, 59, 7000, 9])
    def test_a_tenfold_change_carries_the_previous_row_forward(self, published):
        cleaned = clean(prices(listing(price_agorot=published)), prices(earlier(590)))
        row = cleaned.prices.row(0, named=True)

        assert row["price_agorot"] == 590
        assert row["price_flag"] == "carried_forward"
        assert row["last_seen_date"] == YESTERDAY
        assert row["source_timestamp"].date() == YESTERDAY
        # The row belongs to the new dataset version.
        assert row["dataset_version"] == "2026-11-15"
        assert report_row(cleaned)["carried_forward"] == 1

    def test_a_price_below_the_floor_is_carried_forward(self):
        cleaned = clean(prices(listing(price_agorot=5)), prices(earlier(8)))

        assert cleaned.prices["price_agorot"].to_list() == [8]
        assert cleaned.prices["price_flag"].to_list() == ["carried_forward"]
        assert cleaned.changes["reason"].to_list() == ["below_floor"]

    def test_the_original_last_seen_date_is_kept_across_runs(self):
        three_days_ago = TODAY - timedelta(days=3)
        previous = prices(earlier(590, seen=three_days_ago, flag="carried_forward"))

        cleaned = clean(prices(listing(price_agorot=5900)), previous)

        assert cleaned.prices["last_seen_date"].to_list() == [three_days_ago]
        assert cleaned.prices["price_flag"].to_list() == ["carried_forward"]

    def test_without_a_previous_row_a_price_below_the_floor_is_dropped(self):
        cleaned = clean(prices(listing(price_agorot=5), listing(BREAD, 790)))

        assert cleaned.prices["product_key"].to_list() == [BREAD]
        assert report_row(cleaned)["implausible_dropped"] == 1
        assert report_row(cleaned)["carried_forward"] == 0

    def test_without_a_previous_version_any_price_above_the_floor_is_accepted(self):
        cleaned = clean(prices(listing(price_agorot=590_000)))

        assert cleaned.prices["price_flag"].to_list() == ["ok"]

    def test_the_carried_row_is_listed_with_both_prices(self):
        cleaned = clean(prices(listing(price_agorot=5900)), prices(earlier(590)))

        assert cleaned.changes.schema == CHANGES_SCHEMA
        assert cleaned.changes.rows() == [(MILK, STORE, 590, 5900, "implausible_change")]

    def test_a_good_price_after_a_carried_one_is_accepted_again(self):
        previous = prices(earlier(590, flag="carried_forward"))

        cleaned = clean(prices(listing(price_agorot=610)), previous)

        assert cleaned.prices["price_flag"].to_list() == ["ok"]
        assert cleaned.prices["last_seen_date"].to_list() == [TODAY]

    def test_a_product_missing_from_the_new_files_is_not_carried(self):
        cleaned = clean(
            prices(listing(BREAD, 790)), prices(earlier(590), earlier(780, product_key=BREAD))
        )

        assert cleaned.prices["product_key"].to_list() == [BREAD]

    def test_the_previous_row_is_matched_by_product_and_store(self):
        previous = prices(earlier(59, store_id="osher_ad:22"))

        cleaned = clean(prices(listing(price_agorot=590)), previous)

        assert cleaned.prices["price_flag"].to_list() == ["ok"]


class TestLargePlausibleChanges:
    @pytest.mark.parametrize("published", [1180, 295, 5899, 60])
    def test_the_change_is_accepted_and_listed(self, published):
        cleaned = clean(prices(listing(price_agorot=published)), prices(earlier(590)))

        assert cleaned.prices["price_agorot"].to_list() == [published]
        assert cleaned.prices["price_flag"].to_list() == ["ok"]
        assert cleaned.prices["last_seen_date"].to_list() == [TODAY]
        assert cleaned.changes.rows() == [(MILK, STORE, 590, published, "large_change")]
        assert report_row(cleaned)["large_changes"] == 1

    @pytest.mark.parametrize("published", [590, 600, 1179, 296])
    def test_an_ordinary_change_is_not_listed(self, published):
        cleaned = clean(prices(listing(price_agorot=published)), prices(earlier(590)))

        assert cleaned.prices["price_agorot"].to_list() == [published]
        assert cleaned.changes.height == 0
        assert report_row(cleaned)["large_changes"] == 0


class TestStaleness:
    @pytest.mark.parametrize(("days_old", "kept"), [(0, True), (6, True), (7, True), (8, False)])
    def test_listings_older_than_seven_days_are_excluded(self, days_old, kept):
        cleaned = clean(prices(listing(seen=TODAY - timedelta(days=days_old))))

        assert (cleaned.prices.height == 1) is kept
        assert report_row(cleaned)["stale_excluded"] == (0 if kept else 1)

    def test_the_cutoff_is_configurable(self):
        old = prices(listing(seen=TODAY - timedelta(days=3)))

        assert clean(old, config=CleaningConfig(staleness_days=2)).prices.height == 0
        assert clean(old, config=CleaningConfig(staleness_days=3)).prices.height == 1

    def test_a_row_carried_forward_for_too_long_is_excluded(self):
        nine_days_ago = TODAY - timedelta(days=9)
        previous = prices(earlier(590, seen=nine_days_ago, flag="carried_forward"))

        cleaned = clean(prices(listing(price_agorot=5900)), previous)

        assert cleaned.prices.height == 0
        assert report_row(cleaned)["carried_forward"] == 1
        assert report_row(cleaned)["stale_excluded"] == 1


class TestReport:
    def test_counts_are_per_chain_and_add_up(self):
        current = prices(
            listing(MILK, 590),
            listing(MILK, 600, hour=19),
            listing(BREAD, 0),
            listing(MILK, 590, store_id="yesh_hesed:611"),
            listing(BREAD, 5, store_id="yesh_hesed:611"),
            listing(MILK, 590, store_id="maayan_2000:38", seen=TODAY - timedelta(days=30)),
        )

        cleaned = clean(current)

        assert cleaned.report.schema == REPORT_SCHEMA
        assert cleaned.report.rows() == [
            ("maayan_2000", 1, 0, 0, 0, 0, 0, 1, 0),
            ("osher_ad", 3, 1, 1, 0, 0, 0, 0, 1),
            ("yesh_hesed", 2, 0, 0, 1, 0, 0, 0, 1),
        ]
        for row in cleaned.report.iter_rows(named=True):
            removed = (
                row["non_positive_dropped"]
                + row["duplicates_dropped"]
                + row["implausible_dropped"]
                + row["stale_excluded"]
            )
            assert row["rows_in"] - removed == row["rows_out"]
        assert cleaned.report["rows_out"].sum() == cleaned.prices.height

    def test_an_empty_table_gives_an_empty_report(self):
        cleaned = clean(prices())

        assert cleaned.prices.schema == PRICES_SCHEMA
        assert cleaned.prices.height == cleaned.report.height == cleaned.changes.height == 0

    def test_the_schema_of_the_prices_is_unchanged(self):
        cleaned = clean(prices(listing(price_agorot=5900)), prices(earlier(590)))

        assert cleaned.prices.schema == PRICES_SCHEMA
        assert cleaned.prices.columns == PRICES_SCHEMA.names()


class TestConfig:
    def test_the_defaults(self):
        config = CleaningConfig()

        assert (config.min_price_agorot, config.staleness_days) == (10, 7)
        assert (config.large_change_ratio, config.implausible_ratio) == (2.0, 10.0)

    def test_thresholds_can_be_changed(self):
        config = CleaningConfig(min_price_agorot=100, implausible_ratio=5.0, large_change_ratio=1.5)
        previous = prices(earlier(590), earlier(790, product_key=BREAD))

        cleaned = clean(prices(listing(MILK, 2950), listing(BREAD, 1185)), previous, config=config)

        assert cleaned.prices["price_flag"].to_list() == ["carried_forward", "ok"]
        assert cleaned.changes["reason"].to_list() == ["implausible_change", "large_change"]

    @pytest.mark.parametrize(
        ("fields", "message"),
        [
            ({"min_price_agorot": 0}, "at least 1 agora"),
            ({"large_change_ratio": 1.0}, "1 < large change < implausible"),
            ({"large_change_ratio": 10.0}, "1 < large change < implausible"),
            ({"implausible_ratio": 1.5}, "1 < large change < implausible"),
            ({"staleness_days": -1}, "cannot be negative"),
        ],
    )
    def test_bad_thresholds_are_rejected(self, fields, message):
        with pytest.raises(ValueError, match=message):
            CleaningConfig(**fields)


class TestOnRealSamples:
    def test_the_sample_dataset_is_cleaned(self, raw_dir):
        tables = build_canonical(raw_dir, "2026-11-15", RUN_AT)

        cleaned = clean(tables["prices"])

        assert cleaned.prices.schema == PRICES_SCHEMA
        assert cleaned.prices.select(pl.struct("product_key", "store_id").is_unique().all()).item()
        assert (cleaned.prices["price_agorot"] >= 10).all()
        assert cleaned.report["rows_in"].sum() == tables["prices"].height
        assert cleaned.report["rows_out"].sum() == cleaned.prices.height

    def test_cleaning_twice_changes_nothing(self, raw_dir):
        tables = build_canonical(raw_dir, "2026-11-15", RUN_AT)
        once = clean(tables["prices"]).prices

        again = clean(once, previous=once)

        assert again.prices.equals(once)
        assert again.changes.height == 0
