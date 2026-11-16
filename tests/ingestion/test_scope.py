"""Tests for restricting the dataset to the Bnei Brak branches."""

import polars as pl
import pytest

from shopping_agent.ingestion.adapters import ADAPTERS, PRICE_SCHEMA, STORE_SCHEMA
from shopping_agent.ingestion.scope import (
    BNEI_BRAK,
    SCOPE_REPORT_SCHEMA,
    ScopeError,
    prices_in_scope,
    restrict_to_scope,
    scope_report,
    stores_in_scope,
)
from support import MAAYAN_PRICES, OSHER_PRICES, RAW, stage_fixture

BRANCHES = {
    "maayan_2000": ["maayan_2000:38", "maayan_2000:39", "maayan_2000:40"],
    "osher_ad": ["osher_ad:11"],
    "yesh_hesed": ["yesh_hesed:219", "yesh_hesed:295", "yesh_hesed:611"],
}


@pytest.fixture(scope="module")
def stores() -> pl.DataFrame:
    return pl.concat([adapter.read_stores(RAW) for adapter in ADAPTERS])


@pytest.fixture(scope="module")
def prices() -> pl.DataFrame:
    return pl.concat([adapter.read_prices(RAW) for adapter in ADAPTERS])


class TestStores:
    def test_only_bnei_brak_branches_of_the_three_chains_are_kept(self, stores):
        kept = stores_in_scope(stores)

        assert stores.height == 84
        assert kept.height == 7
        assert kept["city"].unique().to_list() == [BNEI_BRAK]
        for chain, branches in BRANCHES.items():
            assert sorted(kept.filter(chain=chain)["store_id"]) == branches

    def test_the_layout_is_unchanged(self, stores):
        assert stores_in_scope(stores).schema == STORE_SCHEMA

    def test_a_store_without_a_city_is_left_out(self, stores):
        assert stores["city"].null_count() > 0
        assert stores_in_scope(stores)["city"].null_count() == 0

    def test_other_cities_can_be_chosen(self, stores):
        kept = stores_in_scope(stores.filter(chain="osher_ad"), cities={BNEI_BRAK, "9600"})

        assert set(kept["city"]) == {BNEI_BRAK, "9600"}

    def test_a_chain_with_no_branch_in_scope_fails_loudly(self, stores):
        elsewhere = stores.with_columns(
            city=pl.when(chain="osher_ad").then(pl.lit("1")).otherwise("city")
        )

        with pytest.raises(ScopeError, match="no branches in 6100 for: osher_ad"):
            stores_in_scope(elsewhere)

    def test_an_empty_scope_fails_loudly(self, stores):
        with pytest.raises(ScopeError, match="no cities were given"):
            stores_in_scope(stores, cities=())


class TestPrices:
    def test_only_rows_of_the_stores_in_scope_are_kept(self, stores, prices):
        kept = prices_in_scope(prices, stores_in_scope(stores))

        # In the samples only Yesh Hesed's price rows belong to a Bnei Brak branch.
        assert prices.height == 160 + 218 + 300
        assert kept.height == 160
        assert kept["store_id"].unique().to_list() == ["yesh_hesed:611"]
        assert kept.schema == PRICE_SCHEMA

    def test_rows_keep_their_order_and_content(self, stores, prices):
        kept = prices_in_scope(prices, stores_in_scope(stores))

        assert kept.equals(prices.filter(store_id="yesh_hesed:611"))

    def test_prices_of_every_chain_are_kept_when_listed(self, stores, tmp_path):
        # Move the first store of each sample price file to a Bnei Brak branch.
        stage_fixture(tmp_path, OSHER_PRICES, storeid="11")
        stage_fixture(tmp_path, MAAYAN_PRICES, storeid="38")
        staged = pl.concat([adapter.read_prices(tmp_path) for adapter in ADAPTERS[1:]])

        kept = prices_in_scope(staged, stores_in_scope(stores))

        assert set(kept["store_id"]) == {"osher_ad:11", "maayan_2000:38"}
        assert 0 < kept.height < staged.height

    def test_a_price_of_an_unknown_store_is_dropped(self, stores, prices):
        unknown = prices.head(1).with_columns(store_id=pl.lit("yesh_hesed:99999"))

        assert prices_in_scope(unknown, stores_in_scope(stores)).height == 0


class TestReport:
    def test_branch_count_per_chain(self, stores, prices):
        scoped = restrict_to_scope(stores, prices)

        assert scoped.report.schema == SCOPE_REPORT_SCHEMA
        assert scoped.report.rows() == [
            ("maayan_2000", 3, 0, 0),
            ("osher_ad", 1, 0, 0),
            ("yesh_hesed", 3, 1, 160),
        ]

    def test_the_report_counts_what_was_kept(self, stores, prices):
        scoped = restrict_to_scope(stores, prices)

        assert scoped.report["branches"].sum() == scoped.stores.height == 7
        assert scoped.report["price_rows"].sum() == scoped.prices.height == 160

    def test_a_store_listed_twice_is_counted_once(self, stores, prices):
        kept = stores_in_scope(stores)

        report = scope_report(pl.concat([kept, kept]), prices_in_scope(prices, kept))

        assert report["branches"].to_list() == [3, 1, 3]


def test_restrict_to_scope_returns_stores_prices_and_report(stores, prices):
    scoped = restrict_to_scope(stores, prices)

    assert scoped.stores.equals(stores_in_scope(stores))
    assert scoped.prices.equals(prices_in_scope(prices, scoped.stores))
    assert set(scoped.prices["store_id"]) <= set(scoped.stores["store_id"])
