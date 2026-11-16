"""Tests for the subset enumeration behind ``decide()``.

The NumPy engine is checked against a plain-Python reference that follows the
definitions of architecture section 5.5 line by line.
"""

import time
from itertools import combinations
from math import comb, floor

import numpy as np
import pytest

from shopping_agent import decision
from shopping_agent.decision import (
    DEFAULT_PENALTY_AGOROT,
    DEFAULT_TRAVEL_AGOROT_PER_KM,
    MAX_STORES,
    NO_STORE,
    NOT_LISTED,
    Point,
    decide,
    haversine_km,
    shortest_tour,
)

X = NOT_LISTED
HOME = Point(32.0840, 34.8300)
A = Point(32.0900, 34.8350)
B = Point(32.0800, 34.8200)
C = Point(32.0950, 34.8450)
# Rows: milk, bread, eggs. Columns: stores A, B, C.
COST = [
    [590, 550, X],
    [800, X, 760],
    [X, X, 1390],
]


def reference(cost, home, stores, penalty=1000, per_km=100.0) -> dict[tuple[int, ...], tuple]:
    """Section 5.5, written out subset by subset."""
    rows = {}
    for size in (1, 2, 3):
        for subset in combinations(range(len(stores)), size):
            cheapest = [min((row[s] for s in subset if row[s] != X), default=None) for row in cost]
            fulfilled = sum(value is not None for value in cheapest)
            basket = sum(value for value in cheapest if value is not None)
            km = shortest_tour(home, [stores[s] for s in subset]).distance_km
            travel = floor(per_km * km + 0.5)
            rows[subset] = (fulfilled, basket, km, travel, basket + penalty * (size - 1) + travel)
    return rows


def as_rows(table) -> dict[tuple[int, ...], tuple]:
    rows = {}
    for n in range(len(table)):
        subset = tuple(int(s) for s in table.stores[n] if s != NO_STORE)
        assert len(subset) == table.size[n]
        rows[subset] = (
            int(table.fulfilled[n]),
            int(table.basket_agorot[n]),
            float(table.travel_km[n]),
            int(table.travel_agorot[n]),
            int(table.total_agorot[n]),
        )
    return rows


def assert_matches_reference(cost, home, stores, **settings) -> None:
    table = decide(cost, home, stores, **settings).subsets
    expected = reference(
        cost,
        home,
        stores,
        settings.get("penalty_agorot", 1000),
        settings.get("travel_agorot_per_km", 100.0),
    )
    found = as_rows(table)

    assert found.keys() == expected.keys()
    for subset, (fulfilled, basket, km, travel, total) in expected.items():
        assert found[subset][:2] == (fulfilled, basket), subset
        assert found[subset][2] == pytest.approx(km, abs=1e-9), subset
        assert found[subset][3:] == (travel, total), subset


class TestDefinitions:
    def test_every_subset_of_one_to_three_stores_is_evaluated(self):
        table = decide(COST, HOME, [A, B, C]).subsets

        assert len(table) == 3 + 3 + 1
        assert table.size.tolist() == [1, 1, 1, 2, 2, 2, 3]
        assert table.stores.tolist() == [
            [0, NO_STORE, NO_STORE],
            [1, NO_STORE, NO_STORE],
            [2, NO_STORE, NO_STORE],
            [0, 1, NO_STORE],
            [0, 2, NO_STORE],
            [1, 2, NO_STORE],
            [0, 1, 2],
        ]

    def test_fulfilled_counts_items_listed_in_at_least_one_store(self):
        table = decide(COST, HOME, [A, B, C]).subsets

        assert table.fulfilled.tolist() == [2, 1, 2, 2, 3, 3, 3]

    def test_basket_takes_the_lowest_cost_of_each_fulfilled_item(self):
        table = decide(COST, HOME, [A, B, C]).subsets

        assert table.basket_agorot.tolist() == [
            590 + 800,
            550,
            760 + 1390,
            550 + 800,
            590 + 760 + 1390,
            550 + 760 + 1390,
            550 + 760 + 1390,
        ]

    def test_travel_is_the_shortest_tour_rounded_once(self):
        table = decide(COST, HOME, [A, B, C]).subsets
        there_and_back = 2 * haversine_km(HOME, A)
        three = shortest_tour(HOME, [A, B, C]).distance_km

        assert table.travel_km[0] == pytest.approx(there_and_back)
        assert table.travel_km[6] == pytest.approx(three)
        assert table.travel_agorot[0] == floor(100 * there_and_back + 0.5)
        assert table.travel_agorot.dtype == np.int64
        assert table.travel_km.dtype == np.float64

    def test_total_is_basket_plus_penalty_plus_travel(self):
        table = decide(COST, HOME, [A, B, C]).subsets
        penalty = DEFAULT_PENALTY_AGOROT * (table.size - 1)

        assert DEFAULT_PENALTY_AGOROT == 1000
        assert DEFAULT_TRAVEL_AGOROT_PER_KM == 100
        assert penalty.tolist() == [0, 0, 0, 1000, 1000, 1000, 2000]
        assert (table.total_agorot == table.basket_agorot + penalty + table.travel_agorot).all()

    def test_the_whole_table_follows_section_5_5(self):
        assert_matches_reference(COST, HOME, [A, B, C])

    def test_penalty_and_travel_cost_can_be_set(self):
        assert_matches_reference(
            COST, HOME, [A, B, C], penalty_agorot=250, travel_agorot_per_km=37.5
        )
        free = decide(COST, HOME, [A, B, C], penalty_agorot=0, travel_agorot_per_km=0).subsets

        assert (free.total_agorot == free.basket_agorot).all()

    @pytest.mark.parametrize("seed", range(5))
    def test_random_baskets_match_the_reference(self, seed):
        rng = np.random.default_rng(seed)
        items, store_count = int(rng.integers(1, 12)), int(rng.integers(1, 9))
        cost = rng.integers(100, 5000, size=(items, store_count))
        cost[rng.random(cost.shape) < 0.4] = X
        stores = [
            Point(32.08 + float(dy), 34.83 + float(dx))
            for dy, dx in rng.uniform(-0.04, 0.04, size=(store_count, 2))
        ]

        assert_matches_reference(cost.tolist(), HOME, stores)


class TestEdges:
    def test_fewer_than_three_stores(self):
        one = decide([[590], [800]], HOME, [A]).subsets
        two = decide([[590, 550]], HOME, [A, B]).subsets

        assert one.size.tolist() == [1]
        assert two.size.tolist() == [1, 1, 2]
        assert two.basket_agorot.tolist() == [590, 550, 550]

    def test_no_stores_in_range(self):
        table = decide(np.zeros((2, 0), dtype=np.int64), HOME, []).subsets

        assert len(table) == 0
        assert table.stores.shape == (0, 3)

    def test_no_items(self):
        table = decide(np.zeros((0, 2), dtype=np.int64), HOME, [A, B]).subsets

        assert table.fulfilled.tolist() == [0, 0, 0]
        assert table.basket_agorot.tolist() == [0, 0, 0]

    def test_an_item_listed_nowhere_is_never_fulfilled(self):
        table = decide([[X, X, X], [800, 790, 760]], HOME, [A, B, C]).subsets

        assert set(table.fulfilled.tolist()) == {1}

    def test_a_store_that_lists_nothing_fulfils_nothing(self):
        table = decide([[590, X], [800, X]], HOME, [A, B]).subsets

        assert table.fulfilled.tolist() == [2, 0, 2]
        assert table.basket_agorot.tolist() == [1390, 0, 1390]

    def test_large_costs_do_not_overflow(self):
        huge = 10**12
        table = decide([[huge, X]] * 50, HOME, [A, B]).subsets

        assert table.basket_agorot.tolist() == [50 * huge, 0, 50 * huge]


class TestInputChecks:
    @pytest.mark.parametrize(
        ("cost", "stores", "error", "message"),
        [
            ([590, 800], [A, B], ValueError, "one row per item and one column per store"),
            ([[590, 800]], [A], ValueError, "2 store columns but 1 stores"),
            ([[590.5, 800.0]], [A, B], TypeError, "whole agorot"),
            ([[590, 0]], [A, B], ValueError, "must be positive, or -1 when not listed; got 0"),
            ([[590, -5]], [A, B], ValueError, "must be positive, or -1 when not listed; got -5"),
        ],
    )
    def test_a_malformed_cost_matrix_is_rejected(self, cost, stores, error, message):
        with pytest.raises(error, match=message):
            decide(cost, HOME, stores)

    def test_more_than_forty_stores_are_rejected(self):
        stores = [A] * (MAX_STORES + 1)

        with pytest.raises(ValueError, match="at most 40 candidate stores, got 41"):
            decide(np.full((1, len(stores)), 100), HOME, stores)

    @pytest.mark.parametrize(
        ("settings", "error"),
        [
            ({"penalty_agorot": -1}, ValueError),
            ({"penalty_agorot": 10.5}, TypeError),
            ({"travel_agorot_per_km": -1.0}, ValueError),
            ({"travel_agorot_per_km": float("nan")}, ValueError),
        ],
    )
    def test_bad_settings_are_rejected(self, settings, error):
        with pytest.raises(error):
            decide(COST, HOME, [A, B, C], **settings)

    def test_impossible_coordinates_are_rejected(self):
        with pytest.raises(ValueError, match="not a valid latitude and longitude"):
            decide([[590]], HOME, [Point(95.0, 34.8)])


class TestScale:
    def test_forty_stores_give_10_700_subsets_within_the_latency_target(self):
        rng = np.random.default_rng(7)
        cost = rng.integers(100, 5000, size=(30, MAX_STORES))
        cost[rng.random(cost.shape) < 0.3] = X
        stores = [
            Point(32.08 + float(dy), 34.83 + float(dx))
            for dy, dx in rng.uniform(-0.05, 0.05, size=(MAX_STORES, 2))
        ]

        started = time.perf_counter()
        table = decide(cost, HOME, stores).subsets
        elapsed = time.perf_counter() - started

        assert len(table) == comb(40, 1) + comb(40, 2) + comb(40, 3) == 10_700
        # The whole request has 3 seconds; the engine gets a small part of it.
        assert elapsed < 0.5

    def test_thirty_stores_give_4_525_subsets(self):
        cost = np.full((5, 30), 100)

        assert len(decide(cost, HOME, [A] * 30).subsets) == 4_525


def test_the_engine_is_reached_only_through_decide():
    assert "evaluate_subsets" not in decision.__all__
    assert not hasattr(decision, "evaluate_subsets")
    assert callable(decision.decide)
