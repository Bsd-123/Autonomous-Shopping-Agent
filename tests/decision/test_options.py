"""Tests for the options ``decide()`` returns and the recommended one."""

from itertools import combinations

import numpy as np
import pytest

from shopping_agent.decision import (
    NO_STORE,
    NOT_LISTED,
    DecisionConfig,
    Option,
    Point,
    decide,
    shortest_tour,
)

X = NOT_LISTED
HOME = Point(32.0840, 34.8300)
A = Point(32.0900, 34.8350)
B = Point(32.0800, 34.8200)
C = Point(32.0950, 34.8450)
STORES = [A, B, C]
# No penalty and no travel, so totals are basket prices and easy to read.
FREE = DecisionConfig(penalty_agorot=0, travel_agorot_per_km=0.0)


def sizes(decision) -> list[int]:
    return [len(option.stores) for option in decision.options]


class TestOnePerStoreCount:
    def test_each_option_is_the_best_subset_of_its_size(self):
        # Rows: milk, bread, eggs. Each store is cheapest for one item.
        cost = [[100, 500, 500], [500, 100, 500], [500, 500, 100]]

        result = decide(cost, HOME, STORES, config=FREE)

        assert sizes(result) == [1, 2, 3]
        assert [option.total_agorot for option in result.options] == [1100, 700, 300]
        assert [option.fulfilled for option in result.options] == [3, 3, 3]

    def test_most_items_come_before_the_lowest_total(self):
        # Store A is cheap but lacks the eggs; store B has everything for more.
        cost = [[100, 400], [100, 400], [X, 400]]

        one_store = decide(cost, HOME, [A, B], config=FREE).options[0]

        assert one_store.stores == (1,)
        assert one_store.fulfilled == 3
        assert one_store.total_agorot == 1200

    def test_among_equal_items_the_lowest_total_wins(self):
        cost = [[300, 250, 280], [300, 250, 280]]

        one_store = decide(cost, HOME, STORES, config=FREE).options[0]

        assert one_store.stores == (1,)
        assert one_store.total_agorot == 500

    def test_options_match_a_search_over_the_subsets(self):
        rng = np.random.default_rng(3)
        cost = rng.integers(100, 3000, size=(8, 6))
        cost[rng.random(cost.shape) < 0.35] = X
        stores = [
            Point(32.08 + float(dy), 34.83 + float(dx))
            for dy, dx in rng.uniform(-0.04, 0.04, size=(6, 2))
        ]

        result = decide(cost, HOME, stores)
        table = result.subsets

        for option in result.options:
            rows = np.flatnonzero(table.size == len(option.stores))
            best = max(rows, key=lambda row: (table.fulfilled[row], -table.total_agorot[row]))
            assert option.fulfilled == table.fulfilled[best]
            assert option.total_agorot == table.total_agorot[best]


class TestDominatedOptions:
    def test_an_option_no_better_than_a_smaller_one_is_dropped(self):
        # Buying the milk at B saves 10 agorot, far less than the penalty for a second store.
        cost = [[590, 580], [800, 820]]

        result = decide(cost, HOME, [A, B])

        assert sizes(result) == [1]
        assert len(result.subsets) == 3

    def test_an_option_is_kept_when_it_fulfils_more_items(self):
        cost = [[590, X], [X, 820]]

        result = decide(cost, HOME, [A, B])

        assert sizes(result) == [1, 2]
        assert [option.fulfilled for option in result.options] == [1, 2]
        assert result.options[1].total_agorot > result.options[0].total_agorot

    def test_an_option_is_kept_when_it_costs_less(self):
        cost = [[100, 5000], [5000, 100]]

        result = decide(cost, HOME, [A, B])

        assert sizes(result) == [1, 2]
        assert result.options[1].total_agorot < result.options[0].total_agorot

    def test_an_equal_total_with_no_more_items_is_dropped(self):
        cost = [[100, 600], [600, 100]]
        config = DecisionConfig(penalty_agorot=500, travel_agorot_per_km=0.0)

        result = decide(cost, HOME, [A, B], config=config)

        # One store costs 700; two stores cost 200 + 500.
        table = result.subsets
        assert table.total_agorot[table.size == 2].tolist() == [700]
        assert result.options[0].total_agorot == 700
        assert sizes(result) == [1]

    def test_a_three_store_option_is_compared_with_the_one_store_option_too(self):
        # Two stores are worse than one, but three would be compared with both.
        cost = [[100, 500, 500], [500, 100, 500], [500, 500, 100]]
        config = DecisionConfig(penalty_agorot=450, travel_agorot_per_km=0.0)

        result = decide(cost, HOME, STORES, config=config)

        # 1 store: 1100. 2 stores: 700 + 450 = 1150, dropped. 3 stores: 300 + 900 = 1200, dropped.
        assert sizes(result) == [1]

    def test_the_penalty_decides_which_options_survive(self):
        cost = [[100, 500, 500], [500, 100, 500], [500, 500, 100]]
        config = DecisionConfig(penalty_agorot=390, travel_agorot_per_km=0.0)

        result = decide(cost, HOME, STORES, config=config)

        # 1 store: 1100. 2 stores: 700 + 390 = 1090. 3 stores: 300 + 780 = 1080.
        assert sizes(result) == [1, 2, 3]
        config = DecisionConfig(penalty_agorot=405, travel_agorot_per_km=0.0)
        # 1 store: 1100. 2 stores: 1105, dropped. 3 stores: 300 + 810 = 1110, dropped.
        assert sizes(decide(cost, HOME, STORES, config=config)) == [1]


class TestRecommended:
    def test_the_lowest_total_is_recommended_when_all_items_are_fulfilled(self):
        cost = [[100, 500, 500], [500, 100, 500], [500, 500, 100]]

        result = decide(cost, HOME, STORES, config=FREE)

        assert result.recommended == 2
        assert result.recommended_option is result.options[2]

    def test_more_items_are_recommended_even_at_a_higher_total(self):
        cost = [[590, X], [X, 820]]

        result = decide(cost, HOME, [A, B])

        assert result.recommended == 1
        assert result.recommended_option.fulfilled == 2
        assert result.recommended_option.total_agorot > result.options[0].total_agorot

    def test_a_single_option_is_recommended(self):
        result = decide([[590, 580], [800, 820]], HOME, [A, B])

        assert result.recommended == 0

    def test_nothing_listed_gives_no_options(self):
        result = decide([[X, X], [X, X]], HOME, [A, B])

        assert result.options == ()
        assert result.recommended is None
        assert result.recommended_option is None

    def test_no_stores_in_range_gives_no_options(self):
        result = decide(np.zeros((2, 0), dtype=np.int64), HOME, [])

        assert result.options == ()
        assert result.recommended is None

    @pytest.mark.parametrize("seed", range(6))
    def test_the_recommended_option_is_never_beaten(self, seed):
        rng = np.random.default_rng(seed)
        cost = rng.integers(100, 3000, size=(7, 7))
        cost[rng.random(cost.shape) < 0.4] = X
        stores = [
            Point(32.08 + float(dy), 34.83 + float(dx))
            for dy, dx in rng.uniform(-0.04, 0.04, size=(7, 2))
        ]

        result = decide(cost, HOME, stores)
        table = result.subsets
        chosen = result.recommended_option

        assert chosen.fulfilled == table.fulfilled.max()
        assert chosen.total_agorot == table.total_agorot[table.fulfilled == chosen.fulfilled].min()
        for first, second in combinations(result.options, 2):
            assert len(first.stores) < len(second.stores)
            assert not (
                first.fulfilled >= second.fulfilled and first.total_agorot <= second.total_agorot
            )


class TestOptionContent:
    COST = [[590, 550, X], [800, X, 760], [X, X, X]]

    def test_each_item_is_bought_at_the_cheapest_store_of_the_option(self):
        result = decide(self.COST, HOME, STORES, config=FREE)
        two = next(option for option in result.options if len(option.stores) == 2)

        # Milk at B (550), bread at C (760); nobody lists the third item.
        assert sorted(two.stores) == [1, 2]
        assert two.supplier == (1, 2, NO_STORE)
        assert two.unfulfilled == (2,)
        assert two.fulfilled == 2
        assert two.basket_agorot == 550 + 760

    def test_every_store_of_an_option_supplies_an_item(self):
        result = decide(self.COST, HOME, STORES)

        for option in result.options:
            assert set(option.supplier) - {NO_STORE} == set(option.stores)

    def test_the_parts_add_up_to_the_total(self):
        config = DecisionConfig(penalty_agorot=300, travel_agorot_per_km=50.0)
        cost = [[100, 5000, 5000], [5000, 100, 5000], [5000, 5000, 100]]

        result = decide(cost, HOME, STORES, config=config)

        assert sizes(result) == [1, 2, 3]
        for option in result.options:
            assert option.penalty_agorot == 300 * (len(option.stores) - 1)
            assert option.travel_agorot == config.travel_cost(option.travel_km)
            assert option.total_agorot == (
                option.basket_agorot + option.penalty_agorot + option.travel_agorot
            )
            assert option.basket_agorot == sum(
                cost[item][store] for item, store in enumerate(option.supplier)
            )

    def test_stores_are_listed_in_the_order_of_the_shortest_tour(self):
        cost = [[100, 5000, 5000], [5000, 100, 5000], [5000, 5000, 100]]

        three = decide(cost, HOME, STORES).options[-1]
        tour = shortest_tour(HOME, STORES)

        assert three.stores == tour.order
        assert three.travel_km == pytest.approx(tour.distance_km)

    def test_an_option_is_plain_data(self):
        option = decide(self.COST, HOME, STORES).options[0]

        assert isinstance(option, Option)
        assert all(type(store) is int for store in option.stores + option.supplier)
        assert type(option.total_agorot) is int
        assert type(option.travel_km) is float
        with pytest.raises(AttributeError):
            option.fulfilled = 0

    def test_the_result_is_the_same_on_every_run(self):
        cost = [[300, 300, 300], [300, 300, 300]]

        first = decide(cost, HOME, STORES)

        assert first.options == decide(cost, HOME, STORES).options
        assert sizes(first) == [1]
