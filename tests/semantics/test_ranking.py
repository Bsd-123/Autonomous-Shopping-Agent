"""Tests for the ranking rule: total cost first, unit price only on ties."""

from fractions import Fraction
from itertools import permutations

import pytest

from shopping_agent.contracts import ParsedItem, Quantity
from shopping_agent.semantics import (
    CandidateCost,
    PricedProduct,
    filter_candidates,
    rank_candidates,
    unit_price,
)


def ml(value: float) -> Quantity:
    return Quantity(value=value, unit="ml")


def milk(pack_ml: float, price: int) -> PricedProduct:
    return PricedProduct(price_agorot=price, pack=ml(pack_ml))


def ranked(request: ParsedItem, products: dict[str, PricedProduct]) -> list[str]:
    outcome = filter_candidates(request, products.items())
    return [key for key, _ in rank_candidates(outcome.kept, products)]


class TestUnitPrice:
    def test_packaged_product(self):
        assert unit_price(milk(1000, 590)) == Fraction(59, 100)
        assert unit_price(milk(1500, 850)) == Fraction(17, 30)

    def test_counted_product(self):
        eggs = PricedProduct(price_agorot=1390, pack=Quantity(value=12, unit="unit"))

        assert unit_price(eggs) == Fraction(1390, 12)

    def test_weighed_product_is_priced_per_kilogram(self):
        tomatoes = PricedProduct(price_agorot=690, is_weighed=True)

        assert unit_price(tomatoes) == Fraction(69, 100)

    def test_equal_unit_prices_compare_equal(self):
        # As floats 0.1 + 0.2 is not 0.3; the exact value has no such error.
        assert unit_price(milk(0.3, 3)) == unit_price(milk(0.1, 1)) == 10


class TestRanking:
    def test_candidates_are_ordered_by_total_cost(self):
        request = ParsedItem(item="milk", total_quantity=ml(3000))
        products = {
            "dear": milk(1000, 690),  # 3 packs, 2070
            "cheap": milk(1000, 550),  # 3 packs, 1650
            "middle": milk(1500, 900),  # 2 packs, 1800
        }

        assert ranked(request, products) == ["cheap", "middle", "dear"]

    def test_cheaper_per_pack_wins_over_cheaper_per_litre_when_its_total_is_lower(self):
        request = ParsedItem(item="milk", total_quantity=ml(1000))
        products = {
            "per_litre": milk(1500, 750),  # 5.00 per litre, but the request costs 7.50
            "per_pack": milk(1000, 590),  # 5.90 per litre, and the request costs 5.90
        }

        assert unit_price(products["per_litre"]) < unit_price(products["per_pack"])
        assert ranked(request, products) == ["per_pack", "per_litre"]

    def test_unit_price_breaks_a_tie_in_total_cost(self):
        request = ParsedItem(item="milk", total_quantity=ml(1000))
        products = {
            "one_litre": milk(1000, 600),  # 6.00 per litre
            "litre_and_a_half": milk(1500, 600),  # 4.00 per litre, same total
        }

        assert ranked(request, products) == ["litre_and_a_half", "one_litre"]

    def test_unit_price_never_overrides_a_lower_total(self):
        request = ParsedItem(item="milk", total_quantity=ml(1000))
        products = {
            "bargain_per_litre": milk(1500, 601),  # far cheaper per litre, one agora dearer
            "one_litre": milk(1000, 600),
        }

        assert ranked(request, products) == ["one_litre", "bargain_per_litre"]

    def test_candidates_equal_in_both_keep_their_given_order(self):
        cost = CandidateCost(cost_agorot=590, packs=1)
        products = {"b": milk(1000, 590), "a": milk(1000, 590), "c": milk(1000, 590)}

        result = rank_candidates([("b", cost), ("a", cost), ("c", cost)], products)

        assert [key for key, _ in result] == ["b", "a", "c"]

    def test_the_order_given_does_not_change_a_ranking_without_full_ties(self):
        request = ParsedItem(item="milk", total_quantity=ml(2000))
        products = {
            "a": milk(1000, 590),  # 1180
            "b": milk(2000, 1090),  # 1090
            "c": milk(1000, 545),  # 1090, and the same price per litre
            "d": milk(1500, 800),  # 2 packs, 1600
        }

        orders = {
            tuple(ranked(request, dict(shuffled))) for shuffled in permutations(products.items())
        }

        # "b" and "c" tie on total and on unit price, so only their order may differ.
        assert orders == {("b", "c", "a", "d"), ("c", "b", "a", "d")}

    def test_count_requests_are_ranked_the_same_way(self):
        request = ParsedItem(item="milk", count=2)
        products = {"big": milk(1500, 800), "small": milk(1000, 590)}

        assert ranked(request, products) == ["small", "big"]

    def test_weighed_products_are_ranked_by_the_cost_of_the_weight(self):
        request = ParsedItem(item="tomatoes", total_quantity=Quantity(value=1500, unit="g"))
        products = {
            "dear": PricedProduct(price_agorot=890, is_weighed=True),
            "cheap": PricedProduct(price_agorot=690, is_weighed=True),
        }

        assert ranked(request, products) == ["cheap", "dear"]

    def test_costs_are_returned_with_the_keys(self):
        request = ParsedItem(item="milk", total_quantity=ml(2000))
        products = {"one": milk(1000, 590), "two": milk(2000, 1090)}
        outcome = filter_candidates(request, products.items())

        assert rank_candidates(outcome.kept, products) == (
            ("two", CandidateCost(cost_agorot=1090, packs=1)),
            ("one", CandidateCost(cost_agorot=1180, packs=2)),
        )

    def test_nothing_to_rank(self):
        assert rank_candidates([], {}) == ()

    def test_a_candidate_without_its_product_is_an_error(self):
        cost = CandidateCost(cost_agorot=590, packs=1)

        with pytest.raises(KeyError, match="no product was given for: 'ghost'"):
            rank_candidates([("ghost", cost)], {})
