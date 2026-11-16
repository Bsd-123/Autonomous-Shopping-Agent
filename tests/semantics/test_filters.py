"""Tests for the overbuy filter and the small-pack exclusion."""

import pytest

from shopping_agent.contracts import ParsedItem, Quantity
from shopping_agent.semantics import (
    OVERBUY_LIMIT,
    CandidateCost,
    Exclusion,
    Mismatch,
    PricedProduct,
    candidate_cost,
    filter_candidates,
    quantity_exclusion,
)


def ml(value: float) -> Quantity:
    return Quantity(value=value, unit="ml")


def g(value: float) -> Quantity:
    return Quantity(value=value, unit="g")


def milk(pack_ml: float, price: int = 590) -> PricedProduct:
    return PricedProduct(price_agorot=price, pack=ml(pack_ml))


def exclusion(request: ParsedItem, product: PricedProduct, **profile) -> Exclusion | None:
    cost = candidate_cost(request, product)
    assert isinstance(cost, CandidateCost)
    return quantity_exclusion(request, product, cost, **profile)


TYPICAL_MILK = ml(1000)
TOMATOES = PricedProduct(price_agorot=690, is_weighed=True)


class TestOverbuy:
    def test_the_limit_is_one_and_a_half_times_the_request(self):
        assert OVERBUY_LIMIT == 1.5

    @pytest.mark.parametrize(
        ("requested_ml", "pack_ml", "expected"),
        [
            (3000, 1000, None),  # 3 x 1 L delivers exactly 3 L
            (3000, 2000, None),  # 2 x 2 L delivers 4 L, 1.33 times
            (1000, 1500, None),  # exactly 1.5 times stays
            (1000, 1501, Exclusion.OVERBUY),
            (1000, 2000, Exclusion.OVERBUY),  # twice the request
            (250, 1000, Exclusion.OVERBUY),  # four times the request
            (2500, 2000, Exclusion.OVERBUY),  # 2 x 2 L delivers 4 L, 1.6 times
        ],
    )
    def test_a_candidate_delivering_more_than_the_limit_is_excluded(
        self, requested_ml, pack_ml, expected
    ):
        request = ParsedItem(item="milk", total_quantity=ml(requested_ml))

        assert exclusion(request, milk(pack_ml)) is expected

    def test_float_error_does_not_exclude_a_candidate_at_the_limit(self):
        request = ParsedItem(item="saffron", total_quantity=g(0.2))
        product = PricedProduct(price_agorot=900, pack=g(0.3))

        assert exclusion(request, product) is None

    def test_a_request_by_count_or_pack_size_cannot_overbuy(self):
        assert exclusion(ParsedItem(item="milk", count=5), milk(2000)) is None
        assert exclusion(ParsedItem(item="milk", count=5, pack_size=ml(2000)), milk(2000)) is None

    def test_a_weighed_product_is_bought_to_the_gram(self):
        request = ParsedItem(item="tomatoes", total_quantity=g(50))

        assert exclusion(request, TOMATOES) is None


class TestSmallPack:
    @pytest.mark.parametrize(
        ("pack_ml", "expected"),
        [(250, Exclusion.SMALL_PACK), (799, Exclusion.SMALL_PACK), (800, None), (1000, None)],
    )
    def test_packs_below_0_8_of_the_typical_pack_are_excluded(self, pack_ml, expected):
        request = ParsedItem(item="milk", total_quantity=ml(3000))

        assert exclusion(request, milk(pack_ml), typical_pack=TYPICAL_MILK) is expected

    def test_three_litres_are_not_met_with_twelve_small_cartons(self):
        request = ParsedItem(item="milk", total_quantity=ml(3000))
        small = milk(250, price=250)

        assert candidate_cost(request, small).packs == 12
        assert exclusion(request, small, typical_pack=TYPICAL_MILK) is Exclusion.SMALL_PACK

    def test_without_a_profile_small_packs_stay(self):
        request = ParsedItem(item="soy sauce", total_quantity=ml(3000))

        assert exclusion(request, milk(250)) is None

    def test_the_rule_applies_only_to_a_total_quantity_request(self):
        request = ParsedItem(item="milk", count=2, pack_size=ml(250))

        assert exclusion(request, milk(250), typical_pack=TYPICAL_MILK) is None

    def test_a_large_pack_is_not_a_small_pack(self):
        request = ParsedItem(item="milk", total_quantity=ml(6000))

        assert exclusion(request, milk(3000), typical_pack=TYPICAL_MILK) is None

    def test_a_typical_pack_in_another_unit_is_not_compared(self):
        request = ParsedItem(item="milk", total_quantity=ml(3000))

        assert exclusion(request, milk(250), typical_pack=g(1000)) is None

    def test_a_small_pack_is_reported_as_small_even_when_it_also_overbuys(self):
        request = ParsedItem(item="milk", total_quantity=ml(100))

        assert exclusion(request, milk(250), typical_pack=TYPICAL_MILK) is Exclusion.SMALL_PACK


class TestFilterCandidates:
    def test_each_candidate_is_filtered_on_its_own(self):
        request = ParsedItem(item="milk", total_quantity=ml(1000))
        candidates = [
            ("one_litre", milk(1000, 590)),
            ("two_litres", milk(2000, 1090)),
            ("small", milk(250, 250)),
            ("by_weight", PricedProduct(price_agorot=500, pack=g(1000))),
            ("litre_and_a_half", milk(1500, 850)),
        ]

        outcome = filter_candidates(request, candidates, typical_pack=TYPICAL_MILK)

        assert outcome.kept == (
            ("one_litre", CandidateCost(cost_agorot=590, packs=1)),
            ("litre_and_a_half", CandidateCost(cost_agorot=850, packs=1)),
        )
        assert outcome.rejected == (
            ("two_litres", Exclusion.OVERBUY),
            ("small", Exclusion.SMALL_PACK),
            ("by_weight", Mismatch.UNIT),
        )
        assert outcome.all_failed_on_overbuy is False

    def test_pack_sizes_of_different_products_are_never_combined(self):
        # 3 L could be met with one 2 L and one 1 L pack; each product is costed alone.
        request = ParsedItem(item="milk", total_quantity=ml(3000))

        outcome = filter_candidates(request, [("one", milk(1000, 590)), ("two", milk(2000, 1090))])

        assert dict(outcome.kept) == {
            "one": CandidateCost(cost_agorot=1770, packs=3),
            "two": CandidateCost(cost_agorot=2180, packs=2),
        }

    def test_reports_when_every_otherwise_valid_candidate_failed_on_overbuy(self):
        request = ParsedItem(item="milk", total_quantity=ml(250))
        candidates = [
            ("one_litre", milk(1000)),
            ("two_litres", milk(2000)),
            ("by_weight", PricedProduct(price_agorot=500, pack=g(1000))),
        ]

        outcome = filter_candidates(request, candidates)

        assert outcome.kept == ()
        assert outcome.all_failed_on_overbuy is True

    def test_no_report_when_a_candidate_survives(self):
        request = ParsedItem(item="milk", total_quantity=ml(250))

        outcome = filter_candidates(request, [("litre", milk(1000)), ("quarter", milk(250))])

        assert [key for key, _ in outcome.kept] == ["quarter"]
        assert outcome.all_failed_on_overbuy is False

    def test_no_report_when_nothing_failed_on_overbuy(self):
        request = ParsedItem(item="milk", total_quantity=ml(3000))
        candidates = [
            ("small", milk(250)),
            ("by_weight", PricedProduct(price_agorot=500, pack=g(1000))),
        ]

        outcome = filter_candidates(request, candidates, typical_pack=TYPICAL_MILK)

        assert outcome.kept == ()
        assert outcome.all_failed_on_overbuy is False

    def test_no_candidates_is_not_an_overbuy_failure(self):
        outcome = filter_candidates(ParsedItem(item="milk", total_quantity=ml(250)), [])

        assert outcome.kept == outcome.rejected == ()
        assert outcome.all_failed_on_overbuy is False

    def test_the_profile_reaches_the_cost_function(self):
        by_count = ParsedItem(item="milk", count=2)
        six_tomatoes = ParsedItem(item="tomatoes", count=6)

        packs = filter_candidates(by_count, [("big", milk(2000))], typical_pack=TYPICAL_MILK)
        weighed = filter_candidates(
            six_tomatoes, [("tomato", TOMATOES)], typical_unit_weight=g(120)
        )

        assert packs.rejected == (("big", Mismatch.NOT_TYPICAL_PACK),)
        assert weighed.kept == (
            ("tomato", CandidateCost(cost_agorot=497, weight_g=720, is_estimate=True)),
        )

    def test_candidates_can_come_from_a_generator(self):
        request = ParsedItem(item="milk", total_quantity=ml(1000))

        outcome = filter_candidates(request, ((size, milk(size)) for size in (1000, 2000)))

        assert [key for key, _ in outcome.kept] == [1000]
        assert [key for key, _ in outcome.rejected] == [2000]
