"""Tests for the candidate cost function, one class per request form."""

import pytest

from shopping_agent.contracts import ParsedItem, Quantity
from shopping_agent.semantics import CandidateCost, Mismatch, PricedProduct, candidate_cost


def ml(value: float) -> Quantity:
    return Quantity(value=value, unit="ml")


def g(value: float) -> Quantity:
    return Quantity(value=value, unit="g")


def units(value: float) -> Quantity:
    return Quantity(value=value, unit="unit")


MILK_1L = PricedProduct(price_agorot=590, pack=ml(1000))
MILK_2L = PricedProduct(price_agorot=1090, pack=ml(2000))
TOMATOES = PricedProduct(price_agorot=690, is_weighed=True)  # per kilogram
TYPICAL_MILK = ml(1000)
TYPICAL_TOMATO = g(120)


class TestTotalQuantity:
    """ "3 L milk": ceil(total / pack size) x pack price."""

    def test_exact_multiple(self):
        request = ParsedItem(item="milk", total_quantity=ml(3000))

        assert candidate_cost(request, MILK_1L) == CandidateCost(cost_agorot=1770, packs=3)

    def test_the_pack_count_is_rounded_up(self):
        request = ParsedItem(item="milk", total_quantity=ml(3000))

        assert candidate_cost(request, MILK_2L) == CandidateCost(cost_agorot=2180, packs=2)

    def test_a_request_smaller_than_one_pack_buys_one_pack(self):
        request = ParsedItem(item="milk", total_quantity=ml(250))

        assert candidate_cost(request, MILK_1L) == CandidateCost(cost_agorot=590, packs=1)

    def test_only_packs_of_one_product_are_used(self):
        # 3 L from 2 L packs is two packs of the same product, never 2 L + 1 L.
        cost = candidate_cost(ParsedItem(item="milk", total_quantity=ml(3000)), MILK_2L)

        assert cost.packs * MILK_2L.pack.value == 4000

    def test_float_error_does_not_add_a_pack(self):
        product = PricedProduct(price_agorot=100, pack=g(0.1))
        request = ParsedItem(item="saffron", total_quantity=g(0.3))

        assert candidate_cost(request, product).packs == 3

    def test_units(self):
        eggs = PricedProduct(price_agorot=1390, pack=units(12))
        request = ParsedItem(item="eggs", total_quantity=units(30))

        assert candidate_cost(request, eggs) == CandidateCost(cost_agorot=4170, packs=3)

    def test_a_different_unit_does_not_fit(self):
        request = ParsedItem(item="milk", total_quantity=g(3000))

        assert candidate_cost(request, MILK_1L) is Mismatch.UNIT

    def test_the_typical_pack_is_not_applied_to_this_form(self):
        request = ParsedItem(item="milk", total_quantity=ml(3000))
        small = PricedProduct(price_agorot=250, pack=ml(250))

        assert candidate_cost(request, small, typical_pack=TYPICAL_MILK).packs == 12


class TestPackSizeAndCount:
    """ "milk 1 L x2": count x pack price, with a matching pack size."""

    def test_matching_pack(self):
        request = ParsedItem(item="milk", count=2, pack_size=ml(1000))

        assert candidate_cost(request, MILK_1L) == CandidateCost(cost_agorot=1180, packs=2)

    def test_an_explicit_container_has_a_count_of_one(self):
        request = ParsedItem(item="milk", pack_size=ml(2000))

        assert candidate_cost(request, MILK_2L) == CandidateCost(cost_agorot=1090, packs=1)

    def test_another_pack_size_does_not_fit(self):
        request = ParsedItem(item="milk", count=2, pack_size=ml(1000))

        assert candidate_cost(request, MILK_2L) is Mismatch.PACK_SIZE

    def test_two_small_packs_never_stand_in_for_the_requested_pack(self):
        half = PricedProduct(price_agorot=320, pack=ml(500))
        request = ParsedItem(item="milk", pack_size=ml(1000))

        assert candidate_cost(request, half) is Mismatch.PACK_SIZE

    def test_a_different_unit_does_not_fit(self):
        request = ParsedItem(item="milk", pack_size=g(1000))

        assert candidate_cost(request, MILK_1L) is Mismatch.UNIT


class TestCountOnly:
    """ "milk x2": count x pack price, the pack within 0.8-1.5x of the typical pack."""

    def test_typical_pack(self):
        request = ParsedItem(item="milk", count=2)

        cost = candidate_cost(request, MILK_1L, typical_pack=TYPICAL_MILK)

        assert cost == CandidateCost(cost_agorot=1180, packs=2)

    @pytest.mark.parametrize(
        ("pack_ml", "fits"),
        [(799, False), (800, True), (1000, True), (1500, True), (1501, False), (2000, False)],
    )
    def test_the_pack_must_be_near_the_typical_pack(self, pack_ml, fits):
        product = PricedProduct(price_agorot=500, pack=ml(pack_ml))

        cost = candidate_cost(ParsedItem(item="milk"), product, typical_pack=TYPICAL_MILK)

        if fits:
            assert cost == CandidateCost(cost_agorot=500, packs=1)
        else:
            assert cost is Mismatch.NOT_TYPICAL_PACK

    def test_without_a_profile_any_pack_fits(self):
        request = ParsedItem(item="soy sauce", count=3)

        assert candidate_cost(request, MILK_2L) == CandidateCost(cost_agorot=3270, packs=3)

    def test_a_pack_in_another_unit_than_the_typical_pack_does_not_fit(self):
        by_weight = PricedProduct(price_agorot=500, pack=g(1000))

        cost = candidate_cost(ParsedItem(item="milk"), by_weight, typical_pack=TYPICAL_MILK)

        assert cost is Mismatch.UNIT


class TestWeighedByWeight:
    """ "1.5 kg tomatoes": price per kilogram x grams / 1000, rounded once."""

    def test_weight(self):
        request = ParsedItem(item="tomatoes", total_quantity=g(1500))

        assert candidate_cost(request, TOMATOES) == CandidateCost(cost_agorot=1035, weight_g=1500)

    @pytest.mark.parametrize(
        ("price_per_kg", "grams", "expected"),
        [
            (690, 250, 173),  # 172.5 rounds up
            (690, 150, 104),  # 103.5 rounds up
            (699, 333, 233),  # 232.767
            (699, 100, 70),  # 69.9
            (1, 400, 0),  # 0.4
            (1090, 1234.5, 1346),  # 1345.605
        ],
    )
    def test_the_cost_is_rounded_once_to_whole_agorot(self, price_per_kg, grams, expected):
        product = PricedProduct(price_agorot=price_per_kg, is_weighed=True)
        request = ParsedItem(item="tomatoes", total_quantity=g(grams))

        cost = candidate_cost(request, product)

        assert cost.cost_agorot == expected
        assert isinstance(cost.cost_agorot, int)
        assert cost.is_estimate is False

    def test_a_weight_with_a_count_is_multiplied_before_rounding(self):
        # 3 x 250 g at 6.90 per kg is 517.5 agorot; rounding each 250 g first would give 519.
        request = ParsedItem(item="tomatoes", count=3, pack_size=g(250))

        assert candidate_cost(request, TOMATOES) == CandidateCost(cost_agorot=518, weight_g=750)

    def test_a_volume_does_not_fit_a_weighed_product(self):
        request = ParsedItem(item="tomatoes", total_quantity=ml(500))

        assert candidate_cost(request, TOMATOES) is Mismatch.UNIT


class TestWeighedByCount:
    """ "6 tomatoes": count x typical unit weight, marked as an estimate."""

    def test_count(self):
        request = ParsedItem(item="tomatoes", count=6)

        cost = candidate_cost(request, TOMATOES, typical_unit_weight=TYPICAL_TOMATO)

        # 6 x 120 g = 720 g at 6.90 per kg = 496.8 agorot.
        assert cost == CandidateCost(cost_agorot=497, weight_g=720, is_estimate=True)

    def test_without_a_typical_unit_weight_the_product_does_not_fit(self):
        cost = candidate_cost(ParsedItem(item="tomatoes", count=6), TOMATOES)

        assert cost is Mismatch.NO_TYPICAL_UNIT_WEIGHT

    def test_the_typical_unit_weight_must_be_in_grams(self):
        with pytest.raises(ValueError, match="must be in grams"):
            candidate_cost(ParsedItem(item="tomatoes"), TOMATOES, typical_unit_weight=units(1))

    def test_the_typical_pack_does_not_apply_to_weighed_products(self):
        request = ParsedItem(item="tomatoes", count=2)

        cost = candidate_cost(
            request, TOMATOES, typical_pack=g(1000), typical_unit_weight=TYPICAL_TOMATO
        )

        assert cost == CandidateCost(cost_agorot=166, weight_g=240, is_estimate=True)


class TestPricedProduct:
    def test_a_packaged_product_needs_a_pack(self):
        with pytest.raises(ValueError, match="needs a pack size"):
            PricedProduct(price_agorot=590)

    @pytest.mark.parametrize("price", [0, -5])
    def test_the_price_is_positive(self, price):
        with pytest.raises(ValueError, match="must be positive"):
            PricedProduct(price_agorot=price, pack=ml(1000))

    @pytest.mark.parametrize("price", [5.9, "590", True])
    def test_the_price_is_whole_agorot(self, price):
        with pytest.raises(TypeError, match="whole number of agorot"):
            PricedProduct(price_agorot=price, pack=ml(1000))


@pytest.mark.parametrize(
    ("request_fields", "product", "profile"),
    [
        ({"total_quantity": ml(3000)}, MILK_1L, {}),
        ({"count": 2, "pack_size": ml(1000)}, MILK_1L, {}),
        ({"count": 2}, MILK_1L, {"typical_pack": TYPICAL_MILK}),
        ({"total_quantity": g(1500)}, TOMATOES, {}),
        ({"count": 6}, TOMATOES, {"typical_unit_weight": TYPICAL_TOMATO}),
    ],
)
def test_every_form_gives_whole_agorot_and_one_way_of_buying(request_fields, product, profile):
    cost = candidate_cost(ParsedItem(item="x", **request_fields), product, **profile)

    assert type(cost.cost_agorot) is int
    assert cost.cost_agorot > 0
    assert (cost.packs is None) != (cost.weight_g is None)
