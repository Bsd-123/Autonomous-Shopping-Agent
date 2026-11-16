"""Tests for selecting the candidate stores in range."""

import pytest

from shopping_agent.decision.geo import haversine_km
from shopping_agent.decision.stores import (
    DEFAULT_RADIUS_KM,
    MAX_STORES_OVERALL,
    MAX_STORES_PER_CHAIN,
    Store,
    StoreInRange,
    stores_in_range,
)

HOME = (32.0840, 34.8300)
# One hundredth of a degree of latitude is about 1.11 km.
KM = 0.01 / 1.11195


def store_at(store_id: str, km_north: float, chain: str | None = None) -> Store:
    return Store(store_id, chain or store_id.split(":")[0], HOME[0] + km_north * KM, HOME[1])


def ids(selected) -> list[str]:
    return [store.store_id for store in selected]


class TestRadius:
    def test_the_architecture_defaults(self):
        assert (DEFAULT_RADIUS_KM, MAX_STORES_PER_CHAIN, MAX_STORES_OVERALL) == (5.0, 10, 40)

    def test_only_stores_within_the_radius_are_selected(self):
        stores = [store_at("a:1", 1.0), store_at("a:2", 4.9), store_at("a:3", 5.1)]

        assert ids(stores_in_range(HOME, stores)) == ["a:1", "a:2"]

    def test_the_radius_can_be_changed(self):
        stores = [store_at("a:1", 1.0), store_at("a:2", 4.9), store_at("a:3", 5.1)]

        assert ids(stores_in_range(HOME, stores, radius_km=2.0)) == ["a:1"]
        assert ids(stores_in_range(HOME, stores, radius_km=10.0)) == ["a:1", "a:2", "a:3"]

    def test_a_store_exactly_on_the_radius_is_in_range(self):
        edge = store_at("a:1", 3.0)
        distance = haversine_km(HOME, (edge.lat, edge.lon))

        assert ids(stores_in_range(HOME, [edge], radius_km=distance)) == ["a:1"]
        assert stores_in_range(HOME, [edge], radius_km=distance - 1e-6) == ()

    def test_a_store_at_home_is_in_range(self):
        selected = stores_in_range(HOME, [Store("a:1", "a", *HOME)])

        assert selected[0].distance_km == 0.0

    def test_no_stores_in_range_is_an_empty_selection(self):
        assert stores_in_range(HOME, [store_at("a:1", 30.0)]) == ()
        assert stores_in_range(HOME, []) == ()

    def test_the_result_carries_the_distance_and_is_ordered_nearest_first(self):
        stores = [store_at("a:1", 3.0), store_at("b:1", -1.0), store_at("c:1", 2.0)]

        selected = stores_in_range(HOME, stores)

        assert ids(selected) == ["b:1", "c:1", "a:1"]
        assert isinstance(selected[0], StoreInRange)
        assert selected[0].chain == "b"
        assert [store.distance_km for store in selected] == pytest.approx([1.0, 2.0, 3.0], abs=0.01)
        assert selected[0].distance_km == haversine_km(HOME, (selected[0].lat, selected[0].lon))


class TestStoresWithoutCoordinates:
    @pytest.mark.parametrize(
        ("lat", "lon"),
        [(None, None), (32.08, None), (None, 34.83), (float("nan"), 34.83), (32.08, float("inf"))],
    )
    def test_a_store_without_coordinates_is_never_selected(self, lat, lon):
        stores = [Store("a:1", "a", lat, lon), store_at("a:2", 1.0)]

        assert ids(stores_in_range(HOME, stores)) == ["a:2"]

    def test_it_is_not_selected_even_with_an_unlimited_radius(self):
        assert stores_in_range(HOME, [Store("a:1", "a", None, None)], radius_km=20_000) == ()

    def test_it_does_not_take_a_place_under_the_caps(self):
        stores = [Store("a:0", "a", None, None), store_at("a:1", 1.0), store_at("a:2", 2.0)]

        assert ids(stores_in_range(HOME, stores, per_chain=2)) == ["a:1", "a:2"]


class TestPerChainCap:
    def test_at_most_ten_per_chain_and_the_nearest_are_kept(self):
        stores = [store_at(f"a:{n}", 0.3 * n) for n in range(1, 15)]

        selected = stores_in_range(HOME, stores)

        assert len(selected) == 10
        assert ids(selected) == [f"a:{n}" for n in range(1, 11)]

    def test_the_cap_applies_to_each_chain_separately(self):
        stores = [store_at(f"a:{n}", 0.3 * n) for n in range(1, 13)]
        stores += [store_at(f"b:{n}", 0.3 * n + 0.1) for n in range(1, 4)]

        selected = stores_in_range(HOME, stores)

        assert sum(store.chain == "a" for store in selected) == 10
        assert sum(store.chain == "b" for store in selected) == 3

    def test_the_cap_can_be_changed(self):
        stores = [store_at(f"a:{n}", 0.3 * n) for n in range(1, 6)]

        assert ids(stores_in_range(HOME, stores, per_chain=2)) == ["a:1", "a:2"]


class TestOverallCap:
    def five_chains(self) -> list[Store]:
        # Ten stores per chain; chain "a" is the nearest, chain "e" the farthest.
        return [
            store_at(f"{chain}:{n}", 0.2 + 0.4 * n + 0.05 * offset)
            for offset, chain in enumerate("abcde")
            for n in range(10)
        ]

    def test_at_most_forty_overall(self):
        selected = stores_in_range(HOME, self.five_chains())

        assert len(selected) == 40
        assert {store.chain: 0 for store in selected}.keys() == set("abcde")

    def test_the_cap_is_filled_chain_by_chain_in_turn(self):
        selected = stores_in_range(HOME, self.five_chains())
        per_chain = {chain: sum(s.chain == chain for s in selected) for chain in "abcde"}

        # Eight full rounds of five chains: every chain keeps its eight nearest.
        assert per_chain == dict.fromkeys("abcde", 8)
        for chain in "abcde":
            kept = sorted(s.store_id for s in selected if s.chain == chain)
            assert kept == [f"{chain}:{n}" for n in range(8)]

    def test_a_farther_chain_is_not_cut_out_by_a_chain_with_many_near_branches(self):
        near = [store_at(f"near:{n}", 0.1 * (n + 1)) for n in range(10)]
        far = [store_at(f"far:{n}", 4.0 + 0.05 * n) for n in range(3)]

        selected = stores_in_range(HOME, near + far, overall=6)

        # In turn: near, far, near, far, near, far. By distance alone "far" would get nothing.
        assert sorted(ids(selected)) == ["far:0", "far:1", "far:2", "near:0", "near:1", "near:2"]

    def test_the_last_round_takes_the_nearest_stores_first(self):
        stores = [store_at("a:1", 1.0), store_at("b:1", 2.0), store_at("c:1", 3.0)]

        assert ids(stores_in_range(HOME, stores, overall=2)) == ["a:1", "b:1"]

    def test_a_chain_that_runs_out_leaves_its_turns_to_the_others(self):
        stores = [store_at("a:1", 1.0)] + [store_at(f"b:{n}", 2.0 + 0.1 * n) for n in range(5)]

        selected = stores_in_range(HOME, stores, overall=4)

        assert ids(selected) == ["a:1", "b:0", "b:1", "b:2"]

    def test_three_chains_never_reach_the_overall_cap(self):
        stores = [
            store_at(f"{chain}:{n}", 0.1 + 0.3 * n)
            for chain in ("yesh_hesed", "osher_ad", "maayan_2000")
            for n in range(12)
        ]

        assert len(stores_in_range(HOME, stores)) == 30


class TestDeterminism:
    def test_stores_at_the_same_distance_are_ordered_by_id(self):
        stores = [store_at("b:1", 1.0), store_at("a:1", 1.0), store_at("c:1", 1.0)]

        assert ids(stores_in_range(HOME, stores)) == ["a:1", "b:1", "c:1"]
        assert ids(stores_in_range(HOME, stores[::-1])) == ["a:1", "b:1", "c:1"]

    def test_the_order_of_the_input_does_not_change_the_selection(self):
        stores = TestOverallCap().five_chains()

        assert stores_in_range(HOME, stores) == stores_in_range(HOME, stores[::-1])

    def test_stores_can_come_from_a_generator(self):
        assert ids(stores_in_range(HOME, (store_at(f"a:{n}", n) for n in (1, 2)))) == ["a:1", "a:2"]


class TestInputChecks:
    @pytest.mark.parametrize("radius", [0, -1.0, float("nan"), float("inf")])
    def test_the_radius_must_be_positive(self, radius):
        with pytest.raises(ValueError, match="radius_km must be a positive number"):
            stores_in_range(HOME, [], radius_km=radius)

    @pytest.mark.parametrize("caps", [{"per_chain": 0}, {"overall": 0}])
    def test_the_caps_must_be_at_least_one(self, caps):
        with pytest.raises(ValueError, match="at least 1"):
            stores_in_range(HOME, [], **caps)

    def test_a_store_given_twice_is_rejected(self):
        with pytest.raises(ValueError, match="store a:1 was given twice"):
            stores_in_range(HOME, [store_at("a:1", 1.0), store_at("a:1", 2.0)])

    def test_an_impossible_home_is_rejected(self):
        with pytest.raises(ValueError, match="not a valid latitude and longitude"):
            stores_in_range((95.0, 34.8), [store_at("a:1", 1.0)])
