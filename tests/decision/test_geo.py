"""Tests for Haversine distance and the shortest shopping tour."""

from itertools import permutations
from math import pi

import pytest

from shopping_agent.decision import (
    EARTH_RADIUS_KM,
    Point,
    Tour,
    haversine_km,
    shortest_tour,
)

TEL_AVIV = Point(32.0853, 34.7818)
JERUSALEM = Point(31.7683, 35.2137)
HAIFA = Point(32.7940, 34.9896)
LONDON = Point(51.5074, -0.1278)
PARIS = Point(48.8566, 2.3522)

# Bnei Brak: a home and three stores on one street, west to east.
HOME = Point(32.0840, 34.8300)
WEST = Point(32.0840, 34.8200)
NEAR_EAST = Point(32.0840, 34.8400)
FAR_EAST = Point(32.0840, 34.8500)


class TestHaversine:
    @pytest.mark.parametrize(
        ("origin", "destination", "expected_km"),
        [
            (TEL_AVIV, JERUSALEM, 54.0),
            (TEL_AVIV, HAIFA, 81.2),
            (LONDON, PARIS, 343.6),
        ],
    )
    def test_known_distances_between_cities(self, origin, destination, expected_km):
        assert haversine_km(origin, destination) == pytest.approx(expected_km, abs=0.5)

    def test_known_distances_on_the_globe(self):
        quarter = pi * EARTH_RADIUS_KM / 2

        # One degree along the equator, pole to equator, and opposite points.
        assert haversine_km(Point(0, 0), Point(0, 1)) == pytest.approx(111.195, abs=0.001)
        assert haversine_km(Point(90, 0), Point(0, 0)) == pytest.approx(quarter)
        assert haversine_km(Point(0, 0), Point(0, 180)) == pytest.approx(2 * quarter)
        assert haversine_km(Point(90, 0), Point(-90, 0)) == pytest.approx(2 * quarter)

    def test_a_degree_of_longitude_is_shorter_away_from_the_equator(self):
        at_bnei_brak = haversine_km(Point(32.084, 34.83), Point(32.084, 35.83))

        assert at_bnei_brak == pytest.approx(94.2, abs=0.1)

    def test_distance_to_the_same_point_is_zero(self):
        assert haversine_km(TEL_AVIV, TEL_AVIV) == 0.0

    def test_distance_is_symmetric_and_a_float_in_kilometres(self):
        there, back = haversine_km(TEL_AVIV, HAIFA), haversine_km(HAIFA, TEL_AVIV)

        assert there == pytest.approx(back)
        assert isinstance(there, float)
        assert there != round(there)

    def test_the_date_line_is_crossed_the_short_way(self):
        assert haversine_km(Point(0, 179.5), Point(0, -179.5)) == pytest.approx(111.195, abs=0.001)

    def test_plain_tuples_are_accepted(self):
        assert haversine_km((32.0853, 34.7818), (31.7683, 35.2137)) == pytest.approx(54.0, abs=0.5)

    @pytest.mark.parametrize(
        "point", [(91.0, 0.0), (0.0, 181.0), (float("nan"), 0.0), (0.0, float("inf"))]
    )
    def test_impossible_coordinates_are_rejected(self, point):
        with pytest.raises(ValueError, match="not a valid latitude and longitude"):
            haversine_km(TEL_AVIV, point)
        with pytest.raises(ValueError, match="not a valid latitude and longitude"):
            haversine_km(point, TEL_AVIV)


class TestShortestTour:
    def test_no_stores_is_an_empty_tour(self):
        assert shortest_tour(HOME, []) == Tour(0.0, ())

    def test_one_store_is_there_and_back(self):
        tour = shortest_tour(HOME, [NEAR_EAST])

        assert tour.order == (0,)
        assert tour.distance_km == pytest.approx(2 * haversine_km(HOME, NEAR_EAST))

    def test_two_stores_form_a_triangle(self):
        tour = shortest_tour(HOME, [WEST, NEAR_EAST])
        triangle = (
            haversine_km(HOME, WEST) + haversine_km(WEST, NEAR_EAST) + haversine_km(NEAR_EAST, HOME)
        )

        assert tour.distance_km == pytest.approx(triangle)
        assert tour.order == (0, 1)

    def test_three_stores_use_the_shortest_of_the_six_orders(self):
        # Home lies between WEST and the two eastern stores. Visiting WEST
        # between them crosses home twice more; the shortest tour keeps the
        # eastern stores together.
        stores = [NEAR_EAST, WEST, FAR_EAST]

        tour = shortest_tour(HOME, stores)

        assert tour.order in [(1, 0, 2), (1, 2, 0), (0, 2, 1), (2, 0, 1)]
        assert tour.distance_km == pytest.approx(2 * haversine_km(WEST, FAR_EAST))
        given_order = (
            haversine_km(HOME, NEAR_EAST)
            + haversine_km(NEAR_EAST, WEST)
            + haversine_km(WEST, FAR_EAST)
            + haversine_km(FAR_EAST, HOME)
        )
        assert tour.distance_km < given_order - 1.0

    def test_the_tour_is_never_longer_than_any_visit_order(self):
        stores = [Point(32.10, 34.84), Point(32.07, 34.81), Point(32.09, 34.86)]

        tour = shortest_tour(HOME, stores)

        for order in permutations(range(3)):
            path = [HOME, *(stores[i] for i in order), HOME]
            length = sum(haversine_km(a, b) for a, b in zip(path, path[1:], strict=False))
            assert tour.distance_km <= length + 1e-9
        assert sorted(tour.order) == [0, 1, 2]

    def test_the_result_does_not_depend_on_chance(self):
        stores = [NEAR_EAST, WEST, FAR_EAST]

        assert shortest_tour(HOME, stores) == shortest_tour(HOME, stores)

    def test_a_store_at_home_adds_nothing(self):
        alone = shortest_tour(HOME, [NEAR_EAST])

        assert shortest_tour(HOME, [HOME, NEAR_EAST]).distance_km == pytest.approx(
            alone.distance_km
        )

    def test_more_than_three_stores_are_rejected(self):
        with pytest.raises(ValueError, match="at most 3 stores, got 4"):
            shortest_tour(HOME, [WEST, NEAR_EAST, FAR_EAST, HOME])

    def test_impossible_coordinates_are_rejected(self):
        with pytest.raises(ValueError, match="not a valid latitude and longitude"):
            shortest_tour(HOME, [Point(95.0, 34.8)])
        with pytest.raises(ValueError, match="not a valid latitude and longitude"):
            shortest_tour(Point(95.0, 34.8), [])
