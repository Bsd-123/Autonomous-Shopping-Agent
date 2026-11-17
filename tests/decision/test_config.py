"""Tests for the configurable store penalty F and travel cost c."""

import pytest

from shopping_agent.decision import (
    DEFAULT_PENALTY_AGOROT,
    DEFAULT_TRAVEL_AGOROT_PER_KM,
    PENALTY_VARIABLE,
    TRAVEL_VARIABLE,
    ConfigError,
    DecisionConfig,
    Point,
    decide,
)

HOME = Point(32.0840, 34.8300)
A = Point(32.0900, 34.8350)
B = Point(32.0800, 34.8200)
C = Point(32.0950, 34.8450)
# Each store is the cheapest for one item, so every subset is returned.
COST = [[100, 500, 500], [500, 100, 500], [500, 500, 100]]


class TestDefaults:
    def test_the_architecture_defaults(self):
        config = DecisionConfig()

        assert config.penalty_agorot == DEFAULT_PENALTY_AGOROT == 1000
        assert config.travel_agorot_per_km == DEFAULT_TRAVEL_AGOROT_PER_KM == 100

    def test_an_empty_environment_gives_the_defaults(self):
        assert DecisionConfig.from_env({}) == DecisionConfig()
        assert DecisionConfig.from_env({PENALTY_VARIABLE: " ", TRAVEL_VARIABLE: ""}) == (
            DecisionConfig()
        )


class TestPenalty:
    @pytest.mark.parametrize(("stores", "expected"), [(1, 0), (2, 1000), (3, 2000)])
    def test_the_penalty_is_f_for_each_store_beyond_the_first(self, stores, expected):
        assert DecisionConfig().penalty(stores) == expected

    def test_a_configured_penalty(self):
        assert DecisionConfig(penalty_agorot=250).penalty(3) == 500
        assert DecisionConfig(penalty_agorot=0).penalty(3) == 0

    def test_a_basket_has_at_least_one_store(self):
        with pytest.raises(ValueError, match="at least one store"):
            DecisionConfig().penalty(0)


class TestTravelCost:
    @pytest.mark.parametrize(
        ("per_km", "km", "expected"),
        [
            (100.0, 1.234, 123),  # 123.4
            (100.0, 1.236, 124),  # 123.6
            (100.0, 0.004, 0),
            (100.0, 0.005, 1),  # a half rounds up
            (37.5, 2.0, 75),
            (150.0, 3.333, 500),  # 499.95
            (0.0, 12.0, 0),
        ],
    )
    def test_the_cost_is_rounded_once_to_whole_agorot(self, per_km, km, expected):
        cost = DecisionConfig(travel_agorot_per_km=per_km).travel_cost(km)

        assert cost == expected
        assert type(cost) is int

    def test_rounding_happens_at_the_end_not_per_leg(self):
        # Three legs of 0.004 km cost 0.4 agorot each: 1 agora together, 0 if rounded per leg.
        config = DecisionConfig()

        assert config.travel_cost(3 * 0.004) == 1
        assert sum(config.travel_cost(0.004) for _ in range(3)) == 0

    @pytest.mark.parametrize("km", [-1.0, float("nan"), float("inf")])
    def test_an_impossible_distance_is_rejected(self, km):
        with pytest.raises(ValueError, match="finite and not negative"):
            DecisionConfig().travel_cost(km)


class TestValidation:
    @pytest.mark.parametrize(
        ("fields", "message"),
        [
            ({"penalty_agorot": -1}, "cannot be negative"),
            ({"penalty_agorot": 10.5}, "whole number of agorot"),
            ({"penalty_agorot": True}, "whole number of agorot"),
            ({"travel_agorot_per_km": -1.0}, "finite and not negative"),
            ({"travel_agorot_per_km": float("nan")}, "finite and not negative"),
            ({"travel_agorot_per_km": float("inf")}, "finite and not negative"),
            ({"travel_agorot_per_km": "100"}, "must be a number"),
            ({"penalty_agorot": 10**9}, "cannot exceed"),
            ({"travel_agorot_per_km": 1e30}, "cannot exceed"),
        ],
    )
    def test_bad_values_are_rejected(self, fields, message):
        with pytest.raises(ConfigError, match=message):
            DecisionConfig(**fields)

    def test_the_configuration_is_read_only(self):
        with pytest.raises(AttributeError):
            DecisionConfig().penalty_agorot = 5


class TestFromEnvironment:
    def test_both_values_are_read(self):
        config = DecisionConfig.from_env({PENALTY_VARIABLE: "750", TRAVEL_VARIABLE: "62.5"})

        assert config == DecisionConfig(penalty_agorot=750, travel_agorot_per_km=62.5)

    def test_one_value_can_be_set_alone(self):
        assert DecisionConfig.from_env({PENALTY_VARIABLE: "0"}) == DecisionConfig(penalty_agorot=0)
        assert DecisionConfig.from_env({TRAVEL_VARIABLE: " 80 "}).travel_agorot_per_km == 80

    def test_the_process_environment_is_the_default_source(self, monkeypatch):
        monkeypatch.setenv(PENALTY_VARIABLE, "1500")
        monkeypatch.delenv(TRAVEL_VARIABLE, raising=False)

        assert DecisionConfig.from_env() == DecisionConfig(penalty_agorot=1500)

    @pytest.mark.parametrize(
        ("env", "message"),
        [
            ({PENALTY_VARIABLE: "ten"}, "STORE_PENALTY_AGOROT must be a whole number"),
            ({PENALTY_VARIABLE: "10.5"}, "STORE_PENALTY_AGOROT must be a whole number"),
            ({PENALTY_VARIABLE: "-1"}, "cannot be negative"),
            ({TRAVEL_VARIABLE: "fast"}, "TRAVEL_AGOROT_PER_KM must be a number"),
            ({TRAVEL_VARIABLE: "-3"}, "finite and not negative"),
            ({TRAVEL_VARIABLE: "nan"}, "finite and not negative"),
        ],
    )
    def test_a_bad_value_fails_loudly_and_names_the_variable(self, env, message):
        with pytest.raises(ConfigError, match=message):
            DecisionConfig.from_env(env)


class TestInDecide:
    def test_totals_use_the_configured_penalty_and_travel_cost(self):
        config = DecisionConfig(penalty_agorot=250, travel_agorot_per_km=40.0)

        table = decide(COST, HOME, [A, B, C], config=config).subsets

        assert table.size.tolist() == [1, 1, 1, 2, 2, 2, 3]
        for n in range(len(table)):
            penalty = config.penalty(int(table.size[n]))
            travel = config.travel_cost(float(table.travel_km[n]))
            assert table.travel_agorot[n] == travel
            assert table.total_agorot[n] == table.basket_agorot[n] + penalty + travel

    def test_a_higher_penalty_only_raises_multi_store_totals(self):
        low = decide(COST, HOME, [A, B, C], config=DecisionConfig(penalty_agorot=0)).subsets
        high = decide(COST, HOME, [A, B, C], config=DecisionConfig(penalty_agorot=700)).subsets

        assert (high.total_agorot - low.total_agorot).tolist() == [0, 0, 0, 700, 700, 700, 1400]

    def test_the_engine_rounds_the_same_way_as_the_configuration(self):
        config = DecisionConfig(travel_agorot_per_km=123.4)

        table = decide(COST, HOME, [A, B, C], config=config).subsets

        assert table.travel_agorot.tolist() == [
            config.travel_cost(float(km)) for km in table.travel_km
        ]
