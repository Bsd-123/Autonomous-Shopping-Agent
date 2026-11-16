"""Tests for store geocoding. No test touches the network or waits."""

import json
import logging
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit

import polars as pl
import pytest

from shopping_agent.ingestion.adapters import ADAPTERS, STORE_SCHEMA
from shopping_agent.ingestion.geocoding import (
    NOMINATIM_URL,
    REPORT_SCHEMA,
    Coordinates,
    GeocodeError,
    NominatimGeocoder,
    address_query,
    geocode_report,
    geocode_stores,
    read_overrides,
    with_coordinates,
)
from support import RAW

REPO = Path(__file__).parents[2]
BNEI_BRAK = "6100"
KISHON = Coordinates(32.0907, 34.8266)


class FakeGeocoder:
    """Answers from a table and records what it was asked."""

    def __init__(self, answers: dict[str, Coordinates | None | Exception]) -> None:
        self.answers = answers
        self.asked: list[str] = []

    def __call__(self, query: str) -> Coordinates | None:
        self.asked.append(query)
        answer = self.answers.get(query)
        if isinstance(answer, Exception):
            raise answer
        return answer


def store(store_id: str, address: str | None, city: str | None = BNEI_BRAK) -> dict:
    return {
        "chain": store_id.split(":")[0],
        "store_id": store_id,
        "name": "סניף",
        "address": address,
        "city": city,
    }


def frame(*stores: dict) -> pl.DataFrame:
    schema = {name: STORE_SCHEMA[name] for name in ("chain", "store_id", "name", "address", "city")}
    return pl.DataFrame(list(stores), schema=schema)


def result_of(geocoded: pl.DataFrame, store_id: str) -> tuple:
    row = geocoded.filter(store_id=store_id).row(0, named=True)
    return row["lat"], row["lon"], row["geocode_status"]


QUERY = "הקישון 11, בני ברק, ישראל"
OSHER = store("osher_ad:11", "הקישון 11")


class TestQuery:
    def test_the_locality_code_becomes_the_city_name(self):
        assert address_query("הקישון 11", BNEI_BRAK) == QUERY

    def test_an_unknown_city_is_sent_as_written(self):
        assert address_query(" הרצל 5 ", "רמת גן") == "הרצל 5, רמת גן, ישראל"
        assert address_query("הרצל 5", None) == "הרצל 5, ישראל"

    @pytest.mark.parametrize("address", [None, "", "   "])
    def test_no_address_gives_no_query(self, address):
        assert address_query(address, BNEI_BRAK) is None


class TestGeocodeStores:
    def test_a_new_store_is_geocoded(self):
        geocoder = FakeGeocoder({QUERY: KISHON})

        geocoded = geocode_stores(frame(OSHER), geocoder)

        assert geocoder.asked == [QUERY]
        assert result_of(geocoded, "osher_ad:11") == (*KISHON, "geocoded")
        assert geocoded.columns == [*frame(OSHER).columns, "lat", "lon", "geocode_status"]

    def test_an_unchanged_store_reuses_its_cached_coordinates(self):
        first = geocode_stores(frame(OSHER), FakeGeocoder({QUERY: KISHON}))
        geocoder = FakeGeocoder({})

        second = geocode_stores(frame(OSHER), geocoder, previous=first)

        assert geocoder.asked == []
        assert result_of(second, "osher_ad:11") == (*KISHON, "geocoded")

    def test_only_new_or_changed_stores_are_sent(self):
        unchanged = store("maayan_2000:38", "רבי עקיבא 94")
        previous = geocode_stores(
            frame(OSHER, unchanged),
            FakeGeocoder(
                {QUERY: KISHON, "רבי עקיבא 94, בני ברק, ישראל": Coordinates(32.08, 34.83)}
            ),
        )
        moved = store("osher_ad:11", "הקישון 13")
        new = store("yesh_hesed:611", "השומר 10")
        geocoder = FakeGeocoder({"הקישון 13, בני ברק, ישראל": Coordinates(32.091, 34.827)})

        geocoded = geocode_stores(frame(moved, unchanged, new), geocoder, previous=previous)

        assert geocoder.asked == ["הקישון 13, בני ברק, ישראל", "השומר 10, בני ברק, ישראל"]
        assert result_of(geocoded, "osher_ad:11") == (32.091, 34.827, "geocoded")
        assert result_of(geocoded, "maayan_2000:38") == (32.08, 34.83, "geocoded")
        assert result_of(geocoded, "yesh_hesed:611") == (None, None, "not_found")

    def test_a_changed_city_counts_as_a_change(self):
        previous = geocode_stores(frame(OSHER), FakeGeocoder({QUERY: KISHON}))
        geocoder = FakeGeocoder({})

        geocode_stores(
            frame(store("osher_ad:11", "הקישון 11", "רמת גן")), geocoder, previous=previous
        )

        assert geocoder.asked == ["הקישון 11, רמת גן, ישראל"]

    def test_an_address_the_geocoder_does_not_know_is_not_asked_again(self, caplog):
        with caplog.at_level(logging.WARNING):
            first = geocode_stores(frame(OSHER), FakeGeocoder({QUERY: None}))
        geocoder = FakeGeocoder({QUERY: KISHON})

        second = geocode_stores(frame(OSHER), geocoder, previous=first)

        assert result_of(first, "osher_ad:11") == (None, None, "not_found")
        assert QUERY in caplog.text and "osher_ad:11" in caplog.text
        assert geocoder.asked == []
        assert result_of(second, "osher_ad:11") == (None, None, "not_found")

    def test_a_failed_request_is_logged_with_the_address_and_retried_next_run(self, caplog):
        failing = FakeGeocoder({QUERY: GeocodeError("timed out")})

        with caplog.at_level(logging.WARNING):
            first = geocode_stores(frame(OSHER), failing)
        geocoder = FakeGeocoder({QUERY: KISHON})
        second = geocode_stores(frame(OSHER), geocoder, previous=first)

        assert result_of(first, "osher_ad:11") == (None, None, "error")
        assert QUERY in caplog.text and "timed out" in caplog.text
        assert geocoder.asked == [QUERY]
        assert result_of(second, "osher_ad:11") == (*KISHON, "geocoded")

    def test_one_failure_does_not_stop_the_other_stores(self):
        other = store("yesh_hesed:611", "השומר 10")
        geocoder = FakeGeocoder(
            {QUERY: GeocodeError("down"), "השומר 10, בני ברק, ישראל": Coordinates(32.09, 34.83)}
        )

        geocoded = geocode_stores(frame(OSHER, other), geocoder)

        assert geocoded["geocode_status"].to_list() == ["error", "geocoded"]

    def test_a_store_without_an_address_is_not_sent(self, caplog):
        geocoder = FakeGeocoder({})

        with caplog.at_level(logging.WARNING):
            geocoded = geocode_stores(frame(store("osher_ad:11", None)), geocoder)

        assert geocoder.asked == []
        assert result_of(geocoded, "osher_ad:11") == (None, None, "not_found")
        assert "osher_ad:11 has no address" in caplog.text


class TestOverrides:
    HAND_CHECKED = Coordinates(32.0911, 34.8259)

    def test_an_override_takes_precedence_over_the_geocoder(self):
        geocoder = FakeGeocoder({QUERY: KISHON})

        geocoded = geocode_stores(
            frame(OSHER), geocoder, overrides={"osher_ad:11": self.HAND_CHECKED}
        )

        assert geocoder.asked == []
        assert result_of(geocoded, "osher_ad:11") == (*self.HAND_CHECKED, "override")

    def test_an_override_takes_precedence_over_the_cache(self):
        previous = geocode_stores(frame(OSHER), FakeGeocoder({QUERY: KISHON}))

        geocoded = geocode_stores(
            frame(OSHER),
            FakeGeocoder({}),
            previous=previous,
            overrides={"osher_ad:11": self.HAND_CHECKED},
        )

        assert result_of(geocoded, "osher_ad:11") == (*self.HAND_CHECKED, "override")

    def test_a_removed_override_is_geocoded_again(self):
        overridden = geocode_stores(
            frame(OSHER), FakeGeocoder({}), overrides={"osher_ad:11": self.HAND_CHECKED}
        )
        geocoder = FakeGeocoder({QUERY: KISHON})

        geocoded = geocode_stores(frame(OSHER), geocoder, previous=overridden)

        assert geocoder.asked == [QUERY]
        assert result_of(geocoded, "osher_ad:11") == (*KISHON, "geocoded")

    def test_an_override_supplies_a_store_the_geocoder_cannot_find(self):
        geocoded = geocode_stores(
            frame(store("osher_ad:11", None)),
            FakeGeocoder({}),
            overrides={"osher_ad:11": self.HAND_CHECKED},
        )

        assert result_of(geocoded, "osher_ad:11") == (*self.HAND_CHECKED, "override")

    def test_the_file_is_read(self, tmp_path):
        path = tmp_path / "geocode_overrides.csv"
        path.write_text(
            "store_id,lat,lon,note\nosher_ad:11,32.0911,34.8259,checked on the map\n"
            " yesh_hesed:611 ,32.09,34.83,\n",
            encoding="utf-8",
        )

        assert read_overrides(path) == {
            "osher_ad:11": self.HAND_CHECKED,
            "yesh_hesed:611": Coordinates(32.09, 34.83),
        }

    def test_a_missing_file_means_no_overrides(self, tmp_path):
        assert read_overrides(tmp_path / "geocode_overrides.csv") == {}

    def test_the_file_in_the_repository_is_valid(self):
        assert read_overrides(REPO / "config" / "geocode_overrides.csv") == {}

    @pytest.mark.parametrize(
        ("content", "message"),
        [
            ("store_id,lat\nosher_ad:11,32.09\n", "missing columns: lon"),
            ("store_id,lat,lon\nosher_ad:11,north,34.8\n", "line 2: lat and lon must be numbers"),
            ("store_id,lat,lon\nosher_ad:11,,34.8\n", "line 2: lat and lon must be numbers"),
            ("store_id,lat,lon\nosher_ad:11,132.09,34.8\n", "line 2: not a store with valid"),
            ("store_id,lat,lon\n,32.09,34.8\n", "line 2: not a store with valid"),
            (
                "store_id,lat,lon\nosher_ad:11,32.09,34.8\nosher_ad:11,32.1,34.8\n",
                "line 3: osher_ad:11 appears twice",
            ),
        ],
    )
    def test_a_bad_file_fails_loudly(self, tmp_path, content, message):
        path = tmp_path / "geocode_overrides.csv"
        path.write_text(content, encoding="utf-8")

        with pytest.raises(GeocodeError, match=message):
            read_overrides(path)


class TestReportAndExclusion:
    @pytest.fixture
    def geocoded(self) -> pl.DataFrame:
        stores = frame(
            OSHER,
            store("yesh_hesed:611", "השומר 10"),
            store("yesh_hesed:295", "שלמה המלך 12"),
            store("yesh_hesed:219", "עזרא 3"),
            store("maayan_2000:38", "רבי עקיבא 94"),
        )
        geocoder = FakeGeocoder(
            {
                QUERY: KISHON,
                "השומר 10, בני ברק, ישראל": Coordinates(32.09, 34.83),
                "שלמה המלך 12, בני ברק, ישראל": None,
                "עזרא 3, בני ברק, ישראל": GeocodeError("down"),
            }
        )
        return geocode_stores(
            stores, geocoder, overrides={"maayan_2000:38": Coordinates(32.08, 34.83)}
        )

    def test_stores_without_coordinates_are_counted_per_chain(self, geocoded):
        report = geocode_report(geocoded)

        assert report.schema == REPORT_SCHEMA
        assert report.rows() == [
            ("maayan_2000", 1, 1, 0, 0, 0, 0),
            ("osher_ad", 1, 0, 1, 0, 0, 0),
            ("yesh_hesed", 3, 0, 1, 1, 1, 2),
        ]

    def test_stores_without_coordinates_are_excluded(self, geocoded):
        kept = with_coordinates(geocoded)

        assert kept["store_id"].to_list() == ["osher_ad:11", "yesh_hesed:611", "maayan_2000:38"]
        assert kept.select(pl.col("lat", "lon").null_count()).row(0) == (0, 0)
        excluded = geocoded.height - kept.height
        assert excluded == geocode_report(geocoded)["without_coordinates"].sum() == 2

    def test_the_real_store_files_can_be_geocoded(self):
        stores = pl.concat([adapter.read_stores(RAW) for adapter in ADAPTERS])
        bnei_brak = stores.filter(city=BNEI_BRAK)
        geocoder = FakeGeocoder({})

        geocoded = geocode_stores(bnei_brak, geocoder)

        assert geocoded.height == len(geocoder.asked) == 7
        assert all(query.endswith(", בני ברק, ישראל") for query in geocoder.asked)
        assert geocoded.columns == [*STORE_SCHEMA.names(), "lat", "lon", "geocode_status"]


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class TestNominatim:
    def make(self, body: bytes | Exception = b"[]", **settings):
        clock = FakeClock()
        requests: list[tuple[str, dict]] = []

        def fetch(url: str, headers) -> bytes:
            requests.append((url, dict(headers)))
            if isinstance(body, Exception):
                raise body
            return body

        geocoder = NominatimGeocoder(
            settings.pop("user_agent", "shopping-agent-tests"),
            fetch=fetch,
            clock=clock,
            sleep=clock.sleep,
            **settings,
        )
        return geocoder, clock, requests

    def test_the_request_carries_the_user_agent_and_the_query(self):
        body = json.dumps([{"lat": "32.0907", "lon": "34.8266"}]).encode()
        geocoder, _, requests = self.make(body, user_agent=" my-app/1.0 ")

        found = geocoder(QUERY)

        url, headers = requests[0]
        assert found == KISHON
        assert headers == {"User-Agent": "my-app/1.0"}
        assert url.startswith(NOMINATIM_URL + "?")
        assert parse_qs(urlsplit(url).query) == {
            "q": [QUERY],
            "format": ["jsonv2"],
            "limit": ["1"],
            "countrycodes": ["il"],
        }

    @pytest.mark.parametrize("user_agent", ["", "   "])
    def test_a_user_agent_is_required(self, user_agent):
        with pytest.raises(ValueError, match="identifying User-Agent"):
            NominatimGeocoder(user_agent)

    def test_requests_are_never_closer_than_one_second(self):
        geocoder, clock, requests = self.make()

        geocoder("a")
        clock.now += 0.25
        geocoder("b")
        geocoder("c")

        assert len(requests) == 3
        assert clock.slept == [0.75, 1.0]

    def test_no_waiting_when_a_second_has_already_passed(self):
        geocoder, clock, _ = self.make()

        geocoder("a")
        clock.now += 1.5
        geocoder("b")

        assert clock.slept == []

    def test_a_failed_request_still_counts_towards_the_rate_limit(self):
        geocoder, clock, _ = self.make(URLError("no route"))

        for _ in range(2):
            with pytest.raises(GeocodeError):
                geocoder("a")

        assert clock.slept == [1.0]

    def test_the_interval_cannot_be_set_below_one_second(self):
        with pytest.raises(ValueError, match="at least 1 second apart"):
            NominatimGeocoder("shopping-agent-tests", min_interval=0.5)
        assert self.make(min_interval=2.0)[0] is not None

    def test_an_unknown_address_is_none(self):
        assert self.make(b"[]")[0](QUERY) is None

    @pytest.mark.parametrize(
        "body",
        [
            URLError("no route"),
            TimeoutError("timed out"),
            b"<html>blocked</html>",
            b'{"error": "rate limited"}',
            b'[{"lat": "north", "lon": "34.8"}]',
            b'[{"display_name": "x"}]',
            b'[{"lat": "132.0", "lon": "34.8"}]',
        ],
    )
    def test_a_failed_or_unreadable_answer_raises(self, body):
        with pytest.raises(GeocodeError):
            self.make(body)[0](QUERY)

    def test_it_works_as_the_geocoder_of_geocode_stores(self):
        body = json.dumps([{"lat": "32.0907", "lon": "34.8266"}]).encode()
        geocoder, clock, requests = self.make(body)
        stores = frame(OSHER, store("yesh_hesed:611", "השומר 10"))

        geocoded = geocode_stores(stores, geocoder)

        assert geocoded["geocode_status"].to_list() == ["geocoded", "geocoded"]
        assert len(requests) == 2
        assert clock.slept == [1.0]
