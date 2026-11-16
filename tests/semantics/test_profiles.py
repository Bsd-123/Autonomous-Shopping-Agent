"""Schema check of the category profile file, and tests of the check itself."""

import copy

import pytest

from shopping_agent.contracts import Quantity
from shopping_agent.semantics.profiles import (
    CategoryProfile,
    ProfileError,
    load_profiles,
    parse_profiles,
    read_profiles,
)

WEIGHED = {
    "tomatoes",
    "cucumbers",
    "onions",
    "potatoes",
    "carrots",
    "peppers",
    "bananas",
    "apples",
    "chicken_breast",
}
MILK = {
    "keywords": ["חלב"],
    "exclude_keywords": ["סויה", "קוקוס"],
    "defaults": {"animal": "cow", "flavor": "plain", "storage": "fresh"},
    "free": ["fat_pct", "brand"],
    "typical_pack": {"value": 1000, "unit": "ml"},
}
TOMATOES = {
    "keywords": ["עגבניה", "עגבניות"],
    "exclude_keywords": ["רסק", "מיובשות"],
    "defaults": {},
    "free": [],
    "typical_pack": {"value": 1000, "unit": "g"},
    "weighed": True,
    "typical_unit_weight": {"value": 120, "unit": "g"},
}


def changed(profile: dict, **changes) -> dict:
    result = copy.deepcopy(profile) | changes
    return {key: value for key, value in result.items() if value is not ...}


class TestTheShippedFile:
    """The schema check that runs on every change to ``profiles.yaml``."""

    def test_it_holds_35_profiles(self):
        assert len(load_profiles()) == 35

    def test_every_profile_has_all_its_parts(self):
        for name, profile in load_profiles().items():
            assert isinstance(profile, CategoryProfile), name
            assert profile.keywords, name
            assert profile.exclude_keywords, name
            assert profile.defaults, name
            assert profile.free, name
            assert profile.typical_pack.value > 0, name

    def test_weighed_categories_have_a_typical_unit_weight(self):
        profiles = load_profiles()

        assert {name for name, profile in profiles.items() if profile.weighed} == WEIGHED
        for name in WEIGHED:
            assert profiles[name].typical_unit_weight.unit == "g", name
            assert 20 <= profiles[name].typical_unit_weight.value <= 1000, name
            assert profiles[name].typical_pack == Quantity(value=1000, unit="g"), name
        for name in profiles.keys() - WEIGHED:
            assert profiles[name].typical_unit_weight is None, name

    def test_the_architecture_examples_are_kept(self):
        milk, tomatoes = load_profiles()["milk"], load_profiles()["tomatoes"]

        assert milk.keywords == ("חלב",)
        assert {"סויה", "קוקוס", "שוקולד", "אבקת", "עיזים"} <= set(milk.exclude_keywords)
        assert milk.defaults == {"animal": "cow", "flavor": "plain", "storage": "fresh"}
        assert milk.free == ("fat_pct", "brand")
        assert milk.typical_pack == Quantity(value=1000, unit="ml")
        assert {"עגבניה", "עגבניות"} <= set(tomatoes.keywords)
        assert {"רסק", "מיובשות"} <= set(tomatoes.exclude_keywords)
        assert tomatoes.typical_unit_weight == Quantity(value=120, unit="g")

    def test_no_keyword_belongs_to_two_profiles(self):
        keywords = [keyword for profile in load_profiles().values() for keyword in profile.keywords]

        assert len(keywords) == len(set(keywords))

    def test_a_longer_keyword_is_shielded_from_the_shorter_one(self):
        # "שמן זית" must not also fall under "שמן".
        profiles = load_profiles()

        assert "שמן זית" in profiles["olive_oil"].keywords
        assert "זית" in profiles["oil"].exclude_keywords

    def test_it_is_loaded_once(self):
        assert load_profiles() is load_profiles()


class TestSchema:
    def test_valid_profiles(self):
        profiles = parse_profiles({"milk": MILK, "tomatoes": TOMATOES})

        assert list(profiles) == ["milk", "tomatoes"]
        assert profiles["milk"].weighed is False
        assert profiles["tomatoes"].typical_unit_weight == Quantity(value=120, unit="g")

    @pytest.mark.parametrize(
        "missing", ["keywords", "exclude_keywords", "defaults", "free", "typical_pack"]
    )
    def test_every_part_must_be_written_out(self, missing):
        with pytest.raises(ProfileError, match=f"'milk' is not valid(.|\n)*{missing}"):
            parse_profiles({"milk": changed(MILK, **{missing: ...})})

    @pytest.mark.parametrize(
        ("message", "profile"),
        [
            ("at least 1 item", changed(MILK, keywords=[])),
            ("keywords repeat: חלב", changed(MILK, keywords=["חלב", "חלב"])),
            ("exclude_keywords repeat: סויה", changed(MILK, exclude_keywords=["סויה", "סויה"])),
            ("free repeat: brand", changed(MILK, free=["brand", "brand"])),
            ("also an exclude keyword: חלב", changed(MILK, exclude_keywords=["חלב"])),
            ("both a default and free: flavor", changed(MILK, free=["flavor"])),
            ("Extra inputs", changed(MILK, typical_price=5)),
            ("String should match", changed(MILK, keywords=[" חלב"])),
            ("String should match", changed(MILK, free=["Fat Pct"])),
            ("String should match", changed(MILK, defaults={"Animal": "cow"})),
            ("'ml', 'g' or 'unit'", changed(MILK, typical_pack={"value": 1, "unit": "l"})),
            ("greater than 0", changed(MILK, typical_pack={"value": 0, "unit": "ml"})),
            ("valid boolean", changed(MILK, weighed="yes")),
            ("needs a typical_unit_weight", changed(TOMATOES, typical_unit_weight=...)),
            ("needs a typical_unit_weight", changed(MILK, weighed=True)),
            (
                "pack and unit weight in g",
                changed(TOMATOES, typical_unit_weight={"value": 1, "unit": "unit"}),
            ),
            (
                "pack and unit weight in g",
                changed(TOMATOES, typical_pack={"value": 1, "unit": "unit"}),
            ),
            (
                "only to a weighed category",
                changed(MILK, typical_unit_weight={"value": 120, "unit": "g"}),
            ),
        ],
    )
    def test_rule_violations(self, message, profile):
        with pytest.raises(ProfileError, match=message):
            parse_profiles({"milk": profile})

    def test_a_keyword_shared_by_two_profiles_is_rejected(self):
        with pytest.raises(ProfileError, match="'חלב' belongs to both 'milk' and 'soy_milk'"):
            parse_profiles({"milk": MILK, "soy_milk": changed(MILK, exclude_keywords=[])})

    @pytest.mark.parametrize("name", ["Milk", "soy milk", "1milk", "חלב", 5])
    def test_profile_names_are_snake_case(self, name):
        with pytest.raises(ProfileError, match="lower-case snake_case"):
            parse_profiles({name: MILK})

    @pytest.mark.parametrize("raw", [None, {}, [], "milk"])
    def test_the_file_must_be_a_mapping_of_profiles(self, raw):
        with pytest.raises(ProfileError, match="mapping of category name to profile"):
            parse_profiles(raw)

    def test_profiles_are_read_only(self):
        profile = parse_profiles({"milk": MILK})["milk"]

        with pytest.raises(Exception, match="frozen"):
            profile.weighed = True


class TestReadingAFile:
    VALID = (
        "milk:\n"
        "  keywords: [חלב]\n"
        "  exclude_keywords: [סויה]\n"
        "  defaults: {animal: cow}\n"
        "  free: [brand]\n"
        "  typical_pack: {value: 1000, unit: ml}\n"
    )

    def test_a_valid_file(self, tmp_path):
        path = tmp_path / "profiles.yaml"
        path.write_text(self.VALID, encoding="utf-8")

        assert read_profiles(path)["milk"].keywords == ("חלב",)

    def test_a_missing_file(self, tmp_path):
        with pytest.raises(ProfileError, match="cannot read the profile file"):
            read_profiles(tmp_path / "missing.yaml")

    def test_broken_yaml(self, tmp_path):
        path = tmp_path / "profiles.yaml"
        path.write_text("milk: [unclosed", encoding="utf-8")

        with pytest.raises(ProfileError, match="not valid YAML"):
            read_profiles(path)

    def test_a_profile_written_twice_is_rejected(self, tmp_path):
        path = tmp_path / "profiles.yaml"
        path.write_text(self.VALID + self.VALID, encoding="utf-8")

        with pytest.raises(ProfileError, match="repeated key in the profile file: milk"):
            read_profiles(path)

    def test_a_field_written_twice_is_rejected(self, tmp_path):
        path = tmp_path / "profiles.yaml"
        path.write_text(self.VALID + "  free: [fat_pct]\n", encoding="utf-8")

        with pytest.raises(ProfileError, match="repeated key in the profile file: free"):
            read_profiles(path)
