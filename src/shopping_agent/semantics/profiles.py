"""Category default profiles: the schema and the loader (architecture section 3.1).

The profiles are hand-written in ``profiles.yaml``, next to this module. This
module defines what a valid file looks like and loads it. A file that breaks a
rule does not load: the error names the profile and the rule.
"""

import re
from collections.abc import Mapping
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StringConstraints,
    ValidationError,
    model_validator,
)

from shopping_agent.contracts import Quantity

PROFILES_FILE = "profiles.yaml"

Word = Annotated[str, StringConstraints(min_length=1, max_length=40, pattern=r"^\S(.*\S)?$")]
AttributeName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$", max_length=40)]
AttributeValue = StrictBool | StrictInt | StrictFloat | Word
_PROFILE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class ProfileError(ValueError):
    """The profile file is missing, unreadable or breaks a rule."""


def _repeated(values: list[str]) -> list[str]:
    return sorted({value for value in values if values.count(value) > 1})


class CategoryProfile(BaseModel):
    """One category: how it is recognized and what an unspecified request means.

    Every field except ``weighed`` and ``typical_unit_weight`` must be written
    out in the file, even when empty, so nothing is left out by accident.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    keywords: tuple[Word, ...] = Field(min_length=1)
    exclude_keywords: tuple[Word, ...]
    defaults: Mapping[AttributeName, AttributeValue]
    free: tuple[AttributeName, ...]
    typical_pack: Quantity
    weighed: StrictBool = False
    typical_unit_weight: Quantity | None = None

    @model_validator(mode="after")
    def _rules(self) -> Self:
        for field in ("keywords", "exclude_keywords", "free"):
            if repeated := _repeated(list(getattr(self, field))):
                raise ValueError(f"{field} repeat: {', '.join(repeated)}")
        if both := sorted(set(self.keywords) & set(self.exclude_keywords)):
            raise ValueError(f"a keyword is also an exclude keyword: {', '.join(both)}")
        if both := sorted(set(self.defaults) & set(self.free)):
            raise ValueError(f"an attribute is both a default and free: {', '.join(both)}")
        if self.weighed:
            if self.typical_unit_weight is None:
                raise ValueError("a weighed category needs a typical_unit_weight")
            if self.typical_unit_weight.unit != "g" or self.typical_pack.unit != "g":
                raise ValueError("a weighed category states its pack and unit weight in g")
        elif self.typical_unit_weight is not None:
            raise ValueError("typical_unit_weight belongs only to a weighed category")
        return self


def parse_profiles(raw: object) -> dict[str, CategoryProfile]:
    """Check the content of a profile file and return the profiles by name."""
    if not isinstance(raw, dict) or not raw:
        raise ProfileError("the profile file must be a mapping of category name to profile")
    profiles: dict[str, CategoryProfile] = {}
    for name, body in raw.items():
        if not isinstance(name, str) or not _PROFILE_NAME.fullmatch(name):
            raise ProfileError(f"profile name {name!r} must be lower-case snake_case")
        try:
            profiles[name] = CategoryProfile.model_validate(body)
        except ValidationError as error:
            raise ProfileError(f"profile {name!r} is not valid:\n{error}") from error

    owner: dict[str, str] = {}
    for name, profile in profiles.items():
        for keyword in profile.keywords:
            if keyword in owner:
                raise ProfileError(
                    f"keyword {keyword!r} belongs to both {owner[keyword]!r} and {name!r}"
                )
            owner[keyword] = name
    return profiles


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe YAML loader that refuses a key written twice in one mapping."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[object, object]:
        keys = [self.construct_object(key, deep=True) for key, _ in node.value]
        if repeated := sorted({str(key) for key in keys if keys.count(key) > 1}):
            raise ProfileError(f"repeated key in the profile file: {', '.join(repeated)}")
        return super().construct_mapping(node, deep)


def read_profiles(path: Path) -> dict[str, CategoryProfile]:
    """Load and check a profile file."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise ProfileError(f"cannot read the profile file {path}: {error}") from error
    try:
        raw = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ProfileError(f"the profile file {path} is not valid YAML: {error}") from error
    return parse_profiles(raw)


@cache
def load_profiles() -> Mapping[str, CategoryProfile]:
    """The profiles shipped with the package, loaded once."""
    with resources.as_file(resources.files(__package__) / PROFILES_FILE) as path:
        return read_profiles(path)
