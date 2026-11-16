"""The parsed shopping-list item (architecture section 5.2).

Parse produces a list of these from the user's text, and the client sends
them back unchanged, with any clarify choices, when it resubmits.
"""

from collections.abc import Iterable, Iterator, Mapping
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    StrictBool,
    StrictInt,
    StringConstraints,
    model_validator,
)

# Generous limits: the client can send parsed items, so every size is bounded.
MAX_TEXT_LENGTH = 200
MAX_ATTRIBUTES = 20
MAX_COUNT = 1_000
MAX_QUANTITY = 10_000_000  # 10,000 L in ml, or 10,000 kg in g

Unit = Literal["ml", "g", "unit"]
Text = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_LENGTH)
]
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
AttributeValue = StrictBool | StrictInt | FiniteFloat | Text


class Quantity(BaseModel):
    """An amount in one of the base units used across the pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float = Field(gt=0, le=MAX_QUANTITY, allow_inf_nan=False, strict=True)
    unit: Unit


class Attribute(BaseModel):
    """A product attribute the user stated, such as brand or fat percentage.

    ``strict`` says whether a product must match the value. Every attribute is
    strict in Phase 1; the flag exists so soft preferences need no schema change.

    The value keeps its type, but equality follows Python: ``True`` equals ``1``
    and ``3`` equals ``3.0``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: AttributeValue
    strict: bool = Field(default=True, strict=True)


def _reject_repeated_names(raw: object) -> object:
    """Fail when two attribute names differ only by surrounding whitespace.

    Names are stripped on validation, so the second would silently replace the first.
    """
    if isinstance(raw, Mapping):
        names = [name.strip() for name in raw if isinstance(name, str)]
        if len(set(names)) != len(names):
            raise ValueError("attribute names repeat once surrounding whitespace is removed")
    return raw


class ReadOnlyAttributes(Mapping[str, Attribute]):
    """The attributes of an item. Unlike a dict inside a frozen model, it cannot be changed.

    It can be hashed, copied and pickled, so an item behaves like any other value.
    """

    __slots__ = ("_data",)

    def __init__(
        self, data: Iterable[tuple[str, Attribute]] | Mapping[str, Attribute] = ()
    ) -> None:
        self._data = dict(data)

    def __getitem__(self, key: str) -> Attribute:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __hash__(self) -> int:
        return hash(frozenset(self._data.items()))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"


class ParsedItem(BaseModel):
    """One item of the shopping list.

    The three request forms map to the fields as follows:

    * a size without a count ("3 L milk") sets ``total_quantity``;
    * a size with a count, or an explicit container ("milk 1 L x2",
      "a 3-liter carton"), sets ``pack_size`` and ``count``;
    * a count only ("milk x2", "6 tomatoes") sets ``count``.

    ``attributes`` is read-only: later steps add defaults by validating a new item,
    not with ``model_copy(update=...)``, which skips validation and the read-only wrapper.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    item: Text
    count: int = Field(default=1, ge=1, le=MAX_COUNT, strict=True)
    pack_size: Quantity | None = None
    total_quantity: Quantity | None = None
    attributes: Annotated[
        Mapping[Text, Attribute],
        BeforeValidator(_reject_repeated_names),
        Field(max_length=MAX_ATTRIBUTES),
        AfterValidator(ReadOnlyAttributes),
        PlainSerializer(dict, return_type=dict[str, Attribute]),
    ] = Field(default_factory=ReadOnlyAttributes)

    @model_validator(mode="after")
    def _at_most_one_size(self) -> Self:
        if self.pack_size is not None and self.total_quantity is not None:
            raise ValueError("at most one of pack_size and total_quantity can be set")
        return self
