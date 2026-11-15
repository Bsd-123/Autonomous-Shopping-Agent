"""The parsed shopping-list item (architecture section 5.2).

Parse produces a list of these from the user's text, and the client sends
them back unchanged, with any clarify choices, when it resubmits.
"""

from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

Unit = Literal["ml", "g", "unit"]
AttributeValue = StrictBool | StrictInt | StrictFloat | StrictStr


class Quantity(BaseModel):
    """An amount in one of the base units used across the pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float = Field(gt=0, allow_inf_nan=False)
    unit: Unit


class Attribute(BaseModel):
    """A product attribute the user stated, such as brand or fat percentage.

    ``strict`` says whether a product must match the value. Every attribute is
    strict in Phase 1; the flag exists so soft preferences need no schema change.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: AttributeValue
    strict: bool = True


class ParsedItem(BaseModel):
    """One item of the shopping list.

    The three request forms map to the fields as follows:

    * a size without a count ("3 L milk") sets ``total_quantity``;
    * a size with a count, or an explicit container ("milk 1 L x2",
      "a 3-liter carton"), sets ``pack_size`` and ``count``;
    * a count only ("milk x2", "6 tomatoes") sets ``count``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    item: str = Field(min_length=1)
    count: int = Field(default=1, ge=1)
    pack_size: Quantity | None = None
    total_quantity: Quantity | None = None
    attributes: dict[str, Attribute] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _at_most_one_size(self) -> Self:
        if self.pack_size is not None and self.total_quantity is not None:
            raise ValueError("at most one of pack_size and total_quantity can be set")
        return self
