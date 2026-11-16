"""The ``POST /basket`` request and its two responses (architecture section 5.1).

The request carries either the list as text or the parsed items the client got
back from a clarify response, with the user's choices. The response is either a
basket result or a clarify response; ``kind`` tells them apart.
"""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from shopping_agent.contracts.base import (
    DEFAULT_RADIUS_KM,
    MAX_ITEMS,
    MAX_STORES_PER_BASKET,
    Contract,
    DatasetVersion,
    ItemIndex,
    Latitude,
    Location,
    Longitude,
    RadiusKm,
    require_unique,
)
from shopping_agent.contracts.nodes import (
    ClarifyChoice,
    ClarifyQuestion,
    Explanation,
    ListText,
    Notice,
    Options,
    ParsedItems,
    Recommended,
    StoresInRange,
    check_choices,
    check_options,
)


class BasketRequest(Contract):
    """Exactly one of ``list_text`` and ``parsed_items``, plus where the user is."""

    list_text: ListText | None = None
    parsed_items: ParsedItems | None = None
    choices: tuple[ClarifyChoice, ...] = Field(default=(), max_length=MAX_ITEMS)
    lat: Latitude
    lon: Longitude
    radius_km: RadiusKm = DEFAULT_RADIUS_KM

    @property
    def location(self) -> Location:
        return Location(lat=self.lat, lon=self.lon)

    @model_validator(mode="after")
    def _one_form_of_the_list(self) -> Self:
        if (self.list_text is None) == (self.parsed_items is None):
            raise ValueError("exactly one of list_text and parsed_items must be given")
        if self.parsed_items is None:
            if self.choices:
                raise ValueError("clarify choices come only with parsed_items")
        else:
            check_choices(self.choices, len(self.parsed_items))
        return self


class ClarifyResponse(Contract):
    """The parsed items and every question of the list, shown on one screen."""

    kind: Literal["clarify"] = "clarify"
    dataset_version: DatasetVersion
    parsed_items: ParsedItems
    simplified_parsing: bool = Field(default=False, strict=True)
    questions: tuple[ClarifyQuestion, ...] = Field(min_length=1, max_length=MAX_ITEMS)

    @model_validator(mode="after")
    def _questions_refer_to_items(self) -> Self:
        require_unique((question.item_index for question in self.questions), "questions for item")
        for question in self.questions:
            if question.item_index >= len(self.parsed_items):
                raise ValueError(
                    f"a question refers to item {question.item_index}, not in the list"
                )
            if question.item != self.parsed_items[question.item_index]:
                raise ValueError(f"question {question.item_index} does not carry its list item")
        return self


class BasketResult(Contract):
    """Up to three basket options, one per store count, with their explanation.

    ``unmatched`` items could not be searched and appear in no option. An item
    that is not listed in an option's stores is in that option's ``unfulfilled``.
    Both are normal fields, not errors (section 5.6).
    """

    kind: Literal["basket"] = "basket"
    dataset_version: DatasetVersion
    parsed_items: ParsedItems
    simplified_parsing: bool = Field(default=False, strict=True)
    stores: StoresInRange = ()
    options: Options = ()
    recommended: Recommended | None = None
    unmatched: tuple[ItemIndex, ...] = Field(default=(), max_length=MAX_ITEMS)
    summary: Explanation
    option_explanations: tuple[Explanation, ...] = Field(
        default=(), max_length=MAX_STORES_PER_BASKET
    )
    notice: Notice | None = None

    @model_validator(mode="after")
    def _result_is_consistent(self) -> Self:
        check_options(self.options, self.recommended)
        if len(self.option_explanations) != len(self.options):
            raise ValueError("there must be one explanation per option")
        require_unique(self.unmatched, "unmatched items")
        require_unique((store.store_id for store in self.stores), "stores in range")

        all_items = set(range(len(self.parsed_items)))
        if not set(self.unmatched) <= all_items:
            raise ValueError("unmatched refers to an item that is not in the list")
        searched = all_items - set(self.unmatched)
        known_stores = {store.store_id for store in self.stores}
        for option in self.options:
            if option.item_indices != searched:
                raise ValueError("an option must account for every matched item and no other")
            if not set(option.store_ids) <= known_stores:
                raise ValueError("an option uses a store that is not in range")

        if (self.notice == "no_stores_in_range") != (not self.stores):
            raise ValueError("notice is no_stores_in_range exactly when no store is in range")
        if self.notice is not None and self.options:
            raise ValueError("a result with a notice has no options")
        return self


BasketResponse = Annotated[BasketResult | ClarifyResponse, Field(discriminator="kind")]
