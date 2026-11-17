"""Typed input and output of the six graph nodes (architecture section 5).

    Parse -> Search -> Clarify -> Compare -> Decide -> Explain

Each node takes one ``...Input`` model and returns one ``...Output`` model.
A model that breaks a structural or business rule cannot be built: validation
raises, and the graph's error handler reports it (section 5.6).

Items are referred to by ``item_index``, their position in the parsed list.
Money is whole agorot.
"""

from collections.abc import Sequence
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from shopping_agent.contracts.base import (
    MAX_CANDIDATES,
    MAX_EXPLANATION_LENGTH,
    MAX_ITEMS,
    MAX_LIST_TEXT_LENGTH,
    MAX_STORES_IN_RANGE,
    MAX_STORES_PER_BASKET,
    Agorot,
    Contract,
    DatasetVersion,
    ItemIndex,
    Kilometres,
    Label,
    Location,
    ProductKey,
    RadiusKm,
    StoreId,
    require_item_order,
    require_unique,
)
from shopping_agent.contracts.parsed_item import MAX_QUANTITY, ParsedItem

ListText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_LIST_TEXT_LENGTH)
]
ParsedItems = Annotated[tuple[ParsedItem, ...], Field(min_length=1, max_length=MAX_ITEMS)]
Explanation = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_EXPLANATION_LENGTH)
]

# "llm" is the normal path; "rules" is the fallback used when the LLM is
# unavailable, which the client labels "simplified parsing, please check".
Parser = Literal["llm", "rules"]
ItemStatus = Literal["matched", "needs_clarification", "unmatched"]
# The three Clarify rules of section 5.4.
ClarifyReason = Literal["no_candidates", "ambiguous", "overbuy"]
Notice = Literal["no_stores_in_range", "nothing_listed"]


class ClarifyChoice(Contract):
    """The product the user picked for one item on the clarify screen."""

    item_index: ItemIndex
    product_key: ProductKey


def check_choices(choices: Sequence[ClarifyChoice], item_count: int) -> None:
    """Raise unless every choice points at a different item of the list."""
    require_unique((choice.item_index for choice in choices), "clarify choices for an item")
    for choice in choices:
        if choice.item_index >= item_count:
            raise ValueError(f"clarify choice refers to item {choice.item_index}, not in the list")


# --- 1. Parse ---------------------------------------------------------------


class ParseInput(Contract):
    list_text: ListText


class ParseOutput(Contract):
    items: ParsedItems
    parser: Parser


# --- 2. Search --------------------------------------------------------------


class Candidate(Contract):
    """A product that can fulfil an item, and how much of it covers the request.

    A packaged product is bought as a whole number of ``packs``. A weighed
    product is bought by ``weight_g``; when the weight was derived from a count
    ("6 tomatoes") it is an estimate.
    """

    product_key: ProductKey
    name: Label
    score: float = Field(allow_inf_nan=False, strict=True)
    packs: int | None = Field(default=None, ge=1, strict=True)
    weight_g: float | None = Field(
        default=None, gt=0, le=MAX_QUANTITY, allow_inf_nan=False, strict=True
    )
    is_estimate: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def _one_way_to_buy(self) -> Self:
        if (self.packs is None) == (self.weight_g is None):
            raise ValueError("exactly one of packs and weight_g must be set")
        if self.is_estimate and self.weight_g is None:
            raise ValueError("only a weighed candidate can be an estimate")
        return self


Candidates = Annotated[tuple[Candidate, ...], Field(max_length=MAX_CANDIDATES)]


class SearchInput(Contract):
    items: ParsedItems
    choices: tuple[ClarifyChoice, ...] = ()

    @model_validator(mode="after")
    def _choices_refer_to_items(self) -> Self:
        check_choices(self.choices, len(self.items))
        return self


class ItemSearchResult(Contract):
    """What Search found for one item.

    * ``matched``: ``candidates`` is the item's equivalence class.
    * ``needs_clarification``: a Clarify rule fired; ``candidates`` are the
      options to show the user, and may be empty when nothing passed the filters.
    * ``unmatched``: Search failed for the item; the basket continues without it.
    """

    item_index: ItemIndex
    item: ParsedItem
    category: Label | None = None
    status: ItemStatus
    clarify_reason: ClarifyReason | None = None
    candidates: Candidates = ()

    @model_validator(mode="after")
    def _status_matches_content(self) -> Self:
        require_unique((candidate.product_key for candidate in self.candidates), "candidates")
        if (self.status == "needs_clarification") != (self.clarify_reason is not None):
            raise ValueError("clarify_reason is set exactly when the status is needs_clarification")
        if self.status == "matched" and not self.candidates:
            raise ValueError("a matched item needs at least one candidate")
        if self.status == "unmatched" and self.candidates:
            raise ValueError("an unmatched item has no candidates")
        if self.clarify_reason == "no_candidates" and self.candidates:
            raise ValueError("the no_candidates rule cannot come with candidates")
        if self.clarify_reason in ("ambiguous", "overbuy") and not self.candidates:
            raise ValueError(f"the {self.clarify_reason} rule needs candidates to show")
        return self


SearchResults = Annotated[tuple[ItemSearchResult, ...], Field(min_length=1, max_length=MAX_ITEMS)]


def _check_result_order(results: Sequence[ItemSearchResult]) -> None:
    require_item_order((result.item_index for result in results), "search results")


class ItemResults(Contract):
    """A contract that carries one search result per list item, in list order."""

    results: SearchResults

    @model_validator(mode="after")
    def _one_result_per_item(self) -> Self:
        _check_result_order(self.results)
        return self


class SearchOutput(ItemResults):
    """What Search found for every item of the list."""


# --- 3. Clarify -------------------------------------------------------------


class ClarifyInput(ItemResults):
    """The search results Clarify inspects."""


class ClarifyQuestion(Contract):
    """One ambiguous item and the products the user can choose between."""

    item_index: ItemIndex
    item: ParsedItem
    reason: ClarifyReason
    options: Candidates = ()

    @model_validator(mode="after")
    def _options_match_reason(self) -> Self:
        require_unique((option.product_key for option in self.options), "clarify options")
        if (self.reason == "no_candidates") != (not self.options):
            raise ValueError("options are empty exactly when the reason is no_candidates")
        return self


class ClarifyOutput(Contract):
    """All questions of the list together; none means the graph continues."""

    questions: tuple[ClarifyQuestion, ...] = Field(default=(), max_length=MAX_ITEMS)

    @model_validator(mode="after")
    def _one_question_per_item(self) -> Self:
        require_unique((question.item_index for question in self.questions), "questions for item")
        return self


# --- 4. Compare -------------------------------------------------------------


class CompareInput(ItemResults):
    location: Location
    radius_km: RadiusKm

    @model_validator(mode="after")
    def _ready_to_compare(self) -> Self:
        if any(result.status == "needs_clarification" for result in self.results):
            raise ValueError("an item still needs clarification and cannot be compared")
        return self


class StoreInRange(Contract):
    store_id: StoreId
    chain: Label
    name: Label
    location: Location
    distance_km: Kilometres


class ItemStoreCost(Contract):
    """A non-empty cell of the cost matrix: the cheapest way to cover an item at a store."""

    item_index: ItemIndex
    store_id: StoreId
    product_key: ProductKey
    cost_agorot: Annotated[int, Field(gt=0, strict=True)]
    is_estimate: bool = Field(default=False, strict=True)


StoresInRange = Annotated[tuple[StoreInRange, ...], Field(max_length=MAX_STORES_IN_RANGE)]
Costs = Annotated[tuple[ItemStoreCost, ...], Field(max_length=MAX_ITEMS * MAX_STORES_IN_RANGE)]


def _check_cost_matrix(stores: Sequence[StoreInRange], costs: Sequence[ItemStoreCost]) -> None:
    require_unique((store.store_id for store in stores), "stores in range")
    require_unique(((cost.item_index, cost.store_id) for cost in costs), "cost cells")
    known = {store.store_id for store in stores}
    for cost in costs:
        if cost.store_id not in known:
            raise ValueError(f"a cost refers to store {cost.store_id}, which is not in range")


class CompareOutput(Contract):
    """The stores in range and the sparse cost matrix; a missing cell means not listed."""

    stores: StoresInRange = ()
    costs: Costs = ()

    @model_validator(mode="after")
    def _costs_refer_to_stores(self) -> Self:
        _check_cost_matrix(self.stores, self.costs)
        return self


# --- 5. Decide --------------------------------------------------------------


class DecideInput(Contract):
    """The cost matrix, the items to cover and the user's location.

    ``item_indices`` are the items Search matched; unmatched items are left out.
    """

    item_indices: tuple[ItemIndex, ...] = Field(max_length=MAX_ITEMS)
    stores: StoresInRange = ()
    costs: Costs = ()
    location: Location

    @model_validator(mode="after")
    def _costs_refer_to_items_and_stores(self) -> Self:
        require_unique(self.item_indices, "item indices")
        _check_cost_matrix(self.stores, self.costs)
        unknown = {cost.item_index for cost in self.costs} - set(self.item_indices)
        if unknown:
            raise ValueError(f"costs refer to items that are not to be decided: {sorted(unknown)}")
        return self


class BasketLine(ItemStoreCost):
    """One item bought at one store: the cost cell the basket chose."""


class BasketOption(Contract):
    """The best basket for one store count (section 5.5).

    ``total_agorot`` is the basket price plus the extra-store penalty and the
    travel cost. ``savings_agorot`` is null when there is nothing to compare with.
    """

    store_ids: tuple[StoreId, ...] = Field(min_length=1, max_length=MAX_STORES_PER_BASKET)
    lines: tuple[BasketLine, ...] = Field(min_length=1, max_length=MAX_ITEMS)
    unfulfilled: tuple[ItemIndex, ...] = Field(default=(), max_length=MAX_ITEMS)
    basket_agorot: Agorot
    penalty_agorot: Agorot
    travel_agorot: Agorot
    total_agorot: Agorot
    savings_agorot: int | None = Field(default=None, strict=True)

    @property
    def fulfilled_count(self) -> int:
        return len(self.lines)

    @property
    def item_indices(self) -> frozenset[int]:
        """Every item the option accounts for, fulfilled or not."""
        return frozenset((*(line.item_index for line in self.lines), *self.unfulfilled))

    @model_validator(mode="after")
    def _basket_is_consistent(self) -> Self:
        require_unique(self.store_ids, "stores of an option")
        require_unique(
            (*(line.item_index for line in self.lines), *self.unfulfilled),
            "items of an option",
        )
        supplying = {line.store_id for line in self.lines}
        if supplying != set(self.store_ids):
            raise ValueError("every store of an option supplies an item, and no other store does")
        if self.basket_agorot != sum(line.cost_agorot for line in self.lines):
            raise ValueError("basket_agorot is not the sum of the lines")
        if len(self.store_ids) == 1 and self.penalty_agorot != 0:
            raise ValueError("a one-store option carries no extra-store penalty")
        if self.total_agorot != self.basket_agorot + self.penalty_agorot + self.travel_agorot:
            raise ValueError("total_agorot is not basket + penalty + travel")
        return self


Options = Annotated[tuple[BasketOption, ...], Field(max_length=MAX_STORES_PER_BASKET)]
Recommended = Annotated[int, Field(ge=0, lt=MAX_STORES_PER_BASKET, strict=True)]


def check_options(options: Sequence[BasketOption], recommended: int | None) -> None:
    """Raise unless there is one option per store count and the right one is recommended."""
    require_unique((len(option.store_ids) for option in options), "options with a store count")
    if len({option.item_indices for option in options}) > 1:
        raise ValueError("every option accounts for the same items, fulfilled or not")
    if not options:
        if recommended is not None:
            raise ValueError("nothing can be recommended without options")
        return
    if recommended is None or recommended >= len(options):
        raise ValueError("recommended must point at one of the options")
    best = min(options, key=lambda option: (-option.fulfilled_count, option.total_agorot))
    chosen = options[recommended]
    if (chosen.fulfilled_count, chosen.total_agorot) != (best.fulfilled_count, best.total_agorot):
        raise ValueError("the recommended option fulfils the most items, then costs the least")


class DecideOutput(Contract):
    """Up to three options, one per store count, and the index of the recommended one."""

    options: Options = ()
    recommended: Recommended | None = None

    @model_validator(mode="after")
    def _options_are_consistent(self) -> Self:
        check_options(self.options, self.recommended)
        return self


# --- 6. Explain -------------------------------------------------------------


class ExplainInput(Contract):
    dataset_version: DatasetVersion
    results: SearchResults
    stores: StoresInRange = ()
    options: Options = ()
    recommended: Recommended | None = None

    @model_validator(mode="after")
    def _options_are_consistent(self) -> Self:
        _check_result_order(self.results)
        check_options(self.options, self.recommended)
        return self


class ExplainOutput(Contract):
    """The text shown with the result: a summary, and one explanation per option."""

    summary: Explanation
    option_explanations: tuple[Explanation, ...] = Field(
        default=(), max_length=MAX_STORES_PER_BASKET
    )
    notice: Notice | None = None
