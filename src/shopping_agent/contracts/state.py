"""The state the graph carries from node to node.

It holds the request and the output of every node that has run. A node reads
what it needs, and its output is stored under its own name. The state checks
that outputs appear in the order of the flow and agree with each other, so a
node never runs on something an earlier node did not produce.
"""

from typing import Annotated, Self

from pydantic import StringConstraints, model_validator

from shopping_agent.contracts.api import BasketRequest
from shopping_agent.contracts.base import Contract, DatasetVersion
from shopping_agent.contracts.nodes import (
    ClarifyChoice,
    ClarifyOutput,
    CompareOutput,
    DecideOutput,
    ExplainOutput,
    ParsedItems,
    ParseOutput,
    SearchOutput,
)

TraceId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


class GraphState(Contract):
    """One request's journey through Parse, Search, Clarify, Compare, Decide and Explain.

    A request with ``parsed_items`` enters at Search, so ``parse`` stays empty.
    When Clarify returns questions the graph stops there and the later outputs
    stay empty.
    """

    trace_id: TraceId
    dataset_version: DatasetVersion
    request: BasketRequest
    parse: ParseOutput | None = None
    search: SearchOutput | None = None
    clarify: ClarifyOutput | None = None
    compare: CompareOutput | None = None
    decide: DecideOutput | None = None
    explain: ExplainOutput | None = None

    @property
    def items(self) -> ParsedItems | None:
        """The parsed items: from the request when it carries them, else from Parse."""
        if self.request.parsed_items is not None:
            return self.request.parsed_items
        return self.parse.items if self.parse is not None else None

    @property
    def choices(self) -> tuple[ClarifyChoice, ...]:
        return self.request.choices

    @property
    def needs_clarification(self) -> bool:
        return self.clarify is not None and bool(self.clarify.questions)

    @model_validator(mode="after")
    def _outputs_follow_the_flow(self) -> Self:
        if self.parse is not None and self.request.parsed_items is not None:
            raise ValueError("Parse does not run when the request carries parsed_items")
        steps = [
            ("search", self.search, "the parsed items", self.items),
            ("clarify", self.clarify, "search", self.search),
            ("compare", self.compare, "clarify", self.clarify),
            ("decide", self.decide, "compare", self.compare),
            ("explain", self.explain, "decide", self.decide),
        ]
        for name, output, needed_name, needed in steps:
            if output is not None and needed is None:
                raise ValueError(f"{name} cannot be set before {needed_name}")
        if self.compare is not None and self.needs_clarification:
            raise ValueError("compare cannot run while clarify questions are open")
        return self

    @model_validator(mode="after")
    def _outputs_agree(self) -> Self:
        items = self.items
        if self.search is not None and items is not None:
            if tuple(result.item for result in self.search.results) != items:
                raise ValueError("search results must carry the parsed items, one each, in order")
        if self.clarify is not None and self.search is not None:
            waiting = {
                result.item_index
                for result in self.search.results
                if result.status == "needs_clarification"
            }
            asked = {question.item_index for question in self.clarify.questions}
            if asked != waiting:
                raise ValueError("clarify asks about exactly the items that need clarification")
        if self.compare is not None and self.search is not None:
            matched = {
                result.item_index for result in self.search.results if result.status == "matched"
            }
            costed = {cost.item_index for cost in self.compare.costs}
            if not costed <= matched:
                raise ValueError("compare has costs for items that search did not match")
            if self.decide is not None:
                for option in self.decide.options:
                    if option.item_indices != matched:
                        raise ValueError("an option must account for every matched item")
        if self.explain is not None and self.decide is not None:
            if len(self.explain.option_explanations) != len(self.decide.options):
                raise ValueError("there must be one explanation per option")
        return self
