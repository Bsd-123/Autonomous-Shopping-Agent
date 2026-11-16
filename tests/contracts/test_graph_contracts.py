"""Tests for the request, node and graph-state contracts.

One valid request is carried through all six nodes; every rule is then broken
on its own to show that it raises.
"""

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from shopping_agent import contracts
from shopping_agent.contracts import (
    DEFAULT_RADIUS_KM,
    MAX_ITEMS,
    MAX_RADIUS_KM,
    MAX_STORES_IN_RANGE,
    BasketLine,
    BasketOption,
    BasketRequest,
    BasketResponse,
    BasketResult,
    Candidate,
    ClarifyInput,
    ClarifyOutput,
    ClarifyQuestion,
    ClarifyResponse,
    CompareInput,
    CompareOutput,
    DecideInput,
    DecideOutput,
    ExplainInput,
    ExplainOutput,
    GraphState,
    ItemSearchResult,
    ItemStoreCost,
    Location,
    ParsedItem,
    ParseInput,
    ParseOutput,
    SearchInput,
    SearchOutput,
    StoreInRange,
)

VERSION = "2026-11-15"
HOME = {"lat": 32.0853, "lon": 34.8338}
MILK = ParsedItem(item="חלב", count=2)
TOMATOES = ParsedItem(item="עגבניות", count=6)
ITEMS = (MILK, TOMATOES)

MILK_CANDIDATE = Candidate(product_key="7290000000015", name="חלב 3% 1 ליטר", score=0.9, packs=2)
TOMATO_CANDIDATE = Candidate(
    product_key="osher_ad:116", name="עגבניה", score=0.8, weight_g=720.0, is_estimate=True
)
OSHER = StoreInRange(
    store_id="osher_ad:11",
    chain="osher_ad",
    name="בני ברק",
    location=Location(lat=32.09, lon=34.84),
    distance_km=0.8,
)
MAAYAN = StoreInRange(
    store_id="maayan_2000:38",
    chain="maayan_2000",
    name="רבי עקיבא",
    location=Location(lat=32.08, lon=34.83),
    distance_km=0.6,
)


def matched(index: int, item: ParsedItem, candidate: Candidate) -> ItemSearchResult:
    return ItemSearchResult(item_index=index, item=item, status="matched", candidates=(candidate,))


def line(index: int, store: StoreInRange, cost: int, key: str = "p") -> BasketLine:
    return BasketLine(item_index=index, store_id=store.store_id, product_key=key, cost_agorot=cost)


def option(*lines: BasketLine, unfulfilled=(), penalty=0, travel=0, **changes) -> BasketOption:
    basket = sum(each.cost_agorot for each in lines)
    fields = {
        "store_ids": tuple(dict.fromkeys(each.store_id for each in lines)),
        "lines": lines,
        "unfulfilled": unfulfilled,
        "basket_agorot": basket,
        "penalty_agorot": penalty,
        "travel_agorot": travel,
        "total_agorot": basket + penalty + travel,
    }
    return BasketOption(**(fields | changes))


RESULTS = (matched(0, MILK, MILK_CANDIDATE), matched(1, TOMATOES, TOMATO_CANDIDATE))
COSTS = (
    ItemStoreCost(item_index=0, store_id="osher_ad:11", product_key="p", cost_agorot=1180),
    ItemStoreCost(item_index=0, store_id="maayan_2000:38", product_key="p", cost_agorot=1240),
    ItemStoreCost(
        item_index=1, store_id="maayan_2000:38", product_key="q", cost_agorot=430, is_estimate=True
    ),
)
ONE_STORE = option(line(0, MAAYAN, 1240), line(1, MAAYAN, 430), travel=120)
TWO_STORES = option(line(0, OSHER, 1180), line(1, MAAYAN, 430), penalty=1000, travel=200)
ONLY_MILK = option(line(0, OSHER, 1180), unfulfilled=(1,), travel=160)


def rejects(model: type[BaseModel], message: str, **fields) -> None:
    with pytest.raises(ValidationError, match=message):
        model(**fields)


class TestRequest:
    def test_text_request(self):
        request = BasketRequest(list_text="  חלב\nעגבניות ", **HOME)

        assert request.list_text == "חלב\nעגבניות"
        assert request.radius_km == DEFAULT_RADIUS_KM == 5
        assert request.location == Location(**HOME)

    def test_resubmitted_items_with_choices(self):
        request = BasketRequest(
            parsed_items=ITEMS, choices=[{"item_index": 1, "product_key": "osher_ad:116"}], **HOME
        )

        assert request.choices[0].product_key == "osher_ad:116"

    def test_exactly_one_form_of_the_list(self):
        rejects(BasketRequest, "exactly one of list_text and parsed_items", **HOME)
        rejects(
            BasketRequest,
            "exactly one of list_text and parsed_items",
            list_text="חלב",
            parsed_items=ITEMS,
            **HOME,
        )

    def test_choices_must_refer_to_resubmitted_items(self):
        choice = {"item_index": 0, "product_key": "p"}

        rejects(BasketRequest, "only with parsed_items", list_text="חלב", choices=[choice], **HOME)
        rejects(
            BasketRequest,
            "not in the list",
            parsed_items=ITEMS,
            choices=[{"item_index": 2, "product_key": "p"}],
            **HOME,
        )
        rejects(BasketRequest, "repeat", parsed_items=ITEMS, choices=[choice, choice], **HOME)

    @pytest.mark.parametrize(
        "fields",
        [
            {"lat": 91.0, "lon": 34.8},
            {"lat": 32.0, "lon": 181.0},
            {"lat": float("nan"), "lon": 34.8},
            {"lat": "32.0", "lon": 34.8},
            {**HOME, "radius_km": 0},
            {**HOME, "radius_km": MAX_RADIUS_KM + 1},
            {**HOME, "list_text": "   "},
            {**HOME, "unknown": 1},
        ],
    )
    def test_bad_values_are_rejected(self, fields):
        with pytest.raises(ValidationError):
            BasketRequest(**({"list_text": "חלב"} | fields))

    def test_the_list_length_is_bounded(self):
        with pytest.raises(ValidationError):
            BasketRequest(parsed_items=[MILK] * (MAX_ITEMS + 1), **HOME)
        with pytest.raises(ValidationError):
            BasketRequest(parsed_items=[], **HOME)


class TestParse:
    def test_input_and_output(self):
        assert ParseInput(list_text="חלב").list_text == "חלב"
        assert ParseOutput(items=ITEMS, parser="rules").parser == "rules"

    def test_violations(self):
        rejects(ParseInput, "at least 1", list_text="")
        rejects(ParseOutput, "at least 1", items=(), parser="llm")
        rejects(ParseOutput, "'llm' or 'rules'", items=ITEMS, parser="guess")


class TestSearch:
    def test_input_and_output(self):
        assert SearchInput(items=ITEMS).choices == ()
        assert SearchOutput(results=RESULTS).results[1].candidates == (TOMATO_CANDIDATE,)

    def test_choices_must_refer_to_items(self):
        rejects(
            SearchInput,
            "not in the list",
            items=ITEMS,
            choices=[{"item_index": 5, "product_key": "p"}],
        )

    def test_a_candidate_is_bought_by_packs_or_by_weight(self):
        base = {"product_key": "p", "name": "חלב", "score": 0.5}

        rejects(Candidate, "exactly one of packs and weight_g", **base)
        rejects(Candidate, "exactly one of packs and weight_g", **base, packs=1, weight_g=500.0)
        rejects(Candidate, "only a weighed candidate", **base, packs=1, is_estimate=True)
        rejects(Candidate, "greater than or equal to 1", **base, packs=0)

    @pytest.mark.parametrize(
        ("message", "fields"),
        [
            ("at least one candidate", {"status": "matched"}),
            ("has no candidates", {"status": "unmatched", "candidates": (MILK_CANDIDATE,)}),
            ("clarify_reason is set exactly", {"status": "needs_clarification"}),
            (
                "clarify_reason is set exactly",
                {
                    "status": "matched",
                    "clarify_reason": "ambiguous",
                    "candidates": (MILK_CANDIDATE,),
                },
            ),
            (
                "cannot come with candidates",
                {
                    "status": "needs_clarification",
                    "clarify_reason": "no_candidates",
                    "candidates": (MILK_CANDIDATE,),
                },
            ),
            (
                "needs candidates to show",
                {"status": "needs_clarification", "clarify_reason": "overbuy"},
            ),
            (
                "candidates repeat",
                {"status": "matched", "candidates": (MILK_CANDIDATE, MILK_CANDIDATE)},
            ),
        ],
    )
    def test_status_must_match_content(self, message, fields):
        rejects(ItemSearchResult, message, item_index=0, item=MILK, **fields)

    def test_an_unmatched_item_is_a_normal_result(self):
        result = ItemSearchResult(item_index=0, item=MILK, status="unmatched")

        assert result.candidates == ()

    def test_results_follow_the_list_order(self):
        rejects(SearchOutput, "order of the list items", results=RESULTS[::-1])
        rejects(SearchOutput, "order of the list items", results=RESULTS[1:])


class TestClarify:
    QUESTION = ClarifyQuestion(
        item_index=1, item=TOMATOES, reason="ambiguous", options=(TOMATO_CANDIDATE,)
    )

    def test_input_and_output(self):
        assert ClarifyInput(results=RESULTS).results == RESULTS
        assert ClarifyOutput().questions == ()
        assert ClarifyOutput(questions=(self.QUESTION,)).questions[0].reason == "ambiguous"

    def test_options_match_the_reason(self):
        base = {"item_index": 0, "item": MILK}

        assert ClarifyQuestion(**base, reason="no_candidates").options == ()
        rejects(ClarifyQuestion, "options are empty exactly", **base, reason="overbuy")
        rejects(
            ClarifyQuestion,
            "options are empty exactly",
            **base,
            reason="no_candidates",
            options=(MILK_CANDIDATE,),
        )

    def test_one_question_per_item(self):
        rejects(ClarifyOutput, "repeat", questions=(self.QUESTION, self.QUESTION))

    def test_response_carries_the_items_and_all_questions(self):
        response = ClarifyResponse(
            dataset_version=VERSION, parsed_items=ITEMS, questions=(self.QUESTION,)
        )

        assert response.kind == "clarify"
        assert response.simplified_parsing is False

    def test_response_violations(self):
        base = {"dataset_version": VERSION, "parsed_items": ITEMS}
        beyond = self.QUESTION.model_copy(update={"item_index": 2})
        wrong_item = self.QUESTION.model_copy(update={"item_index": 0})

        rejects(ClarifyResponse, "at least 1", **base, questions=())
        rejects(ClarifyResponse, "not in the list", **base, questions=(beyond,))
        rejects(ClarifyResponse, "does not carry its list item", **base, questions=(wrong_item,))


class TestCompare:
    def test_input_and_output(self):
        compare_input = CompareInput(results=RESULTS, location=HOME, radius_km=5.0)
        output = CompareOutput(stores=(OSHER, MAAYAN), costs=COSTS)

        assert compare_input.location == Location(**HOME)
        assert len(output.costs) == 3

    def test_no_stores_in_range_is_a_normal_output(self):
        assert CompareOutput() == CompareOutput(stores=(), costs=())

    def test_items_waiting_for_clarification_cannot_be_compared(self):
        waiting = ItemSearchResult(
            item_index=1,
            item=TOMATOES,
            status="needs_clarification",
            clarify_reason="no_candidates",
        )

        rejects(
            CompareInput,
            "still needs clarification",
            results=(RESULTS[0], waiting),
            location=HOME,
            radius_km=5.0,
        )

    def test_cost_matrix_violations(self):
        rejects(CompareOutput, "not in range", stores=(OSHER,), costs=COSTS)
        rejects(CompareOutput, "stores in range repeat", stores=(OSHER, OSHER))
        rejects(CompareOutput, "cost cells repeat", stores=(OSHER,), costs=(COSTS[0], COSTS[0]))
        with pytest.raises(ValidationError):
            CompareOutput(stores=(OSHER,) * (MAX_STORES_IN_RANGE + 1))

    def test_a_cost_is_positive_whole_agorot(self):
        base = {"item_index": 0, "store_id": "osher_ad:11", "product_key": "p"}

        rejects(ItemStoreCost, "greater than 0", **base, cost_agorot=0)
        rejects(ItemStoreCost, "valid integer", **base, cost_agorot=11.8)
        rejects(ItemStoreCost, "should match pattern", **(base | {"store_id": "11"}), cost_agorot=5)


class TestDecide:
    def test_input(self):
        decide_input = DecideInput(
            item_indices=(0, 1), stores=(OSHER, MAAYAN), costs=COSTS, location=HOME
        )

        assert decide_input.item_indices == (0, 1)
        rejects(
            DecideInput,
            "not to be decided",
            item_indices=(0,),
            stores=(OSHER, MAAYAN),
            costs=COSTS,
            location=HOME,
        )

    def test_options_and_recommendation(self):
        output = DecideOutput(options=(ONE_STORE, TWO_STORES), recommended=0)

        assert output.options[0].total_agorot == 1240 + 430 + 120
        assert output.options[1].total_agorot == 1180 + 430 + 1000 + 200
        assert output.options[0].fulfilled_count == 2

    def test_nothing_listed_gives_no_options(self):
        assert DecideOutput() == DecideOutput(options=(), recommended=None)

    def test_most_items_are_recommended_before_lowest_total(self):
        # The one-store option skips the tomatoes and is cheaper, but covers less.
        assert ONLY_MILK.total_agorot < TWO_STORES.total_agorot
        assert DecideOutput(options=(ONLY_MILK, TWO_STORES), recommended=1).recommended == 1
        rejects(
            DecideOutput, "fulfils the most items", options=(ONLY_MILK, TWO_STORES), recommended=0
        )
        rejects(
            DecideOutput, "fulfils the most items", options=(ONE_STORE, TWO_STORES), recommended=1
        )

    def test_recommendation_violations(self):
        rejects(DecideOutput, "must point at one of the options", options=(ONE_STORE,))
        rejects(
            DecideOutput, "must point at one of the options", options=(ONE_STORE,), recommended=1
        )
        rejects(DecideOutput, "without options", recommended=0)
        rejects(DecideOutput, "store count repeat", options=(ONE_STORE, ONE_STORE), recommended=0)
        rejects(
            DecideOutput,
            "accounts for the same items",
            options=(option(line(0, OSHER, 1180)), TWO_STORES),
            recommended=1,
        )

    @pytest.mark.parametrize(
        ("message", "changes"),
        [
            ("not basket \\+ penalty \\+ travel", {"total_agorot": 1}),
            ("not the sum of the lines", {"basket_agorot": 1}),
            ("every store of an option supplies", {"store_ids": ("maayan_2000:38", "osher_ad:11")}),
            ("every store of an option supplies", {"store_ids": ("osher_ad:11",)}),
            ("stores of an option repeat", {"store_ids": ("maayan_2000:38",) * 2}),
            ("items of an option repeat", {"unfulfilled": (1,)}),
            ("no extra-store penalty", {"penalty_agorot": 1000, "total_agorot": 2790}),
            ("greater than or equal to 0", {"travel_agorot": -1}),
        ],
    )
    def test_option_violations(self, message, changes):
        with pytest.raises(ValidationError, match=message):
            option(line(0, MAAYAN, 1240), line(1, MAAYAN, 430), travel=120, **changes)

    def test_an_option_needs_a_line_and_at_most_three_stores(self):
        with pytest.raises(ValidationError):
            option()
        stores = [
            StoreInRange(**(OSHER.model_dump() | {"store_id": f"osher_ad:{n}"})) for n in range(4)
        ]
        with pytest.raises(ValidationError, match="at most 3"):
            option(*(line(n, store, 100) for n, store in enumerate(stores)), penalty=3000)


def result(**changes) -> BasketResult:
    fields = {
        "dataset_version": VERSION,
        "parsed_items": ITEMS,
        "stores": (OSHER, MAAYAN),
        "options": (ONE_STORE, TWO_STORES),
        "recommended": 0,
        "summary": "הסל הזול ביותר בחנות אחת",
        "option_explanations": ("חנות אחת", "שתי חנויות"),
    }
    return BasketResult(**(fields | changes))


class TestExplainAndResult:
    def test_input_and_output(self):
        explain_input = ExplainInput(
            dataset_version=VERSION,
            results=RESULTS,
            stores=(OSHER, MAAYAN),
            options=(ONE_STORE, TWO_STORES),
            recommended=0,
        )
        output = ExplainOutput(summary="סיכום", option_explanations=("א", "ב"))

        assert explain_input.recommended == 0
        assert output.notice is None
        rejects(
            ExplainInput,
            "must point at one of the options",
            dataset_version=VERSION,
            results=RESULTS,
            options=(ONE_STORE,),
        )

    def test_result(self):
        basket = result()

        assert basket.kind == "basket"
        assert basket.unmatched == ()

    def test_unmatched_items_are_a_normal_field(self):
        basket = result(
            options=(option(line(0, OSHER, 1180)),),
            option_explanations=("חנות אחת",),
            unmatched=(1,),
        )

        assert basket.unmatched == (1,)

    def test_no_stores_in_range(self):
        base = {"stores": (), "options": (), "recommended": None, "option_explanations": ()}

        assert result(**base, notice="no_stores_in_range").options == ()
        with pytest.raises(ValidationError, match="exactly when no store is in range"):
            result(**base)
        with pytest.raises(ValidationError, match="exactly when no store is in range"):
            result(notice="no_stores_in_range")

    def test_nothing_listed(self):
        empty = {"options": (), "recommended": None, "option_explanations": ()}

        assert result(**empty, notice="nothing_listed").notice == "nothing_listed"
        with pytest.raises(ValidationError, match="a result with a notice has no options"):
            result(notice="nothing_listed")

    @pytest.mark.parametrize(
        ("message", "changes"),
        [
            ("one explanation per option", {"option_explanations": ("חנות אחת",)}),
            ("fulfils the most items", {"recommended": 1}),
            ("account for every matched item", {"unmatched": (1,)}),
            ("not in the list", {"unmatched": (2,)}),
            ("not in range", {"stores": (MAAYAN,)}),
            ("unmatched items repeat", {"unmatched": (1, 1)}),
        ],
    )
    def test_result_violations(self, message, changes):
        with pytest.raises(ValidationError, match=message):
            result(**changes)

    def test_the_response_is_told_apart_by_kind(self):
        adapter = TypeAdapter(BasketResponse)
        question = ClarifyQuestion(item_index=0, item=MILK, reason="no_candidates")
        clarify = ClarifyResponse(dataset_version=VERSION, parsed_items=ITEMS, questions=[question])

        assert adapter.validate_json(result().model_dump_json()) == result()
        assert adapter.validate_json(clarify.model_dump_json()) == clarify
        with pytest.raises(ValidationError):
            adapter.validate_python({"kind": "other"})


def state(**fields) -> GraphState:
    request = fields.pop("request", BasketRequest(list_text="חלב\nעגבניות", **HOME))
    return GraphState(trace_id="trace-1", dataset_version=VERSION, request=request, **fields)


FULL_RUN = {
    "parse": ParseOutput(items=ITEMS, parser="llm"),
    "search": SearchOutput(results=RESULTS),
    "clarify": ClarifyOutput(),
    "compare": CompareOutput(stores=(OSHER, MAAYAN), costs=COSTS),
    "decide": DecideOutput(options=(ONE_STORE, TWO_STORES), recommended=0),
    "explain": ExplainOutput(summary="סיכום", option_explanations=("א", "ב")),
}


class TestGraphState:
    def test_a_request_runs_through_all_six_nodes(self):
        current = state()
        assert current.items is None

        for name, output in FULL_RUN.items():
            current = GraphState.model_validate(current.model_dump() | {name: output})

        assert current.items == ITEMS
        assert current.needs_clarification is False
        assert GraphState.model_validate_json(current.model_dump_json()) == current

    def test_resubmitted_items_enter_at_search(self):
        request = BasketRequest(parsed_items=ITEMS, **HOME)

        current = state(request=request, search=FULL_RUN["search"])

        assert current.items == ITEMS
        assert current.choices == ()
        with pytest.raises(ValidationError, match="Parse does not run"):
            state(request=request, parse=FULL_RUN["parse"])

    def test_the_graph_stops_at_open_questions(self):
        waiting = ItemSearchResult(
            item_index=1,
            item=TOMATOES,
            status="needs_clarification",
            clarify_reason="no_candidates",
        )
        search = SearchOutput(results=(RESULTS[0], waiting))
        clarify = ClarifyOutput(
            questions=[ClarifyQuestion(item_index=1, item=TOMATOES, reason="no_candidates")]
        )
        run = {"parse": FULL_RUN["parse"], "search": search, "clarify": clarify}

        assert state(**run).needs_clarification is True
        with pytest.raises(ValidationError, match="while clarify questions are open"):
            state(**run, compare=CompareOutput())
        with pytest.raises(ValidationError, match="exactly the items that need clarification"):
            state(**(run | {"clarify": ClarifyOutput()}))

    @pytest.mark.parametrize("missing", ["parse", "search", "clarify", "compare", "decide"])
    def test_a_node_cannot_run_before_the_one_it_depends_on(self, missing):
        run = {name: output for name, output in FULL_RUN.items() if name != missing}

        with pytest.raises(ValidationError, match="cannot be set before"):
            state(**run)

    def test_outputs_must_agree_with_each_other(self):
        other = SearchOutput(results=(RESULTS[0], matched(1, MILK, MILK_CANDIDATE)))
        unmatched = SearchOutput(
            results=(RESULTS[0], ItemSearchResult(item_index=1, item=TOMATOES, status="unmatched"))
        )

        with pytest.raises(ValidationError, match="must carry the parsed items"):
            state(**(FULL_RUN | {"search": other}))
        with pytest.raises(ValidationError, match="items that search did not match"):
            state(**(FULL_RUN | {"search": unmatched}))
        with pytest.raises(ValidationError, match="one explanation per option"):
            state(**(FULL_RUN | {"explain": ExplainOutput(summary="סיכום")}))

    def test_an_option_must_account_for_every_matched_item(self):
        partial = DecideOutput(options=(option(line(0, OSHER, 1180)),), recommended=0)

        with pytest.raises(ValidationError, match="account for every matched item"):
            state(**(FULL_RUN | {"decide": partial, "explain": ExplainOutput(summary="סיכום")}))

    def test_the_trace_id_is_a_plain_token(self):
        with pytest.raises(ValidationError):
            GraphState(trace_id="bad id!", dataset_version=VERSION, request=state().request)


def test_contracts_are_read_only_and_reject_unknown_fields():
    with pytest.raises(ValidationError, match="frozen"):
        ONE_STORE.total_agorot = 0
    with pytest.raises(ValidationError, match="Extra inputs"):
        Location(lat=32.0, lon=34.8, alt=10.0)


def test_every_node_has_an_input_and_an_output_model():
    for node in ("Parse", "Search", "Clarify", "Compare", "Decide", "Explain"):
        assert issubclass(getattr(contracts, f"{node}Input"), contracts.Contract)
        assert issubclass(getattr(contracts, f"{node}Output"), contracts.Contract)
    assert all(hasattr(contracts, name) for name in contracts.__all__)
