# Agent contracts

The typed models shared by every workstream (architecture sections 2, 5 and 5.6). They live in `src/shopping_agent/contracts/` and are imported from the package:

```python
from shopping_agent.contracts import BasketRequest, SearchOutput, GraphState
```

Every model rejects unknown fields and is read-only. A model that breaks a rule cannot be built: validation raises `pydantic.ValidationError`, which the graph's error handler reports with the trace ID.

## Conventions

- Items are referred to by `item_index`, their position in the parsed list.
- Money is whole agorot (`int`).
- `store_id` is `<chain>:<store number>`; `product_key` is the key of the `products` table.
- Unmatched and unfulfilled items are normal fields, not errors.

## API (`api.py`)

| Model | Rules |
|---|---|
| `BasketRequest` | Exactly one of `list_text` and `parsed_items`; `choices` only with `parsed_items`, one per item; `lat`, `lon`; `radius_km` defaults to 5 and is at most 50. |
| `ClarifyResponse` (`kind: "clarify"`) | The parsed items and at least one question; each question carries its own list item. |
| `BasketResult` (`kind: "basket"`) | Up to three options, one explanation each; every option accounts for every matched item; `unmatched` items appear in no option; `notice` is `no_stores_in_range` exactly when no store is in range, and a result with a notice has no options. |

`BasketResponse` is the union of the two responses, told apart by `kind`.

## Nodes (`nodes.py`)

| Node | Input | Output | Main rules |
|---|---|---|---|
| Parse | `ParseInput` (`list_text`) | `ParseOutput` (`items`, `parser`) | 1 to 100 items; `parser` is `llm` or `rules` (the fallback). |
| Search | `SearchInput` (`items`, `choices`) | `SearchOutput` (`results`) | One `ItemSearchResult` per item, in list order. `matched` needs candidates; `unmatched` has none; `needs_clarification` comes with one of the three clarify reasons. A `Candidate` is bought by `packs` or by `weight_g`, never both. |
| Clarify | `ClarifyInput` (`results`) | `ClarifyOutput` (`questions`) | At most one question per item; no questions means the graph continues. |
| Compare | `CompareInput` (`results`, `location`, `radius_km`) | `CompareOutput` (`stores`, `costs`) | No item may still need clarification. At most 40 stores; `costs` is the sparse cost matrix, one cell per item and store, each for a store in range. |
| Decide | `DecideInput` (`item_indices`, `stores`, `costs`, `location`) | `DecideOutput` (`options`, `recommended`) | One option per store count, up to three. In a `BasketOption` every store supplies an item, the basket is the sum of its lines, a one-store option has no penalty, and the total is basket + penalty + travel. The recommended option fulfils the most items, then costs the least. |
| Explain | `ExplainInput` | `ExplainOutput` (`summary`, `option_explanations`, `notice`) | Text only; the API combines it with the decision into a `BasketResult`. |

## Graph state (`state.py`)

`GraphState` holds the request, `trace_id`, `dataset_version` and the output of each node that has run (`parse`, `search`, `clarify`, `compare`, `decide`, `explain`). It checks that:

- outputs appear in the order of the flow; a request with `parsed_items` enters at Search and has no `parse` output;
- nothing runs after Clarify while questions are open;
- the outputs agree: search results carry the parsed items, Clarify asks about exactly the items that need it, costs and options refer only to matched items, and there is one explanation per option.

## Choices to revisit

- The limits (100 items, 200 candidates per item, 50 km radius) are generous defaults in `base.py`.
- `savings_agorot` may be negative or null; how savings are computed belongs to Decide.
- The penalty and travel amounts are not checked against `F` and `c`, which are configuration.
