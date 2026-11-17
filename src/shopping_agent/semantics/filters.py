"""The quantity filters applied to each candidate on its own (architecture section 3.2).

Two filters apply to a request for a total quantity ("3 L milk"):

* **Small packs.** In a category with a profile, a pack smaller than 0.8 times
  the typical pack is excluded, so the request is not met with twelve small
  cartons.
* **Overbuy.** A candidate that would deliver more than 1.5 times the requested
  quantity is excluded.

Each candidate is judged alone, with packs of that one product only; packs of
different products are never combined. When every candidate that was valid in
all other respects fails on overbuy, the outcome says so, and the item goes to
Clarify.
"""

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from enum import StrEnum

from shopping_agent.contracts import ParsedItem, Quantity
from shopping_agent.semantics.cost import (
    SIZESIZE_TOLERANCE,
    TYPICAL_PACK_MIN_RATIO,
    CandidateCost,
    Mismatch,
    PricedProduct,
    candidate_cost,
)

OVERBUY_LIMIT = 1.5
SMALL_PACK_RATIO = TYPICAL_PACK_MIN_RATIO


class Exclusion(StrEnum):
    """Why a candidate that fits the request is still filtered out."""

    SMALL_PACK = "small_pack"
    OVERBUY = "overbuy"


def quantity_exclusion(
    request: ParsedItem,
    product: PricedProduct,
    cost: CandidateCost,
    *,
    typical_pack: Quantity | None = None,
) -> Exclusion | None:
    """Return why a fitting candidate is filtered out, or ``None`` when it stays.

    ``cost`` is the candidate's cost for the request. Only a request for a
    total quantity of a packaged product can be excluded: a weighed product is
    bought to the gram, and a request by count or pack size names no total.
    """
    total = request.total_quantity
    if total is None or product.is_weighed or product.pack is None or cost.packs is None:
        return None
    pack = product.pack
    if (
        typical_pack is not None
        and typical_pack.unit == pack.unit
        and pack.value < SMALL_PACK_RATIO * typical_pack.value * (1 - SIZE_TOLERANCE)
    ):
        return Exclusion.SMALL_PACK
    if cost.packs * pack.value > OVERBUY_LIMIT * total.value * (1 + SIZE_TOLERANCE):
        return Exclusion.OVERBUY
    return None


@dataclass(frozen=True, slots=True)
class FilterOutcome[Key: Hashable]:
    """The candidates of one item after the quantity rules.

    ``kept`` holds each surviving candidate with its cost, in the order given.
    ``rejected`` holds the others with the reason. ``all_failed_on_overbuy`` is
    true when nothing survived and every candidate that fitted the request and
    passed the small-pack rule failed on overbuy: the signal for Clarify.
    """

    kept: tuple[tuple[Key, CandidateCost], ...]
    rejected: tuple[tuple[Key, Mismatch | Exclusion], ...]

    @property
    def all_failed_on_overbuy(self) -> bool:
        reasons = [reason for _, reason in self.rejected]
        return not self.kept and Exclusion.OVERBUY in reasons


def filter_candidates[Key: Hashable](
    request: ParsedItem,
    candidates: Iterable[tuple[Key, PricedProduct]],
    *,
    typical_pack: Quantity | None = None,
    typical_unit_weight: Quantity | None = None,
) -> FilterOutcome[Key]:
    """Cost every candidate for the request and apply the quantity filters.

    ``candidates`` pairs a key of the caller's choice, such as the product key,
    with the product. Each candidate is costed and filtered on its own.
    """
    kept: list[tuple[Key, CandidateCost]] = []
    rejected: list[tuple[Key, Mismatch | Exclusion]] = []
    for key, product in candidates:
        cost = candidate_cost(
            request, product, typical_pack=typical_pack, typical_unit_weight=typical_unit_weight
        )
        if isinstance(cost, Mismatch):
            rejected.append((key, cost))
            continue
        exclusion = quantity_exclusion(request, product, cost, typical_pack=typical_pack)
        if exclusion is None:
            kept.append((key, cost))
        else:
            rejected.append((key, exclusion))
    return FilterOutcome(tuple(kept), tuple(rejected))
