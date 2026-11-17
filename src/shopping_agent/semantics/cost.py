"""The cost of covering one request with one candidate product (architecture section 3.2).

A request takes one of five forms, and each has its own cost:

* Total quantity ("3 L milk"): ceil(total / pack size) x pack price.
* Pack size and count ("milk 1 L x2"): count x pack price; the pack size
  must match.
* Count only ("milk x2"): count x pack price; the pack must be within
  0.8-1.5x of the category's typical pack.
* Weighed, by weight ("1.5 kg tomatoes"): price per kilogram x grams / 1000,
  rounded once.
* Weighed, by count ("6 tomatoes"): as by weight, for count x typical unit
  weight; the result is an estimate.

Packs of different products are never combined: the cost is always for one
product. Every cost is a whole number of agorot.

The overbuy limit and the small-pack exclusion are separate filters and are
not applied here.
"""

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from math import ceil, isclose

from shopping_agent.contracts import ParsedItem, Quantity

# A count-only request accepts packs from 0.8 to 1.5 times the category's typical pack.
TYPICAL_PACK_MIN_RATIO = 0.8
TYPICAL_PACK_MAX_RATIO = 1.5
GRAMS_PER_KILOGRAM = 1000
# Sizes are compared as floats; this absorbs representation error only.
_TOLERANCE = 1e-9


class Mismatch(StrEnum):
    """Why a product cannot cover a request."""

    UNIT = "unit"  # the request is in a unit the product is not sold in
    PACK_SIZE = "pack_size"  # the requested pack size is not the product's
    NOT_TYPICAL_PACK = "not_typical_pack"  # count only, and the pack is far from typical
    NO_TYPICAL_UNIT_WEIGHT = "no_typical_unit_weight"  # weighed by count, weight unknown


@dataclass(frozen=True, slots=True)
class PricedProduct:
    """A product as listed at one store.

    For a packaged product ``price_agorot`` is the price of one pack of size
    ``pack``. For a weighed product it is the price per kilogram, and ``pack``
    is not used.
    """

    price_agorot: int
    pack: Quantity | None = None
    is_weighed: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.price_agorot, bool) or not isinstance(self.price_agorot, int):
            raise TypeError("price_agorot must be a whole number of agorot")
        if self.price_agorot <= 0:
            raise ValueError("price_agorot must be positive")
        if not self.is_weighed and self.pack is None:
            raise ValueError("a packaged product needs a pack size")


@dataclass(frozen=True, slots=True)
class CandidateCost:
    """What covering the request with the product costs, and how much is bought.

    A packaged product is bought as ``packs`` whole packs; a weighed one as
    ``weight_g`` grams. ``is_estimate`` marks a weight derived from a count.
    """

    cost_agorot: int
    packs: int | None = None
    weight_g: float | None = None
    is_estimate: bool = False


def _weighed_cost(price_per_kg: int, grams: float, is_estimate: bool) -> CandidateCost:
    # Exact decimal arithmetic, with the single rounding at the end.
    exact = Decimal(price_per_kg) * Decimal(str(grams)) / GRAMS_PER_KILOGRAM
    # Anything bought costs at least one agora; a cost of zero means "not listed" downstream.
    rounded = int(exact.quantize(Decimal(1), rounding=ROUND_HALF_UP))
    return CandidateCost(
        cost_agorot=max(1, rounded),
        weight_g=grams,
        is_estimate=is_estimate,
    )


def _weighed(
    request: ParsedItem, product: PricedProduct, typical_unit_weight: Quantity | None
) -> CandidateCost | Mismatch:
    size = request.total_quantity or request.pack_size
    if size is not None:
        if size.unit != "g":
            return Mismatch.UNIT
        # "1.5 kg tomatoes" is the whole amount, whatever the count; "tomatoes 1 kg x2"
        # is two packs of a kilogram. This matches the packaged case.
        multiplier = 1 if request.total_quantity is not None else request.count
        return _weighed_cost(product.price_agorot, size.value * multiplier, is_estimate=False)
    if typical_unit_weight is None:
        return Mismatch.NO_TYPICAL_UNIT_WEIGHT
    if typical_unit_weight.unit != "g":
        raise ValueError("typical_unit_weight must be in grams")
    grams = request.count * typical_unit_weight.value
    return _weighed_cost(product.price_agorot, grams, is_estimate=True)


def _packaged(
    request: ParsedItem, product: PricedProduct, pack: Quantity, typical_pack: Quantity | None
) -> CandidateCost | Mismatch:
    if request.total_quantity is not None:
        if request.total_quantity.unit != pack.unit:
            return Mismatch.UNIT
        packs = max(1, ceil(request.total_quantity.value / pack.value - _TOLERANCE))
    elif request.pack_size is not None:
        if request.pack_size.unit != pack.unit:
            return Mismatch.UNIT
        if not isclose(request.pack_size.value, pack.value, rel_tol=_TOLERANCE):
            return Mismatch.PACK_SIZE
        packs = request.count
    else:
        if typical_pack is not None:
            if typical_pack.unit != pack.unit:
                return Mismatch.UNIT
            ratio = pack.value / typical_pack.value
            if not (
                TYPICAL_PACK_MIN_RATIO - _TOLERANCE <= ratio <= TYPICAL_PACK_MAX_RATIO + _TOLERANCE
            ):
                return Mismatch.NOT_TYPICAL_PACK
        packs = request.count
    return CandidateCost(cost_agorot=packs * product.price_agorot, packs=packs)


def candidate_cost(
    request: ParsedItem,
    product: PricedProduct,
    *,
    typical_pack: Quantity | None = None,
    typical_unit_weight: Quantity | None = None,
) -> CandidateCost | Mismatch:
    """Return the cost of covering ``request`` with ``product``, or why it cannot.

    ``typical_pack`` and ``typical_unit_weight`` come from the item's category
    profile and are ``None`` when it has no profile. Without a typical pack a
    count-only request accepts any pack size.
    """
    if product.is_weighed:
        return _weighed(request, product, typical_unit_weight)
    assert product.pack is not None  # checked when the product was built
    return _packaged(request, product, product.pack, typical_pack)
