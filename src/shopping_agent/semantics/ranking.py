"""The one ranking rule for candidates (architecture section 3.2).

Candidates are ranked by the total cost of covering the request. The unit
price, computed here at query time, only breaks ties between candidates that
cost the same. A larger pack that is cheaper per litre therefore loses to a
smaller one whenever the smaller one covers the request for less.
"""

from collections.abc import Hashable, Iterable, Mapping
from fractions import Fraction

from shopping_agent.semantics.cost import CandidateCost, PricedProduct

GRAMS_PER_KILOGRAM = 1000


def unit_price(product: PricedProduct) -> Fraction:
    """The price in agorot of one base unit of the product: a millilitre, a gram or a unit.

    The value is exact, so two equal unit prices always compare equal. A weighed
    product is priced per kilogram, so its unit price is that price over 1,000.
    """
    if product.is_weighed:
        return Fraction(product.price_agorot, GRAMS_PER_KILOGRAM)
    if product.pack is None:
        raise ValueError("a packaged product needs a pack size")
    return Fraction(product.price_agorot) / Fraction(str(product.pack.value))


def rank_candidates[Key: Hashable](
    costed: Iterable[tuple[Key, CandidateCost]], products: Mapping[Key, PricedProduct]
) -> tuple[tuple[Key, CandidateCost], ...]:
    """Order candidates from the cheapest way of covering the request to the dearest.

    ``costed`` pairs each candidate's key with its cost for the request, as
    ``filter_candidates`` returns in ``kept``; ``products`` gives the product
    behind each key. Candidates with the same total cost are ordered by unit
    price, and those equal in both keep the order they were given in.
    """
    candidates = list(costed)
    missing = [key for key, _ in candidates if key not in products]
    if missing:
        raise KeyError(f"no product was given for: {', '.join(map(repr, missing))}")
    return tuple(
        sorted(candidates, key=lambda item: (item[1].cost_agorot, unit_price(products[item[0]])))
    )
