"""Per-chain ingestion adapters for the Phase 1 chains."""

from shopping_agent.ingestion.adapters.base import ChainAdapter, RawFormatError
from shopping_agent.ingestion.adapters.layout import PRICE_SCHEMA, STORE_SCHEMA
from shopping_agent.ingestion.adapters.maayan_2000 import Maayan2000Adapter
from shopping_agent.ingestion.adapters.osher_ad import OsherAdAdapter
from shopping_agent.ingestion.adapters.yesh_hesed import YeshHesedAdapter

ADAPTERS: tuple[ChainAdapter, ...] = (
    YeshHesedAdapter(),
    OsherAdAdapter(),
    Maayan2000Adapter(),
)

__all__ = [
    "ADAPTERS",
    "PRICE_SCHEMA",
    "STORE_SCHEMA",
    "ChainAdapter",
    "Maayan2000Adapter",
    "OsherAdAdapter",
    "RawFormatError",
    "YeshHesedAdapter",
]
