"""Osher Ad adapter."""

from shopping_agent.ingestion.adapters.base import ChainAdapter


class OsherAdAdapter(ChainAdapter):
    chain = "osher_ad"
    source_folder = "osherad"
    source_name = "osher_ad"
    # Osher Ad writes the Hebrew for "unknown" where it has no value.
    null_markers = frozenset({"לא ידוע"})
