"""Maayan 2000 adapter."""

from shopping_agent.ingestion.adapters.base import ChainAdapter


class Maayan2000Adapter(ChainAdapter):
    chain = "maayan_2000"
    source_folder = "maayan2000"
    source_name = "maayan_2000"
    # Maayan 2000 fills unknown text with the Hebrew for "unknown" or a run
    # of X characters.
    null_markers = frozenset({"לא ידוע", "XXXXX"})
