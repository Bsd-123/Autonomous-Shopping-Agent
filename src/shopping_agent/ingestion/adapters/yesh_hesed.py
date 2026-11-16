"""Yesh Hesed adapter.

Yesh Hesed has no files of its own: it is a sub-chain inside Shufersal's
files. The adapter reads the Shufersal folder and keeps only the Yesh Hesed
stores and the prices that belong to them.
"""

from pathlib import Path

import polars as pl

from shopping_agent.ingestion.adapters.base import ChainAdapter, require_rows

SUB_CHAIN_NAME = "יש חסד"


class YeshHesedAdapter(ChainAdapter):
    chain = "yesh_hesed"
    source_folder = "shufersal"
    source_name = "shufersal"

    def select_raw_stores(self, raw: pl.DataFrame) -> pl.DataFrame:
        return raw.filter(pl.col("subchainname") == SUB_CHAIN_NAME)

    def read_prices(self, raw_dir: Path, stores: pl.DataFrame | None = None) -> pl.DataFrame:
        """Read the prices of the Yesh Hesed stores.

        The rows have to be selected by store, so the store file is read here when
        ``stores`` is not given.
        """
        return super().read_prices(raw_dir, self.read_stores(raw_dir) if stores is None else stores)

    def _own_rows(self, raw: pl.DataFrame, stores: pl.DataFrame) -> pl.DataFrame:
        # Other sub-chains hold about 94% of the rows. They are dropped before
        # parsing, so they cost no memory and only their store numbers are
        # validated: a row whose number cannot be read could belong to any store.
        store_ids = self._store_ids(raw)
        own = raw.filter(store_ids.is_in(stores["store_id"].implode()))
        return require_rows(own, f"{self.chain} price rows for its stores", self.source_folder)
