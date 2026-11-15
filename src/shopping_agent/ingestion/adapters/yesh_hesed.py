"""Yesh Hesed adapter.

Yesh Hesed has no files of its own: it is a sub-chain inside Shufersal's
files. The adapter reads the Shufersal folder and keeps only the Yesh Hesed
stores and the prices that belong to them.
"""

from pathlib import Path

import polars as pl

from shopping_agent.ingestion.adapters.base import ChainAdapter

SUB_CHAIN_NAME = "יש חסד"


class YeshHesedAdapter(ChainAdapter):
    chain = "yesh_hesed"
    source_folder = "shufersal"
    source_name = "shufersal"

    def select_raw_stores(self, raw: pl.DataFrame) -> pl.DataFrame:
        return raw.filter(pl.col("subchainname") == SUB_CHAIN_NAME)

    def read_prices(self, raw_dir: Path) -> pl.DataFrame:
        stores = self.read_stores(raw_dir).select("store_id")
        return super().read_prices(raw_dir).join(stores, on="store_id", how="semi")
