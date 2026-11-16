"""Helpers shared by the adapter tests: fixture paths, staging, and layout assertions."""

import shutil
from pathlib import Path

import polars as pl

from shopping_agent.ingestion.adapters import PRICE_SCHEMA, STORE_SCHEMA
from shopping_agent.ingestion.adapters.base import RAW_STORE_COLUMNS

RAW = Path(__file__).parents[2] / "fixtures" / "raw"
OSHER_PRICES = RAW / "osherad" / "price_file_osher_ad.csv"
OSHER_STORES = RAW / "osherad" / "store_file_osher_ad.csv"
SHUFERSAL_PRICES = RAW / "shufersal" / "price_full_file_shufersal.csv"
SHUFERSAL_STORES = RAW / "shufersal" / "store_file_shufersal.csv"

PLACEHOLDERS = ["''", "", "לא ידוע", "XXXXX"]


def stage_fixture(raw_dir: Path, source: Path, **first_row: str) -> Path:
    """Copy a fixture CSV under ``raw_dir``, replacing cells of its first row.

    The first row of a fixture holds every value, and the rows below repeat it
    wherever their cells are empty, so a replaced value spreads down the file.
    """
    frame = pl.read_csv(source, infer_schema=False)
    first = pl.int_range(pl.len()) == 0
    frame = frame.with_columns(
        pl.when(first).then(pl.lit(value)).otherwise(pl.col(column)).alias(column)
        for column, value in first_row.items()
    )
    target = raw_dir / source.parent.name / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(target)
    return target


def stage_shufersal(raw_dir: Path, *, stores: bool = True, **first_row: str) -> None:
    """Stage the Shufersal sample, whose first row belongs to a store outside Yesh Hesed."""
    stage_fixture(raw_dir, SHUFERSAL_PRICES, **first_row)
    if stores:
        shutil.copy(SHUFERSAL_STORES, raw_dir / "shufersal" / SHUFERSAL_STORES.name)


def write_header_only(raw_dir: Path, source: Path, columns: tuple[str, ...]) -> None:
    """Write the file ``source`` names under ``raw_dir`` with a header and no rows."""
    target = raw_dir / source.parent.name / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(",".join(columns) + "\n", encoding="utf-8")


def write_store_snapshots(raw_dir: Path, *snapshots: tuple[str, str, str]) -> None:
    """Write an Osher Ad store file with one store per (date, time, store number)."""
    rows = [
        f"snapshot{index}.xml,{date},{time},1,1,{number},store {number},address,6100,0"
        for index, (date, time, number) in enumerate(snapshots)
    ]
    lines = [",".join(RAW_STORE_COLUMNS), *rows]
    target = raw_dir / OSHER_STORES.parent.name / OSHER_STORES.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def row(frame: pl.DataFrame, **where: str) -> dict:
    matches = frame.filter(**where)
    assert matches.height >= 1, where
    return matches.row(0, named=True)


def assert_no_placeholders(frame: pl.DataFrame) -> None:
    text = frame.select(pl.col(pl.String))
    leaked = text.select(pl.all().is_in(PLACEHOLDERS).sum()).row(0)
    assert sum(leaked) == 0


def assert_store_layout(stores: pl.DataFrame, chain: str) -> None:
    assert stores.schema == STORE_SCHEMA
    assert stores.columns == STORE_SCHEMA.names()
    assert stores["chain"].unique().to_list() == [chain]
    assert stores["store_id"].is_unique().all()
    assert stores["store_id"].str.starts_with(f"{chain}:").all()
    assert stores.select(pl.col("store_id", "name", "source_timestamp").null_count().sum()).row(
        0
    ) == (0, 0, 0)
    assert_no_placeholders(stores)


def assert_price_layout(prices: pl.DataFrame, chain: str) -> None:
    assert prices.schema == PRICE_SCHEMA
    assert prices.columns == PRICE_SCHEMA.names()
    assert prices["chain"].unique().to_list() == [chain]
    required = ["store_id", "item_code", "name", "is_weighed", "price", "source_file"]
    assert sum(prices.select(pl.col(required).null_count()).row(0)) == 0
    assert (prices["price"] > 0).all()
    assert_no_placeholders(prices)
