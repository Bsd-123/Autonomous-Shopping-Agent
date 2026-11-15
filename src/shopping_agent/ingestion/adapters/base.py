"""Shared reading logic for the per-chain adapters.

The Kaggle dataset publishes one folder per chain with CSV files converted
from the chain's XML price files. The CSVs are compressed row to row:

* an empty cell means "same value as the row above";
* a cell holding two single quotes means the value is missing.

``ChainAdapter`` expands that encoding and maps the raw columns to the common
layout in ``layout.py``. Everything that differs between chains lives in the
subclasses: the source folder, placeholder values, and row selection.
"""

from pathlib import Path
from typing import ClassVar

import polars as pl

from shopping_agent.ingestion.adapters.layout import PRICE_SCHEMA, STORE_SCHEMA

DATASET_MISSING = "''"

RAW_STORE_COLUMNS = (
    "file_name",
    "lastupdatedate",
    "lastupdatetime",
    "subchainid",
    "subchainname",
    "storeid",
    "storename",
    "address",
    "city",
    "zipcode",
)

RAW_PRICE_COLUMNS = (
    "file_name",
    "storeid",
    "priceupdatetime",
    "itemcode",
    "itemname",
    "manufacturename",
    "unitqty",
    "quantity",
    "unitofmeasure",
    "bisweighted",
    "qtyinpackage",
    "itemprice",
)


class RawFormatError(ValueError):
    """A raw file is missing or does not have the expected columns."""


def read_expanded(
    path: Path, required: tuple[str, ...], null_markers: frozenset[str] = frozenset()
) -> pl.DataFrame:
    """Read a raw dataset CSV as text and undo its row-to-row compression."""
    if not path.is_file():
        raise RawFormatError(f"raw file not found: {path}")
    raw = pl.read_csv(path, infer_schema=False)
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise RawFormatError(f"{path.name} is missing columns: {', '.join(missing)}")

    markers = sorted(null_markers | {DATASET_MISSING, ""})
    text = pl.all()
    return (
        raw.select(required)
        .with_columns(pl.when(text != "").then(text).name.keep())
        .with_columns(text.forward_fill())
        .with_columns(text.str.strip_chars())
        .with_columns(pl.when(~text.is_in(markers)).then(text).name.keep())
    )


def _file_published_at() -> pl.Expr:
    """When the chain published the price file, read from its name.

    Names end with a date and a time of four or six digits, for example
    ``Price7290103152017-001-006-20261114-090012.xml``. The item's own
    ``priceupdatetime`` is the last price change and can be years old, so it
    says nothing about how fresh the listing is.
    """
    date = pl.col("file_name").str.extract(r"-(\d{8})-\d{4,6}(?:\.|$)", 1)
    time = pl.col("file_name").str.extract(r"-\d{8}-(\d{4,6})(?:\.|$)", 1).str.pad_end(6, "0")
    return (date + time).str.to_datetime("%Y%m%d%H%M%S", strict=False, time_unit="ms")


class ChainAdapter:
    """Reads one chain's raw files and emits rows in the common layout."""

    chain: ClassVar[str]
    source_folder: ClassVar[str]
    source_name: ClassVar[str]
    null_markers: ClassVar[frozenset[str]] = frozenset()

    def read_stores(self, raw_dir: Path) -> pl.DataFrame:
        path = Path(raw_dir) / self.source_folder / f"store_file_{self.source_name}.csv"
        raw = read_expanded(path, RAW_STORE_COLUMNS, self.null_markers)
        # The file can hold several daily snapshots; only the newest is current.
        updated = pl.concat_str("lastupdatedate", "lastupdatetime", separator=" ")
        raw = raw.with_columns(updated.alias("updated"))
        raw = raw.filter(pl.col("updated") == raw["updated"].max())
        raw = self.select_raw_stores(raw)
        return raw.select(
            chain=pl.lit(self.chain),
            store_id=self._store_id(),
            name="storename",
            address="address",
            city="city",
            zipcode="zipcode",
            source_timestamp=pl.col("updated").str.to_datetime(strict=False, time_unit="ms"),
            source_file="file_name",
        ).cast(STORE_SCHEMA)

    def read_prices(self, raw_dir: Path) -> pl.DataFrame:
        folder = Path(raw_dir) / self.source_folder
        candidates = [
            folder / f"{kind}_{self.source_name}.csv" for kind in ("price_full_file", "price_file")
        ]
        paths = [path for path in candidates if path.is_file()]
        if not paths:
            raise RawFormatError(f"no price files for {self.chain} in {folder}")
        raw = pl.concat(
            [read_expanded(path, RAW_PRICE_COLUMNS, self.null_markers) for path in paths]
        )
        return raw.select(
            chain=pl.lit(self.chain),
            store_id=self._store_id(),
            item_code="itemcode",
            name="itemname",
            manufacturer="manufacturename",
            is_weighed=pl.col("bisweighted") == "1",
            quantity=pl.col("quantity").cast(pl.Float64, strict=False),
            unit="unitqty",
            unit_of_measure="unitofmeasure",
            qty_in_package=pl.col("qtyinpackage").cast(pl.Float64, strict=False),
            price=pl.col("itemprice").cast(pl.Decimal(12, 2), strict=False),
            price_updated_at=pl.col("priceupdatetime").str.to_datetime(
                strict=False, time_unit="ms"
            ),
            source_timestamp=_file_published_at(),
            source_file="file_name",
        ).cast(PRICE_SCHEMA)

    def select_raw_stores(self, raw: pl.DataFrame) -> pl.DataFrame:
        """Hook for chains whose files hold more than the chain itself."""
        return raw

    def _store_id(self) -> pl.Expr:
        # Chains pad store numbers differently ("011" and "11"), so the number
        # is normalized and namespaced by chain to stay unique across chains.
        number = pl.col("storeid").cast(pl.Int64, strict=False).cast(pl.String)
        return pl.concat_str(pl.lit(f"{self.chain}:"), number)
