"""Shared reading logic for the per-chain adapters.

The Kaggle dataset publishes one folder per chain with CSV files converted
from the chain's XML price files. The CSVs are compressed row to row:

* an empty cell means "same value as the row above";
* a cell holding two single quotes means the value is missing.

``ChainAdapter`` expands that encoding and maps the raw columns to the common
layout in ``layout.py``. Everything that differs between chains lives in the
subclasses: the source folder, placeholder values, and row selection.
"""

from collections.abc import Mapping
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


PRICE_PATTERN = r"^-?\d+(\.\d{1,2})?$"


class RawFormatError(ValueError):
    """A raw file is missing, lacks expected columns, or holds values that do not parse."""


def require_parsed(raw: pl.DataFrame, parsed: pl.DataFrame, columns: Mapping[str, str]) -> None:
    """Fail when a value present in a raw column is null after parsing.

    ``raw`` and ``parsed`` must be row-aligned. ``columns`` maps each raw column
    to the output column parsed from it.
    """
    for raw_column, column in columns.items():
        lost = raw[raw_column].is_not_null() & parsed[column].is_null()
        if lost.any():
            bad = raw.filter(lost)
            raise RawFormatError(
                f"{bad['file_name'][0]}: {bad.height} {raw_column} value(s) do not parse, "
                f"first: {bad[raw_column][0]!r}"
            )


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
        undated = raw["updated"].null_count()
        if undated:
            raise RawFormatError(f"{path.name} has {undated} store row(s) without an update time")
        raw = raw.filter(pl.col("updated") == raw["updated"].max())
        raw = self.select_raw_stores(raw)
        stores = raw.select(
            chain=pl.lit(self.chain),
            store_id=self._store_id(),
            name="storename",
            address="address",
            city="city",
            zipcode="zipcode",
            source_timestamp=pl.col("updated").str.to_datetime(strict=False, time_unit="ms"),
            source_file="file_name",
        )
        require_parsed(raw, stores, {"storeid": "store_id", "updated": "source_timestamp"})
        return stores.cast(STORE_SCHEMA)

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
        weighed = pl.col("bisweighted")
        price = pl.col("itemprice")
        prices = raw.select(
            chain=pl.lit(self.chain),
            store_id=self._store_id(),
            item_code="itemcode",
            name="itemname",
            manufacturer="manufacturename",
            is_weighed=pl.when(weighed == "1").then(True).when(weighed == "0").then(False),
            quantity=pl.col("quantity").cast(pl.Float64, strict=False),
            unit="unitqty",
            unit_of_measure="unitofmeasure",
            qty_in_package=pl.col("qtyinpackage").cast(pl.Float64, strict=False),
            # The Decimal cast truncates extra digits, so the format is checked first.
            price=pl.when(price.str.contains(PRICE_PATTERN)).then(
                price.cast(pl.Decimal(12, 2), strict=False)
            ),
            source_timestamp=pl.col("priceupdatetime").str.to_datetime(
                strict=False, time_unit="ms"
            ),
            source_file="file_name",
        )
        require_parsed(
            raw,
            prices,
            {
                "storeid": "store_id",
                "bisweighted": "is_weighed",
                "quantity": "quantity",
                "qtyinpackage": "qty_in_package",
                "itemprice": "price",
                "priceupdatetime": "source_timestamp",
            },
        )
        return prices.cast(PRICE_SCHEMA)

    def select_raw_stores(self, raw: pl.DataFrame) -> pl.DataFrame:
        """Hook for chains whose files hold more than the chain itself."""
        return raw

    def _store_id(self) -> pl.Expr:
        # Chains pad store numbers differently ("011" and "11"), so the number
        # is normalized and namespaced by chain to stay unique across chains.
        number = pl.col("storeid").cast(pl.Int64, strict=False).cast(pl.String)
        return pl.concat_str(pl.lit(f"{self.chain}:"), number)
