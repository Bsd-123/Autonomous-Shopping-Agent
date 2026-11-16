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
from typing import ClassVar, NamedTuple

import polars as pl

from shopping_agent.ingestion.adapters.layout import (
    ISRAEL_TIME_ZONE,
    PRICE_SCHEMA,
    STORE_SCHEMA,
)

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

REQUIRED_SETTINGS = ("chain", "source_folder", "source_name")

# At most two decimal places, because the Decimal cast would silently truncate
# more. The sign is not judged here: plausible prices are the cleaning step's job.
PRICE_PATTERN = r"^-?\d+(\.\d{1,2})?$"


class RawFormatError(ValueError):
    """A raw file is missing, lacks expected columns, or holds values that do not parse."""


class Parsed(NamedTuple):
    """An output column computed from one raw column and checked for parse failures."""

    source: str
    expression: pl.Expr


def parse_local_time(raw_time: pl.Expr) -> pl.Expr:
    """Parse a published timestamp, which carries no offset, as Israel local time.

    A time that does not exist (the hour skipped when clocks go forward) becomes
    null and so fails the parse check; an ambiguous one (the repeated hour) is
    read as the earlier of the two.
    """
    return raw_time.str.to_datetime(strict=False, time_unit="ms").dt.replace_time_zone(
        ISRAEL_TIME_ZONE, ambiguous="earliest", non_existent="null"
    )


def require_parsed(raw: pl.DataFrame, parsed: pl.DataFrame, sources: Mapping[str, str]) -> None:
    """Fail when a value present in a raw column is null after parsing.

    ``raw`` and ``parsed`` must be row-aligned (they may be the same frame).
    ``sources`` maps each parsed column to the raw column it was parsed from.
    """
    for column, source in sources.items():
        lost = raw[source].is_not_null() & parsed[column].is_null()
        if lost.any():
            bad = raw.filter(lost)
            raise RawFormatError(
                f"{bad['file_name'][0]}: {bad.height} {source} value(s) do not parse, "
                f"first: {bad[source][0]!r}"
            )


def build_frame(
    raw: pl.DataFrame, columns: Mapping[str, str | pl.Expr | Parsed], schema: pl.Schema
) -> pl.DataFrame:
    """Select ``columns`` from ``raw`` in the order and types of ``schema``.

    Each column is a raw column name, an expression, or a ``Parsed`` column. The
    parsed columns are the only place a conversion can lose a value, and each is
    checked here, so a column added to the layout cannot skip the check.
    """
    frame = raw.select(
        **{
            name: spec.expression if isinstance(spec, Parsed) else spec
            for name, spec in columns.items()
        }
    )
    require_parsed(
        raw,
        frame,
        {name: spec.source for name, spec in columns.items() if isinstance(spec, Parsed)},
    )
    return frame.select(schema.names()).cast(schema)


def read_expanded(
    path: Path, required: tuple[str, ...], null_markers: frozenset[str] = frozenset()
) -> pl.DataFrame:
    """Read the required columns of a raw CSV as text and undo its row-to-row compression."""
    if not path.is_file():
        raise RawFormatError(f"raw file not found: {path}")
    lazy = pl.scan_csv(path, infer_schema=False)
    present = lazy.collect_schema().names()
    missing = [column for column in required if column not in present]
    if missing:
        raise RawFormatError(f"{path.name} is missing columns: {', '.join(missing)}")

    markers = sorted(null_markers | {DATASET_MISSING, ""})
    text = pl.all()
    # One lazy plan, so the large files are not copied once per step.
    expanded = pl.when(text != "").then(text).forward_fill().str.strip_chars()
    return (
        lazy.select(list(required))
        .select(pl.when(~expanded.is_in(markers)).then(expanded).name.keep())
        .collect()
    )


class ChainAdapter:
    """Reads one chain's raw files and emits rows in the common layout."""

    chain: ClassVar[str]
    source_folder: ClassVar[str]
    source_name: ClassVar[str]
    null_markers: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        unset = [name for name in REQUIRED_SETTINGS if not hasattr(cls, name)]
        if unset:
            raise TypeError(f"{cls.__name__} must set: {', '.join(unset)}")

    def read_stores(self, raw_dir: Path) -> pl.DataFrame:
        path = raw_dir / self.source_folder / f"store_file_{self.source_name}.csv"
        raw = read_expanded(path, RAW_STORE_COLUMNS, self.null_markers)
        updated = pl.concat_str("lastupdatedate", "lastupdatetime", separator=" ")
        raw = raw.with_columns(updated.alias("updated"))
        undated = raw["updated"].null_count()
        if undated:
            raise RawFormatError(f"{path.name} has {undated} store row(s) without an update time")
        # Every row is validated, so an unreadable time in an older snapshot is not
        # skipped. The newest snapshot is picked by parsed time, never by text order.
        snapshots = raw.with_columns(parse_local_time(pl.col("updated")).alias("updated_at"))
        require_parsed(snapshots, snapshots, {"updated_at": "updated"})
        # The file can hold several daily snapshots; only the newest is current.
        newest = snapshots.filter(pl.col("updated_at") == pl.col("updated_at").max())
        return build_frame(
            self.select_raw_stores(newest),
            {
                "chain": pl.lit(self.chain),
                "store_id": Parsed("storeid", self._store_id()),
                "name": "storename",
                "address": "address",
                "city": "city",
                "zipcode": "zipcode",
                "source_timestamp": "updated_at",
                "source_file": "file_name",
            },
            STORE_SCHEMA,
        )

    def read_prices(self, raw_dir: Path) -> pl.DataFrame:
        return self._parse_prices(self._read_raw_prices(raw_dir))

    def select_raw_stores(self, raw: pl.DataFrame) -> pl.DataFrame:
        """Hook for chains whose files hold more than the chain itself."""
        return raw

    def _read_raw_prices(self, raw_dir: Path) -> pl.DataFrame:
        folder = raw_dir / self.source_folder
        candidates = [
            folder / f"{kind}_{self.source_name}.csv" for kind in ("price_full_file", "price_file")
        ]
        paths = [path for path in candidates if path.is_file()]
        if not paths:
            raise RawFormatError(f"no price files for {self.chain} in {folder}")
        return pl.concat(
            [read_expanded(path, RAW_PRICE_COLUMNS, self.null_markers) for path in paths]
        )

    def _parse_prices(self, raw: pl.DataFrame) -> pl.DataFrame:
        weighed = pl.col("bisweighted")
        price = pl.col("itemprice")
        return build_frame(
            raw,
            {
                "chain": pl.lit(self.chain),
                "store_id": Parsed("storeid", self._store_id()),
                "item_code": "itemcode",
                "name": "itemname",
                "manufacturer": "manufacturename",
                "is_weighed": Parsed(
                    "bisweighted",
                    pl.when(weighed == "1").then(True).when(weighed == "0").then(False),
                ),
                "quantity": Parsed("quantity", pl.col("quantity").cast(pl.Float64, strict=False)),
                "unit": "unitqty",
                "unit_of_measure": "unitofmeasure",
                "qty_in_package": Parsed(
                    "qtyinpackage", pl.col("qtyinpackage").cast(pl.Float64, strict=False)
                ),
                "price": Parsed(
                    "itemprice",
                    pl.when(price.str.contains(PRICE_PATTERN)).then(
                        price.cast(pl.Decimal(12, 2), strict=False)
                    ),
                ),
                "source_timestamp": Parsed(
                    "priceupdatetime", parse_local_time(pl.col("priceupdatetime"))
                ),
                "source_file": "file_name",
            },
            PRICE_SCHEMA,
        )

    def _store_id(self) -> pl.Expr:
        # Chains pad store numbers differently ("011" and "11"), so the number
        # is normalized and namespaced by chain to stay unique across chains.
        number = pl.col("storeid").cast(pl.Int64, strict=False).cast(pl.String)
        return pl.concat_str(pl.lit(f"{self.chain}:"), number)
