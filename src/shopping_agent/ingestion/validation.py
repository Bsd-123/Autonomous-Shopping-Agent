"""Pandera validation of the canonical tables (architecture section 4).

Every table has a Pandera schema: its exact columns, in order, with their
types, and the rules its values follow. ``validate_tables`` checks all four
tables and collects every violation before failing, so one run reports the
whole drift. A violation fails the run: nothing is built or deployed from it.
"""

from collections.abc import Mapping

import pandera.polars as pa
import polars as pl
from pandera.errors import SchemaError, SchemaErrors

from shopping_agent.ingestion.adapters.layout import SOURCE_TIME

VALIDATION_NOT_RUN = "not_validated"
VALIDATION_PASSED = "passed"
VALIDATION_FAILED = "failed"
VALIDATION_STATUSES = (VALIDATION_NOT_RUN, VALIDATION_PASSED, VALIDATION_FAILED)

PRICE_FLAGS = ("ok", "carried_forward")
BASE_UNITS = ("ml", "g", "unit")
STORE_ID_PATTERN = r"^[a-z0-9][a-z0-9_]*:[0-9]+$"
PolarsType = type[pl.DataType] | pl.DataType
UTC_TIME = pl.Datetime("ms", "UTC")
MAX_FAILURES_SHOWN = 10


class SchemaDriftError(ValueError):
    """A canonical table does not match its schema.

    ``failures`` lists what was found, one line per broken rule.
    """

    def __init__(self, failures: list[str]) -> None:
        self.failures = failures
        shown = failures[:MAX_FAILURES_SHOWN]
        more = len(failures) - len(shown)
        text = "; ".join(shown) + (f"; and {more} more" if more > 0 else "")
        super().__init__(f"schema validation failed: {text}")


def _required(dtype: PolarsType, *checks: pa.Check, unique: bool = False) -> pa.Column:
    return pa.Column(dtype, list(checks), nullable=False, unique=unique)


def _optional(dtype: PolarsType, *checks: pa.Check) -> pa.Column:
    return pa.Column(dtype, list(checks), nullable=True)


def _schema(columns: dict[str, pa.Column]) -> pa.DataFrameSchema:
    # strict: no unknown column. ordered: the columns come in this order.
    # No coercion: a column of another type is drift, not something to repair.
    return pa.DataFrameSchema(columns, strict=True, ordered=True, coerce=False)


PRODUCTS = _schema(
    {
        "product_key": _required(pl.String, pa.Check.str_length(min_value=1), unique=True),
        "barcode": _optional(pl.String, pa.Check.str_matches(r"^[0-9]{13}$")),
        "name": _optional(pl.String),
        "name_normalized": _optional(pl.String),
        "brand": _optional(pl.String),
        "category": _optional(pl.String),
        "size_value": _optional(pl.Float64, pa.Check.ge(0)),
        "size_unit": _optional(pl.String),
        "base_quantity": _optional(pl.Float64, pa.Check.gt(0)),
        "base_unit": _optional(pl.String, pa.Check.isin(BASE_UNITS)),
        "is_weighed": _required(pl.Boolean),
        "attributes": _optional(pl.String),
    }
)

PRICES = _schema(
    {
        "product_key": _required(pl.String, pa.Check.str_length(min_value=1)),
        "store_id": _required(pl.String, pa.Check.str_matches(STORE_ID_PATTERN)),
        # Zero and negative prices are a matter for the cleaning rules, not for the schema.
        "price_agorot": _required(pl.Int64),
        "price_flag": _required(pl.String, pa.Check.isin(PRICE_FLAGS)),
        "last_seen_date": _required(pl.Date),
        "source_timestamp": _required(SOURCE_TIME),
        "ingestion_timestamp": _required(UTC_TIME),
        "dataset_version": _required(pl.String, pa.Check.str_length(min_value=1)),
    }
)

STORES = _schema(
    {
        "store_id": _required(pl.String, pa.Check.str_matches(STORE_ID_PATTERN), unique=True),
        "chain": _required(pl.String, pa.Check.str_length(min_value=1)),
        "name": _optional(pl.String),
        "address": _optional(pl.String),
        "city": _optional(pl.String),
        "lat": _optional(pl.Float64, pa.Check.in_range(-90, 90)),
        "lon": _optional(pl.Float64, pa.Check.in_range(-180, 180)),
        "geocode_status": _required(pl.String, pa.Check.str_length(min_value=1)),
    }
)

INGESTION_RUNS = _schema(
    {
        "dataset_version": _required(pl.String, pa.Check.str_length(min_value=1)),
        "chain": _required(pl.String, pa.Check.str_length(min_value=1), unique=True),
        "validation_status": _required(pl.String, pa.Check.isin(VALIDATION_STATUSES)),
        "rows_in": _required(pl.Int64, pa.Check.ge(0)),
        "rows_rejected": _required(pl.Int64, pa.Check.ge(0)),
        "ingestion_timestamp": _required(UTC_TIME),
        "newest_source_timestamp": _optional(SOURCE_TIME),
    }
)

PANDERA_SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "products": PRODUCTS,
    "prices": PRICES,
    "stores": STORES,
    "ingestion_runs": INGESTION_RUNS,
}


def _failures(table: str, error: SchemaErrors | SchemaError) -> list[str]:
    cases = error.failure_cases
    if not isinstance(cases, pl.DataFrame) or cases.is_empty():
        return [f"{table}: {str(error).splitlines()[0]}"]
    lines: list[str] = []
    grouped = cases.group_by("column", "check", maintain_order=True).agg(
        pl.len().alias("count"), pl.col("failure_case").first().alias("example")
    )
    for row in grouped.iter_rows(named=True):
        where = f"{table}.{row['column']}" if row["column"] else table
        lines.append(f"{where}: {row['check']} (for example {row['example']!r})")
    return lines


def validate_tables(tables: Mapping[str, pl.DataFrame]) -> None:
    """Check every canonical table against its schema; raise ``SchemaDriftError`` on drift."""
    failures: list[str] = []
    for name, schema in PANDERA_SCHEMAS.items():
        if name not in tables:
            failures.append(f"{name}: the table is missing")
            continue
        try:
            schema.validate(tables[name], lazy=True)
        except (SchemaErrors, SchemaError) as error:
            failures.extend(_failures(name, error))
    unknown = sorted(set(tables) - set(PANDERA_SCHEMAS))
    failures.extend(f"{name}: not a canonical table" for name in unknown)
    if failures:
        raise SchemaDriftError(failures)
