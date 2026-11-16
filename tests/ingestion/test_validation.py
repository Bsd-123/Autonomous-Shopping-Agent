"""Tests for the Pandera schemas and for a run that fails on drift."""

import polars as pl
import pytest

from canonical_support import RUN_AT, VERSION
from shopping_agent.ingestion import canonical
from shopping_agent.ingestion.__main__ import main
from shopping_agent.ingestion.canonical import SCHEMAS, build_canonical, write_canonical
from shopping_agent.ingestion.validation import (
    PANDERA_SCHEMAS,
    SchemaDriftError,
    validate_tables,
)
from support import OSHER_PRICES, stage_fixture


@pytest.fixture
def tables(raw_dir) -> dict[str, pl.DataFrame]:
    return build_canonical(raw_dir, VERSION, RUN_AT)


def drifted(tables, table: str, change) -> dict[str, pl.DataFrame]:
    return {**tables, table: change(tables[table])}


class TestSchemas:
    def test_each_canonical_table_has_a_pandera_schema(self):
        assert list(PANDERA_SCHEMAS) == list(SCHEMAS)

    def test_the_pandera_schemas_have_the_declared_columns_and_types(self):
        for name, schema in SCHEMAS.items():
            columns = PANDERA_SCHEMAS[name].columns

            assert list(columns) == schema.names(), name
            for column, dtype in schema.items():
                assert columns[column].dtype.type == dtype, f"{name}.{column}"

    def test_an_empty_table_of_the_declared_schema_is_valid(self):
        validate_tables({name: pl.DataFrame(schema=schema) for name, schema in SCHEMAS.items()})

    def test_the_tables_of_a_run_are_valid(self, tables):
        validate_tables(tables)

    def test_a_valid_run_is_recorded_as_passed(self, tables):
        assert tables["ingestion_runs"]["validation_status"].unique().to_list() == ["passed"]


class TestDrift:
    @pytest.mark.parametrize(
        ("table", "change", "message"),
        [
            # Structure
            (
                "prices",
                lambda t: t.with_columns(unit_price=pl.lit(1.0)),
                "prices: column_in_schema",
            ),
            ("stores", lambda t: t.drop("lat"), "stores: column_in_dataframe"),
            ("products", lambda t: t.rename({"name": "item_name"}), "products: column_in"),
            ("prices", lambda t: t.select(t.columns[::-1]), "prices: column_ordered"),
            # Types
            (
                "prices",
                lambda t: t.with_columns(pl.col("price_agorot").cast(pl.Float64)),
                "prices.price_agorot: dtype",
            ),
            (
                "prices",
                lambda t: t.with_columns(pl.col("last_seen_date").cast(pl.String)),
                "prices.last_seen_date: dtype",
            ),
            (
                "prices",
                lambda t: t.with_columns(pl.col("source_timestamp").dt.replace_time_zone(None)),
                "prices.source_timestamp: dtype",
            ),
            (
                "products",
                lambda t: t.with_columns(pl.col("is_weighed").cast(pl.Int8)),
                "products.is_weighed: dtype",
            ),
            # Values
            (
                "prices",
                lambda t: t.with_columns(price_flag=pl.lit("sale")),
                "prices.price_flag: isin",
            ),
            (
                "prices",
                lambda t: t.with_columns(price_agorot=pl.lit(None, dtype=pl.Int64)),
                "prices.price_agorot: not_nullable",
            ),
            (
                "prices",
                lambda t: t.with_columns(store_id=pl.lit("Osher Ad 6")),
                "prices.store_id: str_matches",
            ),
            (
                "stores",
                lambda t: t.with_columns(lat=pl.lit(132.0)),
                "stores.lat: in_range",
            ),
            (
                "stores",
                lambda t: pl.concat([t, t.head(1)]),
                "stores.store_id: field_uniqueness",
            ),
            (
                "products",
                lambda t: t.with_columns(base_unit=pl.lit("litre")),
                "products.base_unit: isin",
            ),
            (
                "products",
                lambda t: t.with_columns(is_weighed=pl.lit(None, dtype=pl.Boolean)),
                "products.is_weighed: not_nullable",
            ),
            (
                "ingestion_runs",
                lambda t: t.with_columns(rows_in=pl.lit(-1, dtype=pl.Int64)),
                "ingestion_runs.rows_in: greater_than_or_equal_to",
            ),
            (
                "ingestion_runs",
                lambda t: t.with_columns(validation_status=pl.lit("fine")),
                "ingestion_runs.validation_status: isin",
            ),
        ],
    )
    def test_a_drifted_table_fails(self, tables, table, change, message):
        with pytest.raises(SchemaDriftError, match=message):
            validate_tables(drifted(tables, table, change))

    def test_every_violation_of_a_run_is_reported_together(self, tables):
        broken = drifted(tables, "prices", lambda t: t.with_columns(price_flag=pl.lit("sale")))
        broken = drifted(broken, "stores", lambda t: t.drop("lon"))

        with pytest.raises(SchemaDriftError) as raised:
            validate_tables(broken)

        assert len(raised.value.failures) == 2
        assert "prices.price_flag" in str(raised.value)
        assert "stores: column_in_dataframe" in str(raised.value)

    def test_the_message_shows_an_example_of_the_bad_value(self, tables):
        broken = drifted(tables, "prices", lambda t: t.with_columns(price_flag=pl.lit("sale")))

        with pytest.raises(SchemaDriftError, match="for example 'sale'"):
            validate_tables(broken)

    def test_a_missing_or_unknown_table_fails(self, tables):
        without_stores = {name: table for name, table in tables.items() if name != "stores"}

        with pytest.raises(SchemaDriftError, match="stores: the table is missing"):
            validate_tables(without_stores)
        with pytest.raises(SchemaDriftError, match="baskets: not a canonical table"):
            validate_tables({**tables, "baskets": tables["stores"]})

    def test_a_zero_price_is_left_to_the_cleaning_rules(self, tables):
        zero = pl.lit(0, dtype=pl.Int64)

        validate_tables(drifted(tables, "prices", lambda t: t.with_columns(price_agorot=zero)))


@pytest.fixture
def drifting_build(monkeypatch):
    """Make the price builder emit a float price and an unknown flag, as after a code drift."""
    original = canonical.build_prices

    def build(*args, **kwargs):
        return original(*args, **kwargs).with_columns(
            pl.col("price_agorot").cast(pl.Float64), price_flag=pl.lit("sale")
        )

    monkeypatch.setattr(canonical, "build_prices", build)


class TestARunWithASchemaViolation:
    def test_the_build_raises(self, raw_dir, drifting_build):
        with pytest.raises(SchemaDriftError, match="prices.price_agorot: dtype"):
            build_canonical(raw_dir, VERSION, RUN_AT)

    def test_no_dataset_is_written(self, raw_dir, tmp_path, drifting_build):
        with pytest.raises(SchemaDriftError):
            write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)

        assert [path.name for path in tmp_path.iterdir()] == ["rejected"]
        assert not list(tmp_path.glob("v=*"))

    def test_the_failed_status_is_recorded(self, raw_dir, tmp_path, drifting_build):
        with pytest.raises(SchemaDriftError):
            write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)

        folder = tmp_path / "rejected" / "v=2026-11-15"
        assert [path.name for path in folder.iterdir()] == ["ingestion_runs.parquet"]
        runs = pl.read_parquet(folder / "ingestion_runs.parquet")
        assert runs["validation_status"].to_list() == ["failed"] * 3
        assert runs["chain"].to_list() == ["maayan_2000", "osher_ad", "yesh_hesed"]
        assert runs["rows_in"].to_list() == [300, 218, 160]
        assert runs["dataset_version"].unique().to_list() == [VERSION]

    def test_the_command_exits_with_an_error(self, raw_dir, tmp_path, capsys, drifting_build):
        status = main(["--raw", str(raw_dir), "--out", str(tmp_path), "--dataset-version", VERSION])

        captured = capsys.readouterr()
        assert status == 1
        assert captured.out == ""
        assert captured.err.startswith("error: schema validation failed: ")
        assert "prices.price_flag" in captured.err
        assert len(captured.err.strip().splitlines()) == 1

    def test_a_previous_good_dataset_is_kept(self, raw_dir, tmp_path, monkeypatch):
        write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)
        original = canonical.build_prices
        monkeypatch.setattr(
            canonical,
            "build_prices",
            lambda *args, **kwargs: original(*args, **kwargs).with_columns(
                price_flag=pl.lit("sale")
            ),
        )

        with pytest.raises(SchemaDriftError):
            write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)

        runs = pl.read_parquet(tmp_path / "v=2026-11-15" / "ingestion_runs.parquet")
        assert runs["validation_status"].unique().to_list() == ["passed"]
        assert sorted(path.name for path in tmp_path.iterdir()) == ["rejected", "v=2026-11-15"]

    def test_a_good_run_leaves_no_rejected_folder(self, raw_dir, tmp_path):
        write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)

        assert [path.name for path in tmp_path.iterdir()] == ["v=2026-11-15"]


class TestADriftedFile:
    """A raw file whose format changed fails the run before any table exists."""

    def test_a_renamed_column_in_a_price_file_fails_the_run(self, tmp_path, capsys):
        raw = tmp_path / "raw"
        staged = stage_fixture(raw, OSHER_PRICES)
        text = staged.read_text(encoding="utf-8")
        staged.write_text(text.replace("itemprice", "item_price", 1), encoding="utf-8")
        out = tmp_path / "out"

        status = main(["--raw", str(raw), "--out", str(out), "--dataset-version", VERSION])

        assert status == 1
        assert "error:" in capsys.readouterr().err
        assert not out.exists() or not list(out.glob("v=*"))
