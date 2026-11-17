"""Tests for the prices table partitioned by chain through the folder layout."""

from pathlib import Path

import duckdb
import polars as pl
import pytest

from canonical_support import RUN_AT, VERSION
from shopping_agent.ingestion.adapters import OsherAdAdapter
from shopping_agent.ingestion.canonical import (
    PRICES_SCHEMA,
    CanonicalError,
    build_canonical,
    price_files,
    read_table,
    write_canonical,
)

CHAINS = ["maayan_2000", "osher_ad", "yesh_hesed"]
ROWS = {"maayan_2000": 300, "osher_ad": 218, "yesh_hesed": 160}


@pytest.fixture
def folder(raw_dir, tmp_path) -> Path:
    return write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)


def glob_of(folder: Path) -> str:
    return (folder / "prices" / "*" / "*.parquet").as_posix()


class TestLayout:
    def test_prices_have_one_folder_per_chain(self, folder):
        prices = folder / "prices"

        assert sorted(path.name for path in prices.iterdir()) == [
            f"chain={chain}" for chain in CHAINS
        ]
        for chain in CHAINS:
            assert [path.name for path in (prices / f"chain={chain}").iterdir()] == [
                "prices.parquet"
            ]

    def test_there_is_no_single_prices_file(self, folder):
        assert not (folder / "prices.parquet").exists()
        assert (folder / "prices").is_dir()

    def test_the_other_tables_stay_single_files(self, folder):
        for name in ("products", "stores", "ingestion_runs"):
            assert (folder / f"{name}.parquet").is_file()

    def test_each_file_holds_only_its_chain(self, folder):
        for chain in CHAINS:
            part = pl.read_parquet(
                folder / "prices" / f"chain={chain}" / "prices.parquet", hive_partitioning=False
            )

            assert part.height == ROWS[chain]
            assert part["store_id"].str.starts_with(f"{chain}:").all()
            assert part.schema == PRICES_SCHEMA

    def test_the_chain_is_not_repeated_inside_the_files(self, folder):
        for file in price_files(folder):
            assert "chain" not in pl.read_parquet_schema(file)

    def test_price_files_lists_them_in_chain_order(self, folder):
        assert [file.parent.name for file in price_files(folder)] == [
            f"chain={chain}" for chain in CHAINS
        ]
        assert [file.parent.name for file in price_files(folder, "osher_ad")] == ["chain=osher_ad"]
        assert price_files(folder, "no_such_chain") == []

    def test_a_run_of_one_chain_writes_one_folder(self, raw_dir, tmp_path):
        folder = write_canonical(raw_dir, tmp_path, VERSION, RUN_AT, [OsherAdAdapter()])

        assert [path.name for path in (folder / "prices").iterdir()] == ["chain=osher_ad"]


class TestReadTable:
    def test_the_whole_table_comes_back_unchanged(self, raw_dir, folder):
        built = build_canonical(raw_dir, VERSION, RUN_AT)["prices"]

        read = read_table(folder, "prices")

        assert read.schema == PRICES_SCHEMA
        assert read.height == built.height == sum(ROWS.values())
        order = ["store_id", "product_key", "source_timestamp", "price_agorot"]
        assert read.sort(order).equals(built.sort(order))

    @pytest.mark.parametrize("chain", CHAINS)
    def test_one_chain_can_be_read_alone(self, folder, chain):
        prices = read_table(folder, "prices", chain=chain)

        assert prices.height == ROWS[chain]
        assert prices["store_id"].str.starts_with(f"{chain}:").all()

    def test_a_chain_without_prices_gives_an_empty_table(self, folder):
        prices = read_table(folder, "prices", chain="no_such_chain")

        assert prices.height == 0
        assert prices.schema == PRICES_SCHEMA

    def test_the_other_tables_are_read_too(self, folder):
        assert read_table(folder, "stores").height == 24 + 38 + 22
        assert read_table(folder, "ingestion_runs")["chain"].to_list() == CHAINS

    def test_bad_requests_are_rejected(self, folder):
        with pytest.raises(CanonicalError, match="unknown table 'baskets'"):
            read_table(folder, "baskets")
        with pytest.raises(CanonicalError, match="only prices is partitioned by chain"):
            read_table(folder, "stores", chain="osher_ad")

    @pytest.mark.parametrize("chain", ["*", "osher_*", "../prices", ""])
    def test_a_chain_is_a_name_and_never_a_pattern(self, folder, chain):
        with pytest.raises(CanonicalError, match="is not a chain name"):
            read_table(folder, "prices", chain=chain)


class TestDuckDB:
    def test_the_folder_name_becomes_the_chain_column(self, folder):
        rows = duckdb.sql(
            f"SELECT chain, count(*) FROM read_parquet('{glob_of(folder)}', "
            "hive_partitioning = true) GROUP BY chain ORDER BY chain"
        ).fetchall()

        assert rows == [(chain, ROWS[chain]) for chain in CHAINS]

    def test_a_query_for_one_chain_returns_rows_from_that_chains_file_only(self, folder):
        files = duckdb.sql(
            f"SELECT DISTINCT filename FROM read_parquet('{glob_of(folder)}', "
            "hive_partitioning = true, filename = true) WHERE chain = 'osher_ad'"
        ).fetchall()

        assert len(files) == 1
        assert Path(files[0][0]).parent.name == "chain=osher_ad"

    def test_a_query_for_one_chain_reads_only_that_chains_file(self, folder):
        query = (
            f"SELECT count(*), min(price_agorot) FROM read_parquet('{glob_of(folder)}', "
            "hive_partitioning = true) WHERE chain = 'osher_ad'"
        )

        plan = "\n".join(row[-1] for row in duckdb.sql(f"EXPLAIN ANALYZE {query}").fetchall())

        # DuckDB reports how many of the files it scans and names the ones it read.
        assert "Scanning Files: 1/3" in plan
        assert "chain=osher_ad/prices.parquet" in plan
        assert "chain=maayan_2000" not in plan
        assert "chain=yesh_hesed" not in plan

    def test_a_file_of_another_chain_is_never_opened(self, folder):
        # Yesh Hesed's file is replaced with bytes that are not Parquet: opening it
        # would fail the query. (The first file of the listing, Maayan 2000's, is left
        # whole, because DuckDB reads its footer to learn the column names.)
        (folder / "prices" / "chain=yesh_hesed" / "prices.parquet").write_bytes(b"not parquet")
        query = (
            f"SELECT count(*), min(price_agorot) FROM read_parquet('{glob_of(folder)}', "
            "hive_partitioning = true) WHERE chain = '{chain}'"
        )

        ((count, cheapest),) = duckdb.sql(query.format(chain="osher_ad")).fetchall()

        assert count == ROWS["osher_ad"]
        assert cheapest > 0
        with pytest.raises(duckdb.Error):
            duckdb.sql(query.format(chain="yesh_hesed")).fetchall()

    def test_prices_join_the_other_tables(self, folder):
        rows = duckdb.sql(
            f"SELECT count(*) FROM read_parquet('{glob_of(folder)}', hive_partitioning = true) p "
            f"JOIN read_parquet('{(folder / 'stores.parquet').as_posix()}') s USING (store_id) "
            "WHERE p.chain = 'yesh_hesed' AND s.chain = 'yesh_hesed'"
        ).fetchall()

        assert rows == [(ROWS["yesh_hesed"],)]


def test_a_chain_that_cannot_be_a_folder_name_is_rejected(raw_dir, tmp_path, monkeypatch):
    from shopping_agent.ingestion import canonical

    original = canonical.build_canonical

    def with_a_bad_store_id(*args, **kwargs):
        tables = original(*args, **kwargs)
        tables["prices"] = tables["prices"].with_columns(store_id=pl.lit("../escape:1"))
        return tables

    monkeypatch.setattr(canonical, "build_canonical", with_a_bad_store_id)

    with pytest.raises(CanonicalError, match="cannot partition prices by the chain"):
        write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)

    assert list(tmp_path.iterdir()) == []
