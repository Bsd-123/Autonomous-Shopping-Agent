"""Tests for writing the canonical dataset: the folder, its version and its timestamp."""

import os
import stat
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from canonical_support import RUN_AT, VERSION
from shopping_agent.ingestion.adapters import ADAPTERS, OsherAdAdapter
from shopping_agent.ingestion.canonical import (
    SCHEMAS,
    CanonicalError,
    build_canonical,
    build_prices,
    read_table,
    require_integrity,
    write_canonical,
)
from support import RAW


def test_write_canonical_writes_the_four_tables_to_a_dated_folder(raw_dir, tmp_path):
    folder = write_canonical(raw_dir, tmp_path / "canonical", VERSION, RUN_AT)

    assert folder == tmp_path / "canonical" / "v=2026-11-15"
    assert sorted(path.name for path in folder.iterdir()) == [
        "ingestion_runs.parquet",
        "prices",
        "products.parquet",
        "stores.parquet",
    ]
    for name, schema in SCHEMAS.items():
        table = read_table(folder, name)
        assert dict(table.schema) == schema
        assert table.height > 0


def test_dataset_version_defaults_to_the_run_date(raw_dir, tmp_path):
    folder = write_canonical(raw_dir, tmp_path / "canonical", ingestion_timestamp=RUN_AT)

    assert folder.name == "v=2026-11-15"


class TestWriteIsAtomic:
    @staticmethod
    def fail_on_the_third_write(monkeypatch):
        writes = []
        original = pl.DataFrame.write_parquet

        def flaky(self, *args, **kwargs):
            writes.append(1)
            if len(writes) == 3:
                raise OSError("disk full")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(pl.DataFrame, "write_parquet", flaky)

    def test_a_failed_new_version_leaves_nothing_behind(self, raw_dir, tmp_path, monkeypatch):
        self.fail_on_the_third_write(monkeypatch)

        with pytest.raises(OSError, match="disk full"):
            write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)

        assert list(tmp_path.iterdir()) == []

    def test_a_failed_rerun_keeps_the_previous_dataset_whole(self, raw_dir, tmp_path, monkeypatch):
        write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)
        self.fail_on_the_third_write(monkeypatch)

        with pytest.raises(OSError, match="disk full"):
            write_canonical(raw_dir, tmp_path, VERSION, RUN_AT + timedelta(hours=6))

        assert [path.name for path in tmp_path.iterdir()] == ["v=2026-11-15"]
        for name in ("prices", "ingestion_runs"):
            table = read_table(tmp_path / "v=2026-11-15", name)
            assert table["ingestion_timestamp"].unique().to_list() == [RUN_AT]

    def test_a_rerun_replaces_every_file_of_the_dataset(self, raw_dir, tmp_path):
        later = RUN_AT + timedelta(hours=6)
        write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)

        write_canonical(raw_dir, tmp_path, VERSION, later)

        assert [path.name for path in tmp_path.iterdir()] == ["v=2026-11-15"]
        for name in ("prices", "ingestion_runs"):
            table = read_table(tmp_path / "v=2026-11-15", name)
            assert table["ingestion_timestamp"].unique().to_list() == [later]

    def test_a_failed_swap_restores_the_previous_dataset(self, raw_dir, tmp_path, monkeypatch):
        write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)
        original = Path.rename

        def refuse_staging(self, target):
            if self.name.endswith(".tmp"):
                raise OSError("rename failed")
            return original(self, target)

        monkeypatch.setattr(Path, "rename", refuse_staging)

        with pytest.raises(OSError, match="rename failed"):
            write_canonical(raw_dir, tmp_path, VERSION, RUN_AT + timedelta(hours=6))

        assert [path.name for path in tmp_path.iterdir()] == ["v=2026-11-15"]
        prices = read_table(tmp_path / "v=2026-11-15", "prices")
        assert prices["ingestion_timestamp"].unique().to_list() == [RUN_AT]

    def test_a_failed_swap_of_a_new_version_leaves_nothing_behind(
        self, raw_dir, tmp_path, monkeypatch
    ):
        original = Path.rename

        def refuse_staging(self, target):
            if self.name.endswith(".tmp"):
                raise OSError("rename failed")
            return original(self, target)

        monkeypatch.setattr(Path, "rename", refuse_staging)

        with pytest.raises(OSError, match="rename failed"):
            write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)

        assert list(tmp_path.iterdir()) == []

    def test_no_version_folder_is_visible_while_the_files_are_written(
        self, raw_dir, tmp_path, monkeypatch
    ):
        seen = []
        original = pl.DataFrame.write_parquet

        def spy(self, file, *args, **kwargs):
            seen.append(sorted(path.name for path in tmp_path.iterdir()))
            return original(self, file, *args, **kwargs)

        monkeypatch.setattr(pl.DataFrame, "write_parquet", spy)

        write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)

        assert seen and all(not name.startswith("v=") for names in seen for name in names)


class TestDatasetVersion:
    @pytest.mark.parametrize("version", ["2026-11-15", "2026-11-15.2", "rc_1", "7"])
    def test_a_plain_token_is_accepted(self, raw_dir, version):
        assert build_canonical(raw_dir, version, RUN_AT)["prices"]["dataset_version"][0] == version

    @pytest.mark.parametrize("version", ["", " ", "a b", "../x", "a/b", "a\\b", ".hidden", "-x"])
    def test_anything_else_is_rejected_before_a_file_is_written(self, raw_dir, tmp_path, version):
        with pytest.raises(CanonicalError, match="invalid dataset version"):
            write_canonical(raw_dir, tmp_path / "out", version, RUN_AT)

        assert not (tmp_path / "out").exists()

    def test_the_default_is_the_utc_date_of_the_run(self, raw_dir, tmp_path):
        # 00:30 in Israel is still the day before in UTC.
        israel = timezone(timedelta(hours=2))

        folder = write_canonical(
            raw_dir, tmp_path, ingestion_timestamp=datetime(2026, 11, 15, 0, 30, tzinfo=israel)
        )

        assert folder.name == "v=2026-11-14"


class TestIngestionTimestamp:
    def test_a_timestamp_without_a_time_zone_is_rejected(self, raw_dir, tmp_path):
        naive = datetime(2026, 11, 15, 3, 0)

        with pytest.raises(CanonicalError, match="timezone-aware"):
            build_canonical(raw_dir, VERSION, naive)
        with pytest.raises(CanonicalError, match="timezone-aware"):
            build_prices(OsherAdAdapter().read_prices(RAW), VERSION, naive)
        with pytest.raises(CanonicalError, match="timezone-aware"):
            write_canonical(raw_dir, tmp_path, ingestion_timestamp=naive)


def test_overlapping_runs_of_one_version_never_mix_their_files(raw_dir, tmp_path, monkeypatch):
    first_a, first_b, a_finished = threading.Event(), threading.Event(), threading.Event()
    original = pl.DataFrame.write_parquet
    errors: dict[str, str] = {}

    def interleaved(self, file, *args, **kwargs):
        original(self, file, *args, **kwargs)
        run = threading.current_thread().name
        # Run A stops after its first file until B has written one, so their writes
        # interleave. B then waits until A has finished and publishes last.
        if run == "A" and not first_a.is_set():
            first_a.set()
            first_b.wait(30)
        elif run == "B" and not first_b.is_set():
            first_b.set()
            a_finished.wait(30)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", interleaved)

    def run(name, adapters, hour):
        try:
            write_canonical(raw_dir, tmp_path, VERSION, RUN_AT.replace(hour=hour), adapters)
        except Exception as error:  # reported through the assertion below
            errors[name] = repr(error)

    run_a = threading.Thread(target=run, name="A", args=("A", ADAPTERS, 3))
    run_b = threading.Thread(target=run, name="B", args=("B", [OsherAdAdapter()], 9))
    run_a.start()
    first_a.wait(30)
    run_b.start()
    run_a.join()
    a_finished.set()
    run_b.join()

    assert errors == {}
    folder = tmp_path / "v=2026-11-15"
    tables = {name: read_table(folder, name) for name in SCHEMAS}
    require_integrity(tables)
    # Every file comes from run B, the last to finish, which covers only Osher Ad.
    assert set(tables["ingestion_runs"]["chain"]) == {"osher_ad"}
    assert tables["prices"]["ingestion_timestamp"].unique().to_list() == [RUN_AT.replace(hour=9)]
    assert [path.name for path in tmp_path.iterdir()] == ["v=2026-11-15"]


def test_leftovers_of_a_killed_run_do_not_block_the_next_run(raw_dir, tmp_path):
    write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)
    for leftover in (".v=2026-11-15.old", ".v=2026-11-15.killed.tmp"):
        (tmp_path / leftover).mkdir()
        (tmp_path / leftover / "prices.parquet").write_text("half a file")
    later = RUN_AT + timedelta(hours=6)

    write_canonical(raw_dir, tmp_path, VERSION, later)

    prices = read_table(tmp_path / "v=2026-11-15", "prices")
    assert prices["ingestion_timestamp"].unique().to_list() == [later]


@pytest.mark.skipif(os.name != "posix", reason="Windows has no permission bits to compare")
def test_the_dataset_folder_has_the_mode_of_a_plain_folder(raw_dir, tmp_path):
    # tempfile.mkdtemp would make it owner-only, which a container user that is not the
    # owner could not read.
    plain = tmp_path / "plain"
    plain.mkdir()

    folder = write_canonical(raw_dir, tmp_path, VERSION, RUN_AT)

    assert stat.S_IMODE(folder.stat().st_mode) == stat.S_IMODE(plain.stat().st_mode)
