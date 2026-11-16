"""Tests for the command line entry point."""

import re
import runpy
import sys

import pytest

import shopping_agent.ingestion.__main__ as entry_point
from shopping_agent.ingestion.__main__ import main
from shopping_agent.ingestion.canonical import read_table


def test_it_writes_the_dataset_and_reports_the_folder(raw_dir, tmp_path, capsys):
    status = main(
        ["--raw", str(raw_dir), "--out", str(tmp_path), "--dataset-version", "2026-11-15"]
    )

    folder = tmp_path / "v=2026-11-15"
    assert status == 0
    assert capsys.readouterr().out.strip() == f"wrote canonical dataset to {folder}"
    assert read_table(folder, "prices").height == 218 + 300 + 160


def test_the_version_defaults_to_the_run_date(raw_dir, tmp_path):
    assert main(["--raw", str(raw_dir), "--out", str(tmp_path)]) == 0

    (folder,) = tmp_path.iterdir()
    assert re.fullmatch(r"v=\d{4}-\d{2}-\d{2}", folder.name)


def test_a_missing_raw_folder_is_reported_without_a_traceback(tmp_path, capsys):
    status = main(["--raw", str(tmp_path / "missing"), "--out", str(tmp_path / "out")])

    error = capsys.readouterr().err
    assert status == 1
    assert error.startswith("error: raw file not found")
    assert "Traceback" not in error
    assert not (tmp_path / "out").exists()


def test_an_invalid_version_is_reported_and_writes_nothing(raw_dir, tmp_path, capsys):
    out = tmp_path / "a" / "b"

    status = main(["--raw", str(raw_dir), "--out", str(out), "--dataset-version", "../../../x"])

    assert status == 1
    assert "invalid dataset version" in capsys.readouterr().err
    assert not out.exists()
    assert not (tmp_path / "x").exists()


def test_running_the_module_as_a_script_exits_with_the_status(raw_dir, tmp_path, monkeypatch):
    arguments = ["--raw", str(raw_dir), "--out", str(tmp_path), "--dataset-version", "2026-11-15"]
    monkeypatch.setattr(sys, "argv", ["shopping_agent.ingestion", *arguments])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(entry_point.__file__, run_name="__main__")

    assert exit_info.value.code == 0
    assert (tmp_path / "v=2026-11-15").is_dir()
