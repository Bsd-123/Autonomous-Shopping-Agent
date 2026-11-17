"""Shared test fixtures."""

from pathlib import Path

import pytest

FIXTURES_RAW = Path(__file__).parent / "fixtures" / "raw"


@pytest.fixture
def raw_dir() -> Path:
    """Sample files cut from each chain's real data."""
    return FIXTURES_RAW
