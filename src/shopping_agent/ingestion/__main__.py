"""Command line entry point: ``python -m shopping_agent.ingestion``."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from shopping_agent.ingestion.adapters import RawFormatError
from shopping_agent.ingestion.canonical import CanonicalError, write_canonical


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the canonical Parquet dataset.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"), help="raw dataset folder")
    parser.add_argument("--out", type=Path, default=Path("data/canonical"), help="output folder")
    parser.add_argument(
        "--dataset-version", help="version of the dataset, default today's date (UTC)"
    )
    args = parser.parse_args(argv)
    try:
        folder = write_canonical(args.raw, args.out, args.dataset_version)
    except (RawFormatError, CanonicalError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote canonical dataset to {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
