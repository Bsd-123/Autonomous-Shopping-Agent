"""Command line entry point: ``python -m shopping_agent.ingestion``."""

import argparse
from pathlib import Path

from shopping_agent.ingestion.canonical import write_canonical


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical Parquet dataset.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"), help="raw dataset folder")
    parser.add_argument("--out", type=Path, default=Path("data/canonical"), help="output folder")
    parser.add_argument("--version", help="dataset version, default today's date (UTC)")
    args = parser.parse_args()
    folder = write_canonical(args.raw, args.out, args.version)
    print(f"wrote canonical dataset to {folder}")


if __name__ == "__main__":
    main()
