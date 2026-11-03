"""Download the pinned auxiliary dict_uk release without replacing existing files."""

import argparse
import sys
from collections.abc import Sequence
from http.client import HTTPException
from pathlib import Path

from crf.utils.dictionary import fetch_dictionary
from crf.utils.logging import configure_logging


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/dict_uk/source.json"),
        help="manifest directly under repository data/dict_uk (default: %(default)s)",
    )
    arguments = parser.parse_args(argv)
    configure_logging()
    try:
        destination = fetch_dictionary(
            Path(__file__).resolve().parent.parent, source_path=arguments.source
        )
    except (OSError, ValueError, HTTPException) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
