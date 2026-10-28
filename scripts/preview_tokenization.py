"""Preview the low-level tokenizer on supplied text (not a pipeline CLI)."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict

from crf.data.tokenization import tokenize_paragraph


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Paragraph to preview as JSON")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            [asdict(item) for item in tokenize_paragraph(args.text)], ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
