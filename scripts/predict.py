"""Predict raw text with one recorded standalone model bundle."""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from crf.inference import predict
from crf.utils.logging import configure_logging

logger = logging.getLogger("crf.predict")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    configure_logging()
    try:
        result = predict(args.model, args.input, Path.cwd())
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    print(
        json.dumps(
            dict(
                status="complete",
                run_id=result.run_id,
                predictions=str(result.predictions_path),
                sentence_count=result.sentence_count,
                token_count=result.token_count,
                label_space=result.label_space,
            ),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
