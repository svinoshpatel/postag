"""Run pipeline stages from the repository root."""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from crf.pipeline import DevelopmentIncomplete, develop
from crf.utils.config import load_config
from crf.utils.logging import configure_logging

logger = logging.getLogger("crf.cli")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    development = commands.add_parser("develop", help="Run development search")
    development.add_argument("--config", type=Path, required=True)
    development.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override configuration seed (default 42)",
    )
    args = parser.parse_args(argv)
    configure_logging()
    try:
        root = Path.cwd()
        config = load_config(args.config, repository_root=root, seed=args.seed)
        result = develop(config, root)
    except DevelopmentIncomplete as exc:
        logger.error("%s", exc)
        print(json.dumps(dict(status="failed", run_id=exc.run_id, stage=exc.stage)))
        return 1
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    print(
        json.dumps(
            dict(
                status="search_complete_pending_freeze",
                run_id=result.run_id,
                trials_path=str(result.trials_path),
            ),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
