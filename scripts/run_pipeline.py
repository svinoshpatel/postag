"""Run pipeline stages from the repository root."""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from crf.final_evaluation import evaluate_core
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
    evaluation = commands.add_parser("evaluate", help="Evaluate a frozen run")
    evaluation.add_argument("--run-id", required=True)
    evaluation.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    configure_logging()
    try:
        root = Path.cwd()
        if args.command == "evaluate":
            core = evaluate_core(root, args.run_id, seed=args.seed)
            print(
                json.dumps(
                    dict(
                        status="predictions_complete_pending_bootstrap",
                        run_id=core.experiment.run_id,
                    )
                )
            )
            return 0
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
                status="complete",
                run_id=result.run_id,
                run_path=str(result.run_path),
            ),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
