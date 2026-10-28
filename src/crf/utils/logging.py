"""Configure pipeline diagnostics on stderr without changing unrelated loggers."""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Install one stderr handler for the crf hierarchy; callers own stdout data."""
    logger = logging.getLogger("crf")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
