from __future__ import annotations

import logging
import sys


def setup_logging() -> None:
    """Configures structured, sanitized logging for DermAI."""
    log_format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Silence overly verbose external loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("passlib").setLevel(logging.ERROR)
