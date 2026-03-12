"""
Centralized logging configuration - Windows compatible.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging(log_level: str = "INFO", log_file: str = "logs/system.log"):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Console handler - force UTF-8 on Windows
    if sys.platform == "win32":
        import io
        stream = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True
        )
    else:
        stream = sys.stdout

    ch = logging.StreamHandler(stream)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file handler
    fh = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024,
        backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    for noisy in ["websockets", "asyncio", "urllib3", "httpx"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)