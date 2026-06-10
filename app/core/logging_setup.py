"""
Single logging configuration for the entire application.
Call configure_logging() once at startup. All modules use
logging.getLogger(__name__) — never basicConfig().
"""
import logging
import sys
from pythonjsonlogger import jsonlogger


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:
        return  # Already configured

    handler = logging.StreamHandler(sys.stdout)
    fmt = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    handler.setFormatter(fmt)

    file_handler = logging.FileHandler("app.log", mode="a")
    file_handler.setFormatter(fmt)

    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)
    root.addHandler(file_handler)

    # Quiet noisy third-party loggers
    for noisy in ("urllib3", "insightface", "onnxruntime", "mediapipe",
                  "deepface", "pymilvus"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
