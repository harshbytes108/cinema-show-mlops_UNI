import json
import logging
import sys
from datetime import datetime, timezone


def get_json_logger(name: str = "mp183_api") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_prediction(logger: logging.Logger, payload: dict, prediction: float, latency_ms: float) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "prediction",
        "input": payload,
        "prediction": round(float(prediction), 4),
        "latency_ms": round(float(latency_ms), 2),
    }
    logger.info(json.dumps(record))
