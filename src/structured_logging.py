"""Privacy-aware JSON logging correlated with OpenTelemetry traces."""

import hashlib
import logging
import sys

from pythonjsonlogger.json import JsonFormatter

from src.config import LOG_LEVEL
from src.tracing import trace_id


def configure_logging(stream=None):
    logger = logging.getLogger("distributed_rag")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(event)s %(trace_id)s %(message)s"
    ))
    logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False
    return logger


LOGGER = configure_logging()


def query_metadata(query):
    return {
        "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest()[:16],
        "query_length": len(query),
    }


def log_event(event, level=logging.INFO, **metadata):
    LOGGER.log(level, event, extra={"event": event, "trace_id": trace_id(), **metadata})
