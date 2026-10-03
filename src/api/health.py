"""Liveness and readiness checks with intentionally different semantics."""

from src.db import get_connection


def dependency_readiness(service):
    checks = {
        "postgres": False,
        "embedding": False,
        "generation": False,
        "security": False,
    }
    with get_connection() as connection:
        checks["postgres"] = connection.execute("SELECT 1").fetchone()[0] == 1
    pipeline = getattr(service, "retrieval_pipeline", None)
    checks["embedding"] = bool(
        getattr(getattr(pipeline, "retriever", None), "embedding_service", None)
    )
    generator = getattr(service, "generator", None)
    ready = getattr(generator, "ready", None)
    checks["generation"] = bool(ready() if callable(ready) else generator is not None)
    checks["security"] = getattr(service, "security_mode", None) in {
        "baseline", "structured", "layered",
    }
    return checks
