"""Minimal HTTP boundary used only for repeatable load experiments."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import METRICS_PORT
from src.runtime_metrics import (
    GOODPUT, HTTP_REQUESTS, IN_FLIGHT, RESILIENCE_EVENTS, start_metrics_server,
)
from src.resilience import DependencyUnavailableError


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


def create_app(service=None):
    @asynccontextmanager
    async def lifespan(application):
        if service is None:
            from src.rag_service import RAGService
            start_metrics_server(METRICS_PORT)
            application.state.rag = RAGService()
        yield

    application = FastAPI(title="Distributed RAG load boundary", lifespan=lifespan)
    application.state.rag = service

    @application.get("/health")
    def health():
        return {"status": "ok"}

    @application.get("/ready")
    def ready():
        if application.state.rag is None:
            raise HTTPException(status_code=503, detail="RAG service is loading")
        return {"status": "ready"}

    @application.post("/query")
    def query(payload: QueryRequest):
        if application.state.rag is None:
            HTTP_REQUESTS.labels(status="unavailable").inc()
            raise HTTPException(status_code=503, detail="RAG service is loading")
        IN_FLIGHT.inc()
        try:
            result = application.state.rag.answer(payload.question, payload.top_k)
            if not result.answer:
                HTTP_REQUESTS.labels(status="invalid").inc()
                raise HTTPException(status_code=502, detail="Empty RAG answer")
            HTTP_REQUESTS.labels(status="success").inc()
            GOODPUT.inc()
            return {
                "status": getattr(result, "status", "normal"),
                "mode": (getattr(result, "service_modes", []) or ["normal"])[0],
                "degraded_components": getattr(result, "degraded_components", []),
                "answer": result.answer,
                "sources": [
                    {"title": chunk.title, "chunk": chunk.chunk_index}
                    for chunk in result.chunks
                ],
                "trace_id": result.trace_id or None,
                "output_tokens": result.output_tokens,
                "response_cache_hit": result.response_cache_hit,
            }
        except HTTPException:
            raise
        except DependencyUnavailableError as error:
            HTTP_REQUESTS.labels(status="unavailable").inc()
            RESILIENCE_EVENTS.labels(
                mode=error.mode.value,
                component=type(error).__name__,
            ).inc()
            raise HTTPException(
                status_code=503,
                detail={"mode": error.mode.value, "error": type(error).__name__},
            ) from error
        except Exception as error:
            HTTP_REQUESTS.labels(status="error").inc()
            raise HTTPException(status_code=500, detail=type(error).__name__) from error
        finally:
            IN_FLIGHT.dec()

    return application


app = create_app()
