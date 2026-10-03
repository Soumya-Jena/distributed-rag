"""Deployment-facing RAG API with explicit health semantics."""

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response

from src.api.health import dependency_readiness
from src.api.schemas import QueryRequest, QueryResponse
from src.config import METRICS_PORT, METRICS_SERVER_ENABLED
from src.rag_service import RAGService
from src.resilience import DependencyUnavailableError
from src.runtime_metrics import start_metrics_server


def create_app(service=None, readiness_check=dependency_readiness):
    @asynccontextmanager
    async def lifespan(application):
        if application.state.rag is None:
            application.state.rag = RAGService()
        if METRICS_SERVER_ENABLED:
            start_metrics_server(METRICS_PORT)
        application.state.started = True
        yield
        application.state.started = False
        close = getattr(application.state.rag.generator, "close", None)
        if callable(close):
            close()

    application = FastAPI(
        title="Distributed RAG API",
        version="1.0",
        lifespan=lifespan,
    )
    application.state.rag = service
    application.state.started = service is not None

    @application.get("/live")
    def live():
        return {"status": "alive"}

    @application.get("/ready")
    def ready():
        if not application.state.started or application.state.rag is None:
            raise HTTPException(status_code=503, detail="service is starting")
        try:
            checks = readiness_check(application.state.rag)
        except Exception as error:
            raise HTTPException(status_code=503, detail="dependency check failed") from error
        if not all(checks.values()):
            raise HTTPException(status_code=503, detail={"checks": checks})
        return {"status": "ready", "checks": checks}

    @application.post("/api/v1/query", response_model=QueryResponse)
    def query(payload: QueryRequest, request: Request, response: Response):
        request_id = request.headers.get("x-request-id") or uuid4().hex
        response.headers["x-request-id"] = request_id
        try:
            result = application.state.rag.answer(payload.question, payload.top_k)
        except DependencyUnavailableError as error:
            raise HTTPException(
                status_code=503,
                detail={"mode": error.mode.value, "error": type(error).__name__},
            ) from error
        if not result.answer:
            raise HTTPException(status_code=502, detail="empty RAG answer")
        modes = getattr(result, "service_modes", []) or ["normal"]
        return QueryResponse(
            request_id=request_id,
            status=getattr(result, "status", "normal"),
            mode=modes[0],
            degraded_components=getattr(result, "degraded_components", []),
            answer=result.answer,
            sources=[
                {"title": chunk.title, "chunk": chunk.chunk_index}
                for chunk in result.chunks
            ],
            trace_id=result.trace_id or None,
            output_tokens=result.output_tokens,
            response_cache_hit=result.response_cache_hit,
        )

    return application


app = create_app()
