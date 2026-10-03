"""Stable public API request and response schemas."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceResponse(BaseModel):
    title: str
    chunk: int


class QueryResponse(BaseModel):
    request_id: str
    status: str
    mode: str
    degraded_components: list[str]
    answer: str
    sources: list[SourceResponse]
    trace_id: str | None = None
    output_tokens: int = 0
    response_cache_hit: bool = False
