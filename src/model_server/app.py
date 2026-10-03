"""Private HTTP boundary for the expensive generation model."""

from contextlib import asynccontextmanager
from threading import BoundedSemaphore

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.config import GENERATION_BULKHEAD_SLOTS, MAX_NEW_TOKENS
from src.model_server.generator import build_generator


class Message(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=100_000)

    @field_validator("role")
    @classmethod
    def valid_role(cls, value):
        if value not in {"system", "user", "assistant"}:
            raise ValueError("unsupported message role")
        return value


class GenerateRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=64)
    max_new_tokens: int = Field(default=MAX_NEW_TOKENS, ge=1, le=2048)


def create_app(generator=None):
    @asynccontextmanager
    async def lifespan(application):
        if application.state.generator is None:
            application.state.generator = build_generator()
        application.state.ready = True
        yield
        application.state.ready = False

    application = FastAPI(
        title="Internal RAG model server",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    application.state.generator = generator
    application.state.ready = generator is not None
    application.state.slots = BoundedSemaphore(GENERATION_BULKHEAD_SLOTS)

    @application.get("/live")
    def live():
        return {"status": "alive"}

    @application.get("/ready")
    def ready():
        if not application.state.ready or application.state.generator is None:
            raise HTTPException(status_code=503, detail="model is loading")
        return {
            "status": "ready",
            "model": application.state.generator.model_name,
        }

    @application.post("/internal/generate")
    def generate(request: GenerateRequest):
        if not application.state.ready:
            raise HTTPException(status_code=503, detail="model is not ready")
        if not application.state.slots.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="generation capacity exhausted")
        try:
            generator = application.state.generator
            text = generator.generate(
                [message.model_dump() for message in request.messages],
                max_new_tokens=request.max_new_tokens,
            )
            return {
                "text": text,
                "model": generator.model_name,
                "input_tokens": generator.last_input_token_count,
                "output_tokens": generator.last_output_token_count,
            }
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=503, detail="generation failed") from error
        finally:
            application.state.slots.release()

    return application


app = create_app()
