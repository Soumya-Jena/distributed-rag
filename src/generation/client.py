"""Local and HTTP implementations of the generation contract."""

import httpx

from src.config import (
    GENERATION_BACKEND,
    GENERATION_MODEL,
    GENERATION_TIMEOUT_SECONDS,
    GENERATION_URL,
    MAX_NEW_TOKENS,
)
from src.generator import LocalGenerator


class LocalGenerationClient:
    def __init__(self, generator=None):
        self.generator = generator or LocalGenerator()

    @property
    def model_name(self):
        return self.generator.model_name

    @property
    def last_input_token_count(self):
        return self.generator.last_input_token_count

    @property
    def last_output_token_count(self):
        return self.generator.last_output_token_count

    def count_tokens(self, text):
        return self.generator.count_tokens(text)

    def generate(self, messages, *, max_new_tokens=MAX_NEW_TOKENS):
        return self.generator.generate(messages, max_new_tokens=max_new_tokens)


class HTTPGenerationClient:
    def __init__(
        self,
        base_url=GENERATION_URL,
        timeout_seconds=GENERATION_TIMEOUT_SECONDS,
        client=None,
        model_name=GENERATION_MODEL,
    ):
        self.model_name = model_name
        self.last_input_token_count = 0
        self.last_output_token_count = 0
        self._owns_client = client is None
        self.client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds, connect=min(5.0, timeout_seconds)),
        )

    def generate(self, messages, *, max_new_tokens=MAX_NEW_TOKENS):
        response = self.client.post(
            "/internal/generate",
            json={"messages": messages, "max_new_tokens": max_new_tokens},
        )
        response.raise_for_status()
        payload = response.json()
        text = payload["text"]
        self.last_input_token_count = int(payload.get("input_tokens", 0))
        self.last_output_token_count = int(payload.get("output_tokens", 0))
        return text

    def count_tokens(self, text):
        # The tokenizer belongs to the model server. This estimate is used only
        # for local context budgeting; authoritative counts return with generation.
        return len(text.split())

    def ready(self):
        response = self.client.get("/ready")
        response.raise_for_status()
        return response.json().get("status") == "ready"

    def close(self):
        if self._owns_client:
            self.client.close()


def create_generation_client(backend=GENERATION_BACKEND):
    if backend == "local":
        return LocalGenerationClient()
    if backend == "http":
        return HTTPGenerationClient()
    raise ValueError(f"Unknown generation backend: {backend}")
