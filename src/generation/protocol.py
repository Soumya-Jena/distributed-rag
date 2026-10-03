"""Deployment-neutral generation contract."""

from typing import Protocol


class GenerationClient(Protocol):
    last_input_token_count: int
    last_output_token_count: int
    model_name: str

    def generate(self, messages, *, max_new_tokens: int) -> str:
        ...

    def count_tokens(self, text: str) -> int:
        ...
