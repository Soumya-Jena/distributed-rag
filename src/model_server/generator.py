"""Model-server-owned generator construction."""

import os

from src.generator import LocalGenerator


class FakeGenerator:
    """Cheap deterministic adapter for container and contract smoke tests."""

    model_name = "fake-smoke-generator"
    last_input_token_count = 0
    last_output_token_count = 0

    def generate(self, messages, max_new_tokens=350):
        text = "EVIDENCE_STATUS: SUPPORTED\n\nSmoke-test answer [S1]"
        self.last_input_token_count = sum(
            len(item.get("content", "").split()) for item in messages
        )
        self.last_output_token_count = len(text.split())
        return text


def build_generator():
    if os.getenv("MODEL_SERVER_FAKE", "false").lower() in {"1", "true", "yes"}:
        return FakeGenerator()
    return LocalGenerator()
