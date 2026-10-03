"""Generation clients shared by orchestration and deployment adapters."""

from src.generation.client import (
    HTTPGenerationClient,
    LocalGenerationClient,
    create_generation_client,
)
from src.generation.protocol import GenerationClient

__all__ = [
    "GenerationClient",
    "HTTPGenerationClient",
    "LocalGenerationClient",
    "create_generation_client",
]
