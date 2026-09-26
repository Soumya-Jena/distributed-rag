"""Deterministic fake service for validating load tooling, never RAG capacity."""

import threading
import time
import os
from types import SimpleNamespace

from src.load_test_api import create_app


class DeterministicLoadService:
    def __init__(self):
        self.cache_enabled = os.getenv("HARNESS_CACHE_ENABLED", "false").lower() == "true"
        self.compute_slot = threading.Semaphore(1)
        self.cache = set()
        self.cache_lock = threading.Lock()

    def answer(self, question, _top_k):
        with self.cache_lock:
            cached = self.cache_enabled and question in self.cache
            if self.cache_enabled:
                self.cache.add(question)
        if cached:
            time.sleep(0.01)
        else:
            with self.compute_slot:
                time.sleep(0.20)
        return SimpleNamespace(
            answer="EVIDENCE_STATUS: SUPPORTED\n\nDeterministic harness answer. [S1]",
            chunks=[SimpleNamespace(title="load-harness", chunk_index=0)],
            trace_id="b" * 32, output_tokens=8, response_cache_hit=cached,
        )


app = create_app(DeterministicLoadService())
