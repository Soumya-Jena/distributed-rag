"""Locust workload with explicit unique and repeated working sets."""

import itertools
import json
import os
import random
import threading
from pathlib import Path

from locust import HttpUser, between, events, task


WORKLOAD = json.loads(Path("datasets/evaluation/load_workload.json").read_text(encoding="utf-8"))
MODE = os.getenv("LOAD_MODE", "unique").lower()
QUESTIONS = [item for item in WORKLOAD if item["mode"] == MODE]
if not QUESTIONS:
    raise RuntimeError(f"No workload questions for LOAD_MODE={MODE}")
_cycle_lock = threading.Lock()
CATEGORY_WEIGHTS = {
    "normal": 50, "paraphrase": 20, "identifier": 10,
    "multi_part": 10, "unsupported": 5, "security": 5,
}
_by_category = {
    category: itertools.cycle(
        item for item in QUESTIONS if item["category"] == category
    )
    for category in CATEGORY_WEIGHTS
    if any(item["category"] == category for item in QUESTIONS)
}


@events.test_start.add_listener
def validate_mode(environment, **_kwargs):
    print(f"Load mode={MODE}; questions={len(QUESTIONS)}")


class RAGUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task
    def ask_question(self):
        if MODE == "unique":
            with _cycle_lock:
                categories = list(_by_category)
                category = random.choices(
                    categories, weights=[CATEGORY_WEIGHTS[name] for name in categories], k=1
                )[0]
                item = next(_by_category[category])
        else:
            item = random.choice(QUESTIONS)
        with self.client.post(
            "/query", json={"question": item["question"]},
            catch_response=True, name=f"/query [{MODE}]",
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            try:
                payload = response.json()
            except ValueError:
                response.failure("Response was not JSON")
                return
            if not payload.get("answer"):
                response.failure("Missing answer")
            elif "sources" not in payload:
                response.failure("Missing sources")
