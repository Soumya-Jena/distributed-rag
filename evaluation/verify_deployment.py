"""Run a small, repeatable deployment smoke test against the packaged API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/day-19/smoke-results.json"),
    )
    args = parser.parse_args()

    request_id = "deployment-smoke-001"
    with httpx.Client(base_url=args.url, timeout=30.0) as client:
        live = client.get("/live")
        ready = client.get("/ready")
        query = client.post(
            "/api/v1/query",
            headers={"x-request-id": request_id},
            json={
                "question": "How does PostgreSQL streaming replication work?",
                "top_k": 3,
            },
        )

    query_body = query.json() if query.headers.get("content-type", "").startswith("application/json") else {}
    result = {
        "live_status": live.status_code,
        "ready_status": ready.status_code,
        "readiness": ready.json(),
        "query_status": query.status_code,
        "request_id_round_trip": query.headers.get("x-request-id") == request_id,
        "answer_status": query_body.get("status"),
        "source_count": len(query_body.get("sources", [])),
        "trace_id_present": bool(query_body.get("trace_id")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not all((live.is_success, ready.is_success, query.is_success)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
