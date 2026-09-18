"""Build a deterministic 100-request cache workload from frozen queries."""

import json
from pathlib import Path


SOURCE = Path("datasets/evaluation/query_transform_questions.jsonl")
OUTPUT = Path("datasets/evaluation/cache_workload.jsonl")


def main():
    with SOURCE.open(encoding="utf-8") as file:
        source = [json.loads(line) for line in file if line.strip()]
    rows = []
    request = 1
    for record in source[:40]:
        rows.append({
            "id": f"cache-{request:03d}", "kind": "unique",
            "source_id": record["id"], "query": record["question"],
        })
        request += 1
    for record in source[:30]:
        rows.append({
            "id": f"cache-{request:03d}", "kind": "exact_repeat",
            "source_id": record["id"], "query": record["question"],
        })
        request += 1
    for record in source[:20]:
        rows.append({
            "id": f"cache-{request:03d}", "kind": "near_repeat",
            "source_id": record["id"], "query": record["semantic"],
        })
        request += 1
    for record in source[40:50]:
        rows.append({
            "id": f"cache-{request:03d}", "kind": "new_query",
            "source_id": record["id"], "query": record["alternate"],
        })
        request += 1
    if len(rows) != 100:
        raise RuntimeError(f"Expected 100 requests, got {len(rows)}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
    print(f"Saved {len(rows)} requests to {OUTPUT}")


if __name__ == "__main__":
    main()
