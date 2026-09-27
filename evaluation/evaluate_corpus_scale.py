"""Benchmark exact vector, GIN lexical, and hybrid retrieval at one corpus size."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import psutil

from src.scale_db import get_scale_connection


OUTPUT_DIR = Path("experiments/day-16")


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(values, value)) if values else 0.0


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = [item for item in csv.DictReader(handle) if item["stage"] != str(row["stage"])]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(row)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def timed_fetch(conn, statement: str, params: tuple) -> tuple[list, float]:
    started = time.perf_counter()
    rows = conn.execute(statement, params).fetchall()
    return rows, (time.perf_counter() - started) * 1000


def vector_search(conn, embedding) -> tuple[list, float]:
    return timed_fetch(
        conn,
        """
        SELECT id, 1 - (embedding <=> %s) AS score
        FROM benchmark_chunks
        ORDER BY embedding <=> %s
        LIMIT 20
        """,
        (embedding, embedding),
    )


def lexical_search(conn, query: str) -> tuple[list, float]:
    return timed_fetch(
        conn,
        """
        SELECT id, ts_rank_cd(search_vector, websearch_to_tsquery('english', %s)) AS score
        FROM benchmark_chunks
        WHERE search_vector @@ websearch_to_tsquery('english', %s)
        ORDER BY score DESC, id
        LIMIT 20
        """,
        (query, query),
    )


def hybrid_search(conn, query: str, embedding) -> tuple[list, float, float, float, float]:
    started = time.perf_counter()
    vector_rows, vector_ms = vector_search(conn, embedding)
    lexical_rows, lexical_ms = lexical_search(conn, query)
    fusion_started = time.perf_counter()
    scores: dict[int, float] = {}
    for rank, row in enumerate(vector_rows, start=1):
        scores[row[0]] = scores.get(row[0], 0.0) + 1 / (60 + rank)
    for rank, row in enumerate(lexical_rows, start=1):
        scores[row[0]] = scores.get(row[0], 0.0) + 1 / (60 + rank)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:20]
    fusion_ms = (time.perf_counter() - fusion_started) * 1000
    total_ms = (time.perf_counter() - started) * 1000
    return ranked, total_ms, vector_ms, lexical_ms, fusion_ms


def storage_snapshot(conn, stage: str, chunks: int) -> dict:
    values = conn.execute(
        """
        SELECT
            pg_table_size('benchmark_chunks'),
            pg_indexes_size('benchmark_chunks'),
            pg_total_relation_size('benchmark_chunks'),
            pg_database_size(current_database())
        """
    ).fetchone()
    index_rows = conn.execute(
        """
        SELECT indexrelname, pg_relation_size(indexrelid)
        FROM pg_stat_user_indexes
        WHERE relname = 'benchmark_chunks'
        ORDER BY indexrelname
        """
    ).fetchall()
    return {
        "stage": stage,
        "chunks": chunks,
        "table_bytes": values[0],
        "indexes_bytes": values[1],
        "total_relation_bytes": values[2],
        "database_bytes": values[3],
        "bytes_per_chunk": round(values[2] / chunks, 3),
        "index_sizes_json": json.dumps({name: size for name, size in index_rows}, sort_keys=True),
    }


def save_explain(conn, stage: str, query_text: str, embedding) -> None:
    plans = {}
    plans["vector_exact"] = conn.execute(
        """
        EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
        SELECT id FROM benchmark_chunks ORDER BY embedding <=> %s LIMIT 20
        """,
        (embedding,),
    ).fetchone()[0]
    plans["lexical_gin"] = conn.execute(
        """
        EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
        SELECT id FROM benchmark_chunks
        WHERE search_vector @@ websearch_to_tsquery('english', %s)
        LIMIT 20
        """,
        (query_text,),
    ).fetchone()[0]
    path = OUTPUT_DIR / "plans" / f"{stage.lower()}-explain.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plans, indent=2) + "\n", encoding="utf-8")


def benchmark(stage: str, expected_chunks: int, query_limit: int) -> dict:
    process = psutil.Process()
    memory_before = psutil.virtual_memory()
    disk_before = psutil.disk_io_counters()
    cpu_before = psutil.cpu_percent(interval=0.2)

    with get_scale_connection() as conn:
        conn.execute("SET statement_timeout = '30s'")
        chunks = conn.execute("SELECT COUNT(*) FROM benchmark_chunks").fetchone()[0]
        if chunks != expected_chunks:
            raise RuntimeError(f"Expected {expected_chunks:,} chunks, found {chunks:,}")
        queries = conn.execute(
            "SELECT query_id, query_text, embedding FROM benchmark_queries ORDER BY query_id LIMIT %s",
            (query_limit,),
        ).fetchall()
        if not queries:
            raise RuntimeError("No benchmark queries found; ingest the 10K tier first")

        detailed = []
        for pass_name in ("first_pass", "warm_repeat"):
            for query_id, query_text, embedding in queries:
                _, vector_ms = vector_search(conn, embedding)
                _, lexical_ms = lexical_search(conn, query_text)
                _, hybrid_ms, hybrid_vector_ms, hybrid_lexical_ms, fusion_ms = hybrid_search(
                    conn, query_text, embedding
                )
                detailed.append(
                    {
                        "stage": stage,
                        "chunks": chunks,
                        "pass": pass_name,
                        "query_id": query_id,
                        "vector_ms": round(vector_ms, 6),
                        "lexical_ms": round(lexical_ms, 6),
                        "hybrid_ms": round(hybrid_ms, 6),
                        "hybrid_vector_ms": round(hybrid_vector_ms, 6),
                        "hybrid_lexical_ms": round(hybrid_lexical_ms, 6),
                        "fusion_ms": round(fusion_ms, 6),
                    }
                )

        save_explain(conn, stage, queries[0][1], queries[0][2])
        storage = storage_snapshot(conn, stage, chunks)

    write_rows(OUTPUT_DIR / "raw" / f"{stage.lower()}-latencies.csv", detailed)
    append_csv(OUTPUT_DIR / "storage.csv", storage)
    memory_after = psutil.virtual_memory()
    disk_after = psutil.disk_io_counters()
    cpu_after = psutil.cpu_percent(interval=0.2)
    append_csv(
        OUTPUT_DIR / "resources.csv",
        {
            "stage": stage,
            "chunks": chunks,
            "cpu_percent_before": cpu_before,
            "cpu_percent_after": cpu_after,
            "memory_percent_before": memory_before.percent,
            "memory_percent_after": memory_after.percent,
            "process_rss_mib": round(process.memory_info().rss / 1024**2, 3),
            "disk_read_mib": round((disk_after.read_bytes - disk_before.read_bytes) / 1024**2, 3),
            "disk_write_mib": round((disk_after.write_bytes - disk_before.write_bytes) / 1024**2, 3),
        },
    )

    warm = [row for row in detailed if row["pass"] == "warm_repeat"]
    first = [row for row in detailed if row["pass"] == "first_pass"]
    summary = {"stage": stage, "chunks": chunks, "queries": len(queries)}
    for name in ("vector", "lexical", "hybrid"):
        values = [row[f"{name}_ms"] for row in warm]
        first_values = [row[f"{name}_ms"] for row in first]
        summary[f"{name}_p50_ms"] = round(percentile(values, 50), 6)
        summary[f"{name}_p95_ms"] = round(percentile(values, 95), 6)
        summary[f"{name}_p99_ms"] = round(percentile(values, 99), 6)
        summary[f"{name}_first_pass_p50_ms"] = round(percentile(first_values, 50), 6)
    summary["hybrid_vector_p50_ms"] = round(
        percentile([row["hybrid_vector_ms"] for row in warm], 50), 6
    )
    summary["hybrid_lexical_p50_ms"] = round(
        percentile([row["hybrid_lexical_ms"] for row in warm], 50), 6
    )
    summary["fusion_p50_ms"] = round(
        percentile([row["fusion_ms"] for row in warm], 50), 6
    )
    summary["reranker_p50_ms"] = 0.0
    append_csv(OUTPUT_DIR / "latency-summary.csv", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--expected-chunks", type=int, required=True)
    parser.add_argument("--queries", type=int, default=50)
    args = parser.parse_args()
    result = benchmark(args.stage, args.expected_chunks, args.queries)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
