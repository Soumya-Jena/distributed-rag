import csv
from pathlib import Path
from statistics import mean


OUTPUT_DIR = Path("experiments/day-16")


def read(name):
    with (OUTPUT_DIR / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(name, rows):
    with (OUTPUT_DIR / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    latency = read("latency-summary.csv")
    storage = {row["stage"]: row for row in read("storage.csv")}
    ingestion = {row["stage"]: row for row in read("ingestion-throughput.csv")}
    comparison = []
    for latency_row in latency:
        stage = latency_row["stage"]
        storage_row = storage[stage]
        ingestion_row = ingestion[stage]
        comparison.append(
            {
                "stage": stage,
                "chunks": latency_row["chunks"],
                "incremental_chunks_ingested": ingestion_row["inserted_chunks"],
                "ingestion_chunks_per_second": ingestion_row["chunks_per_second"],
                "db_rows_per_second": ingestion_row["rows_per_second"],
                "total_relation_mib": round(int(storage_row["total_relation_bytes"]) / 1024**2, 3),
                "bytes_per_chunk": storage_row["bytes_per_chunk"],
                "vector_p50_ms": latency_row["vector_p50_ms"],
                "vector_p95_ms": latency_row["vector_p95_ms"],
                "vector_p99_ms": latency_row["vector_p99_ms"],
                "lexical_p50_ms": latency_row["lexical_p50_ms"],
                "lexical_p95_ms": latency_row["lexical_p95_ms"],
                "lexical_p99_ms": latency_row["lexical_p99_ms"],
                "hybrid_p50_ms": latency_row["hybrid_p50_ms"],
                "hybrid_p95_ms": latency_row["hybrid_p95_ms"],
                "hybrid_p99_ms": latency_row["hybrid_p99_ms"],
                "hit_at_1": "N/A (synthetic mechanics track)",
                "hit_at_5": "N/A (synthetic mechanics track)",
                "mrr": "N/A (synthetic mechanics track)",
            }
        )
    write("scale-comparison.csv", comparison)

    baseline = read("track-a-real-50q-retrieval.csv")
    write(
        "semantic-baseline-summary.csv",
        [
            {
                "track": "A-real-frozen-corpus",
                "questions": len(baseline),
                "hit_at_1": round(mean(float(row["hit@1"]) for row in baseline), 6),
                "hit_at_3": round(mean(float(row["hit@3"]) for row in baseline), 6),
                "hit_at_5": round(mean(float(row["hit@5"]) for row in baseline), 6),
                "recall_at_5": round(mean(float(row["recall@5"]) for row in baseline), 6),
                "mrr": round(mean(float(row["reciprocal_rank"]) for row in baseline), 6),
                "scope": "baseline only; no synthetic scaling quality claim",
            }
        ],
    )
    degradation = []
    for row in baseline:
        reciprocal = float(row["reciprocal_rank"])
        baseline_rank = round(1 / reciprocal) if reciprocal else "not_found"
        degradation.append(
            {
                "id": row["id"],
                "question": row["question"],
                "baseline_rank": baseline_rank,
                "scaled_rank": "N/A",
                "rank_change": "N/A",
                "reason": "No genuine scaled semantic corpus; synthetic vectors excluded from quality analysis",
            }
        )
    write("rank-degradation.csv", degradation)


if __name__ == "__main__":
    main()
