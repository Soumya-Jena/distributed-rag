"""Stream deterministic synthetic vectors into the isolated scale database.

This is Track B data. It measures PostgreSQL/pgvector mechanics and must not be
used to make semantic retrieval-quality claims.
"""

import argparse
import csv
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

from src.scale_db import ensure_scale_database, get_scale_connection


DIMENSIONS = 384
SEED = 42
DEFAULT_TARGETS = (10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)
OUTPUT_DIR = Path("experiments/day-16")
MANIFEST = Path("datasets/scaling/manifest.json")
TOPICS = (
    "postgres replication WAL standby durability",
    "postgres MVCC snapshot vacuum transaction",
    "kubernetes scheduler pod node affinity",
    "kafka partition consumer offset replay",
    "distributed cache invalidation consistency",
    "observability metrics traces structured logs",
    "vector retrieval cosine similarity ranking",
    "database indexing query planning buffers",
)


def vector_text(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in vector) + "]"


def safety_reason(memory_limit: float, min_disk_gib: float) -> str | None:
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(Path.cwd())
    if memory.percent >= memory_limit:
        return f"host memory reached {memory.percent:.1f}%"
    free_gib = disk.free / 1024**3
    if free_gib < min_disk_gib:
        return f"free disk fell to {free_gib:.1f} GiB"
    return None


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


def populate_queries(conn, count: int = 50) -> None:
    existing = conn.execute("SELECT COUNT(*) FROM benchmark_queries").fetchone()[0]
    if existing >= count:
        return
    rows = conn.execute(
        """
        SELECT id, content, embedding
        FROM benchmark_chunks
        ORDER BY id
        LIMIT %s
        """,
        (count,),
    ).fetchall()
    with conn.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO benchmark_queries (query_id, query_text, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (query_id) DO NOTHING
            """,
            [(index, row[1], row[2]) for index, row in enumerate(rows, start=1)],
        )
    conn.commit()


def ingest_to_target(
    target: int,
    batch_size: int,
    memory_limit: float,
    min_disk_gib: float,
) -> dict:
    run_id = f"scale-{target}"
    with get_scale_connection() as conn:
        current = conn.execute("SELECT COUNT(*) FROM benchmark_chunks").fetchone()[0]
        conn.execute(
            """
            INSERT INTO scale_ingestion_runs
                (run_id, target_chunks, inserted_chunks, status, message)
            VALUES (%s, %s, %s, 'running', 'streaming bounded batches')
            ON CONFLICT (run_id) DO UPDATE SET
                inserted_chunks = EXCLUDED.inserted_chunks,
                status = 'running', message = EXCLUDED.message, updated_at = NOW()
            """,
            (run_id, target, current),
        )
        conn.commit()

        if current >= target:
            populate_queries(conn)
            conn.execute(
                """UPDATE scale_ingestion_runs SET inserted_chunks=%s,
                status='complete', message='target already present', updated_at=NOW()
                WHERE run_id=%s""",
                (current, run_id),
            )
            conn.commit()
            return {
                "stage": f"S{DEFAULT_TARGETS.index(target) + 1}" if target in DEFAULT_TARGETS else target,
                "target_chunks": target,
                "start_chunks": current,
                "inserted_chunks": 0,
                "vector_generation_seconds": 0.0,
                "db_write_seconds": 0.0,
                "total_seconds": 0.0,
                "chunks_per_second": 0.0,
                "embeddings_per_second": 0.0,
                "rows_per_second": 0.0,
                "batch_size": batch_size,
                "peak_rss_mib": psutil.Process().memory_info().rss / 1024**2,
                "status": "already_complete",
            }

        stage = f"S{DEFAULT_TARGETS.index(target) + 1}" if target in DEFAULT_TARGETS else str(target)
        process = psutil.Process()
        peak_rss = process.memory_info().rss
        vector_seconds = 0.0
        write_seconds = 0.0
        started = time.perf_counter()
        inserted = 0

        try:
            while current < target:
                reason = safety_reason(memory_limit, min_disk_gib)
                if reason:
                    conn.execute(
                        """UPDATE scale_ingestion_runs SET inserted_chunks=%s,
                        status='stopped', message=%s, updated_at=NOW() WHERE run_id=%s""",
                        (current, reason, run_id),
                    )
                    conn.commit()
                    raise RuntimeError(f"Safe stop before {target:,}: {reason}")

                size = min(batch_size, target - current)
                generation_started = time.perf_counter()
                rng = np.random.default_rng(SEED + current)
                vectors = rng.standard_normal((size, DIMENSIONS), dtype=np.float32)
                vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
                rows = []
                for offset, vector in enumerate(vectors):
                    sequence = current + offset + 1
                    topic = TOPICS[sequence % len(TOPICS)]
                    content = (
                        f"Synthetic database mechanics record {sequence}. "
                        f"Technical topic: {topic}. Batch-safe seeded benchmark text."
                    )
                    rows.append((f"synthetic-{sequence:09d}", stage, content, vector_text(vector)))
                vector_seconds += time.perf_counter() - generation_started

                write_started = time.perf_counter()
                with conn.cursor().copy(
                    "COPY benchmark_chunks (source_id, scale_stage, content, embedding) FROM STDIN"
                ) as copy:
                    for row in rows:
                        copy.write_row(row)
                current += size
                inserted += size
                conn.execute(
                    """UPDATE scale_ingestion_runs SET inserted_chunks=%s,
                    status='running', updated_at=NOW() WHERE run_id=%s""",
                    (current, run_id),
                )
                conn.commit()
                write_seconds += time.perf_counter() - write_started
                peak_rss = max(peak_rss, process.memory_info().rss)

            conn.execute(
                """UPDATE scale_ingestion_runs SET inserted_chunks=%s,
                status='complete', message='target reached', updated_at=NOW() WHERE run_id=%s""",
                (current, run_id),
            )
            conn.commit()
            populate_queries(conn)
        except Exception as exc:
            conn.execute(
                """UPDATE scale_ingestion_runs SET inserted_chunks=%s,
                status='failed', message=%s, updated_at=NOW() WHERE run_id=%s""",
                (current, str(exc)[:500], run_id),
            )
            conn.commit()
            raise

    total = time.perf_counter() - started
    return {
        "stage": stage,
        "target_chunks": target,
        "start_chunks": target - inserted,
        "inserted_chunks": inserted,
        "vector_generation_seconds": round(vector_seconds, 6),
        "db_write_seconds": round(write_seconds, 6),
        "total_seconds": round(total, 6),
        "chunks_per_second": round(inserted / total, 3),
        "embeddings_per_second": round(inserted / vector_seconds, 3),
        "rows_per_second": round(inserted / write_seconds, 3),
        "batch_size": batch_size,
        "peak_rss_mib": round(peak_rss / 1024**2, 3),
        "status": "complete",
    }


def write_manifest(targets: list[int], batch_size: int) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "track": "B-database-mechanics",
                "data_kind": "synthetic seeded random unit vectors and templated text",
                "semantic_quality_claims_allowed": False,
                "dimensions": DIMENSIONS,
                "seed": SEED,
                "batch_size": batch_size,
                "planned_scale_ladder": list(DEFAULT_TARGETS),
                "requested_targets": targets,
                "token_counts": "not applicable to synthetic mechanics track",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", type=int, default=list(DEFAULT_TARGETS))
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--memory-limit", type=float, default=90.0)
    parser.add_argument("--min-disk-gib", type=float, default=10.0)
    args = parser.parse_args()
    ensure_scale_database()
    write_manifest(args.targets, args.batch_size)
    for target in args.targets:
        row = ingest_to_target(target, args.batch_size, args.memory_limit, args.min_disk_gib)
        append_csv(OUTPUT_DIR / "ingestion-throughput.csv", row)
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
