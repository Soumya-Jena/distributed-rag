import argparse
import csv
import re

from pathlib import Path

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.db import get_connection


def valid_label(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise argparse.ArgumentTypeError("invalid experiment label")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, type=valid_label)
    parser.add_argument("--ingestion-seconds", required=True, type=float)
    args = parser.parse_args()

    with get_connection() as conn:
        stats = conn.execute(
            """
            SELECT
                COUNT(*) AS chunks,
                AVG(token_count) AS avg_tokens,
                MIN(token_count) AS min_tokens,
                MAX(token_count) AS max_tokens,
                pg_total_relation_size('chunks') AS relation_bytes
            FROM chunks;
            """
        ).fetchone()

    row = {
        "label": args.label,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunks": stats[0],
        "avg_tokens": float(stats[1]),
        "min_tokens": stats[2],
        "max_tokens": stats[3],
        "relation_bytes": stats[4],
        "ingestion_seconds": args.ingestion_seconds,
    }

    output = Path("experiments/day-05") / f"{args.label}-corpus.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    print(f"Chunks          : {row['chunks']}")
    print(f"Avg tokens      : {row['avg_tokens']:.2f}")
    print(f"Min tokens      : {row['min_tokens']}")
    print(f"Max tokens      : {row['max_tokens']}")
    print(f"Relation bytes  : {row['relation_bytes']}")
    print(f"Ingestion       : {row['ingestion_seconds']:.3f}s")
    print(f"Saved to        : {output}")


if __name__ == "__main__":
    main()
