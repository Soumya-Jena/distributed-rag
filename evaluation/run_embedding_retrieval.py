import os
import subprocess
import sys
from pathlib import Path


MODELS = [
    ("embedding-minilm", "sentence-transformers/all-MiniLM-L6-v2"),
    ("embedding-bge-small", "BAAI/bge-small-en-v1.5"),
    ("embedding-e5-small-v2", "intfloat/e5-small-v2"),
]
OUTPUT = Path("experiments/day-06")


def run(arguments, environment):
    print(f"\n> {' '.join(arguments)}", flush=True)
    subprocess.run(arguments, check=True, env=environment)


def main():
    for label, model in MODELS:
        environment = os.environ.copy()
        environment["EMBEDDING_MODEL"] = model
        environment["CHUNK_SIZE"] = "100"
        environment["CHUNK_OVERLAP"] = "20"
        print(f"\n=== {label}: {model} ===", flush=True)
        run(
            [
                sys.executable,
                "-m",
                "src.ingest",
                "datasets/raw",
                "--metrics-output",
                str(OUTPUT / f"{label}-embedding.csv"),
            ],
            environment,
        )
        run(
            [
                sys.executable,
                "-m",
                "evaluation.evaluate_retrieval",
                "--label",
                label,
                "--output-dir",
                str(OUTPUT),
            ],
            environment,
        )


if __name__ == "__main__":
    main()
