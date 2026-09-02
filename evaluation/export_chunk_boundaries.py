import csv

from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.chunking import chunk_text
from src.config import EMBEDDING_MODEL
from src.document_loader import discover_documents, load_document


CONFIGURATIONS = [
    (100, 20),
    (150, 30),
    (200, 40),
    (240, 40),
]


def main():
    model = SentenceTransformer(EMBEDDING_MODEL)
    output_directory = Path("experiments/day-05/boundaries")
    output_directory.mkdir(parents=True, exist_ok=True)

    for chunk_size, overlap in CONFIGURATIONS:
        label = f"chunk-{chunk_size}-{overlap}"
        output = output_directory / f"{label}-chunks.csv"
        rows = []

        for document in discover_documents(Path("datasets/raw")):
            chunks = chunk_text(
                load_document(document),
                model,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            for chunk in chunks:
                rows.append(
                    {
                        "source": document.name,
                        "chunk_index": chunk["chunk_index"],
                        "token_count": chunk["token_count"],
                        "content": chunk["content"],
                    }
                )

        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"{label}: {len(rows)} chunks -> {output}")


if __name__ == "__main__":
    main()
