"""Inspect attack-bearing chunks and source provenance in the isolated DB."""

import os

import psycopg

from src.config import DATABASE_URL
from src.security_eval import assert_security_database, security_database_url


def main():
    target = os.getenv("SECURITY_DATABASE_URL") or security_database_url(DATABASE_URL)
    assert_security_database(target)
    with psycopg.connect(target) as conn:
        rows = conn.execute(
            "SELECT d.title, c.chunk_index, c.content "
            "FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE d.source_path LIKE %s ORDER BY d.title, c.chunk_index",
            ("%poisoned%",),
        ).fetchall()
    for title, index, content in rows:
        print(title, index, "ATTACK_MARKER" in content, len(content), repr(content[-180:]))


if __name__ == "__main__":
    main()
