"""Record provenance on isolated security documents without changing chunks."""

import os

import psycopg

from src.config import DATABASE_URL
from src.security_eval import assert_security_database, security_database_url


def main():
    target = os.getenv("SECURITY_DATABASE_URL") or security_database_url(DATABASE_URL)
    assert_security_database(target)
    with psycopg.connect(target) as conn:
        paths = [row[0] for row in conn.execute("SELECT source_path FROM documents")]
        if not paths or any(
            not path.replace("\\", "/").startswith("datasets/security/")
            for path in paths
        ):
            raise RuntimeError("Unexpected corpus in ragdb_security")
        for path in paths:
            poisoned = "/poisoned/" in path.replace("\\", "/")
            conn.execute(
                "UPDATE documents SET metadata = metadata || %s::jsonb "
                "WHERE source_path = %s",
                (
                    '{"source_type":"synthetic_attack_document",'
                    '"publisher":"local_security_experiment",'
                    '"trust_level":"untrusted"}'
                    if poisoned else
                    '{"source_type":"local_document",'
                    '"publisher":"local_security_experiment",'
                    '"trust_level":"unreviewed"}',
                    path,
                ),
            )
        conn.commit()
    print(f"Annotated provenance for {len(paths)} isolated documents")


if __name__ == "__main__":
    main()
