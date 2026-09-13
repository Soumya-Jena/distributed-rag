"""Create the isolated test database without touching the normal corpus."""

import os

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

from src.config import DATABASE_URL
from src.security_eval import assert_security_database, security_database_url


def main():
    target = os.getenv("SECURITY_DATABASE_URL") or security_database_url(DATABASE_URL)
    assert_security_database(target)
    admin_info = conninfo_to_dict(target)
    admin_info["dbname"] = "postgres"
    with psycopg.connect(**admin_info, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", ("ragdb_security",)
        ).fetchone()
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier("ragdb_security")))
            print("Created ragdb_security")
        else:
            print("ragdb_security already exists")
    with psycopg.connect(target) as conn:
        has_documents = conn.execute(
            "SELECT to_regclass('public.documents') IS NOT NULL"
        ).fetchone()[0]
        if has_documents:
            paths = [row[0] for row in conn.execute("SELECT source_path FROM documents")]
            unexpected = [
                path for path in paths
                if not path.replace("\\", "/").startswith("datasets/security/")
            ]
            if unexpected:
                raise RuntimeError(
                    "Existing ragdb_security contains non-security documents; "
                    "refusing to mix corpora"
                )
    print("Security database is isolated. Ingest datasets/security with DATABASE_URL set to it.")


if __name__ == "__main__":
    main()
