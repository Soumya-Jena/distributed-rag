"""Apply idempotent schema migrations as an explicit one-off command."""

from pathlib import Path

import psycopg

from src.config import DATABASE_URL


SCHEMA = Path(__file__).resolve().parents[1] / "schema.sql"


def migrate(database_url=DATABASE_URL):
    schema = SCHEMA.read_text(encoding="utf-8")
    with psycopg.connect(database_url) as connection:
        connection.execute(schema)
        connection.commit()


def main():
    migrate()
    print("Database migrations applied.")


if __name__ == "__main__":
    main()
