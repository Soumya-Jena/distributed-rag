import os
from pathlib import Path
from urllib.parse import urlparse

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg import sql


load_dotenv()

DATABASE_URL_SCALE = os.getenv(
    "DATABASE_URL_SCALE",
    "postgresql://rag:rag@localhost:5432/ragdb_scale",
)
EXPECTED_DATABASE = "ragdb_scale"
SCHEMA_PATH = Path(__file__).with_name("scale_schema.sql")


def database_name(url: str = DATABASE_URL_SCALE) -> str:
    return urlparse(url).path.lstrip("/")


def assert_scale_database(url: str = DATABASE_URL_SCALE) -> None:
    actual = database_name(url)
    if actual != EXPECTED_DATABASE:
        raise RuntimeError(
            f"Refusing scalability work on database {actual!r}; "
            f"DATABASE_URL_SCALE must target {EXPECTED_DATABASE!r}."
        )


def get_scale_connection():
    assert_scale_database()
    conn = psycopg.connect(DATABASE_URL_SCALE)
    register_vector(conn)
    return conn


def ensure_scale_database() -> None:
    """Create the isolated database if needed, then apply its schema."""
    assert_scale_database()
    parsed = urlparse(DATABASE_URL_SCALE)
    admin_url = DATABASE_URL_SCALE.replace(f"/{EXPECTED_DATABASE}", "/postgres", 1)
    with psycopg.connect(admin_url, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (EXPECTED_DATABASE,)
        ).fetchone()
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(EXPECTED_DATABASE)))

    with psycopg.connect(DATABASE_URL_SCALE) as conn:
        conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.commit()

