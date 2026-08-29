import psycopg

from pgvector.psycopg import register_vector

from src.config import DATABASE_URL


def get_connection():
    conn = psycopg.connect(DATABASE_URL)

    # The vector extension must already exist before
    # registering the pgvector Python types.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

    register_vector(conn)

    return conn


def initialize_database():
    with open("src/schema.sql", "r", encoding="utf-8") as file:
        schema = file.read()

    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(schema)
        conn.commit()

    print("Database schema initialized.")