import psycopg

from pgvector.psycopg import register_vector

from src.config import DATABASE_CONNECT_TIMEOUT, DATABASE_URL


def get_connection():
    conn = psycopg.connect(DATABASE_URL, connect_timeout=DATABASE_CONNECT_TIMEOUT)
    register_vector(conn)
    return conn


def initialize_database():
    from src.database.migrate import migrate

    migrate()
    print("Database schema initialized.")
