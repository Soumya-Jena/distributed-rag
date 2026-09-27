import pytest

from src.scale_db import assert_scale_database, database_name


def test_database_name():
    assert database_name("postgresql://rag:rag@localhost:5432/ragdb_scale") == "ragdb_scale"


def test_scale_guard_accepts_only_isolated_database():
    assert_scale_database("postgresql://rag:rag@localhost:5432/ragdb_scale")
    with pytest.raises(RuntimeError, match="Refusing scalability work"):
        assert_scale_database("postgresql://rag:rag@localhost:5432/ragdb")

