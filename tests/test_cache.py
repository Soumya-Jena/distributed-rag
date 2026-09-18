import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import redis

from src.cache import Cache, fingerprint, hash_text, normalize_query
from src.query_transformer import QueryTransformer
from src.response_cache import ResponseCache
from src.retrieval_cache import RetrievalCache


class FakeRedis:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, _ttl, value):
        self.values[key] = value

    def delete(self, key):
        self.values.pop(key, None)

    def ping(self):
        return True

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    def eval(self, _script, _keys, key, token):
        if self.values.get(key) == token:
            self.delete(key)
            return 1
        return 0


class BrokenRedis(FakeRedis):
    def get(self, key):
        raise redis.RedisError("offline")

    def set(self, key, value, nx=False, ex=None):
        raise redis.RedisError("offline")


class FakeGenerator:
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    def __init__(self):
        self.calls = 0

    def generate(self, _messages, max_new_tokens=96):
        self.calls += 1
        return "SEMANTIC: explain MVCC\nTECHNICAL: PostgreSQL MVCC\nALTERNATE: how MVCC works"


class CacheTests(unittest.TestCase):
    def test_query_normalization_is_conservative(self):
        self.assertEqual(normalize_query("  How   does WAL work?  "), "How does WAL work?")
        self.assertNotEqual(normalize_query("WAL"), normalize_query("wal"))

    def test_hash_and_fingerprint_are_deterministic(self):
        self.assertEqual(hash_text("x"), hash_text("x"))
        self.assertEqual(fingerprint({"b": 2, "a": 1}), fingerprint({"a": 1, "b": 2}))

    def test_json_cache_and_lock(self):
        cache = Cache(client=FakeRedis())
        self.assertIsNone(cache.get_json("missing"))
        self.assertTrue(cache.set_json("key", {"answer": 1}, 60))
        self.assertEqual(cache.get_json("key"), {"answer": 1})
        with cache.lock("lock:key") as acquired:
            self.assertTrue(acquired)
            with cache.lock("lock:key") as second:
                self.assertFalse(second)
        with cache.lock("lock:key") as acquired_again:
            self.assertTrue(acquired_again)

    def test_cache_failure_is_fail_open(self):
        cache = Cache(client=BrokenRedis())
        value, stats = cache.lookup_json("key")
        self.assertIsNone(value)
        self.assertFalse(stats.available)
        self.assertFalse(cache.set_json("key", {"x": 1}, 60))

    def test_transform_cache_uses_normalized_exact_query(self):
        client = FakeRedis()
        generator = FakeGenerator()
        transformer = QueryTransformer(
            generator=generator, cache=Cache(client=client)
        )
        first = transformer.transform(" Explain   MVCC ")
        second = transformer.transform("Explain MVCC")
        self.assertEqual(first, second)
        self.assertEqual(generator.calls, 1)
        self.assertTrue(transformer.last_cache_stats.hit)

    def test_corpus_version_changes_retrieval_key(self):
        cache = RetrievalCache(Cache(client=FakeRedis()), "hybrid", False)
        with patch("src.retrieval_cache.get_corpus_version", return_value=20):
            old_key, _ = cache.key("What is MVCC?", 20, 5)
        with patch("src.retrieval_cache.get_corpus_version", return_value=21):
            new_key, _ = cache.key("What is MVCC?", 20, 5)
        self.assertNotEqual(old_key, new_key)
        self.assertTrue(old_key.startswith("retrieval:20:"))
        self.assertTrue(new_key.startswith("retrieval:21:"))

    def test_response_cache_requires_grounding_and_valid_citation(self):
        valid = SimpleNamespace(
            answer="EVIDENCE_STATUS: SUPPORTED\nMVCC provides snapshots. [S1]",
            chunks=[object()],
        )
        invalid = SimpleNamespace(
            answer="EVIDENCE_STATUS: SUPPORTED\nMVCC provides snapshots.",
            chunks=[object()],
        )
        self.assertTrue(ResponseCache.eligible(valid))
        self.assertFalse(ResponseCache.eligible(invalid))


if __name__ == "__main__":
    unittest.main()
