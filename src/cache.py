"""Fail-open Redis caching with deterministic, versioned keys."""

import hashlib
import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import redis

from src.config import CACHE_ENABLED, REDIS_URL
from src.db import get_connection


LOGGER = logging.getLogger(__name__)


def normalize_query(query):
    """Normalize insignificant whitespace while preserving case and identifiers."""
    return " ".join(query.strip().split())


def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(config):
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def get_corpus_version():
    with get_connection() as conn:
        row = conn.execute(
            "SELECT corpus_version FROM corpus_config WHERE id = 1"
        ).fetchone()
    if not row:
        raise RuntimeError("Corpus configuration is missing. Re-ingest the corpus.")
    return int(row[0])


@dataclass(frozen=True)
class CacheStats:
    hit: bool = False
    cache_lookup_seconds: float = 0.0
    compute_seconds: float = 0.0
    available: bool = True

    def to_dict(self):
        return asdict(self)


class Cache:
    """A generic JSON cache. Redis failures are warnings, never request failures."""

    def __init__(self, url=REDIS_URL, enabled=CACHE_ENABLED, client=None):
        self.enabled = enabled
        self.client = client or redis.Redis.from_url(
            url, decode_responses=True, socket_connect_timeout=0.25,
            socket_timeout=0.5,
        )

    def get_json(self, key):
        if not self.enabled:
            return None
        try:
            value = self.client.get(key)
            return None if value is None else json.loads(value)
        except (redis.RedisError, json.JSONDecodeError) as error:
            LOGGER.warning("Cache read failed; computing normally: %s", error)
            return None

    def lookup_json(self, key):
        started = time.perf_counter()
        if not self.enabled:
            return None, CacheStats(
                cache_lookup_seconds=time.perf_counter() - started,
                available=False,
            )
        try:
            value = self.client.get(key)
            elapsed = time.perf_counter() - started
            if value is None:
                return None, CacheStats(cache_lookup_seconds=elapsed)
            return json.loads(value), CacheStats(
                hit=True, cache_lookup_seconds=elapsed
            )
        except (redis.RedisError, json.JSONDecodeError) as error:
            LOGGER.warning("Cache lookup failed; computing normally: %s", error)
            return None, CacheStats(
                cache_lookup_seconds=time.perf_counter() - started,
                available=False,
            )

    def set_json(self, key, value, ttl):
        if not self.enabled:
            return False
        try:
            self.client.set(
                key, json.dumps(value, separators=(",", ":")), ex=ttl
            )
            return True
        except redis.RedisError as error:
            LOGGER.warning("Cache write failed; continuing normally: %s", error)
            return False

    def delete(self, key):
        if not self.enabled:
            return False
        try:
            self.client.delete(key)
            return True
        except redis.RedisError as error:
            LOGGER.warning("Cache delete failed: %s", error)
            return False

    def ping(self):
        if not self.enabled:
            return False
        try:
            return bool(self.client.ping())
        except redis.RedisError:
            return False

    @contextmanager
    def lock(self, key, ttl=120):
        """Best-effort SET-NX lock used to prevent response-cache stampedes."""
        token = uuid.uuid4().hex
        acquired = False
        if self.enabled:
            try:
                acquired = bool(self.client.set(key, token, nx=True, ex=ttl))
            except redis.RedisError as error:
                LOGGER.warning("Cache lock unavailable: %s", error)
        try:
            yield acquired
        finally:
            if acquired:
                try:
                    self.client.eval(
                        "if redis.call('get', KEYS[1]) == ARGV[1] then "
                        "return redis.call('del', KEYS[1]) else return 0 end",
                        1, key, token,
                    )
                except redis.RedisError as error:
                    LOGGER.warning("Cache lock release failed: %s", error)
