import json
import time
from src.utils.logger import logger

class CacheLayer:
    """Redis cache with in-memory fallback for low-latency inference."""

    def __init__(self):
        self.backend = "memory"
        self.store = {}
        self.ttl_store = {}
        try:
            import redis
            self.client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.client.ping()
            self.backend = "redis"
            logger.info("CacheLayer connected to Redis.")
        except Exception:
            logger.warning("Redis unavailable. Using in-memory cache fallback.")

    def get(self, key):
        if self.backend == "redis":
            val = self.client.get(key)
            return json.loads(val) if val else None
        # In-memory with TTL check
        if key in self.store:
            if key in self.ttl_store and time.time() > self.ttl_store[key]:
                del self.store[key]
                del self.ttl_store[key]
                return None
            return self.store[key]
        return None

    def set(self, key, value, ttl=60):
        if self.backend == "redis":
            self.client.setex(key, ttl, json.dumps(value))
        else:
            self.store[key] = value
            self.ttl_store[key] = time.time() + ttl

    def clear(self):
        if self.backend == "redis":
            self.client.flushdb()
        else:
            self.store.clear()
            self.ttl_store.clear()
