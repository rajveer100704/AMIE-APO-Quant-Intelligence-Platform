import json
import redis
import structlog
from typing import Any, Dict, Optional

logger = structlog.get_logger(__name__)

class StateManager:
    """
    Centralized state manager for AMIE-APO.
    Implements a Redis-first strategy with in-memory fallback.
    """
    def __init__(self, host='localhost', port=6379, db=0):
        self.host = host
        self.port = port
        self.db = db
        self.redis_client = None
        self._memory_store = {}
        self.mode = "memory"
        
        self._initialize_backend()

    def _initialize_backend(self):
        try:
            self.redis_client = redis.Redis(
                host=self.host, 
                port=self.port, 
                db=self.db, 
                socket_connect_timeout=2
            )
            self.redis_client.ping()
            self.mode = "redis"
            logger.info("StateManager initialized with Redis backend")
        except (redis.ConnectionError, redis.TimeoutError):
            self.mode = "memory"
            logger.warning("Redis unavailable. Falling back to In-Memory state store")

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Sets a value in the state store."""
        if self.mode == "redis":
            try:
                serialized = json.dumps(value)
                self.redis_client.set(key, serialized, ex=ttl)
            except Exception as e:
                logger.error("Redis set failed", key=key, error=str(e))
                self._memory_store[key] = value
        else:
            self._memory_store[key] = value

    def get(self, key: str) -> Optional[Any]:
        """Retrieves a value from the state store."""
        if self.mode == "redis":
            try:
                data = self.redis_client.get(key)
                return json.loads(data) if data else None
            except Exception as e:
                logger.error("Redis get failed", key=key, error=str(e))
                return self._memory_store.get(key)
        return self._memory_store.get(key)

    def delete(self, key: str):
        """Deletes a key from the state store."""
        if self.mode == "redis":
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.error("Redis delete failed", key=key, error=str(e))
        if key in self._memory_store:
            del self._memory_store[key]

    def get_status(self) -> Dict[str, str]:
        """Returns the current backend status."""
        return {"backend": self.mode}

# Singleton instance
state_manager = StateManager()
