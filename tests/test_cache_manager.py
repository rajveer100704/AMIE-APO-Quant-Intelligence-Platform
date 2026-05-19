import pytest
import time
from unittest.mock import patch, MagicMock
from src.api.cache_manager import CacheLayer

@pytest.mark.unit
def test_cache_layer_memory_mode():
    # Force memory mode by patching redis to raise Exception or import failure
    with patch("redis.Redis", side_effect=Exception("Redis unavailable")):
        cache = CacheLayer()
        assert cache.backend == "memory"
        
        # Test Set/Get
        cache.set("key1", {"data": 123}, ttl=2)
        assert cache.get("key1") == {"data": 123}
        
        # Test Non-existent key
        assert cache.get("key_missing") is None
        
        # Test TTL expiration
        cache.set("key2", {"data": 456}, ttl=0.1)
        time.sleep(0.2)
        assert cache.get("key2") is None
        
        # Test Clear
        cache.set("key3", "value3")
        assert cache.get("key3") == "value3"
        cache.clear()
        assert cache.get("key3") is None

@pytest.mark.unit
def test_cache_layer_redis_mode():
    mock_redis_client = MagicMock()
    mock_redis_client.ping.return_value = True
    mock_redis_client.get.return_value = '{"data": "redis_val"}'
    
    with patch("redis.Redis", return_value=mock_redis_client):
        cache = CacheLayer()
        assert cache.backend == "redis"
        
        # Test Get
        res = cache.get("redis_key")
        assert res == {"data": "redis_val"}
        mock_redis_client.get.assert_called_with("redis_key")
        
        # Test Set
        cache.set("redis_key", {"data": "new_val"}, ttl=10)
        mock_redis_client.setex.assert_called_with("redis_key", 10, '{"data": "new_val"}')
        
        # Test Clear
        cache.clear()
        mock_redis_client.flushdb.assert_called_once()
