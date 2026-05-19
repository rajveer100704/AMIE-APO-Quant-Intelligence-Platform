import pytest
from unittest.mock import patch, MagicMock
from src.state.state_manager import StateManager, state_manager

@pytest.mark.unit
def test_state_manager_memory_mode():
    with patch("redis.Redis", side_effect=Exception("Redis connection error")):
        sm = StateManager()
        assert sm.mode == "memory"
        
        # Test Set/Get
        sm.set("mystate", {"status": "active"})
        assert sm.get("mystate") == {"status": "active"}
        
        # Test Non-existent key
        assert sm.get("missing_state") is None
        
        # Test Delete
        sm.delete("mystate")
        assert sm.get("mystate") is None
        
        # Test get_status
        assert sm.get_status() == {"backend": "memory"}

@pytest.mark.unit
def test_state_manager_redis_mode():
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    mock_redis.get.return_value = '{"status": "running"}'
    
    with patch("redis.Redis", return_value=mock_redis):
        sm = StateManager()
        assert sm.mode == "redis"
        
        # Test Get
        res = sm.get("redis_key")
        assert res == {"status": "running"}
        mock_redis.get.assert_called_with("redis_key")
        
        # Test Set
        sm.set("redis_key", {"status": "updated"}, ttl=300)
        mock_redis.set.assert_called_with("redis_key", '{"status": "updated"}', ex=300)
        
        # Test Delete
        sm.delete("redis_key")
        mock_redis.delete.assert_called_with("redis_key")
        
        # Test get_status
        assert sm.get_status() == {"backend": "redis"}

@pytest.mark.unit
def test_state_manager_redis_exceptions():
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    # Make get/set/delete raise exceptions to trigger fallback memory store logic
    mock_redis.get.side_effect = Exception("Redis error")
    mock_redis.set.side_effect = Exception("Redis error")
    mock_redis.delete.side_effect = Exception("Redis error")
    
    with patch("redis.Redis", return_value=mock_redis):
        sm = StateManager()
        assert sm.mode == "redis"
        
        # Test Set exception falls back to memory
        sm.set("key_err", "val_err")
        # Test Get exception falls back to memory
        assert sm.get("key_err") == "val_err"
        
        # Test Delete exception doesn't crash and deletes from memory
        sm.delete("key_err")
        assert sm.get("key_err") is None

@pytest.mark.unit
def test_global_state_manager():
    # Verify the global singleton exists
    assert state_manager is not None
