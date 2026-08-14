# tests/test_stale_cache.py
import time
import pytest
from data.cache import get_stale_cache_obj, set_cache_obj, get_cache_obj


def test_get_stale_returns_expired_value():
    key = f"test:stale:{time.time()}"
    set_cache_obj(key, {"x": 1}, ttl=1)   # TTL = 1 second
    time.sleep(1.1)
    # Fresh cache returns None (expired)
    assert get_cache_obj(key) is None
    # Stale accessor returns the value and indicates it's past TTL
    result = get_stale_cache_obj(key)
    assert result is not None
    value, expires_at = result
    assert value == {"x": 1}
    assert expires_at < time.time()  # past expiry


def test_get_stale_returns_none_for_missing_key():
    key = "test:stale:never_written_xyz"
    assert get_stale_cache_obj(key) is None


def test_get_stale_returns_fresh_value_too():
    key = f"test:stale:fresh:{time.time()}"
    set_cache_obj(key, {"y": 2}, ttl=3600)
    result = get_stale_cache_obj(key)
    assert result is not None
    value, expires_at = result
    assert value == {"y": 2}
    assert expires_at > time.time()
