# tests/test_resilience_new.py
import threading
import time

import pytest

from data.resilience import CircuitBreaker, TokenBucket


class TestCircuitBreaker:

    def test_closed_by_default(self):
        cb = CircuitBreaker("test", failure_threshold=3, cooldown_secs=60)
        assert not cb.is_open
        assert cb.state() == "closed"

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker("test", failure_threshold=3, cooldown_secs=60)
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open
        cb.record_failure()
        assert cb.is_open
        assert cb.state() == "open"

    def test_success_resets_counter(self):
        cb = CircuitBreaker("test", failure_threshold=2, cooldown_secs=60)
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert not cb.is_open

    def test_half_open_after_cooldown(self):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_secs=0.05)
        cb.record_failure()
        assert cb.is_open
        time.sleep(0.06)
        assert not cb.is_open
        assert cb.state() == "half-open"

    def test_thread_safe(self):
        cb = CircuitBreaker("test", failure_threshold=100, cooldown_secs=60)
        errors = []

        def hammer():
            try:
                for _ in range(50):
                    cb.record_failure()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        assert cb._failures == 200


class TestTokenBucket:

    def test_initial_tokens_allow_immediate_requests(self):
        bucket = TokenBucket(rate=10.0, capacity=5)
        for _ in range(5):
            assert bucket.acquire(block=False)

    def test_sixth_request_fails_non_blocking(self):
        bucket = TokenBucket(rate=1.0, capacity=5)
        for _ in range(5):
            bucket.acquire(block=False)
        assert not bucket.acquire(block=False)

    def test_tokens_refill_over_time(self):
        bucket = TokenBucket(rate=100.0, capacity=1)
        bucket.acquire(block=False)
        assert not bucket.acquire(block=False)
        time.sleep(0.015)
        assert bucket.acquire(block=False)
