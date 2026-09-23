"""Tests du rate limiting — mémoire (une instance) et Redis (plusieurs instances/réplicas)."""

from __future__ import annotations

import asyncio

import pytest
from fakeredis import FakeAsyncRedis

from api.middleware import RateLimiter, RedisRateLimiter


def test_in_memory_rate_limiter_blocks_beyond_quota() -> None:
    async def run() -> list[bool]:
        limiter = RateLimiter(rate_per_minute=3)
        return [await limiter.allow("client-a") for _ in range(5)]

    assert asyncio.run(run()) == [True, True, True, False, False]


def test_in_memory_rate_limiter_clients_are_independent() -> None:
    async def run() -> tuple[bool, bool, bool]:
        limiter = RateLimiter(rate_per_minute=1)
        first = await limiter.allow("client-a")
        second = await limiter.allow("client-a")
        other = await limiter.allow("client-b")
        return first, second, other

    assert asyncio.run(run()) == (True, False, True)


def test_rate_limiter_rejects_non_positive_quota() -> None:
    with pytest.raises(ValueError, match="strictement positif"):
        RateLimiter(rate_per_minute=0)
    with pytest.raises(ValueError, match="strictement positif"):
        RedisRateLimiter(FakeAsyncRedis(), rate_per_minute=-1)


def test_redis_rate_limiter_blocks_beyond_quota() -> None:
    async def run() -> list[bool]:
        limiter = RedisRateLimiter(FakeAsyncRedis(), rate_per_minute=3)
        return [await limiter.allow("client-a") for _ in range(5)]

    assert asyncio.run(run()) == [True, True, True, False, False]


def test_redis_rate_limiter_clients_are_independent() -> None:
    async def run() -> tuple[bool, bool, bool]:
        limiter = RedisRateLimiter(FakeAsyncRedis(), rate_per_minute=1)
        first = await limiter.allow("client-a")
        second = await limiter.allow("client-a")
        other = await limiter.allow("client-b")
        return first, second, other

    assert asyncio.run(run()) == (True, False, True)


def test_redis_rate_limiter_quota_shared_across_instances() -> None:
    """Le cas d'usage central : deux instances (deux réplicas API) partageant le même Redis
    doivent partager le même quota — pas deux quotas séparés comme le ferait RateLimiter."""

    async def run() -> list[bool]:
        shared_store = FakeAsyncRedis()
        limiter_replica_a = RedisRateLimiter(shared_store, rate_per_minute=4)
        limiter_replica_b = RedisRateLimiter(shared_store, rate_per_minute=4)
        results = []
        for i in range(6):
            limiter = limiter_replica_a if i % 2 == 0 else limiter_replica_b
            results.append(await limiter.allow("shared-client"))
        return results

    assert asyncio.run(run()) == [True, True, True, True, False, False]


def test_redis_rate_limiter_window_expires() -> None:
    async def run() -> tuple[bool, bool, bool]:
        limiter = RedisRateLimiter(FakeAsyncRedis(), rate_per_minute=1, window_seconds=1)
        first = await limiter.allow("client-a")
        second = await limiter.allow("client-a")
        await asyncio.sleep(1.1)
        third = await limiter.allow("client-a")
        return first, second, third

    assert asyncio.run(run()) == (True, False, True)
