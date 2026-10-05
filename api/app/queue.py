"""Enqueue background jobs (arq) from API request handlers."""

from typing import Any

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from app.config import get_settings

_pool: ArqRedis | None = None


async def get_queue() -> ArqRedis:
    global _pool
    if _pool is None:
        _pool = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    return _pool


async def enqueue(function: str, *args: Any) -> None:
    """Job functions are registered in app.worker.WorkerSettings."""
    await (await get_queue()).enqueue_job(function, *args)
