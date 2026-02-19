"""
Sliding-window rate limiter backed by Redis.

Each unique key (IP address by default) is allowed at most
RATE_LIMIT_REQUESTS requests within a RATE_LIMIT_WINDOW_SECONDS rolling window.

The counter is stored as a Redis string with a TTL equal to the window size.
On the first request the key is created and the TTL is set; subsequent requests
within the same window simply increment the counter.  When the TTL expires
Redis deletes the key automatically and the window resets.
"""

import os
from fastapi import Request, HTTPException, status
import redis.asyncio as aioredis

RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))


def get_client_ip(request: Request) -> str:
    """Return the real client IP, respecting Railway's X-Forwarded-For header."""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def rate_limit(request: Request) -> None:
    """
    FastAPI dependency.  Raises HTTP 429 when the caller exceeds the limit.
    Reads the Redis client from app.state.redis (set during lifespan startup).
    """
    redis: aioredis.Redis = request.app.state.redis
    ip = get_client_ip(request)
    key = f"rl:{ip}"

    # Atomically increment and set TTL on first touch
    current: int = await redis.incr(key)
    if current == 1:
        await redis.expire(key, RATE_LIMIT_WINDOW_SECONDS)

    if current > RATE_LIMIT_REQUESTS:
        ttl: int = await redis.ttl(key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "limit": RATE_LIMIT_REQUESTS,
                "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
                "retry_after_seconds": ttl,
            },
            headers={"Retry-After": str(ttl)},
        )
