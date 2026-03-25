import logging
import os
from contextlib import asynccontextmanager
from typing import Protocol
from urllib.parse import urlunsplit


class SuppressHealthcheckAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "GET /health " not in record.getMessage()


class LifecycleManager(Protocol):
    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(existing, SuppressHealthcheckAccessFilter) for existing in access_logger.filters):
        access_logger.addFilter(SuppressHealthcheckAccessFilter())

    return logging.getLogger("s2s-endpoint")


def build_lifespan(manager: LifecycleManager):
    @asynccontextmanager
    async def lifespan(app):
        await manager.start()
        try:
            yield
        finally:
            await manager.stop()

    return lifespan


def public_base_url(request) -> str:
    scheme = request.headers.get("x-forwarded-proto", "").strip() or request.url.scheme
    host = request.headers.get("x-forwarded-host", "").strip() or request.headers.get("host", "").strip()
    if not host:
        host = request.url.netloc
    return urlunsplit((scheme, host, "/", "", ""))
