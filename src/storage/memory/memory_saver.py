import psycopg
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing import Optional, Union
import logging
import time

logger = logging.getLogger(__name__)

# Database connection timeout（seconds），Per attempt 15 seconds，Total attempts 2 times
DB_CONNECTION_TIMEOUT = 15
DB_MAX_RETRIES = 2


class MemoryManager:
    """Memory Manager Singleton class"""

    _instance: Optional['MemoryManager'] = None
    _checkpointer: Optional[Union[AsyncPostgresSaver, MemorySaver]] = None
    _pool: Optional[AsyncConnectionPool] = None
    _setup_done: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _connect_with_retry(self, db_url: str) -> Optional[psycopg.Connection]:
        """Database connection with retry, every 15 seconds timeout, total attempts 2 times"""
        last_error = None
        for attempt in range(1, DB_MAX_RETRIES + 1):
            try:
                logger.info(f"Attempting database connection (attempt {attempt}/{DB_MAX_RETRIES})")
                conn = psycopg.connect(db_url, autocommit=True, connect_timeout=DB_CONNECTION_TIMEOUT)
                logger.info(f"Database connection established on attempt {attempt}")
                return conn
            except Exception as e:
                last_error = e
                logger.warning(f"Database connection attempt {attempt} failed: {e}")
                if attempt < DB_MAX_RETRIES:
                    time.sleep(1)  # Brief wait before retry
        logger.error(f"All {DB_MAX_RETRIES} database connection attempts failed, last error: {last_error}")
        return None

    def _setup_schema_and_tables(self, db_url: str) -> bool:
        """Synchronously create schema and tables (execute only once), returns success status"""
        if self._setup_done:
            return True

        conn = self._connect_with_retry(db_url)
        if conn is None:
            return False

        try:
            with conn.cursor() as cur:
                cur.execute("CREATE SCHEMA IF NOT EXISTS memory")
            conn.execute("SET search_path TO memory")
            PostgresSaver(conn).setup()
            self._setup_done = True
            logger.info("Memory schema and tables created")
            return True
        except Exception as e:
            logger.warning(f"Failed to setup schema/tables: {e}")
            return False
        finally:
            conn.close()

    def _get_db_url_safe(self) -> Optional[str]:
        """Safely get db_url, returns None on failure"""
        try:
            from storage.database.db import get_db_url
            db_url = get_db_url()
            if db_url and db_url.strip():
                return db_url
            logger.warning("db_url is empty, will fallback to MemorySaver")
            return None
        except Exception as e:
            logger.warning(f"Failed to get db_url: {e}, will fallback to MemorySaver")
            return None

    def _create_fallback_checkpointer(self) -> MemorySaver:
        """CreateMemoryFallback checkpointer"""
        self._checkpointer = MemorySaver()
        logger.warning("Using MemorySaver as fallback checkpointer (data will not persist across restarts)")
        return self._checkpointer

    def get_checkpointer(self) -> BaseCheckpointSaver:
        """Get checkpointer, prefer PostgresSaver, fallback to MemorySaver on failure"""
        if self._checkpointer is not None:
            return self._checkpointer

        # 1. Try to get db_url
        db_url = self._get_db_url_safe()
        if not db_url:
            return self._create_fallback_checkpointer()

        # 2. Try to connect to database and create schema/tables (with retry)
        if not self._setup_schema_and_tables(db_url):
            return self._create_fallback_checkpointer()

        # 3. Add search_path to connection string
        if "?" in db_url:
            db_url = f"{db_url}&options=-csearch_path%3Dmemory"
        else:
            db_url = f"{db_url}?options=-csearch_path%3Dmemory"

        # 4. Try to create connection pool and checkpointer
        try:
            self._pool = AsyncConnectionPool(
                conninfo=db_url,
                timeout=DB_CONNECTION_TIMEOUT,
                min_size=1,
                max_idle=300,
                check=AsyncConnectionPool.check_connection,
            )
            self._checkpointer = AsyncPostgresSaver(self._pool)
            logger.info("AsyncPostgresSaver initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to create AsyncPostgresSaver: {e}, will fallback to MemorySaver")
            return self._create_fallback_checkpointer()

        return self._checkpointer

_memory_manager: Optional[MemoryManager] = None


def get_memory_saver() -> BaseCheckpointSaver:
    """Get checkpointer, prefer PostgresSaver, fallback to MemorySaver when db_url unavailable or connection fails"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager.get_checkpointer()