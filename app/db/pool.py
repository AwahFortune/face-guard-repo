"""
MySQL connection pool — one pool for the lifetime of the process.
Exposes get_conn() as a context manager so callers never leak connections.
"""
import logging
import time
from contextlib import contextmanager
from typing import Generator, Tuple

import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool

from ..core.config import settings

logger = logging.getLogger(__name__)

_pool: MySQLConnectionPool | None = None


def init_pool() -> None:
    global _pool
    _pool = MySQLConnectionPool(
        pool_name=settings.MYSQL_POOL_NAME,
        pool_size=settings.MYSQL_POOL_SIZE,
        host=settings.MYSQL_HOST,
        port=settings.MYSQL_PORT,
        user=settings.MYSQL_USER,
        password=settings.MYSQL_PASSWORD,
        database=settings.MYSQL_DATABASE,
        autocommit=False,
    )
    logger.info("MySQL pool initialised (size=%d)", settings.MYSQL_POOL_SIZE)
    _ensure_schema()


def _ensure_schema() -> None:
    with get_conn() as (conn, cur):
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id      VARCHAR(64)  PRIMARY KEY,
                embedding    BLOB         NOT NULL,
                nonce        BLOB         NOT NULL,
                hmac         BLOB         NOT NULL,
                det_score    FLOAT,
                model_version VARCHAR(16),
                registration_time BIGINT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS authorization_logs (
                log_id       INT AUTO_INCREMENT PRIMARY KEY,
                user_id      VARCHAR(64)  NOT NULL,
                similarity   FLOAT,
                det_score    FLOAT,
                model_version VARCHAR(16),
                ip_address   VARCHAR(45),
                timestamp    BIGINT,
                status       VARCHAR(15)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_images (
                image_id     INT AUTO_INCREMENT PRIMARY KEY,
                user_id      VARCHAR(64)  NOT NULL,
                image_path   VARCHAR(255),
                det_score    FLOAT,
                registration_time BIGINT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS failed_attempts (
                user_id      VARCHAR(64)  PRIMARY KEY,
                count        INT          NOT NULL DEFAULT 0,
                last_attempt BIGINT       NOT NULL,
                locked_until BIGINT       NOT NULL DEFAULT 0
            )
        """)
        conn.commit()
    logger.info("DB schema verified")


@contextmanager
def get_conn() -> Generator[Tuple, None, None]:
    if _pool is None:
        raise RuntimeError("DB pool not initialised — call init_pool() first")
    try:
        conn = _pool.get_connection()
    except mysql.connector.errors.PoolError as exc:
        raise RuntimeError("DB pool exhausted — all connections in use") from exc
    cur = conn.cursor()
    try:
        yield conn, cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


# ── Brute-force helpers ───────────────────────────────────────────────────────

def get_failed_attempts(user_id: str) -> Tuple[int, int]:
    """Return (count, locked_until_epoch)."""
    with get_conn() as (_, cur):
        cur.execute(
            "SELECT count, locked_until FROM failed_attempts WHERE user_id=%s",
            (user_id,)
        )
        row = cur.fetchone()
    if row is None:
        return 0, 0
    return row[0], row[1]


def record_failed_attempt(user_id: str) -> int:
    """Increment failure counter. Return new count."""
    now = int(time.time())
    with get_conn() as (conn, cur):
        cur.execute("""
            INSERT INTO failed_attempts (user_id, count, last_attempt, locked_until)
            VALUES (%s, 1, %s, 0)
            ON DUPLICATE KEY UPDATE count=count+1, last_attempt=%s
        """, (user_id, now, now))
        conn.commit()
        cur.execute("SELECT count FROM failed_attempts WHERE user_id=%s", (user_id,))
        count = cur.fetchone()[0]

    if count >= settings.MAX_FAILED_ATTEMPTS:
        locked_until = now + settings.LOCKOUT_SECONDS
        with get_conn() as (conn, cur):
            cur.execute(
                "UPDATE failed_attempts SET locked_until=%s WHERE user_id=%s",
                (locked_until, user_id)
            )
            conn.commit()
    return count


def clear_failed_attempts(user_id: str) -> None:
    with get_conn() as (conn, cur):
        cur.execute(
            "UPDATE failed_attempts SET count=0, locked_until=0 WHERE user_id=%s",
            (user_id,)
        )
        conn.commit()
