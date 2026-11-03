# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
import os
import sqlite3
import time
from typing import Optional

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cache (
    key       TEXT PRIMARY KEY,
    blob      BLOB NOT NULL,
    blob_size INTEGER NOT NULL,
    atime     REAL NOT NULL
)
"""

_CACHE_FILENAME = "cache.db"


def _close(conn):
    if conn:
        try:
            conn.close()
        except sqlite3.Error:
            pass


def _open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.execute(_CREATE_TABLE_SQL)
    conn.execute("DROP INDEX IF EXISTS idx_cache_atime")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_atime_key on cache(atime, key)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_blob_size on cache(blob_size)")
    return conn


def _connect(cache_dir: str) -> sqlite3.Connection:
    os.makedirs(cache_dir, exist_ok=True)
    db_path = os.path.join(cache_dir, _CACHE_FILENAME)
    try:
        return _open_db(db_path)
    except sqlite3.Error:
        logger.debug("cache db corrupt, recreating %s", db_path,
                     exc_info=True)
        try:
            os.unlink(db_path)
        except OSError:
            pass
        return _open_db(db_path)


_CACHE_VERSION = b''


def cache_key(compiler_version: str, sm_arch: str, opt_level: int,
              bytecode: bytes) -> str:

    def encode_uint(x: int):
        return int.to_bytes(x, 4, byteorder='big', signed=False)

    version = compiler_version.encode()
    arch = sm_arch.encode()

    h = hashlib.sha256()
    h.update(_CACHE_VERSION)
    h.update(encode_uint(len(version)))
    h.update(version)
    h.update(encode_uint(len(arch)))
    h.update(arch)
    h.update(encode_uint(opt_level))
    h.update(encode_uint(len(bytecode)))
    h.update(bytecode)
    return h.hexdigest()


def cache_lookup(cache_dir: str, key: str) -> Optional[bytes]:
    conn = None
    try:
        conn = _connect(cache_dir)
        row = conn.execute(
            "SELECT blob FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE cache SET atime = ? WHERE key = ?",
            (time.time(), key)
        )
        conn.commit()
        blob = row[0]
        return blob
    except (sqlite3.Error, OSError):
        logger.debug("cache lookup failed for %s", key, exc_info=True)
        return None
    finally:
        _close(conn)


def cache_store(cache_dir: str, key: str, cubin: bytes) -> None:
    conn = None
    try:
        conn = _connect(cache_dir)
        conn.execute(
            "INSERT OR IGNORE INTO cache"
            " (key, blob, blob_size, atime) VALUES (?, ?, ?, ?)",
            (key, cubin, len(cubin), time.time())
        )
        conn.commit()
    except (sqlite3.Error, OSError):
        logger.debug("cache store failed for %s", key, exc_info=True)
    finally:
        _close(conn)


def evict_lru(cache_dir: str, size_limit: int) -> None:
    conn = None
    try:
        conn = _connect(cache_dir)
        row_limit = 100
        while True:
            res = conn.execute("""
            DELETE FROM cache WHERE key IN (SELECT key FROM
                (SELECT key, SUM(blob_size) OVER (ORDER BY atime, key) as cumul_size
                    FROM cache ORDER BY atime, key limit ?)
                WHERE cumul_size <= (SELECT SUM(blob_size) - ? FROM cache)
            )
            """, (row_limit, size_limit))
            if res.rowcount < row_limit:
                break
            row_limit *= 10
        conn.commit()
    except sqlite3.Error:
        logger.debug("cache evict failed", exc_info=True)
    finally:
        _close(conn)
