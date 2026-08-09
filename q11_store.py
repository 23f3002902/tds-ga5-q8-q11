from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

DATA_DIR = "data"
DB_PATH = "data/incidents.db"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def init_db() -> None:
    import os
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            request_hash TEXT NOT NULL,
            state_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS receipts (
            run_id TEXT NOT NULL,
            receipt_id TEXT NOT NULL,
            receipt_hash TEXT NOT NULL,
            response_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, receipt_id),
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );
    """)
    conn.commit()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_run(run_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None:
        return None
    return {
        "request_hash": row["request_hash"],
        "state": json.loads(row["state_json"]),
    }


def save_run(run_id: str, request_hash: str, state: dict) -> None:
    conn = _get_conn()
    ts = now_iso()
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            """INSERT OR REPLACE INTO runs (run_id, request_hash, state_json, created_at, updated_at)
               VALUES (?, ?, ?, COALESCE((SELECT created_at FROM runs WHERE run_id = ?), ?), ?)""",
            (run_id, request_hash, json.dumps(state), run_id, ts, ts),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def load_receipt(run_id: str, receipt_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM receipts WHERE run_id = ? AND receipt_id = ?",
        (run_id, receipt_id),
    ).fetchone()
    if row is None:
        return None
    return {
        "receipt_hash": row["receipt_hash"],
        "response": json.loads(row["response_json"]),
    }


def save_receipt(run_id: str, receipt_id: str, receipt_hash: str, response: dict) -> None:
    conn = _get_conn()
    ts = now_iso()
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            """INSERT OR REPLACE INTO receipts (run_id, receipt_id, receipt_hash, response_json, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (run_id, receipt_id, receipt_hash, json.dumps(response), ts),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
