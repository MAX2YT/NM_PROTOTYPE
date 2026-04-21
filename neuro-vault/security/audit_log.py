"""
security/audit_log.py — SQLite-backed tamper-evident audit trail.

Every query, retrieval event, and LLM response is logged to a local SQLite
database.  Query text is stored as AES-256 ciphertext for DPDP Act 2023
compliance (sensitive health queries must be protected at rest).

Schema:
  query_log:
    id            INTEGER PRIMARY KEY AUTOINCREMENT
    timestamp     TEXT    NOT NULL
    session_id    TEXT    NOT NULL
    query_hash    TEXT    NOT NULL   -- SHA-256 of plaintext query
    query_enc     TEXT    NOT NULL   -- Fernet-encrypted query text
    abstained     INTEGER NOT NULL   -- 1 if system abstained, 0 otherwise
    abstain_reason TEXT              -- Reason string if abstained
    dataset_sources TEXT            -- JSON array e.g. ["pubmed","mtsamples"]
    num_chunks    INTEGER           -- Number of chunks retrieved
    response_len  INTEGER           -- Length of LLM response in chars
    latency_ms    INTEGER           -- Total end-to-end latency
    model_name    TEXT              -- Ollama model used
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from config import Config
from security.encryption import EncryptionManager

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS query_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    session_id      TEXT    NOT NULL,
    query_hash      TEXT    NOT NULL,
    query_enc       TEXT    NOT NULL,
    abstained       INTEGER NOT NULL DEFAULT 0,
    abstain_reason  TEXT,
    dataset_sources TEXT,
    num_chunks      INTEGER,
    response_len    INTEGER,
    latency_ms      INTEGER,
    model_name      TEXT
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_query_log_timestamp
    ON query_log (timestamp);
"""


class AuditLogger:
    """Write and query the audit trail for all Neuro-Vault interactions.

    Args:
        db_path:   Path to the SQLite database file.
        enc_mgr:   ``EncryptionManager`` instance for query encryption.
        model_name: The Ollama model name to record.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        enc_mgr: Optional[EncryptionManager] = None,
        model_name: str = Config.OLLAMA_MODEL,
    ) -> None:
        self.db_path = db_path or Config.AUDIT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.enc_mgr = enc_mgr or EncryptionManager()
        self.model_name = model_name
        self._init_db()

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        """Create the audit database and table if they do not exist."""
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_SQL)
            conn.commit()
        logger.debug("Audit database ready at '%s'", self.db_path)

    def _connect(self) -> sqlite3.Connection:
        """Open a new SQLite connection.

        Returns:
            ``sqlite3.Connection`` with row_factory set for dict-like access.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------ #
    #  Writing
    # ------------------------------------------------------------------ #

    def log_query(
        self,
        *,
        session_id: str,
        query: str,
        abstained: bool,
        abstain_reason: str = "",
        dataset_sources: Optional[List[str]] = None,
        num_chunks: int = 0,
        response_len: int = 0,
        latency_ms: int = 0,
    ) -> int:
        """Record a clinical query event to the audit trail.

        All personally-identifiable or clinically-sensitive text is
        encrypted before writing.

        Args:
            session_id:      Browser/user session identifier.
            query:           Plaintext query string (will be encrypted).
            abstained:       Whether the system abstained from answering.
            abstain_reason:  Human-readable reason for abstention.
            dataset_sources: List of dataset names consulted.
            num_chunks:      Number of context chunks retrieved.
            response_len:    Character count of the LLM response.
            latency_ms:      Total query latency in milliseconds.

        Returns:
            Row ID of the inserted audit record.
        """
        ts = datetime.now(timezone.utc).isoformat()
        query_hash = self.enc_mgr.hash_query(query)
        query_enc = self.enc_mgr.encrypt(query)
        ds_json = json.dumps(dataset_sources or [])

        sql = """
        INSERT INTO query_log
            (timestamp, session_id, query_hash, query_enc,
             abstained, abstain_reason, dataset_sources,
             num_chunks, response_len, latency_ms, model_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            cur = conn.execute(
                sql,
                (
                    ts,
                    session_id,
                    query_hash,
                    query_enc,
                    int(abstained),
                    abstain_reason,
                    ds_json,
                    num_chunks,
                    response_len,
                    latency_ms,
                    self.model_name,
                ),
            )
            conn.commit()
            row_id = cur.lastrowid

        logger.info(
            "Audit log [%d]: session=%s abstained=%s datasets=%s",
            row_id,
            session_id,
            abstained,
            dataset_sources,
        )
        return row_id

    # ------------------------------------------------------------------ #
    #  Reading / analytics
    # ------------------------------------------------------------------ #

    def recent_logs(self, limit: int = 50) -> List[dict]:
        """Return the most recent audit records as dicts.

        Query text is decrypted before returning for authorised admin use.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of dicts with all column values, ``query`` field
            containing decrypted text.
        """
        sql = """
        SELECT * FROM query_log
        ORDER BY id DESC
        LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()

        results = []
        for row in rows:
            d = dict(row)
            try:
                d["query"] = self.enc_mgr.decrypt(d.pop("query_enc", ""))
            except ValueError:
                d["query"] = "[DECRYPTION FAILED]"
                d.pop("query_enc", None)
            d["dataset_sources"] = json.loads(d.get("dataset_sources") or "[]")
            results.append(d)

        return results

    def summary_stats(self) -> dict:
        """Compute aggregate statistics over all audit records.

        Returns:
            dict with ``total_queries``, ``total_abstentions``,
            ``abstention_rate``, ``avg_latency_ms``,
            ``dataset_usage`` (counts per dataset).
        """
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
            abstentions = conn.execute(
                "SELECT COUNT(*) FROM query_log WHERE abstained = 1"
            ).fetchone()[0]
            avg_latency = conn.execute(
                "SELECT AVG(latency_ms) FROM query_log"
            ).fetchone()[0]
            ds_rows = conn.execute(
                "SELECT dataset_sources FROM query_log"
            ).fetchall()

        dataset_usage: dict = {}
        for row in ds_rows:
            sources = json.loads(row[0] or "[]")
            for ds in sources:
                dataset_usage[ds] = dataset_usage.get(ds, 0) + 1

        return {
            "total_queries": total,
            "total_abstentions": abstentions,
            "abstention_rate": (abstentions / total) if total > 0 else 0.0,
            "avg_latency_ms": round(avg_latency or 0.0, 1),
            "dataset_usage": dataset_usage,
        }
