# neurotrace/control/circuit_registry.py

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


@dataclass
class CircuitComponent:
    """
    Single component of a circuit at model level.
    Example: layer_3.attn_head_1, layer_10.mlp_dir_7, sae_feature_42, ...
    """
    layer: int
    component_type: str  # "attention_head", "mlp", "sae_direction", "residual", ...
    index: int           # head index, direction index, etc.
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitCausalMetrics:
    """
    Aggregate causal metrics associated with a discovered circuit.
    These come from the hierarchical causal tester.
    """
    vlo_mean: float
    vlo_std: float = 0.0
    faithfulness: float = 0.0
    causal_scrubbing: float = 0.0
    effect_size_vs_random: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircuitCausalMetrics":
        return cls(
            vlo_mean=data.get("vlo_mean", 0.0),
            vlo_std=data.get("vlo_std", 0.0),
            faithfulness=data.get("faithfulness", 0.0),
            causal_scrubbing=data.get("causal_scrubbing", 0.0),
            effect_size_vs_random=data.get("effect_size_vs_random", 0.0),
        )


@dataclass
class CircuitSemantics:
    """
    Human-oriented semantic metadata.
    """
    task_tag: str
    human_label: str = ""
    description: str = ""
    examples: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircuitSemantics":
        return cls(
            task_tag=data.get("task_tag", ""),
            human_label=data.get("human_label", ""),
            description=data.get("description", ""),
            examples=list(data.get("examples", []) or []),
            tags=list(data.get("tags", []) or []),
        )


@dataclass
class CircuitFeatures:
    """
    Feature-level summary: SAE indices, geometric features, etc.
    This is deliberately generic; raw feature tensors live in other stores.
    """
    sae_indices: Dict[str, List[int]] = field(default_factory=dict)  # e.g. {"layer_12": [15, 42]}
    geometric: Dict[str, Any] = field(default_factory=dict)          # LID, spectral_entropy, etc.
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircuitFeatures":
        return cls(
            sae_indices=data.get("sae_indices", {}) or {},
            geometric=data.get("geometric", {}) or {},
            extra=data.get("extra", {}) or {},
        )


@dataclass
class CircuitRecord:
    """
    Full record of a discovered circuit, suitable for long-term storage.
    """
    circuit_id: str
    model_name: str
    model_revision: str = "unknown"
    components: List[CircuitComponent] = field(default_factory=list)
    features: CircuitFeatures = field(default_factory=CircuitFeatures)
    causal_metrics: CircuitCausalMetrics = field(
        default_factory=lambda: CircuitCausalMetrics(vlo_mean=0.0)
    )
    semantics: CircuitSemantics = field(
        default_factory=lambda: CircuitSemantics(task_tag="unknown")
    )
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().strftime(ISO_FORMAT)
    )

    def to_json(self) -> str:
        def _default(o: Any) -> Any:
            if hasattr(o, "__dict__"):
                return o.__dict__
            if isinstance(o, CircuitComponent):
                return asdict(o)
            if isinstance(o, (CircuitFeatures, CircuitCausalMetrics, CircuitSemantics)):
                return asdict(o)
            raise TypeError(f"Cannot serialize {type(o)}")

        return json.dumps(asdict(self), default=_default)

    @classmethod
    def from_json(cls, raw: str) -> "CircuitRecord":
        data = json.loads(raw)

        components = [
            CircuitComponent(**cmp) for cmp in data.get("components", []) or []
        ]

        features = CircuitFeatures.from_dict(data.get("features", {}) or {})
        causal_metrics = CircuitCausalMetrics.from_dict(
            data.get("causal_metrics", {}) or {}
        )
        semantics = CircuitSemantics.from_dict(data.get("semantics", {}) or {})

        return cls(
            circuit_id=data["circuit_id"],
            model_name=data.get("model_name", "unknown"),
            model_revision=data.get("model_revision", "unknown"),
            components=components,
            features=features,
            causal_metrics=causal_metrics,
            semantics=semantics,
            created_at=data.get("created_at", datetime.utcnow().strftime(ISO_FORMAT)),
        )


class CircuitRegistry:
    """
    Thin persistence layer around an SQLite DB storing circuit records.

    Table schema (single table 'circuits'):
      circuit_id TEXT PRIMARY KEY
      model_name TEXT
      task_tag   TEXT
      vlo_mean   REAL
      faithfulness REAL
      created_at TEXT (ISO UTC)
      blob       TEXT (full JSON record)
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        parent_dir = os.path.dirname(db_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self._lock = threading.RLock()
        # Keep a persistent connection for in-memory databases
        self._persistent_conn: Optional[sqlite3.Connection] = None
        if db_path == ":memory:":
            self._persistent_conn = self._create_connection()
        self._init_db()

    def close(self) -> None:
        """
        Close all connections and checkpoint WAL file.
        Call this before deleting the database file (especially on Windows).
        """
        with self._lock:
            conn = self._connect()
            try:
                # Checkpoint WAL file to main database
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                conn.commit()
            finally:
                conn.close()

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _connect(self) -> sqlite3.Connection:
        """Get a connection (persistent for in-memory, new for file-based)."""
        if self._persistent_conn is not None:
            return self._persistent_conn
        return self._create_connection()

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS circuits (
                        circuit_id TEXT PRIMARY KEY,
                        model_name TEXT NOT NULL,
                        task_tag   TEXT NOT NULL,
                        vlo_mean   REAL,
                        faithfulness REAL,
                        created_at TEXT NOT NULL,
                        blob       TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
            finally:
                # Don't close persistent connection
                if self._persistent_conn is None:
                    conn.close()

    # ----------------------------- Public API -----------------------------

    def upsert(self, record: CircuitRecord) -> None:
        """
        Insert or replace a circuit record.
        """
        blob = record.to_json()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO circuits (
                        circuit_id, model_name, task_tag,
                        vlo_mean, faithfulness, created_at, blob
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(circuit_id) DO UPDATE SET
                        model_name=excluded.model_name,
                        task_tag=excluded.task_tag,
                        vlo_mean=excluded.vlo_mean,
                        faithfulness=excluded.faithfulness,
                        created_at=excluded.created_at,
                        blob=excluded.blob
                    """,
                    (
                        record.circuit_id,
                        record.model_name,
                        record.semantics.task_tag,
                        record.causal_metrics.vlo_mean,
                        record.causal_metrics.faithfulness,
                        record.created_at,
                        blob,
                    ),
                )
                conn.commit()
            finally:
                # Don't close persistent connection
                if self._persistent_conn is None:
                    conn.close()

    def get(self, circuit_id: str) -> Optional[CircuitRecord]:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT blob FROM circuits WHERE circuit_id = ?", (circuit_id,)
                )
                row = cur.fetchone()
            finally:
                # Don't close persistent connection
                if self._persistent_conn is None:
                    conn.close()
        if not row:
            return None
        return CircuitRecord.from_json(row[0])

    def delete(self, circuit_id: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM circuits WHERE circuit_id = ?", (circuit_id,))
                conn.commit()
            finally:
                # Don't close persistent connection
                if self._persistent_conn is None:
                    conn.close()

    def list(
        self,
        model_name: Optional[str] = None,
        task_tag: Optional[str] = None,
        min_vlo: Optional[float] = None,
        min_faithfulness: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[CircuitRecord]:
        """
        List circuits with simple filters. This is the "index scan" path;
        heavy semantic queries should go through vector DB / analysis.
        """
        query = "SELECT blob FROM circuits WHERE 1=1"
        params: List[Any] = []

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        if task_tag:
            query += " AND task_tag = ?"
            params.append(task_tag)

        if min_vlo is not None:
            query += " AND vlo_mean >= ?"
            params.append(min_vlo)

        if min_faithfulness is not None:
            query += " AND faithfulness >= ?"
            params.append(min_faithfulness)

        query += " ORDER BY created_at DESC"

        if limit is not None:
            query += f" LIMIT {int(limit)}"

        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
            finally:
                # Don't close persistent connection
                if self._persistent_conn is None:
                    conn.close()

        return [CircuitRecord.from_json(row[0]) for row in rows]

    def iter_all(self, batch_size: int = 256) -> Iterable[CircuitRecord]:
        """
        Efficient streaming over all circuits, for offline analysis / migration.
        """
        offset = 0
        while True:
            with self._lock:
                conn = self._connect()
                try:
                    cur = conn.execute(
                        "SELECT blob FROM circuits ORDER BY created_at DESC LIMIT ? OFFSET ?",
                        (batch_size, offset),
                    )
                    rows = cur.fetchall()
                finally:
                    # Don't close persistent connection
                    if self._persistent_conn is None:
                        conn.close()
            if not rows:
                break
            for row in rows:
                yield CircuitRecord.from_json(row[0])
            offset += batch_size
