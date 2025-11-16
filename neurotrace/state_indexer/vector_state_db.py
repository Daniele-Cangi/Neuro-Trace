from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import os
import sqlite3

import numpy as np
import torch
from torch import Tensor

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None

logger = logging.getLogger(__name__)


@dataclass
class VectorStateDBConfig:
    """
    Config minimale per il vector DB.
    Puoi integrarlo in NeuroTraceConfig.vector_db se preferisci.
    """
    dim: int = 256            # dimensione vettore globale per esempio
    metric: str = "ip"        # "ip" (inner product) o "l2"
    nlist: int = 100          # num cluster IVF
    nprobe: int = 10          # nprobe per ricerca
    use_ivfpq: bool = True
    sqlite_path: str = "neurotrace_state_meta.sqlite3"


class VectorStateDB:
    """
    Vector DB + metadata store per gli stati interni di NeuroTrace.

    - FAISS per vettori (ANN search)
    - SQLite per metadata (prompt, output, task_tag, ecc.)

    Design:
      - insert(example_id, feature_vec, metadata)
      - query_similar(feature_vec, k, filters)
      - query_by_task, batch_fetch

    Nota: per semplicità usiamo un singolo indice FAISS in RAM.
    Salvataggio/loading del modello può essere aggiunto in Fase 3+.
    """

    def __init__(
        self,
        cfg: VectorStateDBConfig,
    ) -> None:
        if faiss is None:
            raise ImportError(
                "faiss non è installato. Installa faiss-cpu o faiss-gpu per usare VectorStateDB."
            )

        self.cfg = cfg
        self.dim = cfg.dim
        self.metric = cfg.metric

        # id numerici interni per FAISS
        self._next_internal_id: int = 0
        self._id_map: Dict[int, str] = {}   # internal_id -> example_id
        self._id_map_rev: Dict[str, int] = {}  # example_id -> internal_id

        # costruiamo indice FAISS
        self.index = self._build_index()

        # setup SQLite per metadata
        self.sqlite_path = cfg.sqlite_path
        self._init_sqlite()

    # ------------------------------------------------------------------
    # FAISS index
    # ------------------------------------------------------------------

    def _build_index(self):
        if self.metric == "ip":
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            metric = faiss.METRIC_L2

        dim = self.dim
        if self.cfg.use_ivfpq:
            quantizer = faiss.IndexFlatIP(dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFPQ(
                quantizer,
                dim,
                self.cfg.nlist,
                8,   # M (subquantizers)
                8,   # nbits
            )
            index.nprobe = self.cfg.nprobe
        else:
            index = faiss.IndexFlatIP(dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)

        logger.info(
            "VectorStateDB: creato indice FAISS (dim=%d, metric=%s, ivfpq=%s)",
            dim,
            self.metric,
            self.cfg.use_ivfpq,
        )
        return index

    # ------------------------------------------------------------------
    # SQLite metadata
    # ------------------------------------------------------------------

    def _init_sqlite(self) -> None:
        need_init = not os.path.exists(self.sqlite_path)
        conn = sqlite3.connect(self.sqlite_path)
        try:
            if need_init:
                conn.execute(
                    """
                    CREATE TABLE state_metadata (
                        internal_id INTEGER PRIMARY KEY,
                        example_id TEXT UNIQUE,
                        prompt TEXT,
                        output TEXT,
                        task_tag TEXT
                    );
                    """
                )
                conn.commit()
        finally:
            conn.close()

    def _insert_metadata(
        self,
        internal_id: int,
        example_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        conn = sqlite3.connect(self.sqlite_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO state_metadata
                (internal_id, example_id, prompt, output, task_tag)
                VALUES (?, ?, ?, ?, ?);
                """,
                (
                    internal_id,
                    example_id,
                    metadata.get("prompt", ""),
                    metadata.get("output", ""),
                    metadata.get("task_tag", ""),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _fetch_metadata_by_ids(self, internal_ids: List[int]) -> List[Dict[str, Any]]:
        if not internal_ids:
            return []
        conn = sqlite3.connect(self.sqlite_path)
        try:
            placeholders = ",".join("?" for _ in internal_ids)
            rows = conn.execute(
                f"""
                SELECT internal_id, example_id, prompt, output, task_tag
                FROM state_metadata
                WHERE internal_id IN ({placeholders});
                """,
                internal_ids,
            ).fetchall()
        finally:
            conn.close()

        # costruiamo mappa internal_id -> dict
        id2meta = {
            row[0]: {
                "internal_id": row[0],
                "example_id": row[1],
                "prompt": row[2],
                "output": row[3],
                "task_tag": row[4],
            }
            for row in rows
        }
        return [id2meta[i] for i in internal_ids if i in id2meta]

    def _fetch_by_task(self, task_tag: str, limit: int) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.sqlite_path)
        try:
            rows = conn.execute(
                """
                SELECT internal_id, example_id, prompt, output, task_tag
                FROM state_metadata
                WHERE task_tag = ?
                LIMIT ?;
                """,
                (task_tag, limit),
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "internal_id": row[0],
                "example_id": row[1],
                "prompt": row[2],
                "output": row[3],
                "task_tag": row[4],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(
        self,
        example_id: str,
        feature_vector: Tensor,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Inserisce un esempio nel vector DB.

        feature_vector: Tensor[dim] o Tensor[1,dim]
        metadata: prompt, output, task_tag, ecc.
        """
        if feature_vector.dim() == 1:
            vec = feature_vector.unsqueeze(0)
        else:
            vec = feature_vector
        vec = vec.detach().cpu().to(torch.float32).numpy()

        if vec.shape[1] != self.dim:
            raise ValueError(
                f"VectorStateDB.insert: dimensione vettore {vec.shape[1]} != cfg.dim {self.dim}"
            )

        internal_id = self._next_internal_id
        self._next_internal_id += 1

        # mapping
        self._id_map[internal_id] = example_id
        self._id_map_rev[example_id] = internal_id

        # FAISS usa id impliciti in ordine di inserimento
        if isinstance(self.index, faiss.IndexIVF):
            if not self.index.is_trained:
                # su pochi punti di training va bene usare gli stessi dati
                self.index.train(vec)
        self.index.add(vec)

        # metadata
        self._insert_metadata(internal_id, example_id, metadata)

    def query_similar(
        self,
        feature_vector: Tensor,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ritorna i k esempi più simili al vettore dato.

        filters (non ancora implementato seriamente):
          - in futuro: filtri su task_tag, ecc. a livello SQLite.
        """
        if feature_vector.dim() == 1:
            vec = feature_vector.unsqueeze(0)
        else:
            vec = feature_vector
        vec = vec.detach().cpu().to(torch.float32).numpy()

        if vec.shape[1] != self.dim:
            raise ValueError(
                f"VectorStateDB.query_similar: dimensione vettore {vec.shape[1]} != cfg.dim {self.dim}"
            )

        if isinstance(self.index, faiss.IndexIVF) and not self.index.is_trained:
            logger.warning("VectorStateDB: indice IVF non ancora trainato, nessun risultato affidabile.")
            return []

        D, I = self.index.search(vec, k)
        internal_ids = [int(i) for i in I[0] if i != -1]

        metas = self._fetch_metadata_by_ids(internal_ids)
        # aggiungiamo distanze/similarità
        id2dist = {int(I[0][j]): float(D[0][j]) for j in range(len(internal_ids))}
        for m in metas:
            m["score"] = id2dist.get(m["internal_id"], 0.0)

        # applichiamo eventuali filtri (per ora solo task_tag)
        if filters and "task_tag" in filters:
            tag = filters["task_tag"]
            metas = [m for m in metas if m.get("task_tag") == tag]

        return metas

    def query_by_task(self, task_tag: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Ritorna metadata per esempi con certo task_tag.
        Per i vettori, puoi poi usare _internal_id -> index storage.
        """
        return self._fetch_by_task(task_tag, limit)

    def batch_fetch(self, example_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Recupera metadata per lista di example_id.
        """
        internal_ids = [self._id_map_rev[eid] for eid in example_ids if eid in self._id_map_rev]
        return self._fetch_metadata_by_ids(internal_ids)
