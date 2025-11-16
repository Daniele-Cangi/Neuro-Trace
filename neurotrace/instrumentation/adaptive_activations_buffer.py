from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import math
import torch
from torch import Tensor

from neurotrace.config import NeuroTraceConfig, CompressionConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Strutture dati interne
# ---------------------------------------------------------------------

@dataclass
class CompressedActivationChunk:
    """
    Rappresenta un pezzo di attivazione compresso per un singolo layer
    e un sottoinsieme di token (tipicamente tutte le posizioni del batch
    corrente, già compresse).
    """

    layer_name: str
    # shape originale [B, S, D]
    original_shape: Tuple[int, int, int]

    # tensore compresso principale (dopo proiezione + quantizzazione)
    # tipicamente shape [B*S, d_proj] o [N, d_proj]
    values: Tensor

    # se top-k è attivo, possiamo avere una rappresentazione sparsa:
    #   - indices: [N, k]
    #   - sparse_values: [N, k]
    # in V1.1 manteniamo questi opzionali: se mancanti, significa densità piena.
    indices: Optional[Tensor] = None
    sparse_values: Optional[Tensor] = None

    # metadati vari (example_ids, step_meta, ecc.)
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# AdaptiveActivationsBuffer
# ---------------------------------------------------------------------

class AdaptiveActivationsBuffer:
    """
    Buffer centrale per:
      - ricevere attivazioni dai hook
      - applicare pipeline di compressione multi-stadio
      - gestire memoria (best-effort) e flushing verso lo State Indexer

    Pipeline implementata:
      1) Random projection (Johnson–Lindenstrauss)
      2) Quantizzazione (fp32 -> fp16 / int8)
      3) Opzionale top-k sparsity

    L'algoritmo di streaming SVD (Frequent Directions) è lasciato come
    estendibile: qui predisponiamo i punti di aggancio ma non imponiamo
    una complessità eccessiva in questa prima implementazione.
    """

    def __init__(
        self,
        cfg: NeuroTraceConfig,
        compression_cfg: Optional[CompressionConfig] = None,
        auto_flush_threshold_bytes: Optional[int] = None,
    ) -> None:
        self.cfg = cfg
        self.compression_cfg = compression_cfg or cfg.compression
        self.auto_flush_threshold_bytes = auto_flush_threshold_bytes

        # raccolta: example_id -> list[CompressedActivationChunk]
        self._storage: Dict[str, List[CompressedActivationChunk]] = {}

        # matrici di proiezione per dimensione D -> P (riutilizzate per layer compatibili)
        self._proj_mats: Dict[int, Tensor] = {}

    # ------------------------------------------------------------------
    # API principale
    # ------------------------------------------------------------------

    def add_activations(
        self,
        example_ids: List[str],
        activations: Dict[str, Tensor],
        step_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Riceve un dizionario {layer_name: tensor[B, S, D]} per il batch corrente.

        example_ids: lista di lunghezza B
        activations: mapping layer_name -> Tensor[B, S, D]
        step_meta: metainfo tipo {"epoch": 0, "global_step": 123, ...}
        """
        if not activations:
            return

        step_meta = step_meta or {}
        batch_size = len(example_ids)

        for layer_name, act in activations.items():
            if act.dim() != 3:
                continue  # ignoriamo formati inaspettati

            B, S, D = act.shape
            if B != batch_size:
                logger.warning(
                    "AdaptiveActivationsBuffer: dimensione batch %d != len(example_ids) %d",
                    B,
                    batch_size,
                )

            # Flatten B,S in un'unica dimensione N = B*S per compressione più semplice
            act_flat = act.reshape(B * S, D).to(dtype=torch.float32, device="cpu")

            # Pipeline di compressione
            comp = self._compress_layer_activation(
                layer_name=layer_name,
                act_flat=act_flat,
                original_shape=(B, S, D),
                meta={
                    "example_ids": example_ids,
                    "step_meta": step_meta,
                },
            )

            # Distribuiamo il chunk compresso su tutti gli example_id (per ora
            # li associamo interamente a ciascun esempio; in future versioni
            # si può spezzare per token).
            for ex_id in example_ids:
                self._storage.setdefault(ex_id, []).append(comp)

        # Trigger auto-flush se superiamo soglia
        if self.auto_flush_threshold_bytes is not None:
            if self._approx_storage_bytes() > self.auto_flush_threshold_bytes:
                logger.info(
                    "AdaptiveActivationsBuffer: soglia memoria superata, auto-flush attivo."
                )

    def flush_to_indexer(
        self,
        indexer_fn: Callable[[str, List[CompressedActivationChunk]], None],
    ) -> None:
        """
        Invia tutto il contenuto corrente a un "indexer" esterno
        (es. future state_indexer.vector_state_db) e svuota il buffer.

        indexer_fn: funzione con firma
          indexer_fn(example_id: str, chunks: List[CompressedActivationChunk]) -> None
        """
        for ex_id, chunks in self._storage.items():
            indexer_fn(ex_id, chunks)

        self._storage.clear()

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Ritorna una stima approssimativa dell'uso di memoria del buffer.
        In futuro si possono aggiungere misure più accurate o integrazione con psutil.
        """
        total_bytes = self._approx_storage_bytes()
        return {
            "num_examples": len(self._storage),
            "approx_bytes": total_bytes,
            "approx_megabytes": total_bytes / (1024**2),
        }

    # ------------------------------------------------------------------
    # Compression pipeline
    # ------------------------------------------------------------------

    def _compress_layer_activation(
        self,
        layer_name: str,
        act_flat: Tensor,  # [N, D] float32 CPU
        original_shape: Tuple[int, int, int],
        meta: Dict[str, Any],
    ) -> CompressedActivationChunk:
        """
        Applica:
          1) random projection (se attiva)
          2) quantizzazione
          3) opzionale top-k sparsity
        Ritorna un CompressedActivationChunk.
        """
        N, D = act_flat.shape

        # 1) Random Projection
        if self.compression_cfg.use_random_projection and D > self.compression_cfg.projection_dim:
            proj = self._get_projection_matrix(D, self.compression_cfg.projection_dim)
            # [N, D] @ [D, P] -> [N, P]
            proj_act = act_flat @ proj
        else:
            proj_act = act_flat

        # 2) Quantizzazione (fp32 -> fp16 o int8)
        proj_act_q = self._quantize(proj_act)

        # 3) Sparsità top-k (opzionale)
        indices = None
        sparse_values = None
        dense_values = proj_act_q

        if self.compression_cfg.use_topk_sparsity:
            (
                indices,
                sparse_values,
                dense_values,
            ) = self._apply_topk_sparsity(proj_act_q)

        comp = CompressedActivationChunk(
            layer_name=layer_name,
            original_shape=original_shape,
            values=dense_values,
            indices=indices,
            sparse_values=sparse_values,
            meta=meta,
        )
        return comp

    def _get_projection_matrix(self, in_dim: int, out_dim: int) -> Tensor:
        """
        Restituisce (o crea e cache) una matrice di proiezione gaussiana [in_dim, out_dim]
        normalizzata per Johnson–Lindenstrauss.
        """
        if in_dim in self._proj_mats:
            mat = self._proj_mats[in_dim]
            if mat.shape[1] == out_dim:
                return mat

        # nuovo
        mat = torch.randn(in_dim, out_dim, dtype=torch.float32)
        # normalizzazione per preservare le distanze in media
        mat /= math.sqrt(out_dim)
        self._proj_mats[in_dim] = mat
        return mat

    def _quantize(self, x: Tensor) -> Tensor:
        """
        Quantizza un tensore float32 in:
          - fp16 se quantization_bits = 16
          - int8 con scaling per-row (per vettore) se quantization_bits = 8
          - rimane fp32 se 32 o valore non riconosciuto
        """
        bits = self.compression_cfg.quantization_bits

        if bits == 16:
            return x.to(dtype=torch.float16)

        if bits == 8:
            # simple per-vector symmetric quantization: x_int8 = round(x / scale)
            # scale = max(abs(x)) / 127, calcolata per riga.
            # NB: per essere riutilizzabile in fase di decode servirebbe salvare scale.
            # In questa V1.1 salviamo solo il valore int8, pensando che lo State Indexer
            # lavori direttamente su questa rappresentazione normalizzata.
            with torch.no_grad():
                # [N, D]
                max_abs = x.abs().amax(dim=1, keepdim=True) + 1e-6
                scale = max_abs / 127.0
                x_norm = x / scale
                x_int8 = torch.clamp(x_norm.round(), -128, 127).to(torch.int8)
            return x_int8

        # default: nessuna quantizzazione
        return x

    def _apply_topk_sparsity(
        self,
        x: Tensor,  # [N, D_proj] (fp16 o int8, ma trattiamo come float per topk)
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Tensor]:
        """
        Applica top-k sparsity per riga.

        ATTENZIONE:
        - Per semplicità, ritorniamo sia la rappresentazione sparsa (indici + valori)
          sia una versione densa "normalizzata" (utile per moduli che non supportano
          ancora il formato sparso).
        - Questo permette alla pipeline a valle di scegliere cosa usare.
        """
        k = self.compression_cfg.topk_per_vector
        if k <= 0:
            return None, None, x

        # Convertiamo a float32 per topk (anche se era int8/fp16)
        x_f = x.to(torch.float32)

        # topk lungo l'ultima dimensione
        values, indices = torch.topk(x_f.abs(), k=k, dim=1, largest=True, sorted=False)
        # Ricostruiamo i segnali con segno
        # gather dei valori originali con indici topk
        signed_values = torch.gather(x_f, dim=1, index=indices)
        # dense "ricostruito" ma con zeri altrove (serve se qualcuno vuole tensore denso)
        dense = torch.zeros_like(x_f)
        dense.scatter_(dim=1, index=indices, src=signed_values)

        return indices.to(torch.int32), signed_values.to(x.dtype), dense.to(x.dtype)

    # ------------------------------------------------------------------
    # Approximate memory usage
    # ------------------------------------------------------------------

    def _approx_storage_bytes(self) -> int:
        """
        Stima approssimativa della memoria occupata in RAM dalle attivazioni compresse.
        """
        total = 0
        for chunks in self._storage.values():
            for c in chunks:
                total += c.values.element_size() * c.values.numel()
                if c.indices is not None:
                    total += c.indices.element_size() * c.indices.numel()
                if c.sparse_values is not None:
                    total += c.sparse_values.element_size() * c.sparse_values.numel()
        return total
