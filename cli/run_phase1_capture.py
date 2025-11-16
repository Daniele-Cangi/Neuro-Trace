#!/usr/bin/env python
# cli/run_phase1_capture.py

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Tuple

import torch

from neurotrace.config import NeuroTraceConfig
from neurotrace.models.wrapper import TargetModelWrapper

# Dataset opzionale (wikitext). Se non c'è, usiamo frasi dummy.
try:
    from datasets import load_dataset  # type: ignore

    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False


# =====================================================================
# Helpers: dataset, batches, compression
# =====================================================================

def setup_logging(level: str = "info") -> None:
    lvl = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING}.get(
        level, logging.INFO
    )
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=lvl,
    )


def load_text_dataset(num_examples: int) -> List[str]:
    """
    Carica un set di frasi per testare il capture.
    Se `datasets` non è disponibile, crea frasi dummy strutturate.
    """
    if HAS_DATASETS:
        logging.info("Carico dataset wikitext-2-raw-v1 (train)...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts: List[str] = []
        for ex in ds:
            t = (ex.get("text") or "").strip()
            if t:
                texts.append(t)
            if len(texts) >= num_examples:
                break
        if not texts:
            logging.warning("Dataset vuoto, uso frasi dummy.")
            return _make_dummy_texts(num_examples)
        logging.info(f"Caricate {len(texts)} frasi dal dataset.")
        return texts[:num_examples]
    else:
        logging.warning("`datasets` non disponibile, uso frasi dummy.")
        return _make_dummy_texts(num_examples)


def _make_dummy_texts(n: int) -> List[str]:
    base = (
        "NeuroTrace Engine is capturing internal activations from a transformer model "
        "to analyze reasoning circuits and residual stream dynamics."
    )
    return [f"[{i}] {base}" for i in range(n)]


def iter_batches(seq: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


# =====================================================================
# Compression pipeline (random projection + quantization + top-k)
# =====================================================================


class ActivationCompressor:
    """
    Implementa una versione concreta della pipeline di compressione fase 1:
      - random projection (dimensione cfg.compression.projection_dim)
      - quantization (fp16 o int8 + scala)
      - opzionale: top-k sparsity sul vettore proiettato
    Non è ancora Frequent Directions / SVD streaming, ma è già
    una compressione reale e non triviale.
    """

    def __init__(self, cfg: NeuroTraceConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.comp_cfg = cfg.compression
        self._proj_cache: Dict[int, torch.Tensor] = {}  # dim_in -> W[D_in, D_proj]

    def _get_projection_matrix(self, dim_in: int) -> torch.Tensor:
        if dim_in in self._proj_cache:
            return self._proj_cache[dim_in]

        dim_out = self.comp_cfg.projection_dim
        if dim_out >= dim_in:
            # Nessuna riduzione in realtà, ma teniamo la coerenza
            W = torch.eye(dim_in, device=self.device)
        else:
            # Johnson–Lindenstrauss: gaussian random matrix / sqrt(dim_in)
            W = torch.randn(dim_in, dim_out, device=self.device) / math.sqrt(dim_in)

        self._proj_cache[dim_in] = W
        logging.debug(
            f"Creato projection matrix per dim_in={dim_in} -> dim_out={W.shape[1]}"
        )
        return W

    def compress_activations(
        self, activations: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Input:
          activations: dict key->tensor [B, S, D]

        Output:
          dict key->{
             "shape": [B, S, D],
             "proj_dim": int,
             "quantization": "fp16" | "int8" | "none",
             "topk": int | 0,
             "data": {
                 ... tensori compressi ...
             }
          }
        """
        compressed: Dict[str, Dict[str, Any]] = {}

        for key, tensor_cpu in activations.items():
            # Spostiamo su device per compressione
            tensor = tensor_cpu.to(self.device, non_blocking=True)
            bsz, seqlen, dim = tensor.shape

            original_shape = [bsz, seqlen, dim]
            logging.debug(f"Compressione attivazione {key} shape={tensor.shape}")

            # 1) Flatten [B, S, D] -> [N, D]
            flat = tensor.reshape(-1, dim)

            # 2) Random projection (se abilitata)
            if self.comp_cfg.use_random_projection:
                W = self._get_projection_matrix(dim)
                # Converti flat a float32 su CUDA per compatibilità con W
                flat_f32 = flat.to(dtype=torch.float32, device=flat.device)
                proj = flat_f32 @ W  # [N, D_proj]
                proj_dim = proj.shape[-1]
            else:
                proj = flat
                proj_dim = dim

            # 3) Top-k sparsity (opzionale, sul vettore proiettato)
            topk = 0
            topk_indices = None
            topk_values = None
            dense_proj = proj

            if self.comp_cfg.use_topk_sparsity:
                k = min(self.comp_cfg.topk_per_vector, proj_dim)
                if k > 0 and proj_dim > k:
                    # topk lungo l'ultima dimensione
                    vals, idx = torch.topk(proj, k, dim=-1)
                    topk = k
                    topk_indices = idx
                    topk_values = vals
                    dense_proj = None  # usiamo la rappresentazione sparsa
                    logging.debug(f"{key}: applicato top-k sparsity con k={k}")

            # 4) Quantization
            q_mode = "none"
            quant_payload: Dict[str, Any] = {}

            if topk > 0:
                # quantizziamo solo i valori top-k
                if self.comp_cfg.quantization_bits == 16:
                    q_mode = "fp16"
                    q_vals = topk_values.half()
                    quant_payload = {
                        "indices": topk_indices.cpu(),
                        "values": q_vals.cpu(),
                    }
                elif self.comp_cfg.quantization_bits == 8:
                    q_mode = "int8"
                    max_abs = topk_values.abs().max().item() + 1e-8
                    scale = max_abs / 127.0
                    q_vals = (topk_values / scale).clamp(-127, 127).round().to(
                        torch.int8
                    )
                    quant_payload = {
                        "indices": topk_indices.cpu(),
                        "values_q": q_vals.cpu(),
                        "scale": scale,
                    }
                else:
                    quant_payload = {
                        "indices": topk_indices.cpu(),
                        "values": topk_values.cpu(),
                    }
            else:
                # quantizziamo il tensore denso [N, proj_dim]
                if self.comp_cfg.quantization_bits == 16:
                    q_mode = "fp16"
                    q_vals = dense_proj.half()
                    quant_payload = {"dense": q_vals.cpu()}
                elif self.comp_cfg.quantization_bits == 8:
                    q_mode = "int8"
                    max_abs = dense_proj.abs().max().item() + 1e-8
                    scale = max_abs / 127.0
                    q_vals = (dense_proj / scale).clamp(-127, 127).round().to(
                        torch.int8
                    )
                    quant_payload = {
                        "dense_q": q_vals.cpu(),
                        "scale": scale,
                    }
                else:
                    quant_payload = {"dense": dense_proj.cpu()}

            compressed[key] = {
                "shape": original_shape,
                "proj_dim": proj_dim,
                "quantization": q_mode,
                "topk": topk,
                "data": quant_payload,
            }

        return compressed


# =====================================================================
# Main
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1 – Instrumentation & Activation Capture for NeuroTrace Engine"
    )
    p.add_argument("--model", type=str, default="gpt2", help="HF model name or path")
    p.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    p.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--num-examples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out-dir", type=str, default="runs/phase1_capture")
    p.add_argument("--log-level", type=str, default="info", choices=["debug", "info", "warning"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    os.makedirs(args.out_dir, exist_ok=True)
    out_act_dir = os.path.join(args.out_dir, "activations")
    os.makedirs(out_act_dir, exist_ok=True)

    # Config NeuroTrace
    cfg = NeuroTraceConfig(
        model_name_or_path=args.model,
        device=args.device,
        precision=args.precision,
        max_seq_len=args.max_seq_len,
        log_level=args.log_level,
        seed=args.seed,
    )

    logging.info(f"Config NeuroTrace: {cfg}")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)

    # Target model wrapper
    logging.info("Inizializzo TargetModelWrapper...")
    wrapper = TargetModelWrapper(cfg).to(device)
    logging.info("Wrapper inizializzato.")

    # Dataset
    texts = load_text_dataset(args.num_examples)
    logging.info(f"Totale esempi da processare: {len(texts)}")

    compressor = ActivationCompressor(cfg, device=device)

    # Statistiche globali
    run_meta: Dict[str, Any] = {
        "config": cfg.to_dict(),
        "num_examples": len(texts),
        "batch_size": args.batch_size,
        "batches": 0,
        "total_forward_time_sec": 0.0,
        "avg_activations_per_batch": 0.0,
        "avg_compressed_bytes_per_batch": 0.0,
    }

    total_activations_count = 0
    total_compressed_bytes = 0.0
    batch_index = 0

    start_run = time.time()

    for batch_texts in iter_batches(texts, args.batch_size):
        batch_index += 1
        logging.info(f"[Batch {batch_index}] size={len(batch_texts)}")

        # Forward + trace
        t0 = time.time()
        out = wrapper.run_texts(batch_texts, capture_activations=True, max_length=args.max_seq_len)
        t1 = time.time()

        batch_forward_time = t1 - t0
        run_meta["total_forward_time_sec"] += batch_forward_time

        activations: Dict[str, torch.Tensor] = out["internal_states_reference"]
        logging.debug(
            f"[Batch {batch_index}] catturate {len(activations)} mappe di attivazione."
        )

        # Compressione
        comp = compressor.compress_activations(activations)

        # Stima dimensione in byte delle attivazioni compresse
        batch_bytes = _estimate_compressed_size_bytes(comp)
        total_compressed_bytes += batch_bytes
        total_activations_count += len(activations)

        # Salva batch su disco
        batch_path = os.path.join(out_act_dir, f"batch_{batch_index:04d}.pt")
        torch.save(
            {
                "texts": batch_texts,
                "compressed_activations": comp,
                "forward_time_sec": batch_forward_time,
            },
            batch_path,
        )
        logging.info(
            f"[Batch {batch_index}] forward_time={batch_forward_time:.3f}s, "
            f"compressed_size≈{batch_bytes/1024/1024:.2f} MB, saved→{batch_path}"
        )

    end_run = time.time()
    run_meta["batches"] = batch_index
    if batch_index > 0:
        run_meta["avg_activations_per_batch"] = total_activations_count / batch_index
        run_meta["avg_compressed_bytes_per_batch"] = total_compressed_bytes / batch_index

    run_meta["wall_time_sec"] = end_run - start_run

    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    logging.info(f"Run completo. Meta salvato in {meta_path}")
    logging.info(
        f"Tempo totale={run_meta['wall_time_sec']:.2f}s, "
        f"avg compressed per batch≈{run_meta['avg_compressed_bytes_per_batch']/1024/1024:.2f} MB"
    )


def _estimate_compressed_size_bytes(
    compressed: Dict[str, Dict[str, Any]]
) -> int:
    """
    Stima rozza della dimensione dei tensori compressi in memoria.
    Non perfetta, ma sufficiente per capire l'ordine di grandezza.
    """
    total = 0
    for key, entry in compressed.items():
        data = entry.get("data", {})
        for name, val in data.items():
            if isinstance(val, torch.Tensor):
                total += val.numel() * val.element_size()
            # scalar (es. scale per int8)
            elif isinstance(val, (float, int)):
                total += 8
    return total


if __name__ == "__main__":
    main()
