from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor

from neurotrace.config import NeuroTraceConfig, SAEConfig
from neurotrace.instrumentation.adaptive_activations_buffer import (
    CompressedActivationChunk,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Modello SAE base (per-layer)
# ---------------------------------------------------------------------

class LayerSparseAutoencoder(nn.Module):
    """
    SAE per un singolo layer hidden_dim -> dict_size -> hidden_dim.

    È volutamente semplice ma solido:
      - encoder lineare + ReLU
      - decoder lineare
      - loss: MSE(reconstruction, input) + lambda * L1(sparse_codes)

    L'uso tipico in NeuroTrace è:
      - pre-training offline con dataset di attivazioni
      - poi solo .encode(x) in modalità eval
    """

    def __init__(
        self,
        input_dim: int,
        dict_size: int,
        sparsity_lambda: float = 1e-3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.sparsity_lambda = sparsity_lambda

        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # x: [N, D]
        z = torch.relu(self.encoder(x))          # [N, dict_size]
        recon = self.decoder(z)                  # [N, D]
        return {"codes": z, "reconstruction": recon}

    def loss(self, x: Tensor, out: Dict[str, Tensor]) -> Tensor:
        recon = out["reconstruction"]
        codes = out["codes"]
        mse = torch.mean((recon - x) ** 2)
        l1 = torch.mean(torch.abs(codes))
        return mse + self.sparsity_lambda * l1

    @torch.no_grad()
    def encode(self, x: Tensor, top_k: int = 32) -> Dict[str, Tensor]:
        """
        Ritorna sparse codes monosemantici:
          - indices: [N, k]
          - values: [N, k]
        """
        self.eval()
        codes = torch.relu(self.encoder(x))  # [N, dict_size]

        if top_k <= 0 or top_k >= self.dict_size:
            # tutto denso (fallback)
            values, indices = torch.topk(codes, k=min(self.dict_size, 64), dim=1)
        else:
            values, indices = torch.topk(codes, k=top_k, dim=1)

        return {
            "indices": indices,   # int64
            "values": values,     # float32
        }


# ---------------------------------------------------------------------
# SAEFeatureExtractor
# ---------------------------------------------------------------------

@dataclass
class SAEFeatureExtractorConfig:
    """
    Config locale specifico per l'estrattore SAE.
    Se hai già SAEConfig dentro NeuroTraceConfig, puoi fondere questi campi.
    """
    sae_topk_per_token: int = 32
    sae_dict_mult: int = 4        # dict_size = dict_mult * hidden_dim se non definito diversamente
    device: str = "cuda"


class SAEFeatureExtractor:
    """
    Trasforma i CompressedActivationChunk (per example_id) in feature SAE per layer.

    Output principale:
      per example_id -> dict:
        {
          "sae_features": {
             "layer_X": {
                "indices": Tensor[num_tokens, k],
                "values": Tensor[num_tokens, k],
             },
             ...
          }
        }

    Nota: in questa V1.1 assumiamo che:
      - il chunk.values sia denso [N, D_proj]
      - se values è int8/fp16, lo portiamo a float32 per SAE.
    """

    def __init__(
        self,
        cfg: NeuroTraceConfig,
        sae_cfg: Optional[SAEFeatureExtractorConfig] = None,
    ) -> None:
        self.cfg = cfg
        self.sae_cfg = sae_cfg or SAEFeatureExtractorConfig(device=cfg.device)

        self.device = torch.device(self.sae_cfg.device)
        # cache di SAE per layer_name -> LayerSparseAutoencoder
        self._sae_per_layer: Dict[str, LayerSparseAutoencoder] = {}

    # ------------------------------------------------------------------
    # SAE management
    # ------------------------------------------------------------------

    def _get_or_create_sae(
        self,
        layer_name: str,
        input_dim: int,
    ) -> LayerSparseAutoencoder:
        """
        Recupera SAE per un layer, oppure crea un nuovo modello non addestrato.

        In pratica:
          - in un setup reale, loaderesti pesi pre-addestrati
          - qui forniamo auto-creazione per non bloccare il flusso
        """
        if layer_name in self._sae_per_layer:
            sae = self._sae_per_layer[layer_name]
            if sae.input_dim != input_dim:
                logger.warning(
                    "SAEFeatureExtractor: dimensione input cambiata per %s (old=%d, new=%d).",
                    layer_name,
                    sae.input_dim,
                    input_dim,
                )
            return sae

        dict_size = self._infer_dict_size(input_dim)
        sae = LayerSparseAutoencoder(
            input_dim=input_dim,
            dict_size=dict_size,
            sparsity_lambda=getattr(self.cfg.sae, "sparsity_lambda", 1e-3)
            if hasattr(self.cfg, "sae")
            else 1e-3,
        ).to(self.device)

        logger.info(
            "SAEFeatureExtractor: creato nuovo SAE per layer=%s, input_dim=%d, dict_size=%d",
            layer_name,
            input_dim,
            dict_size,
        )
        self._sae_per_layer[layer_name] = sae
        return sae

    def _infer_dict_size(self, input_dim: int) -> int:
        mult = self.sae_cfg.sae_dict_mult
        return mult * input_dim

    # ------------------------------------------------------------------
    # Public API: estrazione feature per example
    # ------------------------------------------------------------------

    def extract_features_for_example(
        self,
        example_id: str,
        chunks: List[CompressedActivationChunk],
    ) -> Dict[str, Any]:
        """
        Data la lista di chunk per un example_id, produce un dizionario:

        {
          "example_id": ...,
          "sae_features": {
             "layer_X": {
                "indices": Tensor[num_tokens, k],
                "values": Tensor[num_tokens, k],
             },
             ...
          }
        }

        NOTA: in V1.1 aggreghiamo semplicemente tutti i token (B,S)
        del batch originale in un unico set per example. In futuro
        si può spezzare per posizione / CoT step, ecc.
        """
        sae_features: Dict[str, Dict[str, Tensor]] = {}

        for chunk in chunks:
            layer_name = chunk.layer_name
            # chunk.values: [N, D_proj] tipicamente
            act = chunk.values

            # portiamo a float32 su device per SAE
            if isinstance(act, torch.Tensor):
                x = act.to(dtype=torch.float32, device=self.device)
            else:
                continue

            N, D = x.shape
            sae = self._get_or_create_sae(layer_name, input_dim=D)

            # Encode in sparse codes
            codes = sae.encode(x, top_k=self.sae_cfg.sae_topk_per_token)

            if layer_name not in sae_features:
                sae_features[layer_name] = {
                    "indices": codes["indices"],
                    "values": codes["values"],
                }
            else:
                # concateniamo se esistono già codici per questo layer
                sae_features[layer_name]["indices"] = torch.cat(
                    [sae_features[layer_name]["indices"], codes["indices"]],
                    dim=0,
                )
                sae_features[layer_name]["values"] = torch.cat(
                    [sae_features[layer_name]["values"], codes["values"]],
                    dim=0,
                )

        return {
            "example_id": example_id,
            "sae_features": sae_features,
        }

    # ------------------------------------------------------------------
    # Utility: flatten per meta-modelli / vector DB
    # ------------------------------------------------------------------

    def build_flat_feature_vector(
        self,
        sae_features: Dict[str, Dict[str, Tensor]],
        max_layers: Optional[int] = None,
    ) -> Tensor:
        """
        Costruisce un vettore denso globale per l'esempio, utile per:
          - meta-modelli di importanza
          - vector DB

        Strategia semplice ma ragionevole:
          - per ciascun layer:
              - media dei top-k values per feature index
              - proiettiamo in vettore fisso (es. somma normalizzata)
          - concat di tutti i layer, poi normalizzazione L2.

        In questa V1.1 implementiamo un aggregato molto diretto.
        Claude/Copilot potranno sostituire con qualcosa di più sofisticato.
        """
        layer_names = sorted(sae_features.keys())
        if max_layers is not None:
            layer_names = layer_names[:max_layers]

        per_layer_vecs: List[Tensor] = []

        for ln in layer_names:
            layer_data = sae_features[ln]
            idx = layer_data["indices"]     # [N, k]
            val = layer_data["values"]      # [N, k]

            # convertiamo a float32 CPU
            idx_f = idx.to(dtype=torch.int64, device="cpu")
            val_f = val.to(dtype=torch.float32, device="cpu")

            # numero feature dictionary ~ max index + 1 (best-effort)
            dict_size_est = int(idx_f.max().item()) + 1 if idx_f.numel() > 0 else 0
            if dict_size_est == 0:
                # layer vuoto (capita se niente catturato)
                per_layer_vecs.append(torch.zeros(16, dtype=torch.float32))
                continue

            # costruiamo un vettore [dict_size_est] aggregando per feature index
            feat_vec = torch.zeros(dict_size_est, dtype=torch.float32)
            # flatten indices e values
            idx_flat = idx_f.reshape(-1)
            val_flat = val_f.reshape(-1)
            feat_vec.index_add_(0, idx_flat, val_flat)

            # normalizzazione
            if feat_vec.norm(p=2) > 0:
                feat_vec = feat_vec / feat_vec.norm(p=2)

            # opzionale: riduciamo dimensione con una proiezione random semplice
            target_dim = min(256, dict_size_est)
            if dict_size_est > target_dim:
                # generiamo on-the-fly una proiezione (per ora non cache, è solo fallback)
                proj = torch.randn(dict_size_est, target_dim) / (target_dim**0.5)
                feat_vec = (feat_vec.unsqueeze(0) @ proj).squeeze(0)

            per_layer_vecs.append(feat_vec)

        if not per_layer_vecs:
            return torch.zeros(128, dtype=torch.float32)

        # concat e normalizzazione finale
        flat = torch.cat(per_layer_vecs, dim=0)
        if flat.norm(p=2) > 0:
            flat = flat / flat.norm(p=2)

        return flat
