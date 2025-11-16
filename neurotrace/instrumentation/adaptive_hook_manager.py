from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from neurotrace.config import NeuroTraceConfig
from neurotrace.instrumentation.adaptive_activations_buffer import (
    AdaptiveActivationsBuffer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Config per l'hook manager (granularità e sampling)
# ---------------------------------------------------------------------

@dataclass
class HookManagerConfig:
    """
    Config di alto livello per la strumentazione adattiva.

    In fase 1.1 usiamo principalmente:
      - hook_layers: pattern / indici layer da includere
      - capture_mode: per ora "block_output" (hidden states per layer)
    In fasi successive:
      - sampling_strategy ecc. verranno usati per gradient-based / importance-based.
    """

    # Se non vuoto: lista di indici layer da catturare (0-based).
    # Se None -> tutti i layer trovati.
    hook_layers: Optional[List[int]] = None

    # "block_output" = cattura l'output del blocco Transformer
    # in futuro: "attn_only", "mlp_only", "qkv", ecc.
    capture_mode: str = "block_output"

    # Strategia di sampling (per ora solo "none" implementato in modo effettivo).
    # Placeholder per future estensioni:
    #   - "activation_variance"
    #   - "gradient_magnitude"
    #   - "importance_adaptive"
    sampling_strategy: str = "none"


class AdaptiveHookManager:
    """
    Gestisce la registrazione di hook sui blocchi Transformer e inoltra
    le attivazioni catturate all'AdaptiveActivationsBuffer.

    È progettato per:
      - funzionare con modelli HF tipo GPT-2 / LLaMA-like
      - usare strategie di sampling future (gradient-based, variance-based, ecc.)
      - lavorare per batch con lista di example_ids allineata alle righe del batch.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: NeuroTraceConfig,
        buffer: AdaptiveActivationsBuffer,
        hook_cfg: Optional[HookManagerConfig] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.buffer = buffer
        self.hook_cfg = hook_cfg or HookManagerConfig()

        self._hook_handles: List[Any] = []
        self._hooks_registered: bool = False

        # Contesto corrente per la cattura
        self._current_example_ids: Optional[List[str]] = None
        self._current_step_meta: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_current_batch(
        self,
        example_ids: List[str],
        step_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Imposta il contesto corrente (lista di example_id e meta info) prima
        di lanciare il forward del modello.

        example_ids: lista di stringhe, lunghezza = batch_size
        step_meta: es. {"phase": "phase1_capture", "epoch": 0, "global_step": 123}
        """
        self._current_example_ids = example_ids
        self._current_step_meta = step_meta or {}

    def register_hooks(self) -> None:
        """
        Registra hook in base alla configurazione corrente.
        Può essere chiamato più volte, ma registra solo la prima.
        """
        if self._hooks_registered:
            return

        logger.info("AdaptiveHookManager: registro hook sui blocchi Transformer...")
        for idx, (name, block) in enumerate(self._iter_transformer_blocks()):
            if self._should_hook_layer(idx):
                handle = block.register_forward_hook(
                    self._make_block_hook(layer_index=idx, layer_name=name)
                )
                self._hook_handles.append(handle)

        self._hooks_registered = True
        logger.info("AdaptiveHookManager: registrati %d hook.", len(self._hook_handles))

    def clear_hooks(self) -> None:
        """
        Rimuove tutti gli hook registrati.
        """
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self._hooks_registered = False

    # ------------------------------------------------------------------
    # Internal: individuazione blocchi Transformer
    # ------------------------------------------------------------------

    def _iter_transformer_blocks(self) -> Iterable[Tuple[str, nn.Module]]:
        """
        Individua blocchi Transformer compatibili nei modelli più comuni.

        Copre:
          - GPT-2: model.transformer.h
          - LLaMA-like: model.model.layers

        Se non trova nulla, non genera blocchi (nessun hook).
        """
        m = self.model

        # GPT-2 style
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            for i, block in enumerate(m.transformer.h):
                yield f"layer_{i}", block
            return

        # LLaMA style
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            for i, block in enumerate(m.model.layers):
                yield f"layer_{i}", block
            return

        logger.warning(
            "AdaptiveHookManager: nessun pattern di blocchi Transformer riconosciuto."
        )
        return

    def _should_hook_layer(self, idx: int) -> bool:
        """
        Decide se hookare un certo layer in base a hook_cfg.hook_layers.
        """
        if self.hook_cfg.hook_layers is None:
            return True
        return idx in self.hook_cfg.hook_layers

    # ------------------------------------------------------------------
    # Hook factory
    # ------------------------------------------------------------------

    def _make_block_hook(self, layer_index: int, layer_name: str):
        """
        Crea una closure di hook che cattura output del blocco e lo inoltra al buffer.
        """
        full_name = f"{layer_name}.block"

        def hook(
            module: nn.Module,
            inputs: Tuple[torch.Tensor, ...],
            output: Any,
        ):
            # Se non abbiamo un contesto valido, non salviamo.
            if self._current_example_ids is None:
                return

            # estrai attivazione in forma tensore [B, S, D]
            if isinstance(output, torch.Tensor):
                act = output
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                act = output[0]
            else:
                return

            if act.dim() != 3:
                # Non è un hidden state standard, lo ignoriamo per ora.
                return

            # Qui si potrebbe applicare una prima euristica di sampling
            # (es. activation_variance) prima di passare al buffer.
            # Per ora inoltriamo sempre: la logica adattiva sta nel buffer.
            activation_dict = {full_name: act}

            # Il buffer gestisce compressione + memoria.
            self.buffer.add_activations(
                example_ids=self._current_example_ids,
                activations=activation_dict,
                step_meta=self._current_step_meta,
            )

        return hook
