from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from neurotrace.config import NeuroTraceConfig


logger = logging.getLogger(__name__)


class ResidualHookHandle:
    """
    Handle per rimuovere un hook sul residual stream.
    Compatibile con il Protocol del controller.
    """

    def __init__(self, hook_handle: Any) -> None:
        self._handle = hook_handle

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class TargetModelWrapper(nn.Module):
    """
    Wrapper unificato attorno al modello da analizzare.

    - Carica un modello HF (es. gpt2) + tokenizer
    - Gestisce device / precision
    - Registra hook sui blocchi Transformer per catturare attivazioni
      (fase 1: hook sui blocchi principali per avere hidden states layer-wise)
    - Espone run_texts(...) che ritorna logits + internal_states_reference
    """

    def __init__(self, cfg: NeuroTraceConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Stato interno per attivazioni catturate
        self._current_activations: Dict[str, torch.Tensor] = {}
        self._hook_handles: List[Any] = []
        self._hooks_registered: bool = False

        # Steering hooks per Control Plane
        self._residual_hooks: Dict[Tuple[int, str], ResidualHookHandle] = {}
        self._block_cache: Dict[int, nn.Module] = {}

        # Device
        self.device = torch.device(cfg.device)

        # Carica modello + tokenizer
        self.tokenizer: PreTrainedTokenizerBase
        self.model: PreTrainedModel
        self._load_model_and_tokenizer()

    # ------------------------------------------------------------------
    # Setup modello
    # ------------------------------------------------------------------

    def _load_model_and_tokenizer(self) -> None:
        logger.info(f"Carico modello: {self.cfg.model_name_or_path}")

        # Precisione
        torch_dtype = torch.float32
        if self.cfg.precision == "fp16" and self.device.type == "cuda":
            torch_dtype = torch.float16
        elif self.cfg.precision == "bf16" and self.device.type == "cuda":
            torch_dtype = torch.bfloat16
        else:
            # su CPU rimaniamo fp32
            torch_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name_or_path)
        if self.tokenizer.pad_token_id is None:
            # GPT-2 non ha pad, usiamo eos come pad per batch
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name_or_path,
            torch_dtype=torch_dtype,
        )

        # Alcuni modelli usano cache; la disattiviamo per hooking più semplice
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"Modello caricato su {self.device} con precisione {self.cfg.precision}"
        )

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _iter_transformer_blocks(self) -> Iterable[Tuple[str, nn.Module]]:
        """
        Cerca blocchi Transformer dentro i modelli più comuni (GPT-like / LLaMA-like).
        Torna una lista di (name, module) su cui registrare hook.

        Non è esaustivo ma copre:
        - GPT-2: model.transformer.h
        - modelli tipo LLaMA: model.model.layers
        (si può estendere in seguito)
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

        # Fallback: niente blocchi trovati
        logger.warning(
            "Non sono riuscito a identificare i blocchi Transformer. "
            "Nessun hook di default verrà registrato."
        )
        return

    def _register_default_hooks(self) -> None:
        """
        Registra hook forward sui blocchi Transformer.

        Ogni block(...) restituisce hidden states [B, S, D].
        Salviamo l'output del blocco come 'layer_{i}.block'.
        """
        if self._hooks_registered:
            return

        logger.info("Registro hook di default sui blocchi Transformer...")

        for name, block in self._iter_transformer_blocks():
            handle = block.register_forward_hook(
                self._make_block_hook(name)
            )
            self._hook_handles.append(handle)

        self._hooks_registered = True
        logger.info(f"Registrati {len(self._hook_handles)} hook.")

    def _make_block_hook(self, layer_name: str):
        """
        Crea una closure che cattura il nome del layer.
        """
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any):
            # output può essere tensor o tuple; gestiamo entrambi
            if isinstance(output, torch.Tensor):
                act = output
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                act = output[0]
            else:
                return

            # attendiamo shape [B, S, D]
            if act.dim() == 3:
                # Spostiamo su CPU per non saturare VRAM
                self._current_activations[f"{layer_name}.block"] = act.detach().to("cpu")
            else:
                # Se la shape non è quella attesa, ignoriamo (per ora)
                logger.debug(
                    f"Hook {layer_name}: output shape inattesa {tuple(act.shape)}"
                )

        return hook

    def _clear_hooks(self) -> None:
        """
        (Al momento non lo usiamo, ma c'è se vorrai disattivare hook a runtime)
        """
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self._hooks_registered = False

    def _reset_activation_buffer(self) -> None:
        self._current_activations = {}

    # ------------------------------------------------------------------
    # Residual Stream Hooks (Control Plane API)
    # ------------------------------------------------------------------

    def _get_block_by_index(self, layer_idx: int) -> nn.Module:
        """
        Recupera il modulo del blocco Transformer per un dato layer index.
        Cache per evitare lookup ripetuti.
        """
        if layer_idx in self._block_cache:
            return self._block_cache[layer_idx]

        for name, block in self._iter_transformer_blocks():
            if name == f"layer_{layer_idx}":
                self._block_cache[layer_idx] = block
                return block

        raise ValueError(
            f"Layer {layer_idx} non trovato. "
            f"Disponibili: {[n for n, _ in self._iter_transformer_blocks()]}"
        )

    def add_residual_hook(
        self,
        layer_idx: int,
        position: str,
        hook_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> ResidualHookHandle:
        """
        Registra un hook sul residual stream per steering attivo.

        Args:
            layer_idx: Indice del layer (0-based, es. 0 per primo layer)
            position: "post_attn" | "post_mlp" | "post_block"
                      - "post_attn": dopo self-attention, prima MLP
                      - "post_mlp": dopo MLP (= output finale del blocco)
                      - "post_block": alias di post_mlp
            hook_fn: Funzione che prende tensor [B, S, D] e ritorna tensor modificato

        Returns:
            ResidualHookHandle con metodo .remove()

        Note:
            - Hook applicati in-place sul residual stream
            - Multipli hook sullo stesso punto vengono composti sequenzialmente
            - Responsabilità del chiamante garantire device matching
        """
        # Normalizza position
        if position == "post_block":
            position = "post_mlp"

        if position not in ("post_attn", "post_mlp"):
            raise ValueError(
                f"position deve essere 'post_attn' o 'post_mlp', ricevuto: {position}"
            )

        block = self._get_block_by_index(layer_idx)

        # Per GPT-2: blocco ha .attn e .mlp (nn.Module)
        # Struttura tipica: residual + attn -> residual + mlp
        # Hook post_mlp: hookiamo l'output del blocco (forward hook)
        # Hook post_attn: hookiamo l'output del modulo attn

        if position == "post_mlp":
            # Hook sull'output finale del blocco
            def wrapper_hook(module: nn.Module, inputs: Any, output: Any) -> Any:
                if isinstance(output, torch.Tensor):
                    return hook_fn(output)
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    # Alcuni blocchi ritornano (hidden, *extra)
                    modified = hook_fn(output[0])
                    return (modified, *output[1:])
                return output

            handle = block.register_forward_hook(wrapper_hook)

        elif position == "post_attn":
            # Hook sull'output del modulo attention
            # GPT-2: block.attn restituisce (attn_output, *optional_outputs)
            if not hasattr(block, "attn"):
                raise AttributeError(
                    f"Layer {layer_idx} non ha attributo 'attn'. "
                    "Supporto post_attn disponibile solo per architetture GPT-2-like."
                )

            attn_module = block.attn

            def attn_wrapper_hook(module: nn.Module, inputs: Any, output: Any) -> Any:
                if isinstance(output, torch.Tensor):
                    return hook_fn(output)
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    modified = hook_fn(output[0])
                    return (modified, *output[1:])
                return output

            handle = attn_module.register_forward_hook(attn_wrapper_hook)

        else:
            raise RuntimeError(f"Unhandled position: {position}")

        # Wrap in ResidualHookHandle
        hook_handle = ResidualHookHandle(handle)

        # Salva nel registry (sovrascrivi se già presente)
        key = (layer_idx, position)
        if key in self._residual_hooks:
            logger.warning(
                f"Sovrascrivendo hook esistente per layer={layer_idx}, position={position}"
            )
            self._residual_hooks[key].remove()

        self._residual_hooks[key] = hook_handle
        return hook_handle

    def remove_all_residual_hooks(self) -> None:
        """
        Rimuove tutti i residual hooks registrati per steering.
        Utile per reset del controller.
        """
        for handle in self._residual_hooks.values():
            handle.remove()
        self._residual_hooks.clear()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_with_trace(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        capture_activations: bool = True,
    ) -> Dict[str, Any]:
        """
        Esegue un forward del modello con opzione di cattura attivazioni.

        Ritorna:
          {
            "logits": Tensor[B, S, V] (CPU),
            "internal_states_reference": Dict[str, Tensor[B, S, D]] (CPU)
          }
        """
        if capture_activations:
            self._reset_activation_buffer()
            self._register_default_hooks()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs.logits.detach().cpu()

        internal_states = dict(self._current_activations)  # shallow copy

        return {
            "logits": logits,
            "internal_states_reference": internal_states,
        }

    @torch.no_grad()
    def run_texts(
        self,
        texts: List[str],
        capture_activations: bool = True,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        API di alto livello usata dalla CLI.

        - Tokenizza lista di testi
        - Manda al modello
        - Ritorna logits + attivazioni (CPU)
        """
        if max_length is None:
            max_length = self.cfg.max_seq_len

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        out = self.forward_with_trace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            capture_activations=capture_activations,
        )

        # aggiungiamo per comodità anche i testi/tokenizzati
        out["texts"] = texts
        out["input_ids"] = input_ids.detach().cpu()
        out["attention_mask"] = attention_mask.detach().cpu()
        return out

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generazione autoregressiva con steering attivo.

        Args:
            prompt: Testo di input
            max_new_tokens: Numero massimo di nuovi token da generare
            temperature: Temperatura per sampling (1.0 = nessun scaling)
            top_k: Top-k sampling (None = disabilitato)
            top_p: Nucleus sampling (None = disabilitato)
            **kwargs: Parametri aggiuntivi per model.generate()

        Returns:
            Testo generato (include prompt originale)

        Note:
            - Eventuali residual hooks attivi verranno applicati durante generazione
            - Per steering deterministico, usa temperature=0.0 (greedy)
        """
        # Tokenizza prompt
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(self.device)

        # Parametri generazione
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

        # Merge con kwargs custom
        gen_kwargs.update(kwargs)

        # Genera (steering hooks attivi automaticamente)
        output_ids = self.model.generate(
            input_ids,
            **gen_kwargs,
        )

        # Decode
        generated_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        return generated_text
