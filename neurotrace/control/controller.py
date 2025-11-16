# neurotrace/control/controller.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import torch

from .circuit_registry import CircuitRegistry, CircuitRecord
from .steering_builder import SteeringBuilder, SteeringSpec, LayerSteeringVector


# ---------------------------- Hook Interfaces -----------------------------


class ResidualHookHandle(Protocol):
    """
    Handle per rimuovere un hook sul residual stream.
    """

    def remove(self) -> None:
        ...


class ModelWrapper(Protocol):
    """
    Interfaccia minima che il tuo TargetModelWrapper dovrebbe soddisfare
    per usare il controller.

    Adattala al tuo wrapper reale (quello che hai già nella repo).
    """

    def add_residual_hook(
        self,
        layer_idx: int,
        position: str,
        hook_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> ResidualHookHandle:
        """
        Registra una funzione che modifica il residual stream in un certo layer.

        Args:
            layer_idx: indice del layer Transformer.
            position: "post_attn" oppure "post_mlp" (in linea con il tuo wrapper).
            hook_fn: f(tensor) -> tensor modificato.

        Returns:
            Handle con .remove() per disattivare il hook.
        """
        ...

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        **kwargs: Any,
    ) -> str:
        """
        Wrapper di alto livello per generazione autoregressiva.
        """
        ...


@dataclass
class ActiveCircuit:
    spec: SteeringSpec
    alpha_per_layer: Dict[int, float]
    handles: Dict[Tuple[int, str], ResidualHookHandle]


@dataclass
class ControlTrace:
    """
    Traccia dell'ultima generazione sotto controllo.
    """
    prompt: str
    output: str
    active_circuits: List[str]
    layer_alphas: Dict[str, Dict[int, float]]  # circuit_id -> {layer: alpha}
    metadata: Dict[str, Any]


class CircuitController:
    """
    Strato di controllo attivo sui circuiti scoperti da NeuroTrace.

    Usa:
      - CircuitRegistry per caricare i CircuitRecord
      - SteeringBuilder per costruire SteeringSpec
      - ModelWrapper per applicare steering sul residual stream
    """

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        registry: CircuitRegistry,
        steering_builder: SteeringBuilder,
        residual_position: str = "post_mlp",
    ) -> None:
        self._model = model_wrapper
        self._registry = registry
        self._builder = steering_builder
        self._residual_position = residual_position

        self._active: Dict[str, ActiveCircuit] = {}
        self._last_trace: Optional[ControlTrace] = None

    # ----------------------------- Public API -----------------------------

    def list_circuits(
        self,
        task_tag: Optional[str] = None,
        min_vlo: Optional[float] = None,
        min_faithfulness: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[CircuitRecord]:
        return self._registry.list(
            task_tag=task_tag,
            min_vlo=min_vlo,
            min_faithfulness=min_faithfulness,
            limit=limit,
        )

    def enable_circuit(
        self,
        circuit_id: str,
        global_alpha: Optional[float] = None,
        per_layer_scaling: Optional[Dict[int, float]] = None,
        overwrite_if_active: bool = True,
    ) -> None:
        """
        Abilita un circuito: costruisce steering vectors, registra hooks sul modello.

        global_alpha: scaling di default per tutti i layer (se non sovrascritto).
        per_layer_scaling: override per singoli layer (moltiplica global_alpha).
        """
        record = self._registry.get(circuit_id)
        if record is None:
            raise ValueError(f"Circuit '{circuit_id}' not found in registry")

        # costruisci SteeringSpec
        spec = self._builder.build_from_circuit(
            record=record,
            per_layer_scaling=None,  # la usiamo sotto con alpha effettivi
        )

        if not spec.layer_vectors:
            raise ValueError(
                f"Circuit '{circuit_id}' has no valid steering layers; "
                "check SAE indices / feature store integration."
            )

        if circuit_id in self._active and not overwrite_if_active:
            raise RuntimeError(f"Circuit '{circuit_id}' already active")

        # se già attivo e overwrite, prima rimuovi i vecchi hook
        if circuit_id in self._active:
            self.disable_circuit(circuit_id)

        # calcola alpha effettivi per layer
        alpha_per_layer: Dict[int, float] = {}
        for layer_idx, lv in spec.layer_vectors.items():
            base = global_alpha if global_alpha is not None else lv.default_alpha
            if per_layer_scaling and layer_idx in per_layer_scaling:
                base *= per_layer_scaling[layer_idx]
            # clamp nei bounds
            lo, hi = lv.alpha_bounds
            alpha_per_layer[layer_idx] = float(max(lo, min(hi, base)))

        # registra i hook
        handles: Dict[Tuple[int, str], ResidualHookHandle] = {}
        for layer_idx, lv in spec.layer_vectors.items():
            alpha_ref = alpha_per_layer[layer_idx]  # snapshot iniziale

            # usiamo un piccolo oggetto per permettere aggiornamenti (mutable alpha)
            alpha_box = {"value": alpha_ref}

            def make_hook(
                layer_vector: LayerSteeringVector,
                alpha_box_ref: Dict[str, float],
            ) -> Callable[[torch.Tensor], torch.Tensor]:
                def hook(t: torch.Tensor) -> torch.Tensor:
                    # t: [batch, seq, hidden_dim]
                    # layer_vector.direction: [hidden_dim]
                    # broadcast
                    if alpha_box_ref["value"] == 0.0:
                        return t
                    # Match dtype and device of input tensor
                    direction = layer_vector.direction.to(dtype=t.dtype, device=t.device)
                    return t + alpha_box_ref["value"] * direction
                return hook

            hook_fn = make_hook(lv, alpha_box)
            handle = self._model.add_residual_hook(
                layer_idx=layer_idx,
                position=self._residual_position,
                hook_fn=hook_fn,
            )
            handles[(layer_idx, self._residual_position)] = handle

            # sostituiamo il valore alpha nel dict con un riferimento mutabile
            alpha_per_layer[layer_idx] = alpha_box  # type: ignore[assignment]

        self._active[circuit_id] = ActiveCircuit(
            spec=spec,
            alpha_per_layer={k: v["value"] for k, v in alpha_per_layer.items()},  # snapshot
            handles=handles,
        )

    def disable_circuit(self, circuit_id: str) -> None:
        """
        Rimuove tutti i hook associati ad un circuito.
        """
        ac = self._active.pop(circuit_id, None)
        if ac is None:
            return
        for handle in ac.handles.values():
            try:
                handle.remove()
            except Exception:
                # non bloccare per errori nello shutdown hooks
                pass

    def set_circuit_alpha(
        self,
        circuit_id: str,
        new_alpha: float,
        per_layer: Optional[Dict[int, float]] = None,
    ) -> None:
        """
        Aggiorna alpha (globale o per layer) per un circuito già attivo.
        Nota: questo assume che i hook usino un alpha box mutabile (come sopra).
        """
        if circuit_id not in self._active:
            raise ValueError(f"Circuit '{circuit_id}' not active")

        ac = self._active[circuit_id]
        per_layer = per_layer or {}

        # NB: in questa versione base, non teniamo il reference all'alpha_box;
        # quello è integrato nel closure del hook. Se vuoi alpha mutabile runtime,
        # puoi estendere ActiveCircuit per salvarli.
        # Qui ci limitiamo a loggare il nuovo valore nel nostro snapshot.
        for layer_idx in ac.spec.layer_vectors.keys():
            scale = per_layer.get(layer_idx, 1.0)
            lv = ac.spec.layer_vectors[layer_idx]
            lo, hi = lv.alpha_bounds
            updated = float(max(lo, min(hi, new_alpha * scale)))
            ac.alpha_per_layer[layer_idx] = updated

        # Per un vero aggiornamento runtime, servirebbe una struttura
        # condivisa (alpha_box) salvata qui; è un punto che puoi far
        # integrare a Copilot/Claude sui tuoi hook reali.

    def clear_all(self) -> None:
        """
        Disattiva tutti i circuiti attivi.
        """
        for cid in list(self._active.keys()):
            self.disable_circuit(cid)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        **kwargs: Any,
    ) -> str:
        """
        Esegue una generazione con i circuiti attualmente attivi.

        Salva una ControlTrace che puoi interrogare con last_trace().
        """
        output = self._model.generate(
            prompt=prompt, max_new_tokens=max_new_tokens, **kwargs
        )

        # snapshot stato attivo
        layer_alphas: Dict[str, Dict[int, float]] = {}
        for cid, ac in self._active.items():
            layer_alphas[cid] = dict(ac.alpha_per_layer)

        self._last_trace = ControlTrace(
            prompt=prompt,
            output=output,
            active_circuits=list(self._active.keys()),
            layer_alphas=layer_alphas,
            metadata={},
        )
        return output

    def last_trace(self) -> Optional[ControlTrace]:
        return self._last_trace

    # ----------------------------- Introspection ---------------------------

    def active_circuits_summary(self) -> Dict[str, Any]:
        """
        Piccolo summary utile per logging / UI.
        """
        summary: Dict[str, Any] = {"count": len(self._active), "circuits": []}
        for cid, ac in self._active.items():
            summary["circuits"].append(
                {
                    "circuit_id": cid,
                    "task_tag": ac.spec.task_tag,
                    "label": ac.spec.semantics_label,
                    "layers": sorted(ac.spec.layer_vectors.keys()),
                    "alpha_per_layer": ac.alpha_per_layer,
                }
            )
        return summary
