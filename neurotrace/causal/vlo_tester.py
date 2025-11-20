# neurotrace/causal/vlo_tester.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class InterventionType(Enum):
    """Tipo di intervento causale."""
    ZERO_ABLATION = "zero_ablation"  # Set activations to zero
    MEAN_ABLATION = "mean_ablation"  # Replace with mean across examples
    RESAMPLE = "resample"  # Replace with random example
    PATCH = "patch"  # Replace with specific counterfactual


@dataclass
class VLOResult:
    """
    Risultato di un test VLO (Value of Learned Organization).

    VLO = logit_diff(clean) - logit_diff(intervened)

    VLO > 0: il componente contribuisce positivamente al task
    VLO < 0: il componente danneggia il task
    |VLO| grande: componente causalmente importante
    """
    # Logit differences
    clean_logit_diff: float
    intervened_logit_diff: float

    # VLO metric
    vlo: float  # clean - intervened

    # Normalized metrics
    faithfulness: float  # |VLO| / |clean_logit_diff|
    effect_size: float  # VLO / std(random_interventions)

    # Metadata
    intervention_type: InterventionType
    component_name: str
    num_examples: int


def compute_vlo(
    clean_logit_diff: float,
    intervened_logit_diff: float,
) -> float:
    """
    Compute VLO = clean - intervened.

    Interpretation:
    - VLO > 0: intervention hurt performance → component was helpful
    - VLO < 0: intervention improved performance → component was harmful
    - VLO ≈ 0: component not causally relevant
    """
    return clean_logit_diff - intervened_logit_diff


class VLOTester:
    """
    Tester per VLO (Value of Learned Organization) su circuiti neurali.

    Workflow:
    1. Run modello su clean examples → clean logits
    2. Intervene su componenti specifici (ablation/patch)
    3. Run modello con intervento → intervened logits
    4. Compute logit difference → VLO metric

    Example:
        tester = VLOTester(model_wrapper)
        result = tester.test_component(
            layer_idx=9,
            component_type="attention_head",
            examples=clean_examples,
            intervention_type=InterventionType.ZERO_ABLATION
        )
        print(f"VLO: {result.vlo:.3f}, Faithfulness: {result.faithfulness:.3f}")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[any] = None,
        device: str = "cuda",
    ) -> None:
        """
        Args:
            model: PyTorch model (es. TargetModelWrapper.model)
            tokenizer: Tokenizer for text processing
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def compute_logit_difference(
        self,
        logits: torch.Tensor,
        target_positions: torch.Tensor,
        correct_token_ids: torch.Tensor,
        incorrect_token_ids: torch.Tensor,
    ) -> float:
        """
        Compute logit difference between correct and incorrect tokens.

        logit_diff = mean(logits[correct] - logits[incorrect])

        Args:
            logits: [batch, seq, vocab]
            target_positions: [batch] - positions to measure
            correct_token_ids: [batch] - IDs of correct tokens
            incorrect_token_ids: [batch] - IDs of incorrect tokens

        Returns:
            Mean logit difference across batch
        """
        batch_size = logits.shape[0]
        logit_diffs = []

        for i in range(batch_size):
            pos = target_positions[i]
            correct_id = correct_token_ids[i]
            incorrect_id = incorrect_token_ids[i]

            logit_correct = logits[i, pos, correct_id]
            logit_incorrect = logits[i, pos, incorrect_id]

            logit_diffs.append((logit_correct - logit_incorrect).item())

        return float(torch.tensor(logit_diffs).mean().item())

    def test_component(
        self,
        layer_idx: int,
        component_type: str,
        component_idx: Optional[int],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_positions: torch.Tensor,
        correct_token_ids: torch.Tensor,
        incorrect_token_ids: torch.Tensor,
        intervention_type: InterventionType = InterventionType.ZERO_ABLATION,
        counterfactual_input_ids: Optional[torch.Tensor] = None,
    ) -> VLOResult:
        """
        Test causal importance of a component via intervention.

        Args:
            layer_idx: Layer index
            component_type: "attention_head", "mlp", "residual"
            component_idx: Head index (for attention), None for full layer
            input_ids: [batch, seq] clean examples
            attention_mask: [batch, seq]
            target_positions: [batch] positions where to measure logits
            correct_token_ids: [batch] correct token IDs
            incorrect_token_ids: [batch] incorrect token IDs
            intervention_type: Type of intervention
            counterfactual_input_ids: [batch, seq] for PATCH intervention

        Returns:
            VLOResult with VLO and faithfulness metrics
        """
        # Batch processing to avoid memory issues with large datasets
        batch_size = 50  # Optimized for 6GB VRAM notebook GPUs
        num_examples = input_ids.shape[0]

        # 1. Clean forward pass (batched)
        clean_logit_diffs = []
        with torch.no_grad():
            for i in range(0, num_examples, batch_size):
                batch_input = input_ids[i:i+batch_size].to(self.device)
                batch_mask = attention_mask[i:i+batch_size].to(self.device)

                batch_outputs = self.model(input_ids=batch_input, attention_mask=batch_mask)
                batch_logits = batch_outputs.logits.cpu()

                batch_logit_diff = self.compute_logit_difference(
                    batch_logits,
                    target_positions[i:i+batch_size],
                    correct_token_ids[i:i+batch_size],
                    incorrect_token_ids[i:i+batch_size],
                )
                clean_logit_diffs.append(batch_logit_diff)

                del batch_outputs, batch_logits
                torch.cuda.empty_cache()

        clean_logit_diff = sum(clean_logit_diffs) / len(clean_logit_diffs)

        # 2. Intervened forward pass (batched)
        intervention_hook = self._create_intervention_hook(
            layer_idx=layer_idx,
            component_type=component_type,
            component_idx=component_idx,
            intervention_type=intervention_type,
            input_ids=input_ids,
            counterfactual_input_ids=counterfactual_input_ids,
        )

        # Register hook
        handle = self._register_intervention_hook(layer_idx, component_type, intervention_hook)

        try:
            intervened_logit_diffs = []
            with torch.no_grad():
                for i in range(0, num_examples, batch_size):
                    batch_input = input_ids[i:i+batch_size].to(self.device)
                    batch_mask = attention_mask[i:i+batch_size].to(self.device)

                    batch_outputs = self.model(input_ids=batch_input, attention_mask=batch_mask)
                    batch_logits = batch_outputs.logits.cpu()

                    batch_logit_diff = self.compute_logit_difference(
                        batch_logits,
                        target_positions[i:i+batch_size],
                        correct_token_ids[i:i+batch_size],
                        incorrect_token_ids[i:i+batch_size],
                    )
                    intervened_logit_diffs.append(batch_logit_diff)

                    del batch_outputs, batch_logits
                    torch.cuda.empty_cache()

            intervened_logit_diff = sum(intervened_logit_diffs) / len(intervened_logit_diffs)

        finally:
            handle.remove()

        # 3. Compute VLO
        vlo = compute_vlo(clean_logit_diff, intervened_logit_diff)

        # 4. Faithfulness
        if abs(clean_logit_diff) > 1e-6:
            faithfulness = abs(vlo) / abs(clean_logit_diff)
        else:
            faithfulness = 0.0

        # 5. Effect size (placeholder, serve baseline random)
        effect_size = abs(vlo)  # Simplified, ideally vlo / std(random)

        component_name = f"layer_{layer_idx}.{component_type}"
        if component_idx is not None:
            component_name += f".{component_idx}"

        return VLOResult(
            clean_logit_diff=clean_logit_diff,
            intervened_logit_diff=intervened_logit_diff,
            vlo=vlo,
            faithfulness=faithfulness,
            effect_size=effect_size,
            intervention_type=intervention_type,
            component_name=component_name,
            num_examples=input_ids.shape[0],
        )

    def _create_intervention_hook(
        self,
        layer_idx: int,
        component_type: str,
        component_idx: Optional[int],
        intervention_type: InterventionType,
        input_ids: torch.Tensor,
        counterfactual_input_ids: Optional[torch.Tensor],
    ) -> Callable:
        """
        Create hook function for intervention.

        For attention_head: component_idx specifies which head to ablate (0-11 for GPT-2)
        For mlp: component_idx is ignored (ablate entire MLP)
        """
        if intervention_type == InterventionType.ZERO_ABLATION:
            def hook(module, input, output):
                if isinstance(output, tuple):
                    # Attention outputs: (hidden_states, *extras)
                    original = output[0]

                    # If targeting specific attention head, only ablate that head
                    if component_type == "attention_head" and component_idx is not None:
                        # For GPT-2: hidden_states shape is [batch, seq, hidden_dim]
                        # We need to zero out the contribution of specific head
                        # Each head output is hidden_dim/num_heads dimensions
                        # This requires reshaping to [batch, seq, num_heads, head_dim]

                        batch_size, seq_len, hidden_dim = original.shape
                        num_heads = 12  # GPT-2 has 12 heads
                        head_dim = hidden_dim // num_heads

                        # Reshape to separate heads
                        reshaped = original.view(batch_size, seq_len, num_heads, head_dim)

                        # Zero out specific head
                        modified = reshaped.clone()
                        modified[:, :, component_idx, :] = 0

                        # Reshape back
                        modified = modified.view(batch_size, seq_len, hidden_dim)

                        return (modified, *output[1:])
                    else:
                        # Ablate entire module (MLP or full attention)
                        modified = torch.zeros_like(original)
                        return (modified, *output[1:])
                else:
                    return torch.zeros_like(output)

        elif intervention_type == InterventionType.MEAN_ABLATION:
            def hook(module, input, output):
                if isinstance(output, tuple):
                    original = output[0]

                    if component_type == "attention_head" and component_idx is not None:
                        # Mean ablation for specific head
                        batch_size, seq_len, hidden_dim = original.shape
                        num_heads = 12
                        head_dim = hidden_dim // num_heads

                        reshaped = original.view(batch_size, seq_len, num_heads, head_dim)
                        modified = reshaped.clone()

                        # Replace head with mean activation
                        head_mean = reshaped[:, :, component_idx, :].mean(dim=0, keepdim=True)
                        modified[:, :, component_idx, :] = head_mean.expand(batch_size, seq_len, head_dim)

                        modified = modified.view(batch_size, seq_len, hidden_dim)
                        return (modified, *output[1:])
                    else:
                        mean_activation = original.mean(dim=0, keepdim=True)
                        modified = mean_activation.expand_as(original)
                        return (modified, *output[1:])
                else:
                    mean_activation = output.mean(dim=0, keepdim=True)
                    return mean_activation.expand_as(output)

        else:
            # Fallback: no intervention (for testing)
            def hook(module, input, output):
                return output

        return hook

    def _register_intervention_hook(
        self,
        layer_idx: int,
        component_type: str,
        hook_fn: Callable,
    ) -> any:
        """
        Register hook on specific component.

        Returns:
            Hook handle with .remove() method
        """
        # Navigate to layer
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            # GPT-2 style
            layer = self.model.transformer.h[layer_idx]
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # LLaMA style
            layer = self.model.model.layers[layer_idx]
        else:
            raise ValueError("Unknown model architecture")

        # Register on component
        if component_type == "attention_head":
            # Hook on attention module output
            module = layer.attn if hasattr(layer, "attn") else layer.self_attn
            handle = module.register_forward_hook(hook_fn)
        elif component_type == "mlp":
            # Hook on MLP module
            module = layer.mlp
            handle = module.register_forward_hook(hook_fn)
        elif component_type == "residual" or component_type == "block":
            # Hook on full block
            handle = layer.register_forward_hook(hook_fn)
        else:
            raise ValueError(f"Unknown component_type: {component_type}")

        return handle

    def test_circuit(
        self,
        components: List[Tuple[int, str, Optional[int]]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_positions: torch.Tensor,
        correct_token_ids: torch.Tensor,
        incorrect_token_ids: torch.Tensor,
        intervention_type: InterventionType = InterventionType.ZERO_ABLATION,
    ) -> List[VLOResult]:
        """
        Test multiple components in a circuit.

        Args:
            components: List of (layer_idx, component_type, component_idx)

        Returns:
            List of VLOResult, one per component
        """
        results = []
        for layer_idx, component_type, component_idx in components:
            result = self.test_component(
                layer_idx=layer_idx,
                component_type=component_type,
                component_idx=component_idx,
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_positions=target_positions,
                correct_token_ids=correct_token_ids,
                incorrect_token_ids=incorrect_token_ids,
                intervention_type=intervention_type,
            )
            results.append(result)
            logger.info(f"{result.component_name}: VLO={result.vlo:.3f}, Faithfulness={result.faithfulness:.3f}")

        return results
