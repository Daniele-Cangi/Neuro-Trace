# neurotrace/visualization/sae_feature_viz.py

"""
Analisi e visualizzazione di SAE features.

Supporta:
- Top features: feature più attive su esempi
- Reconstruction quality: confronto input vs ricostruzione
- Feature activation patterns: heatmap attivazioni per feature
- Feature importance: ranked by activation frequency/magnitude
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Dict

import torch
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from neurotrace.state_indexer.sae_feature_extractor import LayerSparseAutoencoder


class SAEFeatureAnalyzer:
    """
    Analizza e visualizza feature di Sparse Autoencoders.

    Utile per:
    - Capire quali feature si attivano su esempi specifici
    - Valutare qualità di ricostruzione
    - Identificare feature monosemantiche
    """

    def __init__(
        self,
        sae: LayerSparseAutoencoder,
        template: str = "plotly_dark",
        width: int = 1200,
        height: int = 800,
    ):
        """
        Args:
            sae: SAE model da analizzare
            template: Plotly template
            width: Larghezza plot
            height: Altezza plot
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "plotly not installed. Install with: pip install plotly"
            )

        self.sae = sae
        self.template = template
        self.width = width
        self.height = height

    # ========================================================================
    # Reconstruction Quality
    # ========================================================================

    def plot_reconstruction_quality(
        self,
        inputs: torch.Tensor,
        num_samples: int = 5,
        output_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> go.Figure:
        """
        Confronta input vs ricostruzione per valutare SAE quality.

        Args:
            inputs: [N, D] input activations
            num_samples: Numero esempi da plottare
            output_path: Path HTML
            show: Mostra in browser

        Returns:
            Figure
        """
        # Forward pass
        with torch.no_grad():
            result = self.sae(inputs)
            reconstructed = result['reconstruction']
            codes = result['codes']

        # Calcola errori
        mse_per_sample = ((inputs - reconstructed) ** 2).mean(dim=1)
        l1_norm_per_sample = torch.abs(codes).sum(dim=1)

        # Seleziona campioni (best, median, worst)
        num_samples = min(num_samples, inputs.shape[0])
        indices = torch.linspace(0, inputs.shape[0] - 1, num_samples, dtype=torch.long)

        # Subplot: input vs reconstructed per sample
        fig = make_subplots(
            rows=num_samples,
            cols=1,
            subplot_titles=[f"Sample {i} (MSE={mse_per_sample[i]:.4f})" for i in indices],
        )

        for plot_idx, sample_idx in enumerate(indices):
            # Input
            fig.add_trace(
                go.Scatter(
                    y=inputs[sample_idx].cpu().numpy(),
                    mode="lines",
                    name="Input",
                    line=dict(color="blue", width=1),
                    showlegend=(plot_idx == 0),
                ),
                row=plot_idx + 1,
                col=1,
            )

            # Reconstructed
            fig.add_trace(
                go.Scatter(
                    y=reconstructed[sample_idx].cpu().numpy(),
                    mode="lines",
                    name="Reconstructed",
                    line=dict(color="red", width=1, dash="dash"),
                    showlegend=(plot_idx == 0),
                ),
                row=plot_idx + 1,
                col=1,
            )

        fig.update_layout(
            template=self.template,
            width=self.width,
            height=self.height,
            title="SAE Reconstruction Quality",
        )

        if output_path:
            fig.write_html(str(output_path))
        if show:
            fig.show()

        return fig

    def plot_reconstruction_errors(
        self,
        inputs: torch.Tensor,
        output_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> go.Figure:
        """
        Plot distribuzione errori di ricostruzione.

        Args:
            inputs: [N, D] input activations
            output_path: Path HTML
            show: Mostra in browser

        Returns:
            Figure
        """
        with torch.no_grad():
            result = self.sae(inputs)
            reconstructed = result['reconstruction']
            codes = result['codes']

        # Errori per sample
        mse_per_sample = ((inputs - reconstructed) ** 2).mean(dim=1).cpu().numpy()
        l1_norm_per_sample = torch.abs(codes).sum(dim=1).cpu().numpy()
        sparsity_per_sample = (codes != 0).sum(dim=1).float().cpu().numpy()

        # Subplot: MSE, L1, Sparsity distributions
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("MSE Distribution", "L1 Norm Distribution", "Sparsity Distribution"),
        )

        # MSE
        fig.add_trace(
            go.Histogram(
                x=mse_per_sample,
                name="MSE",
                marker=dict(color="red", opacity=0.7),
                nbinsx=30,
            ),
            row=1, col=1,
        )

        # L1
        fig.add_trace(
            go.Histogram(
                x=l1_norm_per_sample,
                name="L1 Norm",
                marker=dict(color="orange", opacity=0.7),
                nbinsx=30,
            ),
            row=1, col=2,
        )

        # Sparsity
        fig.add_trace(
            go.Histogram(
                x=sparsity_per_sample,
                name="Active Features",
                marker=dict(color="green", opacity=0.7),
                nbinsx=30,
            ),
            row=1, col=3,
        )

        fig.update_xaxes(title_text="MSE", row=1, col=1)
        fig.update_xaxes(title_text="L1 Norm", row=1, col=2)
        fig.update_xaxes(title_text="# Active Features", row=1, col=3)

        fig.update_layout(
            template=self.template,
            width=self.width,
            height=self.height,
            title="SAE Reconstruction Error Analysis",
            showlegend=False,
        )

        if output_path:
            fig.write_html(str(output_path))
        if show:
            fig.show()

        return fig

    # ========================================================================
    # Feature Analysis
    # ========================================================================

    def plot_top_features(
        self,
        inputs: torch.Tensor,
        top_k: int = 20,
        output_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> go.Figure:
        """
        Plot top-k feature più attive.

        Args:
            inputs: [N, D] input activations
            top_k: Numero top features
            output_path: Path HTML
            show: Mostra in browser

        Returns:
            Figure
        """
        with torch.no_grad():
            result = self.sae(inputs)
            codes = result['codes']

        # Media attivazione per feature
        feature_activation_mean = codes.abs().mean(dim=0).cpu().numpy()

        # Top-k indices
        top_indices = np.argsort(feature_activation_mean)[-top_k:][::-1]
        top_values = feature_activation_mean[top_indices]

        # Bar chart
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=[f"F{i}" for i in top_indices],
                y=top_values,
                marker=dict(
                    color=top_values,
                    colorscale="Viridis",
                    colorbar=dict(title="Activation"),
                ),
                text=[f"{v:.3f}" for v in top_values],
                textposition="outside",
            )
        )

        fig.update_layout(
            template=self.template,
            width=self.width,
            height=self.height,
            title=f"Top {top_k} Most Active SAE Features",
            xaxis_title="Feature Index",
            yaxis_title="Mean Activation",
        )

        if output_path:
            fig.write_html(str(output_path))
        if show:
            fig.show()

        return fig

    def plot_feature_activation_heatmap(
        self,
        inputs: torch.Tensor,
        max_samples: int = 50,
        max_features: int = 100,
        output_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> go.Figure:
        """
        Heatmap attivazioni: samples × features.

        Args:
            inputs: [N, D] input activations
            max_samples: Limite samples da visualizzare
            max_features: Limite features da visualizzare
            output_path: Path HTML
            show: Mostra in browser

        Returns:
            Figure
        """
        with torch.no_grad():
            result = self.sae(inputs)
            codes = result['codes']

        # Limita dimensioni
        codes_np = codes.cpu().numpy()
        if codes_np.shape[0] > max_samples:
            indices = np.linspace(0, codes_np.shape[0] - 1, max_samples, dtype=int)
            codes_np = codes_np[indices]

        # Seleziona top features (per varianza)
        feature_variance = np.var(codes_np, axis=0)
        top_feature_indices = np.argsort(feature_variance)[-max_features:]
        codes_np = codes_np[:, top_feature_indices]

        # Heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=codes_np,
                x=[f"F{i}" for i in top_feature_indices],
                y=[f"S{i}" for i in range(codes_np.shape[0])],
                colorscale="Viridis",
                colorbar=dict(title="Activation"),
            )
        )

        fig.update_layout(
            template=self.template,
            width=self.width,
            height=self.height,
            title="SAE Feature Activation Heatmap",
            xaxis_title="Feature",
            yaxis_title="Sample",
        )

        if output_path:
            fig.write_html(str(output_path))
        if show:
            fig.show()

        return fig

    def plot_feature_frequency(
        self,
        inputs: torch.Tensor,
        threshold: float = 0.01,
        output_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> go.Figure:
        """
        Plot frequenza di attivazione per feature (quanti esempi attivano ogni feature).

        Args:
            inputs: [N, D] input activations
            threshold: Soglia per considerare feature "attiva"
            output_path: Path HTML
            show: Mostra in browser

        Returns:
            Figure
        """
        with torch.no_grad():
            result = self.sae(inputs)
            codes = result['codes']

        # Conta quanti samples attivano ogni feature
        active_mask = (codes.abs() > threshold).float()
        frequency = active_mask.sum(dim=0).cpu().numpy()  # [dict_size]
        frequency_pct = (frequency / inputs.shape[0]) * 100

        # Ordina per frequenza
        sorted_indices = np.argsort(frequency_pct)[::-1]

        # Plot
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(len(frequency_pct))),
                y=frequency_pct[sorted_indices],
                mode="lines",
                line=dict(color="blue", width=2),
            )
        )

        # Linee di riferimento
        fig.add_hline(y=10, line_dash="dash", line_color="green", annotation_text="10%")
        fig.add_hline(y=1, line_dash="dash", line_color="orange", annotation_text="1%")

        fig.update_layout(
            template=self.template,
            width=self.width,
            height=self.height,
            title=f"SAE Feature Activation Frequency (threshold={threshold})",
            xaxis_title="Feature (sorted by frequency)",
            yaxis_title="% Samples Activating Feature",
        )

        if output_path:
            fig.write_html(str(output_path))
        if show:
            fig.show()

        return fig

    # ========================================================================
    # Feature Interpretation
    # ========================================================================

    def get_top_activating_examples(
        self,
        inputs: torch.Tensor,
        feature_idx: int,
        top_k: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trova top-k esempi che attivano maggiormente una feature specifica.

        Args:
            inputs: [N, D] input activations
            feature_idx: Indice feature da analizzare
            top_k: Numero esempi da ritornare

        Returns:
            (top_examples, top_activations):
                - top_examples: [top_k, D] esempi
                - top_activations: [top_k] valori attivazione
        """
        with torch.no_grad():
            result = self.sae(inputs)
            codes = result['codes']

        # Attivazioni per questa feature
        feature_activations = codes[:, feature_idx]

        # Top-k
        top_values, top_indices = torch.topk(feature_activations.abs(), top_k)
        top_examples = inputs[top_indices]

        return top_examples, top_values
