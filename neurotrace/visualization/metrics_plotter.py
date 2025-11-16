# neurotrace/visualization/metrics_plotter.py

"""
Plot interattivi di metriche con Plotly.

Supporta:
- Training metrics: loss, sparsity, learning rate over time
- VLO metrics: per-component VLO, faithfulness distributions
- Comparative plots: multi-circuit comparison
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from neurotrace.training.sae_trainer import TrainingMetrics
from neurotrace.causal.vlo_tester import VLOResult
from neurotrace.control.circuit_registry import CircuitRecord


@dataclass
class TrainingMetricsPlot:
    """Container per metriche di training da plottare."""
    steps: List[int]
    total_loss: List[float]
    mse_loss: List[float]
    l1_loss: List[float]
    sparsity: List[float]
    learning_rate: Optional[List[float]] = None


@dataclass
class VLOMetricsPlot:
    """Container per metriche VLO da plottare."""
    component_names: List[str]
    vlo_values: List[float]
    faithfulness_values: List[float]
    clean_logit_diff: List[float]
    intervened_logit_diff: List[float]


class MetricsPlotter:
    """
    Plotter per metriche di training e causal discovery.

    Usa Plotly per plot interattivi salvabili come HTML.
    """

    def __init__(
        self,
        template: str = "plotly_dark",  # "plotly", "plotly_dark", "seaborn"
        width: int = 1200,
        height: int = 600,
    ):
        """
        Args:
            template: Plotly template (dark/light theme)
            width: Larghezza plot in pixel
            height: Altezza plot in pixel
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "plotly not installed. Install with: pip install plotly"
            )

        self.template = template
        self.width = width
        self.height = height

    # ========================================================================
    # Training Metrics
    # ========================================================================

    def plot_training_metrics(
        self,
        metrics: TrainingMetricsPlot,
        output_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> go.Figure:
        """
        Plot metriche di training SAE.

        Args:
            metrics: TrainingMetricsPlot con dati
            output_path: Se fornito, salva HTML
            show: Se True, mostra in browser

        Returns:
            Plotly Figure
        """
        # Crea subplot 2x2
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Total Loss",
                "MSE vs L1 Loss",
                "Sparsity (Active Features)",
                "Learning Rate"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
            ],
        )

        # 1. Total Loss
        fig.add_trace(
            go.Scatter(
                x=metrics.steps,
                y=metrics.total_loss,
                mode="lines",
                name="Total Loss",
                line=dict(color="red", width=2),
            ),
            row=1, col=1,
        )

        # 2. MSE vs L1
        fig.add_trace(
            go.Scatter(
                x=metrics.steps,
                y=metrics.mse_loss,
                mode="lines",
                name="MSE Loss",
                line=dict(color="blue", width=2),
            ),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=metrics.steps,
                y=metrics.l1_loss,
                mode="lines",
                name="L1 Loss",
                line=dict(color="orange", width=2),
            ),
            row=1, col=2,
        )

        # 3. Sparsity
        fig.add_trace(
            go.Scatter(
                x=metrics.steps,
                y=metrics.sparsity,
                mode="lines",
                name="Sparsity",
                line=dict(color="green", width=2),
            ),
            row=2, col=1,
        )

        # 4. Learning Rate
        if metrics.learning_rate:
            fig.add_trace(
                go.Scatter(
                    x=metrics.steps,
                    y=metrics.learning_rate,
                    mode="lines",
                    name="Learning Rate",
                    line=dict(color="purple", width=2),
                ),
                row=2, col=2,
            )

        # Layout
        fig.update_xaxes(title_text="Training Step", row=2, col=1)
        fig.update_xaxes(title_text="Training Step", row=2, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        fig.update_yaxes(title_text="Active Features", row=2, col=1)
        fig.update_yaxes(title_text="LR", row=2, col=2, type="log")

        fig.update_layout(
            template=self.template,
            width=self.width,
            height=self.height,
            title_text="SAE Training Metrics",
            showlegend=True,
        )

        # Save/show
        if output_path:
            fig.write_html(str(output_path))
        if show:
            fig.show()

        return fig

    def plot_training_history(
        self,
        history: List[TrainingMetrics],
        output_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> go.Figure:
        """
        Plot storico training da lista di TrainingMetrics.

        Args:
            history: Lista di TrainingMetrics (da SAETrainer.metrics_history)
            output_path: Path HTML output
            show: Mostra in browser

        Returns:
            Figure
        """
        steps = list(range(len(history)))
        total_loss = [m.total_loss for m in history]
        mse_loss = [m.mse_loss for m in history]
        l1_loss = [m.l1_loss for m in history]
        sparsity = [m.sparsity for m in history]
        lr = [m.learning_rate for m in history] if history and hasattr(history[0], "learning_rate") else None

        metrics_plot = TrainingMetricsPlot(
            steps=steps,
            total_loss=total_loss,
            mse_loss=mse_loss,
            l1_loss=l1_loss,
            sparsity=sparsity,
            learning_rate=lr,
        )

        return self.plot_training_metrics(metrics_plot, output_path, show)

    # ========================================================================
    # VLO Metrics
    # ========================================================================

    def plot_vlo_results(
        self,
        vlo_results: List[VLOResult],
        output_path: Optional[str | Path] = None,
        show: bool = False,
        sort_by: str = "vlo",  # "vlo", "faithfulness", "name"
    ) -> go.Figure:
        """
        Plot risultati VLO per componenti.

        Args:
            vlo_results: Lista VLOResult
            output_path: Path HTML
            show: Mostra in browser
            sort_by: Criterio ordinamento

        Returns:
            Figure
        """
        # Ordina
        if sort_by == "vlo":
            vlo_results = sorted(vlo_results, key=lambda r: r.vlo, reverse=True)
        elif sort_by == "faithfulness":
            vlo_results = sorted(vlo_results, key=lambda r: r.faithfulness, reverse=True)
        else:
            vlo_results = sorted(vlo_results, key=lambda r: r.component_name)

        component_names = [r.component_name for r in vlo_results]
        vlo_values = [r.vlo for r in vlo_results]
        faithfulness = [r.faithfulness for r in vlo_results]

        # Subplot: VLO bar + Faithfulness bar
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("VLO per Component", "Faithfulness per Component"),
        )

        # VLO bar chart
        fig.add_trace(
            go.Bar(
                x=component_names,
                y=vlo_values,
                name="VLO",
                marker=dict(
                    color=vlo_values,
                    colorscale="RdYlGn",
                    colorbar=dict(title="VLO", x=0.45),
                ),
                text=[f"{v:.3f}" for v in vlo_values],
                textposition="outside",
            ),
            row=1, col=1,
        )

        # Faithfulness bar chart
        fig.add_trace(
            go.Bar(
                x=component_names,
                y=faithfulness,
                name="Faithfulness",
                marker=dict(
                    color=faithfulness,
                    colorscale="Blues",
                    colorbar=dict(title="Faithfulness", x=1.05),
                ),
                text=[f"{v:.3f}" for v in faithfulness],
                textposition="outside",
            ),
            row=1, col=2,
        )

        # Layout
        fig.update_xaxes(title_text="Component", tickangle=-45, row=1, col=1)
        fig.update_xaxes(title_text="Component", tickangle=-45, row=1, col=2)
        fig.update_yaxes(title_text="VLO", row=1, col=1)
        fig.update_yaxes(title_text="Faithfulness", row=1, col=2, range=[0, 1])

        fig.update_layout(
            template=self.template,
            width=self.width,
            height=self.height,
            title_text="VLO Testing Results",
            showlegend=False,
        )

        if output_path:
            fig.write_html(str(output_path))
        if show:
            fig.show()

        return fig

    def plot_vlo_distribution(
        self,
        vlo_results: List[VLOResult],
        output_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> go.Figure:
        """
        Plot distribuzione VLO e faithfulness con istogrammi.

        Args:
            vlo_results: Lista VLOResult
            output_path: Path HTML
            show: Mostra in browser

        Returns:
            Figure
        """
        vlo_values = [r.vlo for r in vlo_results]
        faithfulness = [r.faithfulness for r in vlo_results]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("VLO Distribution", "Faithfulness Distribution"),
        )

        # VLO histogram
        fig.add_trace(
            go.Histogram(
                x=vlo_values,
                name="VLO",
                marker=dict(color="red", opacity=0.7),
                nbinsx=20,
            ),
            row=1, col=1,
        )

        # Faithfulness histogram
        fig.add_trace(
            go.Histogram(
                x=faithfulness,
                name="Faithfulness",
                marker=dict(color="blue", opacity=0.7),
                nbinsx=20,
            ),
            row=1, col=2,
        )

        # Aggiungi linee per soglie
        fig.add_vline(x=0.5, line_dash="dash", line_color="green", row=1, col=1)
        fig.add_vline(x=0.3, line_dash="dash", line_color="green", row=1, col=2)

        fig.update_xaxes(title_text="VLO", row=1, col=1)
        fig.update_xaxes(title_text="Faithfulness", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)

        fig.update_layout(
            template=self.template,
            width=self.width,
            height=self.height,
            title_text="VLO and Faithfulness Distributions",
            showlegend=False,
        )

        if output_path:
            fig.write_html(str(output_path))
        if show:
            fig.show()

        return fig

    # ========================================================================
    # Circuit Comparison
    # ========================================================================

    def plot_circuit_comparison(
        self,
        circuits: List[CircuitRecord],
        output_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> go.Figure:
        """
        Confronto metriche di più circuiti.

        Args:
            circuits: Lista CircuitRecord
            output_path: Path HTML
            show: Mostra in browser

        Returns:
            Figure
        """
        circuit_ids = [c.circuit_id for c in circuits]
        vlo_means = [c.causal_metrics.vlo_mean for c in circuits]
        faithfulness = [c.causal_metrics.faithfulness for c in circuits]
        num_components = [len(c.components) for c in circuits]

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("VLO Mean", "Faithfulness", "# Components"),
        )

        # VLO mean
        fig.add_trace(
            go.Bar(
                x=circuit_ids,
                y=vlo_means,
                name="VLO Mean",
                marker=dict(color="red"),
                text=[f"{v:.3f}" for v in vlo_means],
                textposition="outside",
            ),
            row=1, col=1,
        )

        # Faithfulness
        fig.add_trace(
            go.Bar(
                x=circuit_ids,
                y=faithfulness,
                name="Faithfulness",
                marker=dict(color="blue"),
                text=[f"{v:.3f}" for v in faithfulness],
                textposition="outside",
            ),
            row=1, col=2,
        )

        # Num components
        fig.add_trace(
            go.Bar(
                x=circuit_ids,
                y=num_components,
                name="Components",
                marker=dict(color="green"),
                text=[str(n) for n in num_components],
                textposition="outside",
            ),
            row=1, col=3,
        )

        fig.update_xaxes(tickangle=-45, row=1, col=1)
        fig.update_xaxes(tickangle=-45, row=1, col=2)
        fig.update_xaxes(tickangle=-45, row=1, col=3)

        fig.update_layout(
            template=self.template,
            width=self.width,
            height=self.height,
            title_text="Circuit Comparison",
            showlegend=False,
        )

        if output_path:
            fig.write_html(str(output_path))
        if show:
            fig.show()

        return fig

    # ========================================================================
    # Generic Plot
    # ========================================================================

    def plot_line(
        self,
        x: List[float],
        y: List[float],
        title: str = "Line Plot",
        xlabel: str = "X",
        ylabel: str = "Y",
        output_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> go.Figure:
        """
        Plot generico linea.

        Args:
            x: Valori asse X
            y: Valori asse Y
            title: Titolo
            xlabel: Label X
            ylabel: Label Y
            output_path: Path HTML
            show: Mostra in browser

        Returns:
            Figure
        """
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                line=dict(width=2),
                marker=dict(size=8),
            )
        )

        fig.update_layout(
            template=self.template,
            width=self.width,
            height=self.height,
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
        )

        if output_path:
            fig.write_html(str(output_path))
        if show:
            fig.show()

        return fig
