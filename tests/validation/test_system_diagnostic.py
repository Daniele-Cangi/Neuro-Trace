#!/usr/bin/env python3
"""
NeuroTrace System Diagnostic Test Suite

Purpose: Comprehensive validation of entire system to understand what works and what doesn't.

Tests:
1. Network Interception Fidelity (are we capturing activations 1:1?)
2. SAE Reconstruction Quality (per layer)
3. Steering Causality (does it actually change outputs?)
4. Atlas Functionality (can we navigate the neural map?)
5. Data Analysis (systematic JSON processing)

This is NOT a unit test - this is a DIAGNOSTIC to understand the system.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# NeuroTrace imports
from neurotrace.config import NeuroTraceConfig
from neurotrace.models.wrapper import TargetModelWrapper
from neurotrace.datasets.ioi_generator import IOIDatasetGenerator
from neurotrace.control import EnhancedSAEFeatureStore, CircuitController, CircuitRegistry, SteeringBuilder
from neurotrace.training.enhanced_sae import EnhancedSAE


@dataclass
class DiagnosticResult:
    """Single diagnostic test result."""
    test_name: str
    status: str  # "PASS", "FAIL", "WARNING", "UNKNOWN"
    score: float  # 0.0-1.0 quality metric
    details: Dict
    critical_issues: List[str]
    recommendations: List[str]


@dataclass
class SystemDiagnostic:
    """Complete system diagnostic report."""
    timestamp: str
    overall_status: str
    tests: List[DiagnosticResult]
    summary: Dict
    action_items: List[str]


class NeuroTraceDiagnostic:
    """Comprehensive system diagnostic."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results: List[DiagnosticResult] = []

        print(f"[DIAGNOSTIC] Initializing on {self.device}")
        print("=" * 80)

    def run_all_diagnostics(self) -> SystemDiagnostic:
        """Run complete diagnostic suite."""
        print("\n[PHASE 1] Network Interception Fidelity")
        print("-" * 80)
        self.test_activation_capture_fidelity()

        print("\n[PHASE 2] SAE Reconstruction Quality")
        print("-" * 80)
        self.test_sae_reconstruction_quality()

        print("\n[PHASE 3] Steering Causality")
        print("-" * 80)
        self.test_steering_causality()

        print("\n[PHASE 4] Neural Atlas Functionality")
        print("-" * 80)
        self.test_atlas_navigation()

        print("\n[PHASE 5] Data Analysis & JSON Processing")
        print("-" * 80)
        self.test_systematic_data_analysis()

        # Generate final report
        return self._generate_report()

    # ==================== PHASE 1: Network Interception ====================

    def test_activation_capture_fidelity(self):
        """
        Test: Are we capturing network activations 1:1 without corruption?

        Method:
        1. Run model forward pass and capture activations via hooks
        2. Re-run same input and verify activations are identical
        3. Test across multiple layers and positions
        4. Measure numerical stability
        """
        print("[TEST 1.1] Activation Capture Consistency")

        issues = []
        recommendations = []

        try:
            # Initialize model
            cfg = NeuroTraceConfig(
                model_name_or_path="gpt2",
                device=str(self.device),
                precision="fp32"
            )
            model_wrapper = TargetModelWrapper(cfg=cfg)
            model = model_wrapper.model
            tokenizer = model_wrapper.tokenizer

            # Test prompt
            test_prompt = "When John and Mary went to the store, John gave a book to"
            inputs = tokenizer(test_prompt, return_tensors="pt").to(self.device)

            # Capture activations twice
            activations_run1 = {}
            activations_run2 = {}

            def capture_hook(name, storage):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        storage[name] = output[0].detach().cpu().clone()
                    else:
                        storage[name] = output.detach().cpu().clone()
                return hook

            # Run 1
            hooks = []
            for i in range(12):  # GPT-2 has 12 layers
                layer = model.transformer.h[i]
                hooks.append(layer.mlp.register_forward_hook(capture_hook(f"layer_{i}_mlp", activations_run1)))

            with torch.no_grad():
                _ = model(**inputs)

            for h in hooks:
                h.remove()

            # Run 2 (same input)
            hooks = []
            for i in range(12):
                layer = model.transformer.h[i]
                hooks.append(layer.mlp.register_forward_hook(capture_hook(f"layer_{i}_mlp", activations_run2)))

            with torch.no_grad():
                _ = model(**inputs)

            for h in hooks:
                h.remove()

            # Compare activations
            max_diff = 0.0
            layer_diffs = {}

            for layer_name in activations_run1.keys():
                act1 = activations_run1[layer_name]
                act2 = activations_run2[layer_name]

                diff = torch.abs(act1 - act2).max().item()
                layer_diffs[layer_name] = diff
                max_diff = max(max_diff, diff)

            # Evaluate
            if max_diff < 1e-6:
                status = "PASS"
                score = 1.0
            elif max_diff < 1e-4:
                status = "WARNING"
                score = 0.8
                recommendations.append("Small numerical differences detected - consider fp64 for critical ops")
            else:
                status = "FAIL"
                score = 0.0
                issues.append(f"Large activation differences detected (max={max_diff:.2e})")
                issues.append("Network capture is NOT deterministic - investigate randomness sources")

            result = DiagnosticResult(
                test_name="Activation Capture Fidelity",
                status=status,
                score=score,
                details={
                    "max_difference": max_diff,
                    "layer_differences": layer_diffs,
                    "num_layers_tested": len(layer_diffs),
                },
                critical_issues=issues,
                recommendations=recommendations,
            )

            print(f"  Status: {status}")
            print(f"  Max Difference: {max_diff:.2e}")
            print(f"  Score: {score:.2%}")

            self.results.append(result)

        except Exception as e:
            self.results.append(DiagnosticResult(
                test_name="Activation Capture Fidelity",
                status="FAIL",
                score=0.0,
                details={"error": str(e)},
                critical_issues=[f"Test crashed: {str(e)}"],
                recommendations=["Fix implementation errors before proceeding"],
            ))
            print(f"  Status: FAIL (crashed: {e})")

    # ==================== PHASE 2: SAE Reconstruction ====================

    def test_sae_reconstruction_quality(self):
        """
        Test: How well do our SAEs reconstruct activations?

        Method:
        1. Load trained Layer 0 SAE
        2. Capture real Layer 0 activations
        3. Encode → Decode through SAE
        4. Measure reconstruction MSE
        5. Check if MSE meets publication standards (<0.05)
        """
        print("[TEST 2.1] SAE Reconstruction Quality (Layer 0)")

        issues = []
        recommendations = []

        try:
            # Load SAE
            checkpoint_path = Path("checkpoints/layer0_sae/final.pt")
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"SAE checkpoint not found: {checkpoint_path}")

            store = EnhancedSAEFeatureStore()
            store.load_sae(str(checkpoint_path), layer=0, device=self.device)
            sae = store.saes[0]

            # Load activations
            activations_dir = Path("runs/deep_ioi_capture")
            latest_run = max([d for d in activations_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
            act_dir = latest_run / "activations"

            if not act_dir.exists():
                raise FileNotFoundError(f"Activations not found: {act_dir}")

            # Test on first 10 batches
            mse_values = []
            l0_sparsity_values = []

            batch_files = sorted(act_dir.glob("batch_*.pt"))[:10]
            for batch_file in batch_files:
                batch_data = torch.load(batch_file, map_location=self.device)
                # Extract layer_0.mlp activations from dict
                activations = batch_data["layer_0.mlp"]

                # Flatten activations [batch, seq, hidden] -> [batch*seq, hidden]
                if activations.dim() == 3:
                    activations = activations.reshape(-1, activations.size(-1))

                # Forward pass (encode + decode)
                with torch.no_grad():
                    output = sae.forward(activations)
                    codes = output['codes']  # [N, dict_size]
                    reconstructed = output['reconstruction']  # [N, hidden_dim]

                # Measure MSE
                mse = torch.mean((activations - reconstructed) ** 2).item()
                mse_values.append(mse)

                # Measure sparsity (L0 norm)
                l0 = (codes != 0).float().sum(dim=1).mean().item()
                l0_sparsity_values.append(l0)

            avg_mse = np.mean(mse_values)
            std_mse = np.std(mse_values)
            avg_l0 = np.mean(l0_sparsity_values)

            # Evaluate
            if avg_mse < 0.05:
                status = "PASS"
                score = 1.0
            elif avg_mse < 0.12:
                status = "WARNING"
                score = 0.7
                recommendations.append("MSE is acceptable but not SOTA quality")
            else:
                status = "FAIL"
                score = 0.3
                issues.append(f"High reconstruction MSE ({avg_mse:.4f}) - SAE quality insufficient")
                recommendations.append("Re-train SAE with more data or better hyperparameters")

            result = DiagnosticResult(
                test_name="SAE Reconstruction Quality",
                status=status,
                score=score,
                details={
                    "avg_mse": avg_mse,
                    "std_mse": std_mse,
                    "avg_l0_sparsity": avg_l0,
                    "batches_tested": len(mse_values),
                    "target_mse": 0.05,
                },
                critical_issues=issues,
                recommendations=recommendations,
            )

            print(f"  Status: {status}")
            print(f"  Avg MSE: {avg_mse:.4f} (target: <0.05)")
            print(f"  Avg L0: {avg_l0:.1f}")
            print(f"  Score: {score:.2%}")

            self.results.append(result)

        except Exception as e:
            self.results.append(DiagnosticResult(
                test_name="SAE Reconstruction Quality",
                status="FAIL",
                score=0.0,
                details={"error": str(e)},
                critical_issues=[f"Test crashed: {str(e)}"],
                recommendations=["Check SAE checkpoint and activation paths"],
            ))
            print(f"  Status: FAIL (crashed: {e})")

    # ==================== PHASE 3: Steering Causality ====================

    def test_steering_causality(self):
        """
        Test: Does steering actually change model outputs?

        Method:
        1. Generate baseline outputs (no steering)
        2. Apply steering with alpha=1.0
        3. Generate steered outputs
        4. Measure difference in outputs (text, logits, activations)
        5. Verify effect is significant and consistent
        """
        print("[TEST 3.1] Steering Causality & Effect Size")

        issues = []
        recommendations = []

        try:
            # Load SAE and setup control plane
            checkpoint_path = Path("checkpoints/layer0_sae/final.pt")
            store = EnhancedSAEFeatureStore()
            store.load_sae(str(checkpoint_path), layer=0, device=self.device)

            cfg = NeuroTraceConfig(
                model_name_or_path="gpt2",
                device=str(self.device),
                precision="fp32"
            )
            model_wrapper = TargetModelWrapper(cfg=cfg)

            # Create simple circuit (top IOI features)
            from neurotrace.control import CircuitRecord, CircuitComponent, CircuitCausalMetrics, CircuitSemantics, CircuitFeatures

            circuit = CircuitRecord(
                circuit_id="diagnostic_test_circuit",
                model_name="gpt2",
                components=[
                    CircuitComponent(layer=0, component_type="sae_direction", index=2586, extra={}),
                    CircuitComponent(layer=0, component_type="sae_direction", index=2081, extra={}),
                ],
                features=CircuitFeatures(sae_indices={"layer_0": [2586, 2081]}),
                causal_metrics=CircuitCausalMetrics(vlo_mean=5.0, vlo_std=0.1, faithfulness=0.7),
                semantics=CircuitSemantics(task_tag="IOI", human_label="Test Circuit", description="Diagnostic test"),
            )

            registry = CircuitRegistry(":memory:")
            registry.upsert(circuit)

            builder = SteeringBuilder(feature_store=store)
            controller = CircuitController(model_wrapper=model_wrapper, registry=registry, steering_builder=builder)

            # Test prompts
            test_prompts = [
                "When John and Mary went to the store, John gave a book to",
                "Alice and Bob were talking. Alice said to",
                "The cat and the dog fought. The cat bit",
            ]

            baseline_outputs = []
            steered_outputs = []
            effect_detected = []

            for prompt in test_prompts:
                # Baseline
                baseline_text = controller.generate(prompt, max_new_tokens=10)
                baseline_outputs.append(baseline_text)

                # Steered
                controller.enable_circuit("diagnostic_test_circuit", global_alpha=1.0)
                steered_text = controller.generate(prompt, max_new_tokens=10)
                controller.disable_circuit("diagnostic_test_circuit")
                steered_outputs.append(steered_text)

                # Compare
                different = baseline_text != steered_text
                effect_detected.append(different)

            # Evaluate
            effect_rate = sum(effect_detected) / len(effect_detected)

            if effect_rate >= 0.8:
                status = "PASS"
                score = 1.0
            elif effect_rate >= 0.5:
                status = "WARNING"
                score = 0.6
                recommendations.append("Steering effect is weak - consider higher alpha or stronger features")
            else:
                status = "FAIL"
                score = 0.2
                issues.append("Steering has minimal effect on outputs")
                issues.append("Check if steering vectors are actually being applied")

            result = DiagnosticResult(
                test_name="Steering Causality",
                status=status,
                score=score,
                details={
                    "effect_rate": effect_rate,
                    "num_prompts": len(test_prompts),
                    "baseline_outputs": baseline_outputs,
                    "steered_outputs": steered_outputs,
                },
                critical_issues=issues,
                recommendations=recommendations,
            )

            print(f"  Status: {status}")
            print(f"  Effect Rate: {effect_rate:.1%}")
            print(f"  Score: {score:.2%}")

            self.results.append(result)

        except Exception as e:
            self.results.append(DiagnosticResult(
                test_name="Steering Causality",
                status="FAIL",
                score=0.0,
                details={"error": str(e)},
                critical_issues=[f"Test crashed: {str(e)}"],
                recommendations=["Fix control plane implementation"],
            ))
            print(f"  Status: FAIL (crashed: {e})")

    # ==================== PHASE 4: Neural Atlas ====================

    def test_atlas_navigation(self):
        """
        Test: Can we use the Neural Atlas to explore the network?

        Questions:
        1. How many layers have trained SAEs?
        2. Can we load and query features?
        3. Can we search for similar features?
        4. Is the atlas actually useful?
        """
        print("[TEST 4.1] Neural Atlas Functionality")

        issues = []
        recommendations = []

        # Check what SAEs exist
        checkpoints_dir = Path("checkpoints")
        available_saes = []

        for layer_dir in checkpoints_dir.glob("layer*_sae"):
            if (layer_dir / "final.pt").exists():
                layer_num = int(layer_dir.name.split("_")[0].replace("layer", ""))
                available_saes.append(layer_num)

        available_saes.sort()

        # Evaluate
        if len(available_saes) == 0:
            status = "FAIL"
            score = 0.0
            issues.append("No SAE checkpoints found - Atlas is empty")
            recommendations.append("Train SAEs for all layers using train_all_layers_sae.py")
        elif len(available_saes) < 12:
            status = "WARNING"
            score = len(available_saes) / 12.0
            issues.append(f"Only {len(available_saes)}/12 layers have SAEs")
            recommendations.append(f"Train SAEs for missing layers: {set(range(12)) - set(available_saes)}")
        else:
            status = "PASS"
            score = 1.0

        result = DiagnosticResult(
            test_name="Neural Atlas Completeness",
            status=status,
            score=score,
            details={
                "available_layers": available_saes,
                "missing_layers": list(set(range(12)) - set(available_saes)),
                "coverage": f"{len(available_saes)}/12",
            },
            critical_issues=issues,
            recommendations=recommendations,
        )

        print(f"  Status: {status}")
        print(f"  Available SAEs: {available_saes}")
        print(f"  Coverage: {len(available_saes)}/12")
        print(f"  Score: {score:.2%}")

        self.results.append(result)

    # ==================== PHASE 5: Data Analysis ====================

    def test_systematic_data_analysis(self):
        """
        Test: Can we systematically analyze all JSON results?

        Current Problem: We have JSON files scattered everywhere but no systematic analysis.

        Solution: Build a JSON aggregator and analyzer.
        """
        print("[TEST 5.1] Systematic JSON Analysis")

        issues = []
        recommendations = []

        # Find all JSON result files
        json_files = list(Path(".").rglob("*results*.json"))
        json_files += list(Path("results").rglob("*.json"))

        if len(json_files) == 0:
            status = "WARNING"
            score = 0.5
            recommendations.append("No JSON result files found - nothing to analyze")
        else:
            # Try to load and categorize
            loaded = 0
            failed = 0
            categories = {}

            for json_file in json_files:
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    loaded += 1

                    # Categorize by type
                    if "hybrid_analysis" in str(json_file):
                        categories.setdefault("hybrid_analysis", []).append(json_file)
                    elif "discovery" in str(json_file):
                        categories.setdefault("discovery", []).append(json_file)
                    else:
                        categories.setdefault("other", []).append(json_file)

                except Exception:
                    failed += 1

            if failed == 0:
                status = "PASS"
                score = 1.0
            else:
                status = "WARNING"
                score = 0.7
                issues.append(f"{failed}/{len(json_files)} JSON files failed to load")

            result = DiagnosticResult(
                test_name="JSON Data Analysis",
                status=status,
                score=score,
                details={
                    "total_json_files": len(json_files),
                    "loaded_successfully": loaded,
                    "failed_to_load": failed,
                    "categories": {k: len(v) for k, v in categories.items()},
                },
                critical_issues=issues,
                recommendations=["Build systematic JSON aggregator tool"] + recommendations,
            )

            print(f"  Status: {status}")
            print(f"  JSON Files Found: {len(json_files)}")
            print(f"  Successfully Loaded: {loaded}")
            print(f"  Score: {score:.2%}")

            self.results.append(result)

    # ==================== Report Generation ====================

    def _generate_report(self) -> SystemDiagnostic:
        """Generate final diagnostic report."""
        import datetime

        # Calculate overall status
        avg_score = np.mean([r.score for r in self.results])

        if avg_score >= 0.8:
            overall_status = "HEALTHY"
        elif avg_score >= 0.5:
            overall_status = "NEEDS_ATTENTION"
        else:
            overall_status = "CRITICAL"

        # Collect all action items
        action_items = []
        for result in self.results:
            action_items.extend(result.critical_issues)
            action_items.extend(result.recommendations)

        # Generate summary
        summary = {
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r.status == "PASS"),
            "warnings": sum(1 for r in self.results if r.status == "WARNING"),
            "failed": sum(1 for r in self.results if r.status == "FAIL"),
            "average_score": avg_score,
        }

        report = SystemDiagnostic(
            timestamp=datetime.datetime.now().isoformat(),
            overall_status=overall_status,
            tests=self.results,
            summary=summary,
            action_items=list(set(action_items)),  # deduplicate
        )

        return report

    def print_report(self, report: SystemDiagnostic):
        """Print formatted diagnostic report."""
        print("\n")
        print("=" * 80)
        print("NEUROTRACE SYSTEM DIAGNOSTIC REPORT")
        print("=" * 80)
        print(f"Timestamp: {report.timestamp}")
        print(f"Overall Status: {report.overall_status}")
        print(f"Average Score: {report.summary['average_score']:.1%}")
        print()
        print(f"Tests Run: {report.summary['total_tests']}")
        print(f"  PASS: {report.summary['passed']}")
        print(f"  WARNING: {report.summary['warnings']}")
        print(f"  FAIL: {report.summary['failed']}")
        print()
        print("-" * 80)
        print("DETAILED RESULTS")
        print("-" * 80)

        for i, test in enumerate(report.tests, 1):
            print(f"\n{i}. {test.test_name}")
            print(f"   Status: {test.status} | Score: {test.score:.1%}")

            if test.critical_issues:
                print("   ISSUES:")
                for issue in test.critical_issues:
                    print(f"     - {issue}")

            if test.recommendations:
                print("   RECOMMENDATIONS:")
                for rec in test.recommendations:
                    print(f"     - {rec}")

        print("\n" + "-" * 80)
        print("ACTION ITEMS")
        print("-" * 80)
        for i, item in enumerate(report.action_items, 1):
            print(f"{i}. {item}")

        print("\n" + "=" * 80)
        print(f"VERDICT: {report.overall_status}")
        print("=" * 80)

    def save_report(self, report: SystemDiagnostic, output_path: str = "diagnostic_report.json"):
        """Save report to JSON."""
        report_dict = asdict(report)
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        print(f"\n[SAVED] Diagnostic report saved to {output_path}")


if __name__ == "__main__":
    print("NeuroTrace System Diagnostic")
    print("=" * 80)
    print("Running comprehensive system tests...")
    print()

    diagnostic = NeuroTraceDiagnostic(device="cuda")
    report = diagnostic.run_all_diagnostics()

    diagnostic.print_report(report)
    diagnostic.save_report(report, "tests/validation/diagnostic_report.json")

    print("\n[DONE] Diagnostic complete.")
