"""
PHASE 9A – VIRUS DETECTOR (Layer 10)

Goal:
Train a lightweight MLP detector to distinguish between CLEAN and ATTACKED
states at Layer 10 of GPT-2 Small.

The detector uses scalar features extracted from the residual stream and model outputs
to classify whether the "virus" (adversarial vector) has been injected.

Features per example:
1. Current Logit Diff
2. Projection on Virus Vector
3. Norm of Hidden State
4. Baseline Logit Diff (Context)
5. Susceptibility Delta (Context)

Usage:
    python phase9a_virus_detector.py
"""

import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    from sklearn.metrics import roc_auc_score, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from transformers import AutoModelForCausalLM, AutoTokenizer
from neurotrace.datasets import IOIDatasetGenerator

# ============================================================================
# 1. CONFIG & SETUP
# ============================================================================

CONFIG = {
    "layer": 10,
    "train_size": 2000,
    "test_size": 500,
    "alpha_attack": 1.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "virus_path": "checkpoints/adversarial_delta_layer10.pt",
    "detector_path": "checkpoints/virus_detector_layer10.pt",
    "config_path": "virus_detector_config.json",
    "seed": 42,
    "batch_size": 64,
    "lr": 1e-3,
    "epochs": 20
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

# ============================================================================
# 2. MODEL & DATA UTILS
# ============================================================================

def load_resources():
    print(f"Loading GPT-2 on {CONFIG['device']}...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(CONFIG['device'])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading Virus Vector from {CONFIG['virus_path']}...")
    virus = torch.load(CONFIG['virus_path'], map_location=CONFIG['device'])
    virus = virus.float()
    
    return model, tokenizer, virus

def get_ioi_data(num_train, num_test):
    print(f"Generating IOI dataset ({num_train} train, {num_test} test)...")
    generator = IOIDatasetGenerator()
    examples = generator.generate(num_examples=num_train + num_test, ensure_diversity=True)
    return examples[:num_train], examples[num_train:]

# ============================================================================
# 3. FEATURE EXTRACTION
# ============================================================================

def extract_features_from_batch(model, tokenizer, examples, virus_vector, layer_idx):
    """
    Generates Clean and Attacked features for a batch of examples.
    Returns:
        X: Tensor of shape [2 * batch_size, num_features]
        y: Tensor of shape [2 * batch_size] (0=Clean, 1=Attacked)
    """
    prompts = [ex.text for ex in examples]
    correct_answers = [ex.correct_answer for ex in examples]
    incorrect_answers = [ex.incorrect_answer for ex in examples]
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    last_token_indices = inputs.attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(len(prompts), device=model.device)
    
    # Normalize virus for projection
    virus_norm = virus_vector.norm()
    virus_unit = virus_vector / (virus_norm + 1e-8)
    
    # --- 1. CLEAN PASS ---
    clean_h_cache = {}
    def get_clean_h_hook(module, inp, out):
        clean_h_cache['h'] = out[0].detach().clone() # [batch, seq, dim]
        return out
    
    handle = model.transformer.h[layer_idx].register_forward_hook(get_clean_h_hook)
    with torch.no_grad():
        outputs_clean = model(**inputs)
    handle.remove()
    
    logits_clean = outputs_clean.logits
    h_clean_all = clean_h_cache['h']
    h_clean = h_clean_all[batch_indices, last_token_indices, :] # [batch, dim]
    
    # --- 2. ATTACKED PASS ---
    attacked_h_cache = {}
    def attack_hook(module, inp, out):
        h = out[0]
        # Save pre-attack h (which is same as clean h, but just to be sure we modify in place)
        # Actually we want the attacked state.
        # h' = h + alpha * virus
        perturbation = CONFIG["alpha_attack"] * virus_vector
        h[batch_indices, last_token_indices, :] += perturbation
        attacked_h_cache['h'] = h.detach().clone() # This is h_attacked
        return (h,) + out[1:]
        
    handle = model.transformer.h[layer_idx].register_forward_hook(attack_hook)
    with torch.no_grad():
        outputs_attacked = model(**inputs)
    handle.remove()
    
    logits_attacked = outputs_attacked.logits
    h_attacked_all = attacked_h_cache['h']
    h_attacked = h_attacked_all[batch_indices, last_token_indices, :]
    
    # --- 3. COMPUTE FEATURES ---
    features_list = []
    labels_list = []
    
    for i in range(len(prompts)):
        # Get token IDs
        corr_id = tokenizer.encode(" " + correct_answers[i])[0]
        incorr_id = tokenizer.encode(" " + incorrect_answers[i])[0]
        
        # Logit Diffs
        ld_clean = (logits_clean[i, last_token_indices[i], corr_id] - 
                    logits_clean[i, last_token_indices[i], incorr_id]).item()
        
        ld_attacked = (logits_attacked[i, last_token_indices[i], corr_id] - 
                       logits_attacked[i, last_token_indices[i], incorr_id]).item()
        
        delta_ld = ld_attacked - ld_clean
        
        # Projections & Norms
        # Clean
        hc = h_clean[i]
        norm_clean = hc.norm().item()
        proj_clean = torch.dot(hc, virus_unit).item()
        
        # Attacked
        ha = h_attacked[i]
        norm_attacked = ha.norm().item()
        proj_attacked = torch.dot(ha, virus_unit).item()
        
        # Construct Feature Vectors
        # Feature Order: [logit_diff, proj_virus, norm_h, base_logit_diff, delta_logit_diff]
        
        # Sample 0: CLEAN
        f_clean = [
            ld_clean,       # Current Logit Diff
            proj_clean,     # Current Proj
            norm_clean,     # Current Norm
            ld_clean,       # Context: Base Logit Diff
            delta_ld        # Context: Susceptibility
        ]
        features_list.append(f_clean)
        labels_list.append(0.0)
        
        # Sample 1: ATTACKED
        f_attacked = [
            ld_attacked,    # Current Logit Diff
            proj_attacked,  # Current Proj
            norm_attacked,  # Current Norm
            ld_clean,       # Context: Base Logit Diff
            delta_ld        # Context: Susceptibility
        ]
        features_list.append(f_attacked)
        labels_list.append(1.0)
        
    return torch.tensor(features_list, dtype=torch.float32), torch.tensor(labels_list, dtype=torch.float32)

def create_dataset(model, tokenizer, examples, virus_vector, layer_idx, batch_size=32):
    all_X = []
    all_y = []
    
    print(f"Extracting features for {len(examples)} examples...")
    for i in range(0, len(examples), batch_size):
        batch_ex = examples[i:i+batch_size]
        X, y = extract_features_from_batch(model, tokenizer, batch_ex, virus_vector, layer_idx)
        all_X.append(X)
        all_y.append(y)
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {i}/{len(examples)}...")
            
    return torch.cat(all_X), torch.cat(all_y)

# ============================================================================
# 4. DETECTOR MODEL
# ============================================================================

class VirusDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    # 1. Load Resources
    model, tokenizer, virus = load_resources()
    train_ex, test_ex = get_ioi_data(CONFIG["train_size"], CONFIG["test_size"])
    
    # 2. Create Datasets
    print("\n--- Creating Training Data ---")
    X_train, y_train = create_dataset(model, tokenizer, train_ex, virus, CONFIG["layer"])
    print(f"Train Data: {X_train.shape}, {y_train.shape}")
    
    print("\n--- Creating Test Data ---")
    X_test, y_test = create_dataset(model, tokenizer, test_ex, virus, CONFIG["layer"])
    print(f"Test Data: {X_test.shape}, {y_test.shape}")
    
    # Move to device for training
    X_train, y_train = X_train.to(CONFIG["device"]), y_train.to(CONFIG["device"])
    X_test, y_test = X_test.to(CONFIG["device"]), y_test.to(CONFIG["device"])
    
    # 3. Train Detector
    print("\n--- Training Detector ---")
    detector = VirusDetector(input_dim=X_train.shape[1]).to(CONFIG["device"])
    optimizer = optim.Adam(detector.parameters(), lr=CONFIG["lr"])
    criterion = nn.BCEWithLogitsLoss()
    
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    
    for epoch in range(CONFIG["epochs"]):
        detector.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            logits = detector(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {epoch_loss / len(train_dl):.4f}")
            
    # 4. Evaluation
    print("\n--- Evaluation ---")
    detector.eval()
    with torch.no_grad():
        test_logits = detector(X_test)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        
        preds = (test_probs > 0.5).astype(int)
        acc = accuracy_score(y_test_np, preds)
        
        if SKLEARN_AVAILABLE:
            auc = roc_auc_score(y_test_np, test_probs)
        else:
            auc = 0.0
            print("sklearn not found, skipping AUC.")
            
    print(f"Virus Detector – Test Acc: {acc:.2%}, Test AUC: {auc:.4f}")
    
    # 5. Save
    print("\n--- Saving ---")
    torch.save(detector.state_dict(), CONFIG["detector_path"])
    print(f"Saved weights to {CONFIG['detector_path']}")
    
    config_data = {
        "input_features": [
            "current_logit_diff",
            "current_proj_virus",
            "current_norm",
            "base_logit_diff",
            "susceptibility_delta"
        ],
        "input_dim": X_train.shape[1],
        "hidden_dim": 16,
        "metrics": {
            "test_acc": acc,
            "test_auc": auc
        }
    }
    
    with open(CONFIG["config_path"], "w") as f:
        json.dump(config_data, f, indent=4)
    print(f"Saved config to {CONFIG['config_path']}")

if __name__ == "__main__":
    main()
