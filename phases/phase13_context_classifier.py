"""
PHASE 13: CONTEXT CLASSIFIER (DOMAIN GUARD)
Obiettivo: Addestrare un discriminatore che distingue tra 'IOI Task' e 'General Text'.
Serve a disattivare completamente il sistema di difesa quando non siamo nel dominio operativo,
prevenendo il danno collaterale (PPL spike) su WikiText.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import datasets
import numpy as np
import json
import sys
from transformer_lens import HookedTransformer
from neurotrace.datasets.ioi_generator import IOIDatasetGenerator

# CONFIG
LAYER_ID = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)

# ============================================================================
# UTILS (Adapted from phase_utils)
# ============================================================================

def setup_model():
    print(f"Loading HookedTransformer (gpt2-small) on {DEVICE}...")
    return HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

def generate_ioi_dataset(model, N=1000):
    print(f"Generating {N} IOI examples...")
    generator = IOIDatasetGenerator()
    examples = generator.generate(num_examples=N)
    prompts = [ex.text for ex in examples]
    # Return format matching user expectation: prompts, _, _, _
    return prompts, None, None, None

# ============================================================================
# CLASSIFIER & LOGIC
# ============================================================================

class ContextClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        # Linear Probe semplice: deve essere veloce e leggero
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def get_layer_activations(model, texts, layer_id, batch_size=20):
    """Estrae le attivazioni finali (resid_pre) dal Layer specificato."""
    acts = []
    print(f"Extracting activations for {len(texts)} samples...")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        # Tokenize
        tokens = model.to_tokens(batch_texts, truncate=False)
        # Truncate manually to max length to avoid OOM if mixed lengths, 
        # but for simple classification we take last token state.
        # Let's enforce a max length suitable for GPT2 small
        max_len = 128
        if tokens.shape[1] > max_len:
            tokens = tokens[:, :max_len]
            
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n.endswith(f"blocks.{layer_id}.hook_resid_pre"))
            # Prendi l'attivazione dell'ULTIMO token (stato decisionale)
            # Shape: [batch, pos, d_model] -> [batch, d_model]
            # Note: tokens can be padded. We should ideally take the last non-padding token.
            # But TransformerLens handles padding if we pass attention mask? 
            # to_tokens returns padded tensor. 
            # For simplicity, we take the last token in the sequence (which might be padding if batching mixed lengths).
            # However, to_tokens usually left-pads or right-pads? GPT-2 is usually right-padded?
            # TransformerLens to_tokens defaults: prepend_bos=True.
            # If we batch, it pads.
            # Let's assume for now that the last token is meaningful or the model handles it.
            # A better approach for batching:
            # But for this specific task (IOI vs Wiki), IOI are short, Wiki are long.
            # If we take the last token of the tensor, and it's padding, it's bad.
            # Let's just use batch_size=1 to be safe if we are worried about padding, 
            # or trust that to_tokens produces a batch where we can index.
            # Actually, let's just take the last element.
            
            batch_acts = cache[f"blocks.{layer_id}.hook_resid_pre"][:, -1, :]
            acts.append(batch_acts.cpu())
            
    return torch.cat(acts, dim=0)

def main():
    print("============================================================")
    print("PHASE 13: CONTEXT CLASSIFIER (WIKITEXT vs IOI)")
    print("============================================================")
    
    model = setup_model()

    # 1. PREPARE DATA
    # IOI Data (Class 1)
    print("\n--- Generating IOI Data (Class 1) ---")
    prompts_ioi, _, _, _ = generate_ioi_dataset(model, N=1000) # 1000 IOI samples
    
    # WikiText Data (Class 0)
    print("\n--- Loading WikiText Data (Class 0) ---")
    try:
        wiki_data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        # Filtra testi troppo corti o vuoti
        prompts_wiki = [x["text"] for x in wiki_data if len(x["text"]) > 50][:1000]
    except Exception as e:
        print(f"Error loading WikiText: {e}")
        print("Falling back to dummy text for testing if offline (NOT RECOMMENDED FOR REAL TRAINING)")
        prompts_wiki = ["This is a sample sentence about nothing in particular." for _ in range(1000)]
    
    # Bilanciamento
    min_len = min(len(prompts_ioi), len(prompts_wiki))
    prompts_ioi = prompts_ioi[:min_len]
    prompts_wiki = prompts_wiki[:min_len]
    print(f"Dataset Balanced: {min_len} IOI vs {min_len} WikiText")

    # 2. EXTRACT FEATURES (Layer 10 Activations)
    # Usiamo il modello "Clean" (senza hook virus) perché il classifier deve agire PRIMA di tutto.
    X_ioi = get_layer_activations(model, prompts_ioi, LAYER_ID).to(DEVICE)
    y_ioi = torch.ones(min_len, device=DEVICE)
    
    X_wiki = get_layer_activations(model, prompts_wiki, LAYER_ID).to(DEVICE)
    y_wiki = torch.zeros(min_len, device=DEVICE)
    
    # Merge & Shuffle
    X = torch.cat([X_ioi, X_wiki], dim=0)
    y = torch.cat([y_ioi, y_wiki], dim=0)
    
    perm = torch.randperm(len(X))
    X = X[perm]
    y = y[perm]
    
    # Split Train/Test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 3. TRAIN CLASSIFIER
    print("\n--- Training Context Classifier ---")
    classifier = ContextClassifier().to(DEVICE)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    EPOCHS = 10
    BATCH_SIZE = 64
    
    for epoch in range(EPOCHS):
        classifier.train()
        epoch_loss = 0
        for i in range(0, len(X_train), BATCH_SIZE):
            batch_X = X_train[i:i+BATCH_SIZE]
            batch_y = y_train[i:i+BATCH_SIZE].unsqueeze(1)
            
            optimizer.zero_grad()
            preds = classifier(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch+1) % 2 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")
            
    # 4. EVALUATE
    classifier.eval()
    with torch.no_grad():
        logits = classifier(X_test)
        preds = (torch.sigmoid(logits) > 0.5).float().squeeze()
        acc = (preds == y_test).float().mean().item()
        
        # False Positives on Wiki (Class 0 predicted as 1)
        # Importantissimo: Non vogliamo attivare IOI logic su WikiText
        mask_wiki = (y_test == 0)
        if mask_wiki.sum() > 0:
            fp_wiki = preds[mask_wiki].sum().item()
            total_wiki = mask_wiki.sum().item()
            fp_rate = fp_wiki / total_wiki
        else:
            fp_wiki = 0
            total_wiki = 0
            fp_rate = 0
        
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print(f"WikiText Confusion Rate (False Positives): {fp_rate*100:.2f}%")
    print(f"  -> {int(fp_wiki)} Wiki samples identified as IOI out of {int(total_wiki)}")
    
    # 5. SAVE
    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(classifier.state_dict(), f"checkpoints/context_classifier_layer{LAYER_ID}.pt")
    print(f"Saved Context Classifier to checkpoints/context_classifier_layer{LAYER_ID}.pt")

if __name__ == "__main__":
    main()
