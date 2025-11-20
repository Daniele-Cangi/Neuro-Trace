import torch
from pathlib import Path

# Load old Layer 0 checkpoint
print("=" * 80)
print("OLD TRAINING (train_layer0_sae.py)")
print("=" * 80)

ckpt_old = torch.load('checkpoints/layer0_sae/final.pt', map_location='cpu', weights_only=False)

print("\nCheckpoint keys:", list(ckpt_old.keys()))
print("\nConfig:", ckpt_old.get('config', {}))

training_state = ckpt_old.get('training_state', {})
print("\nTraining state:")
for k, v in training_state.items():
    print(f"  {k}: {v}")

feature_stats = ckpt_old.get('feature_statistics', {})
print("\nFeature statistics:")
for k, v in feature_stats.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: Tensor{tuple(v.shape)}")
    else:
        print(f"  {k}: {v}")

metrics_history = ckpt_old.get('metrics_history', [])
print(f"\nMetrics history entries: {len(metrics_history)}")
if metrics_history:
    print("First entry:", metrics_history[0])
    print("Last entry:", metrics_history[-1])

# Load new Layer 0 checkpoint
print("\n" + "=" * 80)
print("NEW TRAINING (train_atlas_simple.py)")
print("=" * 80)

ckpt_new = torch.load('checkpoints/all_layers_sae/layer_0/final.pt', map_location='cpu', weights_only=False)

print("\nCheckpoint keys:", list(ckpt_new.keys()))
print("\nConfig:", ckpt_new.get('config', {}))

training_state_new = ckpt_new.get('training_state', {})
print("\nTraining state:")
for k, v in training_state_new.items():
    print(f"  {k}: {v}")

feature_stats_new = ckpt_new.get('feature_statistics', {})
print("\nFeature statistics:")
for k, v in feature_stats_new.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: Tensor{tuple(v.shape)}")
    else:
        print(f"  {k}: {v}")

metrics_history_new = ckpt_new.get('metrics_history', [])
print(f"\nMetrics history entries: {len(metrics_history_new)}")
if metrics_history_new:
    print("First entry:", metrics_history_new[0])
    print("Last entry:", metrics_history_new[-1])

# Compare
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

old_steps = training_state.get('step', 0)
new_steps = training_state_new.get('step', 0)

print(f"\nTotal training steps:")
print(f"  OLD: {old_steps:,}")
print(f"  NEW: {new_steps:,}")
print(f"  Ratio: {old_steps/new_steps if new_steps > 0 else 0:.2f}x")

print(f"\nMetrics history entries:")
print(f"  OLD: {len(metrics_history):,}")
print(f"  NEW: {len(metrics_history_new):,}")
