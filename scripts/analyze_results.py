"""Analyze QUICK_RUN results from Kaggle."""
import pandas as pd
import json

df = pd.read_csv("outputs_topic05/combined_all_results.csv")
print("=== QUICK RUN RESULTS (QUICK_RUN=True, 2 epochs, seed=42 only) ===")
print()
print("Rows:", len(df))
print("Datasets:", df["dataset"].unique().tolist())
print("Models:", df["model"].unique().tolist())
print("Seeds:", df["seed"].unique().tolist())
print("Fractions:", df["fraction"].unique().tolist())
print("LRs:", df["lr"].unique().tolist())
print()

# Results table
print("MODEL COMPARISON (OxfordPet, 2 epochs):")
print("-" * 85)
print(f"  {'Model':30s} | {'IoU':>6s} | {'Dice':>6s} | {'PixAcc':>6s} | {'Lat(ms)':>7s} | {'Params':>12s}")
print("-" * 85)
for _, r in df.iterrows():
    print(f"  {r['model']:30s} | {r['test_iou']:.4f} | {r['test_dice']:.4f} | {r['test_pixel_acc']:.4f} | {r['latency_ms']:7.1f} | {int(r['num_params']):>12,}")

# Per-class IoU
print()
print("PER-CLASS IoU:")
print("-" * 70)
print(f"  {'Model':30s} | {'bg':>7s} | {'pet':>7s} | {'boundary':>8s}")
print("-" * 70)
for _, r in df.iterrows():
    pci = json.loads(r["per_class_iou"])
    print(f"  {r['model']:30s} | {pci[0]:.4f}  | {pci[1]:.4f}  | {pci[2]:.4f}")

# Training time
print()
print("TRAINING TIME (2 epochs):")
for _, r in df.iterrows():
    t = r.get("train_time_s", 0)
    print(f"  {r['model']:30s} | {t:.0f}s ({t/60:.1f}min)")

# Profiling
print()
print("PROFILING:")
prof = pd.read_csv("outputs_topic05/profiling.csv")
print(prof.to_string(index=False))

# Check figures
import os
fig_dir = "outputs_topic05/OxfordPet/figures"
print()
print("FIGURES GENERATED:")
for f in sorted(os.listdir(fig_dir)):
    size_kb = os.path.getsize(os.path.join(fig_dir, f)) / 1024
    print(f"  {f:45s} {size_kb:>8.0f} KB")

# Check history
print()
print("TRAINING HISTORY:")
hist_dir = "outputs_topic05/OxfordPet/history"
for f in sorted(os.listdir(hist_dir)):
    hist = pd.read_csv(os.path.join(hist_dir, f))
    print(f"  {f}")
    print(f"    Epochs: {len(hist)}")
    print(f"    Best val_iou: {hist['val_iou'].max():.4f} (epoch {hist['val_iou'].idxmax()+1})")
    print(f"    Final train_loss: {hist['train_loss'].iloc[-1]:.4f}")
    print(f"    Final val_loss: {hist['val_loss'].iloc[-1]:.4f}")
    print()

# Summary
print("=" * 85)
print("DIAGNOSIS:")
print("=" * 85)
config = json.load(open("outputs_topic05/config.json"))
quick_run = config.get("QUICK_RUN", "Unknown")
print(f"  QUICK_RUN = {quick_run}")
print(f"  Epochs trained = {config.get('QUICK_EPOCHS', '?')}")
print(f"  Datasets run = {df['dataset'].unique().tolist()}")

has_kvasir = "KvasirSEG" in df["dataset"].values
print(f"  KvasirSEG included = {has_kvasir}")
if not has_kvasir:
    print("  >> KvasirSEG MISSING - check if dataset was attached on Kaggle")

n_seeds = len(df["seed"].unique())
print(f"  Seeds used = {n_seeds} (QUICK_RUN uses 1)")

print()
if quick_run == "True":
    print("  STATUS: This is a QUICK_RUN test (2 epochs, 1 seed, 1 fraction, 1 LR)")
    print("  NEXT: Set QUICK_RUN=False and run full training for publication-quality results")
else:
    print("  STATUS: Full training run")
