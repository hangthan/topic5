import pandas as pd
import json
import os

base = "outputs_PA_A"

# Config
config = json.load(open(f"{base}/config.json"))
print("=" * 90)
print("CONFIG")
print("=" * 90)
for k, v in config.items():
    print(f"  {k:25s}: {v}")

# Combined results
df = pd.read_csv(f"{base}/combined_all_results.csv")
print("\n" + "=" * 90)
print("EXPERIMENT SUMMARY")
print("=" * 90)
print(f"  Total runs:   {len(df)}")
print(f"  Datasets:     {df['dataset'].unique().tolist()}")
print(f"  Models:       {df['model'].unique().tolist()}")
print(f"  Seeds:        {df['seed'].unique().tolist()}")
print(f"  Fractions:    {df['fraction'].unique().tolist()}")
print(f"  LRs:          {df['lr'].unique().tolist()}")

# Main results table
print("\n" + "=" * 90)
print("ALL RESULTS")
print("=" * 90)
cols = ["dataset", "model", "seed", "fraction", "lr", "test_iou", "test_dice", "test_pixel_acc", "latency_ms"]
avail = [c for c in cols if c in df.columns]
print(df[avail].to_string(index=False))

# Summary
summary = pd.read_csv(f"{base}/combined_summary_mean_std.csv")
print("\n" + "=" * 90)
print("SUMMARY (mean +/- std across seeds)")
print("=" * 90)
print(summary.to_string(index=False))

# Per-class IoU
print("\n" + "=" * 90)
print("PER-CLASS IoU (best run per model/dataset)")
print("=" * 90)
best = df.sort_values("test_iou", ascending=False).groupby(["dataset", "model"]).head(1)
for _, r in best.iterrows():
    try:
        pci = json.loads(r["per_class_iou"])
        pcd = json.loads(r["per_class_dice"])
        classes = json.loads(r["class_names"])
        print(f"\n  {r['dataset']} | {r['model']} (IoU={r['test_iou']:.4f}):")
        for cls, iou, dice in zip(classes, pci, pcd):
            print(f"    {cls:12s}: IoU={iou:.4f}  Dice={dice:.4f}")
    except Exception:
        pass

# Profiling
print("\n" + "=" * 90)
print("PROFILING")
print("=" * 90)
prof = pd.read_csv(f"{base}/profiling.csv")
print(prof.to_string(index=False))

# Figures
for ds in ["OxfordPet", "KvasirSEG"]:
    fig_dir = f"{base}/{ds}/figures"
    if os.path.exists(fig_dir):
        print(f"\n  {ds} figures:")
        for f in sorted(os.listdir(fig_dir)):
            sz = os.path.getsize(os.path.join(fig_dir, f)) / 1024
            print(f"    {f:50s} {sz:>7.0f} KB")

# Training time
if "train_time_s" in df.columns:
    print("\n" + "=" * 90)
    print("TRAINING TIME")
    print("=" * 90)
    for _, r in df.iterrows():
        t = r.get("train_time_s", 0)
        print(f"  {r['dataset']:12s} | {r['model']:30s} | seed={r['seed']} | frac={r['fraction']} | {t:.0f}s ({t/60:.1f}min)")

print("\n" + "=" * 90)
print("VERDICT")
print("=" * 90)
quick = config.get("QUICK_RUN", "Unknown")
print(f"  QUICK_RUN = {quick}")
n_datasets = len(df["dataset"].unique())
n_runs = len(df)
print(f"  Datasets: {n_datasets}/2", "OK" if n_datasets == 2 else "MISSING!")
print(f"  Runs: {n_runs}")
expected = 3 * 2 * 1 * 1 * 1 if quick == "True" else 3 * 2 * 3 * 3 * 1
print(f"  Expected (QUICK_RUN={quick}): {expected}")
print(f"  Match: {'YES' if n_runs == expected else 'NO - check!'}")
