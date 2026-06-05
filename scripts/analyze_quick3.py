import pandas as pd

df = pd.read_csv('outputs_topic05_3/combined_all_results.csv')
print('Datasets:', df['dataset'].unique().tolist())
print('Models:', df['model'].unique().tolist())

print('-' * 95)
print(f"{'Dataset':15s} | {'Model':30s} | {'IoU':>6s} | {'Dice':>6s} | {'PixAcc':>6s} | {'Lat(ms)':>7s}")
print('-' * 95)
for _, r in df.iterrows():
    print(f"{r['dataset']:15s} | {r['model']:30s} | {r['test_iou']:.4f} | {r['test_dice']:.4f} | {r['test_pixel_acc']:.4f} | {r['latency_ms']:7.1f}")
