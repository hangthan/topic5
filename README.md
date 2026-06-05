# Topic 05 — Investigating Modern Encoder–Decoder Architectures for Image Segmentation

## Mục tiêu

So sánh hiệu năng 3 kiến trúc encoder–decoder hiện đại cho image segmentation:

1. **U-Net** (với ResNet34 pretrained encoder)
2. **SegNet** (với VGG16-BN pretrained encoder)
3. **DeepLabV3+** (với ResNet34 pretrained encoder + ASPP)

## Datasets

| Dataset | Domain | Classes | Size |
|---------|--------|---------|------|
| **Oxford-IIIT Pet** | Natural images | 3 (background, pet, boundary) | ~7,400 images |
| **Kvasir-SEG** | Medical (endoscopy) | 2 (background, polyp) | 1,000 images |

## Cấu trúc Project

```
topic5/
├── kaggle_topic05_segmentation.py    # Main notebook (Kaggle/Colab)
├── local_analysis.py                  # Local analysis scripts
├── README.md                          # This file
├── data/                              # Datasets (auto-download)
│   ├── oxford-iiit-pet/
│   └── kvasir-seg/
└── outputs/                           # Results
    ├── OxfordPet/
    │   ├── all_results.csv
    │   ├── summary_mean_std.csv
    │   ├── history/
    │   ├── checkpoints/
    │   └── figures/
    ├── KvasirSEG/
    │   └── ...
    ├── combined_all_results.csv
    └── combined_summary_mean_std.csv
```

## Cách chạy

### Trên Kaggle (Khuyến nghị)

1. Upload `kaggle_topic05_segmentation.py` lên Kaggle Notebook
2. Thêm Kvasir-SEG dataset vào notebook
3. Chọn **GPU T4 x2** accelerator
4. Đặt `QUICK_RUN = False` để chạy full experiments
5. Run All

### Trên Google Colab

1. Upload file lên Google Drive
2. Mở bằng Colab, chọn Runtime → T4 GPU
3. Chạy notebook

### Trên Local

1. Cài PyTorch CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install albumentations tqdm pandas matplotlib
   ```
2. Đặt `QUICK_RUN = True` để test pipeline
3. Chạy: `python kaggle_topic05_segmentation.py`

## Yêu cầu thư viện

```
torch>=2.0
torchvision>=0.15
albumentations>=1.3
numpy
pandas
matplotlib
tqdm
Pillow
scipy
```

## Experiments

### Controlled Variables
| Factor | Values |
|--------|--------|
| Seeds | 42, 2025, 3407 |
| Data fractions | 1.0, 0.5, 0.25 |
| Learning rates | 3e-4, 1e-3 |
| Optimizer | AdamW (weight_decay=1e-4) |
| LR Scheduler | CosineAnnealingLR |
| Loss | CrossEntropy + DiceLoss |
| Input size | 256×256 |
| Epochs | 25 (with early stopping patience=7) |

### Metrics
- Pixel Accuracy
- Mean IoU (mIoU)
- Mean Dice Score
- Per-class IoU & Dice
- Inference Latency (ms/image)
- Memory Footprint (MB)
