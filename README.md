# Topic 05 — Investigating Modern Encoder–Decoder Architectures for Image Segmentation

## Mục tiêu

So sánh có hệ thống 3 kiến trúc encoder–decoder hiện đại cho bài toán phân đoạn ảnh (image segmentation), đánh giá toàn diện về accuracy, efficiency và robustness khi thiếu dữ liệu.

## Mô hình

| # | Mô hình | Encoder | Decoder | Kết nối E-D |
|---|---------|---------|---------|-------------|
| 1 | **U-Net** | ResNet-34 (pretrained) | 5× ConvTranspose2d | Skip connections (concat) |
| 2 | **SegNet** | VGG16-BN (pretrained) | 5× MaxUnpool2d | Pooling indices |
| 3 | **DeepLabV3+** | ResNet-34 (pretrained) | ASPP + Bilinear | Low-level fusion |

## Datasets

| Dataset | Domain | Classes | Size |
|---------|--------|---------|------|
| **Oxford-IIIT Pet** | Natural images | 3 (background, pet, boundary) | 7,349 images |
| **Kvasir-SEG** | Medical (endoscopy) | 2 (background, polyp) | 1,000 images |

## Cấu trúc nộp bài

```
02VF_Group_##/
├── report/
│   ├── main.pdf                    ← Báo cáo (PDF)
│   ├── main.tex                    ← Source LaTeX
│   ├── refs.bib                    ← Tài liệu tham khảo
│   ├── spconf.sty                  ← IEEE style
│   ├── IEEEbib.bst                 ← Bib style
│   └── figures/                    ← Hình ảnh trong báo cáo
│
├── Topic05_EncoderDecoder.ipynb    ← Notebook chính (có output)
│
├── README.md                       ← File này
│
└── kaggle_output/extracted/        ← Kết quả thí nghiệm
    ├── config.json
    ├── profiling.csv
    ├── combined_all_results.csv
    ├── combined_summary_mean_std.csv
    ├── cross_dataset_comparison.png
    ├── OxfordPet/
    │   ├── all_results.csv
    │   ├── summary_mean_std.csv
    │   ├── figures/ (12 hình)
    │   └── history/ (training logs)
    └── KvasirSEG/
        └── (tương tự)
```

## Cách chạy

### Trên Kaggle (Khuyến nghị — đã dùng cho thí nghiệm chính)

1. Tạo Kaggle Notebook mới
2. Upload notebook `Topic05_EncoderDecoder.ipynb`
3. Thêm Kvasir-SEG dataset (`debeshjha1/kvasirseg`) vào Input
4. Chọn **GPU T4** accelerator
5. Đặt `QUICK_RUN = False` để chạy full 27 runs/dataset
6. Run All → Kết quả xuất ra `/kaggle/working/outputs_topic05/`

> **Thời gian chạy:** ~7 giờ (full experiments, 25 epochs × 54 runs)

### Trên Local (test pipeline)

```bash
# Cài đặt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install albumentations tqdm pandas matplotlib scipy

# Chạy nhanh (2 epochs)
# Mở notebook, đặt QUICK_RUN = True, chạy tất cả cells
```

## Thiết lập thí nghiệm

| Factor | Values |
|--------|--------|
| **Models** | U-Net (ResNet-34), SegNet (VGG16-BN), DeepLabV3+ (ResNet-34) |
| **Datasets** | OxfordPet, Kvasir-SEG |
| **Seeds** | 42, 2025, 3407 |
| **Data fractions** | 1.0, 0.5, 0.25 |
| **Learning rate** | 3×10⁻⁴ (cố định — fair comparison) |
| **Optimizer** | AdamW (weight_decay=10⁻⁴) |
| **LR Scheduler** | CosineAnnealingLR |
| **Loss** | 0.5 × DiceLoss + 0.5 × CrossEntropyLoss |
| **Input size** | 256×256 |
| **Epochs** | 25 (early stopping patience=7) |
| **Mixed Precision** | AMP enabled |

**Tổng:** 3 models × 2 datasets × 3 seeds × 3 fractions = **54 runs**

## Metrics

- **Accuracy:** Mean IoU, Dice Coefficient, Pixel Accuracy, Per-class IoU
- **Efficiency:** Inference Latency (ms), Peak VRAM (MB), FLOPs (GFLOPs)

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

## Kết quả tóm tắt

| Dataset | Best Model | Mean IoU | Latency |
|---------|-----------|----------|---------|
| OxfordPet | **U-Net** | **0.805** ± 0.002 | 7.4 ms |
| Kvasir-SEG | **U-Net** | **0.883** ± 0.004 | 7.4 ms |

U-Net đạt IoU cao nhất trên cả hai datasets. DeepLabV3+ inference nhanh nhất (6.9 ms).
Chi tiết xem báo cáo `report/main.pdf`.
