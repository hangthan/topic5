# Implementation Plan: Local + Kaggle Workflow

## Mục tiêu
Xây dựng pipeline hoàn chỉnh cho Topic 05 sử dụng kết hợp Local (debug/analysis) + Kaggle (full training), chi phí $0.

## Proposed Changes

---

### Phase 1: Setup Local Environment
#### [MODIFY] PyTorch CUDA Installation
- Gỡ `torch 2.3.0+cpu`, cài `torch+cu121` để kích hoạt RTX 3050

---

### Phase 2: Xây dựng Kaggle Notebook (Main Deliverable)

#### [NEW] [kaggle_topic05_segmentation.py](file:///d:/2026/DL/topic5/kaggle_topic05_segmentation.py)
Notebook chính chạy trên Kaggle, bao gồm:

**Models (cải tiến so với demo):**
- **U-Net**: Tự implement với pretrained ResNet34 encoder + decoder với skip connections
- **SegNet**: Tự implement với pretrained VGG16-BN encoder + max-unpool decoder (đúng bản gốc)
- **DeepLabV3+ Compact**: Tự implement với pretrained ResNet34 encoder + ASPP + decoder

**Training Pipeline:**
- Data augmentation (albumentations): HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast
- LR Scheduler: CosineAnnealingLR
- Mixed Precision (AMP) cho tiết kiệm VRAM + tốc độ
- Early stopping (patience=7)
- IMG_SIZE = 256, BATCH_SIZE = 8

**Datasets:**
- OxfordPet (3-class segmentation) — auto download qua torchvision
- KvasirSEG (binary polyp segmentation) — upload lên Kaggle Dataset

**Experiments:**
- 3 models × 2 datasets × 3 seeds × 25 epochs (core)
- Data fraction experiments: 1.0, 0.5, 0.25
- LR experiments: 1e-3, 3e-4

**Metrics & Visualization:**
- IoU, Dice, Pixel Accuracy, Boundary F1
- Confusion Matrix
- Training curves
- Prediction masks + Error maps
- Complexity-aware analysis
- Accuracy-Latency-Memory trade-off plots
- Architecture parameter summary table

---

### Phase 3: Local Analysis Scripts

#### [NEW] [local_analysis.py](file:///d:/2026/DL/topic5/local_analysis.py)
Script phân tích kết quả download từ Kaggle:
- Load CSV results
- Vẽ biểu đồ so sánh chuyên sâu
- Statistical significance tests
- Export figures cho report

---

### Phase 4: Documentation

#### [NEW] [README.md](file:///d:/2026/DL/topic5/README.md)
Hướng dẫn chạy notebook + cấu trúc project

---

## Kaggle Sessions Plan

| Session | Content | Est. Time |
|---------|---------|-----------|
| 1 | OxfordPet: 3 models × 3 seeds × frac=1.0 × lr=3e-4 × 25ep | ~5h |
| 2 | KvasirSEG: 3 models × 3 seeds × frac=1.0 × lr=3e-4 × 25ep | ~4h |
| 3 | Both: fraction={0.5, 0.25} + lr={1e-3} experiments | ~4h |
| 4 | Visualization + complexity analysis + export | ~2h |

Total: ~15h → Nằm trong quota Kaggle 30h/tuần

## Verification Plan

### Automated Tests
- Verify model output shapes `[B, C, H, W]`
- Verify metrics calculation correctness
- Verify checkpoint save/load
- Run QUICK_RUN=True locally trước khi deploy Kaggle

### Manual Verification
- Check training curves hội tụ
- Check prediction masks chất lượng
- Check kết quả mean ± std có reasonable
