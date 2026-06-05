# Đề Xuất Nội Dung Báo Cáo Cuối Kỳ — Topic 05

## Phân Tích Yêu Cầu

### Từ [02VF-Final-projects.pdf](file:///d:/2026/DL/topic5/02VF-Final-projects.pdf)

> [!IMPORTANT]
> **Deadline:** 23:59 PM Monday, June 6, 2026 (Week 20)
> **Điểm:** 5.0 marks (50% điểm thi cuối kỳ)
> **Format:** Báo cáo 2-4 trang (Word/PDF), theo template IEEE 2 cột

#### Yêu cầu chung cho Research Topics (trang 5 PDF):

| # | Yêu cầu | Chúng ta đã làm? |
|---|---------|-------------------|
| 1 | Implement ít nhất 02 DL models; visualize kiến trúc | ✅ 3 models (U-Net, SegNet, DeepLabV3+) |
| 2 | Clearly define hypotheses & research questions | ✅ 5 hypotheses (H1–H5) |
| 3 | Fair comparison (same splits, metrics, training conditions) | ✅ Cùng augmentation, optimizer, loss, seeds |
| 4 | Vary key factors (data amount, hyperparams) | ✅ 3 fractions × 2 LRs |
| 5 | Multiple runs for reliability | ✅ 3 seeds (42, 2025, 3407) → Mean ± Std |
| 6 | Appropriate evaluation metrics | ✅ IoU, Dice, Pixel Acc, Boundary F1 |
| 7 | Inference latency & memory footprint | ✅ Profiling: ms, MB, GFLOPs |
| 8 | Visualizations (dataset, curves, performance, outputs) | ✅ 12 loại biểu đồ |
| 9 | Interpret results, validate/refuse hypotheses | ✅ Critical Discussion section |
| 10 | Discuss trade-offs, limitations, future work | ✅ Trade-off plots + Section 11 notebook |

#### Yêu cầu riêng Topic 5 (trang 3 PDF):

| Yêu cầu | Đáp ứng |
|----------|---------|
| Select ≥ 3 encoder–decoder models | ✅ U-Net(ResNet34), SegNet(VGG16-BN), DeepLabV3+(ResNet34) |
| Train & evaluate on benchmark datasets (different domains) | ✅ OxfordPet (natural) + KvasirSEG (medical) |
| Analyze using segmentation metrics (IoU, Dice) | ✅ IoU, Dice, Pixel Acc, Per-class IoU, Boundary F1 |

### Từ [Template_final_project_report.docx](file:///d:/2026/DL/topic5/Report-templates-extracted/Template_final_project_report.docx)

Cấu trúc bắt buộc:

```
1. Introduction (≥3 paragraphs)
2. Method  
3. Experimental Results and Analysis
4. Conclusion
5. References
```

### Từ [Writing_guidelines_final_project_report.docx](file:///d:/2026/DL/topic5/Report-templates-extracted/Writing_guidelines_final_project_report.docx)

- Font: Times New Roman, 2 cột, 7×9 inches print area
- Abstract: 100–150 words
- Figures: đặt đầu cột, có caption và số thứ tự
- References: IEEE format [1], [2]...

---

## Đề Cương Chi Tiết (2-4 trang, IEEE 2-column)

---

### TITLE
**Investigating Modern Encoder–Decoder Architectures for Image Segmentation**

### Authors
*Tên thành viên nhóm, Trường/Khoa*

### Abstract (~120 words)

> Gợi ý nội dung:
>
> Image segmentation là bài toán quan trọng trong computer vision. Bài báo này khảo sát có hệ thống ba kiến trúc encoder–decoder hiện đại — **U-Net (ResNet-34)**, **SegNet (VGG16-BN)**, và **DeepLabV3+ (ResNet-34)** — trên hai benchmark datasets thuộc hai miền dữ liệu khác nhau: **OxfordPet** (3-class, natural images) và **Kvasir-SEG** (binary, medical endoscopy). Chúng tôi thực hiện 27 thí nghiệm có kiểm soát (3 models × 2 datasets × 3 seeds × 3 data fractions × 1 learning rate) để đánh giá toàn diện về accuracy (Mean IoU, Dice), efficiency (latency, memory, FLOPs), và robustness khi giảm dữ liệu. Kết quả cho thấy U-Net đạt IoU cao nhất (0.805 trên OxfordPet và 0.883 trên Kvasir-SEG), vượt trội nhờ skip connections. DeepLabV3+ bám sát (0.799 và 0.877) đồng thời xử lý inference nhanh nhất (6.85ms). SegNet tốn tài nguyên nhất (42.5 GFLOPs) và có hiệu suất thấp nhất.

### Index Terms
*Encoder–decoder, image segmentation, U-Net, SegNet, DeepLabV3+, transfer learning*

---

### 1. INTRODUCTION (~0.5 trang)

**Paragraph 1 — Problem definition:**
- Image segmentation là gì: gán label cho từng pixel
- Input: ảnh RGB, Output: segmentation mask
- Ứng dụng: y tế (polyp detection), autonomous driving, agriculture

**Paragraph 2 — Significance:**
- Tại sao encoder–decoder quan trọng: encoder trích xuất features, decoder khôi phục resolution
- Thách thức: mỗi kiến trúc có cách xử lý skip connection / upsampling khác nhau → ảnh hưởng accuracy và efficiency
- Cần đánh giá so sánh công bằng (fair comparison) trên nhiều dataset

**Paragraph 3 — What we did:**
- Chọn 3 kiến trúc đại diện: U-Net (dense skip), SegNet (pooling indices), DeepLabV3+ (ASPP)
- Tất cả dùng pretrained encoder (ImageNet)
- Đánh giá trên OxfordPet (natural, 3-class) + Kvasir-SEG (medical, binary)
- 27 runs có kiểm soát (3 seeds, 3 fractions, 1 LR = 3e-4)
- Research Questions & Hypotheses:

| | Research Question | Hypothesis |
|---|---|---|
| RQ1 | Model nào đạt IoU cao nhất trên mỗi domain? | U-Net tốt cho boundary, DeepLabV3+ cho multi-scale |
| RQ2 | Trade-off accuracy vs efficiency? | SegNet chậm & tốn RAM nhất do FLOPs cao |
| RQ3 | Pretrained encoder giúp gì khi ít data? | Degradation ít hơn khi giảm data fraction |

**Paragraph 4 — Report structure:**
- Mục 2 trình bày phương pháp, Mục 3 kết quả và phân tích, Mục 4 kết luận

---

### 2. METHOD (~0.7 trang)

**2.1 Model Architectures**

> [!TIP]
> Dùng 1 figure (Fig. 1) — sơ đồ kiến trúc 3 models side-by-side.
> Lấy từ `figures/` hoặc vẽ bằng draw.io.

| Model | Encoder | Decoder | Đặc trưng |
|-------|---------|---------|-----------|
| U-Net | ResNet-34 (pretrained) | 5-level transpose conv | Skip connections (concat) |
| SegNet | VGG16-BN (pretrained) | MaxUnpool (pooling indices) | Không dùng feature từ encoder |
| DeepLabV3+ | ResNet-34 (pretrained) | ASPP (5 branches) + low-level fusion | Multi-scale context |

Mô tả ngắn gọn từng model (~3 câu/model):
- **U-Net:** Encoder ResNet-34 đã bỏ FC layers. Decoder dùng ConvTranspose2d, concat features từ encoder qua skip connections ở 5 levels. Đây là kiến trúc giữ lại thông tin spatial tốt nhất.
- **SegNet:** Encoder VGG16-BN. Decoder dùng MaxUnpool2d với pooling indices từ encoder (KHÔNG dùng feature maps). Tiết kiệm bộ nhớ truyền nhưng mất thông tin texture.
- **DeepLabV3+:** ASPP module với 5 branches (1×1, dilated 6/12/18, global average pooling). Decoder kết hợp low-level features từ Stage 1 của ResNet. Xử lý tốt objects ở nhiều tỷ lệ.

**2.2 Loss Function & Optimization**
- Hybrid Loss = α × Dice Loss + (1 − α) × CrossEntropy (α = 0.5)
- Optimizer: AdamW (weight_decay = 1e-4)
- Scheduler: CosineAnnealingLR
- AMP (Mixed Precision Training) cho tốc độ
- Early Stopping (patience = 7, monitor val IoU)

**2.3 Data Augmentation**
- Albumentations: HorizontalFlip, VerticalFlip, RandomRotate90, RandomBrightnessContrast, CoarseDropout, ElasticTransform, ShiftScaleRotate
- Resize 256×256, Normalize (ImageNet mean/std)

---

### 3. EXPERIMENTAL RESULTS AND ANALYSIS (~1.5 trang)

**3.1 Datasets**

| Dataset | Domain | Classes | Train/Val/Test | Đặc trưng |
|---------|--------|---------|----------------|-----------|
| OxfordPet [1] | Natural | 3 (bg, pet, boundary) | 70%/15%/15% | Boundary class khó |
| Kvasir-SEG [2] | Medical | 2 (bg, polyp) | 70%/15%/15% | 1000 images, polyp size varies |

> Kèm **Fig. 2**: Dataset samples visualization (từ `dataset_samples.png`)

**3.2 Experimental Setup**
- Hardware: Kaggle T4 GPU × 2 (16GB VRAM)
- Epochs: 25 (full training), EarlyStopping patience=7
- Controlled factors: 3 seeds × 3 data fractions (1.0, 0.5, 0.25) × 1 LR (3e-4) = **27 runs total**

**3.3 Main Results (RQ1)**

> Kèm **Table I**: Overall performance comparison (Mean ± Std across 3 seeds)

*Gợi ý bảng (điền số liệu thật từ full run):*

| Dataset | Model | Mean IoU ↑ | Mean Dice ↑ | Pixel Acc ↑ |
|---------|-------|-----------|------------|------------|
| OxfordPet | U-Net+ResNet34 | **0.805** ± 0.002 | **0.883** ± 0.001 | **0.933** ± 0.000 |
| OxfordPet | SegNet+VGG16 | 0.793 ± 0.001 | 0.875 ± 0.000 | 0.928 ± 0.000 |
| OxfordPet | DeepLabV3++ResNet34 | 0.799 ± 0.001 | 0.879 ± 0.001 | 0.930 ± 0.000 |
| KvasirSEG | U-Net+ResNet34 | **0.883** ± 0.004 | **0.935** ± 0.002 | **0.966** ± 0.002 |
| KvasirSEG | SegNet+VGG16 | 0.874 ± 0.007 | 0.930 ± 0.004 | 0.963 ± 0.002 |
| KvasirSEG | DeepLabV3++ResNet34 | 0.877 ± 0.003 | 0.932 ± 0.002 | 0.964 ± 0.001 |

*→ Phân tích: U-Net xuất sắc dẫn đầu trên cả 2 dataset (vượt qua giả thuyết ban đầu). Khả năng giữ lại thông tin không gian qua skip connections của U-Net đặc biệt hiệu quả cho cả boundary phức tạp (OxfordPet) và các polyp nhỏ (KvasirSEG). DeepLabV3+ bám sát nút nhưng không vượt được U-Net. SegNet tụt hậu.*

> Kèm **Fig. 3**: Model comparison bar chart (từ `model_comparison.png`)

> Kèm **Fig. 4**: Prediction visualization + Error Maps (từ `predictions_*.png`) — chọn 1 hình đẹp nhất

**3.4 Efficiency Trade-offs (RQ2)**

> Kèm **Table II**: Profiling results

| Model | Params (M) | FLOPs (G) | Latency (ms) | Peak VRAM (MB) |
|-------|-----------|-----------|-------------|----------------|
| U-Net+ResNet34 | 24.5 | **10.4** | 7.4 (6.5) | 331 |
| SegNet+VGG16 | 29.5 | 42.5 | 17.4 | **519** |
| DeepLabV3++ResNet34 | 26.7 | 10.4 | **6.9** | 333 |

*→ Phân tích: SegNet tiêu tốn 4× FLOPs so với U-Net/DeepLabV3+, chậm 2.5×. Nguyên nhân: VGG16 encoder không có residual shortcuts + decoder phải thực hiện full convolution reconstruction.*

> Kèm **Fig. 5**: Accuracy-Latency Trade-off scatter plot (từ `accuracy_latency.png`)

**3.5 Data Scarcity Analysis (RQ3)**

*→ Kết quả: IoU degradation khi giảm data từ 100% → 25%. Phân tích slope.*

> Kèm **Fig. 6**: Data fraction degradation line chart (từ `data_fraction_degradation.png` hoặc `complexity_analysis.png`)

**3.6 Training Dynamics**

> Kèm **Fig. 7**: Training curves (loss, IoU, Dice) — chọn 1 dataset (từ `training_curves.png`)

---

### 4. CONCLUSION (~0.3 trang)

**What we did:**
- So sánh có hệ thống 3 kiến trúc encoder–decoder trên 2 datasets (27 runs)
- Đánh giá toàn diện: accuracy, efficiency, robustness

**Key findings:**
- Không có model tối ưu tuyệt đối (validate H5)
- U-Net xuất sắc nhất trên cả hai dataset, chứng minh sức mạnh của kiến trúc có skip connections mật độ cao.
- DeepLabV3+ tốt nhì và là mô hình inference nhanh nhất (phù hợp real-time applications).
- SegNet kém hiệu quả nhất cả về accuracy và efficiency

**Limitations:**
- IMG_SIZE=256 nhỏ hơn thực tế (512–1024)
- Chưa áp dụng Test-Time Augmentation
- DeepLabV3+ dùng output stride /32 thay vì /16 như paper gốc

**Future work:**
- Encoder nhẹ hơn (EfficientNet, MobileNet) cho mobile
- Boundary Loss, Tversky Loss
- ONNX export + INT8 quantization

---

### 5. REFERENCES

```
[1] O. M. Parkhi, A. Vedaldi, A. Zisserman, "Cats and Dogs," 
    IEEE CVPR, 2012.
[2] D. Jha et al., "Kvasir-SEG: A Segmented Polyp Dataset," 
    MMM, 2020.
[3] O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional 
    Networks for Biomedical Image Segmentation," MICCAI, 2015.
[4] V. Badrinarayanan, A. Kendall, R. Cipolla, "SegNet: A Deep 
    Convolutional Encoder-Decoder Architecture for Image 
    Segmentation," IEEE TPAMI, 2017.
[5] L.-C. Chen et al., "Encoder-Decoder with Atrous Separable 
    Convolution for Semantic Image Segmentation," ECCV, 2018.
[6] K. He et al., "Deep Residual Learning for Image Recognition," 
    IEEE CVPR, 2016.
[7] K. Simonyan and A. Zisserman, "Very Deep Convolutional 
    Networks for Large-Scale Image Recognition," ICLR, 2015.
```

---

## Mapping Figures từ Notebook Output

| Figure # | Filename trong output | Mục sử dụng |
|----------|----------------------|-------------|
| Fig. 1 | *(Tự vẽ hoặc tạo diagram kiến trúc)* | 2. Method |
| Fig. 2 | `dataset_samples.png` | 3.1 Datasets |
| Fig. 3 | `model_comparison.png` | 3.3 Main Results |
| Fig. 4 | `predictions_unet_resnet34.png` (crop) | 3.3 Main Results |
| Fig. 5 | `accuracy_latency.png` | 3.4 Efficiency |
| Fig. 6 | `data_fraction_degradation.png` / `complexity_analysis.png` | 3.5 Data Scarcity |
| Fig. 7 | `training_curves.png` | 3.6 Training |

> [!WARNING]
> Giới hạn 4 trang → chọn tối đa **5-6 figures + 2 tables**. 
> Ưu tiên: Table I (main results), Table II (profiling), Fig. 3, Fig. 4, Fig. 5, Fig. 6.

---

## Checklist Nộp Bài

- [ ] File ZIP đặt tên: `02VF_Group_##.zip`
- [ ] Báo cáo 2-4 trang (PDF/Word) theo template IEEE
- [ ] Jupyter Notebook **có output từng cell** (đã chạy complete trên Kaggle)
- [ ] README.md hướng dẫn chạy
- [ ] Dataset (hoặc link download)
- [ ] Nộp qua Google Drive trước 23:59 ngày 06/06/2026
