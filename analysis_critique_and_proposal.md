# Phân tích phản biện & Đề xuất tối ưu — Topic 05: Encoder-Decoder Image Segmentation

---

## I. TỔNG QUAN NOTEBOOK HIỆN TẠI

Notebook [(demo)Topic05_EncoderDecoder_Segmentation_Colab_9plus_KvasirReady.ipynb](file:///d:/2026/DL/topic5/(demo)Topic05_EncoderDecoder_Segmentation_Colab_9plus_KvasirReady.ipynb) triển khai 3 mô hình (U-Net, SegNet, DeepLabV3+ Compact) trên 2 dataset (OxfordPet 3-class, KvasirSEG binary), với controlled experiments framework khá đầy đủ.

### Kết quả hiện tại (QUICK_RUN = True, 1 epoch)

| Dataset | Model | Test IoU | Test Dice | Latency (ms) | Params |
|---------|-------|----------|-----------|---------------|--------|
| OxfordPet | U-Net | 0.374 | 0.501 | 102.7 | 1,927K |
| OxfordPet | SegNet | 0.494 | 0.635 | 48.4 | 196K |
| OxfordPet | DeepLabV3+ | 0.547 | 0.683 | 88.1 | 581K |
| KvasirSEG | U-Net | 0.502 | 0.611 | 157.1 | 1,927K |
| KvasirSEG | SegNet | 0.535 | 0.649 | 46.2 | 196K |
| KvasirSEG | DeepLabV3+ | 0.610 | 0.728 | 89.6 | 581K |

> [!CAUTION]
> Kết quả trên **chỉ chạy 1 epoch** (QUICK_RUN=True), **không có ý nghĩa khoa học** để rút ra kết luận. Nếu nộp bài với kết quả này sẽ bị đánh giá thấp.

---

## II. PHẢN BIỆN PHƯƠNG PHÁP NGHIÊN CỨU

### 🔴 Vấn đề nghiêm trọng (Critical Issues)

#### 1. Kết quả chạy với QUICK_RUN = True — Không có giá trị khoa học

Notebook được nộp ở trạng thái `QUICK_RUN = True`, `QUICK_EPOCHS = 1`. Toàn bộ kết quả chỉ phản ánh 1 epoch training — **mô hình chưa hội tụ**, IoU/Dice rất thấp. Phần Critical Discussion (Cell 25) **để trống** với ghi chú "Điền sau khi chạy full".

**Hệ quả:**
- Không thể kiểm chứng bất kỳ giả thuyết nào (H1-H5)
- Bảng summary có `std_iou = NaN` (chỉ 1 seed)
- Training curves chỉ có 1 điểm → vô nghĩa

#### 2. Kiến trúc mô hình quá đơn giản (Oversimplified Architectures)

```
U-Net:         base=32 → chỉ 1.9M params (gốc: base=64, ~31M params)
SegNet:        Chỉ 1 Conv/block → 196K params (gốc: VGG16 encoder, ~29M params)
DeepLabV3+:    "Compact" — không dùng backbone, chỉ 581K params
```

> [!WARNING]
> **SegNet** gốc sử dụng VGG16 làm encoder (13 conv layers) với maxpool indices unpooling. Phiên bản trong notebook chỉ có **3 conv layers** — đây thực chất **không phải SegNet**, mà là một toy model đơn giản dùng max-unpool. Điều này làm **thiếu tính trung thực** khi claim so sánh SegNet với U-Net.

> [!WARNING]
> **DeepLabV3+** gốc sử dụng backbone (Xception/ResNet) + Atrous Spatial Pyramid Pooling. Phiên bản "Compact" ở đây chỉ có 3 conv layers encoder + ASPP nhỏ. Mô hình **quá nông** để khai thác multi-scale context hiệu quả — đây là yếu tố chính mà DeepLabV3+ claim ưu thế.

#### 3. Không sử dụng Pretrained Backbone

Tất cả 3 mô hình đều train **from scratch** với encoder tự xây. Trong thực tế:
- U-Net hiện đại luôn dùng pretrained encoder (ResNet, EfficientNet)
- DeepLabV3+ tiêu chuẩn dùng ResNet-50/101 hoặc Xception pretrained trên ImageNet
- SegNet dùng VGG16 pretrained

**Hệ quả:** Kết quả không phản ánh đúng năng lực thực của kiến trúc, đặc biệt với dataset nhỏ (KvasirSEG chỉ 1000 ảnh).

#### 4. Input Resolution quá thấp (128×128)

```python
IMG_SIZE: int = 128  # Quá thấp
```

- U-Net gốc: 572×572 → crop 388×388
- DeepLabV3+: thường 512×512 hoặc 321×321
- KvasirSEG: ảnh gốc > 300×300

Ở 128×128, boundary information bị mất nghiêm trọng → ảnh hưởng đặc biệt đến Oxford Pet (có class "boundary") và Kvasir (cần boundary chính xác cho polyp).

---

### 🟡 Vấn đề đáng lưu ý (Significant Issues)

#### 5. Hoàn toàn thiếu Data Augmentation

Notebook **không có bất kỳ augmentation** nào (flip, rotate, color jitter, elastic deformation...). Đây là yếu tố chuẩn trong mọi pipeline segmentation:

```python
# Hiện tại (Cell 8): Chỉ resize + normalize
self.image_tf = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(...)
])
```

Không có augmentation → **overfitting** nhanh, đặc biệt với KvasirSEG chỉ 700 training samples.

#### 6. Không có Learning Rate Scheduler

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
# Không có scheduler → LR cố định suốt training
```

Thiếu scheduler (ReduceLROnPlateau, CosineAnnealing, PolyLR...) khiến:
- Mô hình có thể không hội tụ tốt
- Không khai thác hết tiềm năng khi train nhiều epochs
- DeepLabV3+ gốc dùng Poly LR decay — thiếu thành phần này làm giảm tính trung thực so sánh

#### 7. Thiếu Statistical Significance

Config cho phép chạy 3 seeds `(42, 2025, 3407)` nhưng QUICK_RUN chỉ dùng 1 seed → `std = NaN`. Ngay cả khi chạy full:
- Không có statistical test (paired t-test, Wilcoxon signed-rank)
- Chỉ báo cáo mean ± std mà không kết luận sự khác biệt có ý nghĩa thống kê

#### 8. Metric Computation có sai sót tiềm ẩn

```python
# Cell 16 - calculate_metrics
iou = (intersection + eps) / (union + eps)  # eps = 1e-7
```

Vấn đề: Khi cả `intersection = 0` và `union = 0` (class không xuất hiện trong batch), IoU ≈ 1.0 (do eps/eps ≈ 1). Đây là lỗi → nên **bỏ qua class không có mặt** hoặc dùng masked averaging.

#### 9. Data Split không chuẩn cho OxfordPet

```python
# Cell 9: Oxford Pet
trainval = OxfordIIITPet(split="trainval", ...)
train_ds, val_ds, _ = split_dataset(trainval, seed=seed)  # 70/15/15
test_ds = OxfordIIITPet(split="test", ...)  # Official test set
```

OxfordPet có official train/test split. Notebook dùng random_split cho trainval → **các seed khác nhau tạo ra train/val split khác nhau**, nhưng test set cố định. Điều này OK về nguyên tắc nhưng cần document rõ hơn.

#### 10. Segmentation Complexity Analysis hạn chế

```python
def compute_complexity(mask):
    fg_ratio = fg.sum().item() / mask.numel()
    boundary_ratio = estimate_boundary_ratio(mask)
    return boundary_ratio + 0.3 * (1 - fg_ratio)
```

Metric complexity tự thiết kế, trọng số `0.3` arbitrary, chưa chuẩn hóa. Nên dùng metric chuẩn hơn (connected components, shape complexity, mask area distribution).

---

### 🟠 Vấn đề về trình bày (Presentation Issues)

#### 11. Duplicate Code

`make_fraction_subset()` và `make_loaders()` được định nghĩa **2 lần** (Cell 17 và Cell 20) — thể hiện code chưa clean up.

#### 12. Critical Discussion để trống

Cell 25 có framework tốt (H1-H5, trade-off, limitations) nhưng tất cả **để trống** "Điền sau khi chạy full". Theo yêu cầu trong Notes, phần critical discussion and insights là bắt buộc.

#### 13. Thiếu Architecture Visualization

Notes yêu cầu "visualize the network architectures". Notebook chỉ có `torchinfo.summary()` (text), thiếu:
- Sơ đồ kiến trúc dạng hình
- Feature map visualization
- Gradient-weighted Class Activation Mapping (Grad-CAM)

#### 14. Thiếu Confusion Matrix

Đây là visualization chuẩn cho segmentation, đặc biệt với OxfordPet 3-class, giúp thấy rõ class nào bị nhầm lẫn.

---

## III. ĐỀ XUẤT PHƯƠNG ÁN TỐI ƯU

### A. Kiến trúc mô hình (Model Architecture) — Cải tiến trọng yếu

#### Phương án 1: Dùng Pretrained Backbone (Khuyến nghị mạnh)

```python
import segmentation_models_pytorch as smp

# U-Net với ResNet34 backbone pretrained
model_unet = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=num_classes,
)

# DeepLabV3+ với ResNet34 backbone
model_deeplab = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=num_classes,
)

# FPN (Feature Pyramid Network) — thay thế SegNet
model_fpn = smp.FPN(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=num_classes,
)
```

> [!TIP]
> Thư viện `segmentation_models_pytorch` (smp) cung cấp pretrained models chuẩn. Tuy nhiên, theo yêu cầu đề bài "implement", nên **tự implement kiến trúc** nhưng **sử dụng pretrained encoder** từ `torchvision.models`.

#### Phương án 2: Tự implement nhưng sâu hơn (Cân bằng originality và quality)

```python
# U-Net với ResNet34 encoder (pretrained)
class UNetResNet34(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = torchvision.models.resnet34(pretrained=True)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64ch
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)          # 64ch
        self.enc3 = resnet.layer2   # 128ch
        self.enc4 = resnet.layer3   # 256ch
        self.bottleneck = resnet.layer4  # 512ch
        # Decoder với skip connections
        self.up4 = DecoderBlock(512, 256)
        self.up3 = DecoderBlock(256, 128)
        self.up2 = DecoderBlock(128, 64)
        self.up1 = DecoderBlock(64, 64)
        self.final = nn.Conv2d(64, num_classes, 1)
```

```python
# SegNet với VGG16 encoder (gần đúng bản gốc)
class SegNetVGG(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        vgg = torchvision.models.vgg16_bn(pretrained=True)
        # Chia encoder thành 5 blocks theo VGG16
        self.enc1 = vgg.features[:7]    # 64ch
        self.enc2 = vgg.features[7:14]  # 128ch
        self.enc3 = vgg.features[14:24] # 256ch
        self.enc4 = vgg.features[24:34] # 512ch
        self.enc5 = vgg.features[34:44] # 512ch
        # Decoder mirror
        ...
```

---

### B. Training Pipeline — Cải tiến cần thiết

#### 1. Tăng Resolution

```python
IMG_SIZE: int = 256  # Tối thiểu 256, lý tưởng 320-384
```

Trên T4 GPU (16GB), `batch_size=8` với input 256×256 vẫn chạy tốt.

#### 2. Thêm Data Augmentation (Dùng albumentations)

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

#### 3. Thêm Learning Rate Scheduler

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)
# hoặc PolyLR decay (chuẩn cho DeepLab):
scheduler = torch.optim.lr_scheduler.PolynomialLR(
    optimizer, total_iters=epochs, power=0.9
)
```

#### 4. Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        
    def __call__(self, val_score):
        if self.best_score is None or val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
```

#### 5. Training Configuration đề xuất

```python
@dataclass
class OptimizedConfig:
    IMG_SIZE: int = 256
    BATCH_SIZE: int = 8
    FINAL_EPOCHS: int = 30         # Tăng lên 30
    SEEDS: Tuple[int, ...] = (42, 2025, 3407)
    DATA_FRACTIONS: Tuple[float, ...] = (1.0, 0.5, 0.25)
    LEARNING_RATES: Tuple[float, ...] = (1e-3, 3e-4, 1e-4)
    # Thêm scheduler
    SCHEDULER: str = "cosine"      # "cosine" hoặc "poly"
    # Thêm early stopping
    EARLY_STOP_PATIENCE: int = 7
```

---

### C. Thêm Dataset thứ 3 (Khuyến nghị)

Theo yêu cầu "datasets from different domains", 2 dataset là tối thiểu. Nên thêm 1 dataset nữa:

| Option | Dataset | Domain | Classes | Size | Feasibility |
|--------|---------|--------|---------|------|-------------|
| ⭐ | **CamVid** | Urban driving | 11-32 | 700 imgs | ✅ Nhỏ, chạy nhanh |
| | **ISBI 2012** | Electron microscopy | 2 | 30 imgs | ✅ Rất nhỏ |
| | **Cityscapes** | Urban driving | 19 | 5000 imgs | ⚠️ Lớn, cần thời gian |
| | **DRIVE** | Retinal vessels | 2 | 40 imgs | ✅ Rất nhỏ |

> [!TIP]
> **CamVid** là lựa chọn tốt nhất: domain khác biệt (outdoor/driving vs. natural/medical), multi-class (11+ classes), kích thước vừa phải để chạy trên Colab Free.

---

### D. Evaluation & Analysis — Cải tiến quan trọng

#### 1. Fix Metric Calculation

```python
@torch.no_grad()
def calculate_metrics(preds, targets, num_classes, eps=1e-7):
    if preds.dim() == 4:
        preds = torch.argmax(preds, dim=1)
    
    ious, dices = [], []
    valid_classes = 0
    
    for c in range(num_classes):
        pred_c = preds == c
        target_c = targets == c
        
        # Bỏ qua class không xuất hiện trong batch
        if target_c.sum() == 0 and pred_c.sum() == 0:
            continue
        
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        
        iou = intersection / (union + eps)
        dice = 2 * intersection / (pred_c.sum().float() + target_c.sum().float() + eps)
        
        ious.append(float(iou))
        dices.append(float(dice))
        valid_classes += 1
    
    pixel_acc = (preds == targets).float().mean().item()
    
    return {
        "pixel_acc": pixel_acc,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "per_class_iou": ious,
        "per_class_dice": dices,
    }
```

#### 2. Thêm Confusion Matrix

```python
def plot_confusion_matrix(model, test_loader, num_classes, class_names, device):
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    model.eval()
    for imgs, masks in test_loader:
        imgs = imgs.to(device)
        preds = torch.argmax(model(imgs), dim=1).cpu()
        for t, p in zip(masks.view(-1), preds.view(-1)):
            confusion[t.long(), p.long()] += 1
    
    # Normalize
    confusion_norm = confusion.float() / confusion.sum(dim=1, keepdim=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion_norm, cmap='Blues')
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.colorbar(im)
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.show()
```

#### 3. Thêm Boundary F1-Score

```python
def boundary_f1_score(pred, target, threshold=2):
    """Đánh giá chất lượng segmentation ở vùng biên"""
    from scipy import ndimage
    pred_boundary = pred ^ ndimage.binary_erosion(pred, iterations=threshold)
    target_boundary = target ^ ndimage.binary_erosion(target, iterations=threshold)
    
    tp = (pred_boundary & target_boundary).sum()
    fp = (pred_boundary & ~target_boundary).sum()
    fn = (~pred_boundary & target_boundary).sum()
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return f1
```

#### 4. Statistical Significance Tests

```python
from scipy import stats

def compare_models_significance(results_df, model_a, model_b, metric='test_iou'):
    """Paired t-test giữa 2 models trên cùng seeds"""
    a_scores = results_df[results_df['model'] == model_a][metric].values
    b_scores = results_df[results_df['model'] == model_b][metric].values
    
    t_stat, p_value = stats.ttest_rel(a_scores, b_scores)
    
    print(f"{model_a} vs {model_b}:")
    print(f"  {model_a}: {a_scores.mean():.4f} ± {a_scores.std():.4f}")
    print(f"  {model_b}: {b_scores.mean():.4f} ± {b_scores.std():.4f}")
    print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
    print(f"  Significant (p<0.05): {'Yes' if p_value < 0.05 else 'No'}")
```

---

### E. Visualization — Bổ sung cần thiết

#### 1. Architecture Diagrams

Vẽ sơ đồ kiến trúc bằng `torchviz` hoặc tự vẽ bằng matplotlib/tikz:

```python
from torchviz import make_dot
x = torch.randn(1, 3, 256, 256)
y = model(x)
dot = make_dot(y, params=dict(model.named_parameters()))
dot.render("unet_architecture", format="png")
```

#### 2. Feature Map Visualization

```python
def visualize_feature_maps(model, img, layer_name):
    """Hiển thị feature maps ở các tầng khác nhau"""
    activation = {}
    def hook(model, input, output):
        activation[layer_name] = output.detach()
    
    handle = dict(model.named_modules())[layer_name].register_forward_hook(hook)
    _ = model(img.unsqueeze(0))
    handle.remove()
    
    feat = activation[layer_name][0]
    n_channels = min(16, feat.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < n_channels:
            ax.imshow(feat[i].cpu(), cmap='viridis')
        ax.axis('off')
    plt.suptitle(f'Feature maps at {layer_name}')
    plt.show()
```

#### 3. Data Fraction Degradation Plot

```python
def plot_data_fraction_degradation(summary_df):
    """Biểu đồ suy giảm hiệu năng khi giảm dữ liệu"""
    for ds_name in summary_df['dataset'].unique():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sub = summary_df[summary_df['dataset'] == ds_name]
        
        for model in sub['model'].unique():
            model_data = sub[sub['model'] == model].sort_values('fraction')
            axes[0].plot(model_data['fraction'], model_data['mean_iou'], 
                        marker='o', label=model)
            axes[1].plot(model_data['fraction'], model_data['mean_dice'], 
                        marker='o', label=model)
        
        axes[0].set_xlabel('Data Fraction')
        axes[0].set_ylabel('Mean IoU')
        axes[0].set_title(f'{ds_name} - IoU vs Data Fraction')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Data Fraction')
        axes[1].set_ylabel('Mean Dice')
        axes[1].set_title(f'{ds_name} - Dice vs Data Fraction')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
```

---

### F. Critical Discussion — Framework hoàn chỉnh

Phần discussion cần **điền đầy đủ** với kết quả thực. Dưới đây là template:

```markdown
## 6.1. Phân tích kết quả chính
- So sánh IoU/Dice trung bình ± std trên từng dataset
- Xác định model tốt nhất cho từng domain
- Phân tích per-class performance (class nào khó nhất?)

## 6.2. Kiểm chứng giả thuyết
| Hypothesis | Evidence | Result |
|---|---|---|
| H1: U-Net tốt nhờ skip connections | IoU U-Net vs SegNet (không có skip) | Xác nhận/Bác bỏ + p-value |
| H2: SegNet hiệu quả nhưng kém ở biên | Boundary F1, error maps ở vùng biên | ... |
| H3: DeepLabV3+ tốt nhờ ASPP | IoU ở nhóm "khó" (complexity analysis) | ... |
| H4: Suy giảm khác nhau khi giảm data | Data fraction plots | ... |
| H5: Không có model tối ưu tuyệt đối | Accuracy-Latency-Memory plots | ... |

## 6.3. Cross-domain Analysis
- OxfordPet (3-class, natural) vs KvasirSEG (binary, medical)
- Tại sao model X tốt trên domain A nhưng kém trên domain B?
- Class imbalance ảnh hưởng thế nào?

## 6.4. Trade-offs
- Bảng tổng hợp: IoU | Dice | Latency | Memory | Params
- Mô hình nào phù hợp cho real-time? Cho accuracy-critical?

## 6.5. Limitations
- Compact models vs full-scale
- Chỉ 2 domains, chưa bao quát
- Colab Free time limit
- Thiếu test-time augmentation

## 6.6. Future Work
- Pretrained backbones (ResNet, EfficientNet)
- Attention mechanisms (Attention U-Net, TransUNet)
- Loss engineering (Focal, Tversky, Lovász)
- Knowledge distillation
- Model compression (pruning, quantization)
```

---

## IV. CHECKLIST ĐỐI CHIẾU VỚI YÊU CẦU NOTES

| Yêu cầu (Notes) | Notebook hiện tại | Đề xuất |
|---|---|---|
| ≥ 2 DL models + visualize architecture | ✅ 3 models, ❌ chỉ text summary | ✅ Thêm architecture diagram |
| Clearly define hypotheses and RQs | ✅ 5 RQs, 5 Hs | ✅ OK |
| Fair comparison (same splits, metrics, conditions) | ✅ Framework có | ✅ OK |
| Vary key factors (data amount, hyperparams) | ✅ Config có fractions + LRs | ⚠️ Cần chạy full |
| Multiple runs for reliability | ⚠️ Config 3 seeds, QUICK chỉ 1 | ✅ Chạy full 3 seeds |
| Evaluation metrics (IoU, Dice) | ✅ Pixel Acc, IoU, Dice | ✅ Thêm Boundary F1, Confusion Matrix |
| Latency + memory footprint | ✅ Có | ✅ OK |
| Visualizations (dataset, curves, performance) | ⚠️ Có nhưng 1 epoch | ✅ Thêm feature maps, Grad-CAM |
| Jupyter contains code for all figures | ✅ Có | ✅ OK |
| Interpret results + validate hypotheses | ❌ Để trống | 🔴 **Bắt buộc điền** |
| Discuss trade-offs | ❌ Để trống | 🔴 **Bắt buộc điền** |
| Discuss limitations | ⚠️ Có nhưng surface level | ✅ Chi tiết hơn |
| Suggest improvements | ⚠️ Có nhưng ngắn | ✅ Chi tiết hơn |
| 2-4 page report (Word/LaTeX) | ❓ Chưa thấy | 🔴 **Cần làm** |
| README file | ❓ Chưa thấy | 🔴 **Cần làm** |
| Code runs independently | ⚠️ Phụ thuộc Google Drive mount | ✅ Thêm auto-download option |

---

## V. KẾ HOẠCH THỰC HIỆN ĐỀ XUẤT (Priority Order)

### Phase 1: Critical Fixes (Bắt buộc)
1. ❗ Chạy `QUICK_RUN = False` với đủ epochs (≥15, khuyến nghị 25-30)
2. ❗ Chạy đủ 3 seeds để có mean ± std
3. ❗ Điền Critical Discussion từ kết quả thực
4. ❗ Tăng `IMG_SIZE` lên 256

### Phase 2: Significant Improvements (Nên làm)
5. Thêm Data Augmentation (albumentations)
6. Thêm LR Scheduler (CosineAnnealing)
7. Fix metric calculation (handle missing classes)
8. Cải thiện SegNet architecture cho gần bản gốc hơn
9. Thêm Confusion Matrix visualization

### Phase 3: Excellence Level (Để đạt 9+)
10. Dùng pretrained backbone (ResNet34/VGG16)
11. Thêm dataset thứ 3 (CamVid)
12. Thêm Boundary F1-Score metric
13. Thêm architecture visualization diagrams
14. Thêm statistical significance tests
15. Thêm feature map / Grad-CAM visualization
16. Viết report 2-4 trang + README

---

## VI. ƯỚC TÍNH THỜI GIAN CHẠY TRÊN COLAB T4

| Config | Estimated Time |
|--------|---------------|
| 3 models × 2 datasets × 3 seeds × 3 fractions × 2 LRs × 15 epochs (hiện tại) | ~8-12 giờ |
| 3 models × 2 datasets × 3 seeds × 1 fraction × 1 LR × 25 epochs (tối giản) | ~3-4 giờ |
| 3 models × 3 datasets × 3 seeds × 2 fractions × 2 LRs × 30 epochs (đầy đủ) | ~18-24 giờ |

> [!IMPORTANT]
> Colab Free chỉ cho ~12 giờ liên tục. Nên:
> - Chia nhỏ thí nghiệm thành nhiều sessions
> - Lưu checkpoint sau mỗi dataset
> - Ưu tiên chạy full `fractions × seeds` trên config `(fraction=1.0, lr=3e-4)` trước
> - Thêm resume training từ checkpoint
