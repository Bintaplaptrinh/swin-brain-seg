# BÁO CÁO TỔNG QUAN VÀ ĐÁNH GIÁ KẾT QUẢ DỰ ÁN CADx U NÃO
**Hệ thống Hỗ trợ Chẩn đoán Hình ảnh Y tế (CADx) với Swin Transformer V2 và Swin Hybrid**

---

## 1. TỔNG QUAN DỰ ÁN VÀ KIẾN TRÚC HỆ THỐNG

Dự án phát triển hệ thống CADx (Computer-Aided Diagnosis) tự động hóa quy trình sàng lọc và phân đoạn u não trên các lát cắt ảnh cộng hưởng từ (MRI) 2D từ bộ dữ liệu y tế chuẩn **BRISC 2025**.

### 1.1 Quy trình chẩn đoán 2 giai đoạn (Two-Stage CADx Pipeline)

```mermaid
flowchart TD
    A["Lát cắt ảnh MRI Não (2D Grayscale)"] --> B["Stage 1: Swin Transformer V2-Tiny (Screening)"]
    B -->|P(Tumor) < 0.40| C["KẾT LUẬN: NÃO LÀNH (NO TUMOR)\n→ Dừng pipeline ngay lập tức (Tiết kiệm 100% chi phí tính toán)"]
    B -->|P(Tumor) >= 0.40| D["KẾT LUẬN: CÓ KHỐI U (TUMOR DETECTED)\n→ Chuyển tiếp sang Giai đoạn 2"]
    D --> E["Stage 2: Swin Hybrid (nnU-Net + Swin Transformer 2D)"]
    E --> F["Khôi phục kích thước ảnh gốc (Native Resolution)\n→ Xuất Mặt nạ phân đoạn (Binary Tumor Mask)\n→ Tính Diện tích & Tọa độ Bounding Box"]
```

### 1.2 Đặc tả kiến trúc kỹ thuật

| Thành phần | Giai đoạn 1 (Screening) | Giai đoạn 2 (Segmentation) |
| :--- | :--- | :--- |
| **Mô hình** | Swin Transformer V2-Tiny (`swinv2_cr_tiny_ns_224`) | Swin Hybrid (nnU-Net Residual CNN + 2D Swin Transformer) |
| **Kích thước đầu vào** | 224 × 224 × 3 | 512 × 512 × 1 (Grayscale) |
| **Số lượng tham số** | 27,573,985 (~27.57M) | 38,801,022 (~38.80M) |
| **Độ phức tạp tính toán** | ~4.5 GFLOPs | 237.78 GFLOPs |
| **Độ trễ suy luận** | ~14.8 ms / lát cắt | ~49.60 ms / lát cắt (CUDA AMP) |
| **Nền tảng triển khai** | PyTorch / ONNX Runtime | PyTorch / ONNX Runtime / Native Rust Desktop GUI |

---

## 2. MA TRẬN NHẦM LẪN GIAI ĐOẠN 1 (STAGE 1 CLASSIFIER CONFUSION MATRIX)

Đánh giá trên toàn bộ **1,000 lát cắt ảnh test** của bộ dữ liệu BRISC 2025 (gồm 860 ảnh có u não và 140 ảnh não lành):

### 2.1 Bảng ma trận nhầm lẫn 2×2 (Cấp độ ảnh - Image Level)

| Thực tế \ Dự đoán | Dự đoán Âm tính (Não lành) | Dự đoán Dương tính (Có u) | Tổng số thực tế | Tỷ lệ chính xác |
| :--- | :---: | :---: | :---: | :---: |
| **Não lành (No Tumor)** | **TN = 140** | **FP = 0** | 140 | **100.0%** (Độ đặc hiệu) |
| **Có khối u (Tumor)** | **FN = 0** | **TP = 860** | 860 | **100.0%** (Độ nhạy) |
| **Tổng số dự đoán** | 140 | 860 | **1,000** | **Accuracy = 100.0%** |

### 2.2 Các chỉ số hiệu năng lâm sàng chi tiết

- **Độ chính xác tổng (Accuracy):** $\frac{TP + TN}{Total} = \frac{860 + 140}{1000} = \mathbf{100.0\%}$
- **Độ nhạy / Thu hồi (Sensitivity / Recall):** $\frac{TP}{TP + FN} = \frac{860}{860 + 0} = \mathbf{100.0\%}$
- **Độ đặc hiệu (Specificity):** $\frac{TN}{TN + FP} = \frac{140}{140 + 0} = \mathbf{100.0\%}$
- **Độ chuẩn xác / Giá trị dự báo dương tính (Precision / PPV):** $\frac{TP}{TP + FP} = \frac{860}{860 + 0} = \mathbf{100.0\%}$
- **Giá trị dự báo âm tính (NPV):** $\frac{TN}{TN + FN} = \frac{140}{140 + 0} = \mathbf{100.0\%}$
- **Điểm F1 (F1-Score):** $\mathbf{1.0000}$
- **Ý nghĩa lâm sàng then chốt:**
  - **$\mathbf{FN = 0}$:** Tuyệt đối không bỏ sót bất kỳ bệnh nhân có khối u nào trong quá trình sàng lọc tự động ban đầu.
  - **$\mathbf{TN = 140 / 140}$:** Loại trừ chính xác toàn bộ 100% các ca não lành, giúp hệ thống không phải chạy mô hình phân đoạn Swin Hybrid nặng nề cho các ca này.

---

## 3. MA TRẬN NHẦM LẪN GIAI ĐOẠN 2 (STAGE 2 SEGMENTATION CONFUSION MATRIX)

Đánh giá trên toàn bộ **860 lát cắt có u** ở độ phân giải gốc $512 \times 512$ (tổng cộng $208,642,400$ điểm ảnh được đối soát):

### 3.1 Bảng ma trận nhầm lẫn cấp độ điểm ảnh (Pixel-Level Confusion Matrix)

| Pixel Thực tế \ Dự đoán | Dự đoán Pixel Nền (Background) | Dự đoán Pixel Khối U (Tumor) | Tổng số pixel thực tế | Tỷ lệ chính xác |
| :--- | :---: | :---: | :---: | :---: |
| **Pixel Nền (Background)** | **TN = 204,749,189** | **FP = 287,543** | 205,036,732 | **99.86%** (Đặc hiệu) |
| **Pixel Khối U (Tumor)** | **FN = 419,256** | **TP = 3,186,412** | 3,605,668 | **88.37%** (Độ nhạy) |
| **Tổng pixel dự đoán** | 205,168,445 | 3,473,955 | **208,642,400** | — |

### 3.2 Chỉ số phân đoạn và đo khoảng cách bề mặt (Surface Distance)

- **Độ chuẩn xác cấp Pixel (Pixel Precision):** $\mathbf{91.72\%}$
- **Độ nhạy cấp Pixel (Pixel Recall):** $\mathbf{88.37\%}$
- **Weighted mIoU (Mean Intersection over Union):** $\mathbf{0.81146}$ ($81.15\%$)
- **Dice Similarity Coefficient (DSC):** $\mathbf{0.87722}$ ($87.72\%$)
- **Khoảng cách Hausdorff 95% (HD95):** $\mathbf{16.29\text{ px}}$
- **Sai số bề mặt đối xứng trung bình (ASSD):** $\mathbf{4.68\text{ px}}$
- **Độ tương đồng biên ở ngưỡng 2px (NSD):** $\mathbf{0.75194}$ ($75.19\%$)

---

## 4. PHÂN TÍCH HIỆU NĂNG THEO TỪNG PHÂN LOẠI U NÃO (SUBTYPES BREAKDOWN)

Tập dữ liệu BRISC 2025 gồm 3 phân nhóm u não với đặc trưng bệnh học và độ khó phân đoạn khác nhau:

| Phân nhóm U | Số ảnh Test | Tỷ lệ | mIoU | Dice Score | Đánh giá bệnh học & Khó khăn phân đoạn |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Glioma (U tế bào thần kinh đệm)** | 254 | 29.5% | **0.70489** | **78.90%** | Khó nhất do u thâm nhiễm vào nhu mô não, ranh giới mờ nhạt, cấu trúc biến thiên cao. |
| **Meningioma (U màng não)** | 306 | 35.6% | **0.90494** | **94.66%** | Dễ nhất do u ngoài trục não, ranh giới rõ ràng, hình thái tương đối đồng nhất. |
| **Pituitary (U tuyến yên)** | 300 | 34.9% | **0.80634** | **88.11%** | Độ khó trung bình, vị trí đặc thù tại hố yên, kích thước đa dạng. |
| **Tổng thể có u (Weighted)** | **860** | **100%** | **0.81146** | **87.72%** | **Chất lượng phân đoạn vượt trội, đạt độ tin cậy y tế cao.** |

---

## 5. BẢNG SO SÁNH ĐỐI CHỨNG VỚI CÁC MÔ HÌNH BASELINE (BENCHMARK)

| Mô hình | Epoch | Trọng số | Kích thước | Tham số (M) | GFLOPs | Độ trễ (ms) | Glioma mIoU | Weighted mIoU | Dice |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Swin Hybrid (Val-selected)** | 109 | EMA | 512×512 | 38.80 | 237.78 | 49.81 | **0.70659** | **0.81465** | **0.87959** |
| **Swin Hybrid (Production / Rust)** | 159 | EMA | 512×512 | 38.80 | 237.78 | 49.60 | **0.70489** | **0.81146** | **0.87722** |
| **KAN (Kolmogorov-Arnold)** | 54 | EMA | 256×256 | 8.92 | 32.11 | 26.04 | 0.66709 | 0.79530 | 0.86515 |
| **nnU-Net v2 + Depthwise 7×7** | 23 | EMA | 256×256 | 8.80 | 31.90 | 11.32 | 0.66223 | 0.79378 | 0.86580 |
| **BiRef Historical Baseline** | 38 | EMA | 256×256 | ~10.0 | 32.53 | 12.19 | 0.68229 | 0.80302 | 0.87289 |

> **Nhận xét chuyên môn:**
> - Swin Hybrid vượt trội toàn bộ các mô hình CNN thuần túy và KAN trên bài toán khó nhất là **Glioma (+3.78% so với KAN, +4.27% so với Depthwise)** nhờ khả năng nắm bắt ngữ cảnh toàn cục (Long-range dependencies) của cơ chế Multi-Head Self-Attention từ Swin Transformer.
> - Kỹ thuật EMA (Exponential Moving Average) và cơ chế Attention Decoders giúp tối ưu hóa ranh giới khối u cực kỳ sắc nét.

---

## 6. DANH MỤC CÁC TẬP TIN OUTPUT ĐÃ ĐƯỢC TẠO

Toàn bộ các output cần thiết cho việc làm báo cáo, làm slide thuyết trình và tích hợp vào hệ thống đã được tạo ra đầy đủ:

1. **Ma trận nhầm lẫn trực quan hóa (Vector SVG chất lượng cao):**
   - `outputs/reports/classifier_confusion_matrix.svg` (Ma trận nhầm lẫn Phân loại Giai đoạn 1)
   - `outputs/reports/segmentation_pixel_confusion_matrix.svg` (Ma trận nhầm lẫn Phân đoạn Giai đoạn 2)
   - `outputs/runs/cadx_system_epoch159/classifier_confusion_matrix.svg`
2. **Dữ liệu Ma trận nhầm lẫn (JSON):**
   - `outputs/runs/cadx_system_epoch159/classifier_confusion_matrix.json`
   - `outputs/reports/segmentation_pixel_confusion_matrix.json`
3. **Báo cáo kiểm thử thực nghiệm (Test Metrics JSON):**
   - `outputs/runs/cadx_system_epoch159/test_metrics.json` (Báo cáo hệ thống toàn diện 1,000 ảnh)
   - `outputs/runs/cadx_segmenter_epoch159/test_metrics.json` (Báo cáo phân đoạn 860 ảnh)
   - `outputs/runs/swin_hybrid_512_unfrozen_seed2026/epoch159_test/test_metrics.json`
4. **Bảng phân tích và đối chứng (CSV & JSON):**
   - `outputs/reports/tumor_subtypes_breakdown.csv` & `.json`
   - `outputs/reports/benchmark_models_comparison.csv` & `.json`
5. **Báo cáo chi tiết toàn diện:**
   - `outputs/reports/CADX_PROJECT_EVALUATION_REPORT.md`
