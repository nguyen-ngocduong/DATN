# KẾ HOẠCH MỞ RỘNG QUY MÔ DỮ LIỆU HUẤN LUYỆN

## Mục tiêu

Thực hiện nghiên cứu kinh điển: **"Ảnh hưởng của quy mô dữ liệu huấn luyện đối với hiệu năng phát hiện xâm nhập mạng IoT"** bằng cách so sánh hiệu năng mô hình trên 3 quy mô dữ liệu khác nhau.

---

## Tổng quan kiến trúc

| Nền tảng | Quy mô | Mục tiêu | Thời gian ước tính |
|-----------|--------|----------|---------------------|
| **Google Colab** | 250,000 mẫu | Baseline/Reference | ~25-35 phút |
| **Google Colab** | 500,000 mẫu | Medium Scale | ~40-60 phút |
| **Kaggle** | 1,000,000 mẫu | Full Scale | ~60-90 phút (với GPU) |

---

## Phần 1: Google Colab - 250K và 500K mẫu

### 1.1 Chuẩn bị môi trường Colab

#### Bước 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

#### Bước 2: Cài đặt dependencies
```bash
!pip install -q scikit-learn imbalanced-learn xgboost lightgbm joblib pandas numpy matplotlib seaborn
```

#### Bước 3: Tạo cấu trúc thư mục
```bash
# Tạo thư mục project trên Drive
!mkdir -p /content/drive/MyDrive/cic_iot_project/data
!mkdir -p /content/drive/MyDrive/cic_iot_project/models
!mkdir -p /content/drive/MyDrive/cic_iot_project/plots
!mkdir -p /content/drive/MyDrive/cic_iot_project/phase5_experiments
!mkdir -p /content/drive/MyDrive/cic_iot_project/phase6_evaluation
```

### 1.2 Quy trình chạy cho 250K mẫu

#### Giai đoạn A: Chuẩn bị dữ liệu (Chạy 1 lần, tái sử dụng)

**Phase 2: Preprocessing**
```python
!python 02_preprocessing.py --sample 250000 --random-state 42
```

**Phase 3: Class Balance**
```python
!python 03_balance.py --strategy hybrid --sample-size 250000 --target-majority 30000 --target-minority 8000
```

**Phase 4: Feature Selection**
```python
!python 04_feature_selection.py --top-k 25 --use-balanced
```

#### Giai đoạn B: Huấn luyện mô hình (Phase 5)

```python
!python 05_train_models.py --train-samples 250000 --test-size 0.2 --random-state 123
```

#### Giai đoạn C: Đánh giá (Phase 6)

```python
!python 06_evaluate.py
```

### 1.3 Quy trình chạy cho 500K mẫu

#### Giai đoạn A: Chuẩn bị dữ liệu (Tái sử dụng output 250K làm input)

**Phase 2: Preprocessing với 500K**
```python
!python 02_preprocessing.py --sample 500000 --random-state 42
```

**Phase 3: Class Balance với 500K**
```python
!python 03_balance.py --strategy hybrid --sample-size 500000 --target-majority 30000 --target-minority 8000
```

**Phase 4: Feature Selection với 500K**
```python
!python 04_feature_selection.py --top-k 25 --use-balanced
```

#### Giai đoạn B: Huấn luyện mô hình với 500K**

```python
!python 05_train_models.py --train-samples 500000 --test-size 0.2 --random-state 123
```

#### Giai đoạn C: Đánh giá với 500K**

```python
!python 06_evaluate.py
```

### 1.4 Lưu ý quan trọng cho Colab

| Lưu ý | Chi tiết |
|-------|----------|
| **Copy data về local** | Luôn copy file từ Drive về `/content/` trước khi xử lý để tránh I/O bottleneck |
| **GPU cho XGBoost** | Bật GPU: Runtime → Change runtime type → T4 GPU |
| **Monitor RAM** | Theo dõi RAM usage bằng `!free -h` |
| **Lưu kết quả** | Copy kết quả ngược lại Drive sau khi train xong |

#### Script copy data an toàn:
```python
import shutil
import os

# Copy data từ Drive về local
shutil.copy(
    '/content/drive/MyDrive/cic_iot_project/data/X_train.pkl',
    '/content/X_train_250k.pkl'
)

# Sau khi train xong, copy kết quả về Drive
shutil.copy(
    '/content/phase5_experiments',
    '/content/drive/MyDrive/cic_iot_project/phase5_experiments_250k'
)
```

---

## Phần 2: Kaggle - 1M mẫu

### 2.1 Chuẩn bị trên Kaggle

1. **Tải dataset CICIoT2023 lên Kaggle**
   - Upload file `train.csv` lên Kaggle Dataset
   - Hoặc sử dụng dataset có sẵn: `cic-iot-2023` trên Kaggle

2. **Tạo Notebook mới**
   - New Notebook → Python 3
   - Enable GPU: Settings → Accelerator → GPU P100

3. **Cấu trúc thư mục Kaggle**
   ```
   /kaggle/working/
   ├── train.csv                 # Dataset gốc
   ├── data/                     # Output data
   ├── models/                   # Output models
   ├── plots/                    # Output plots
   ├── phase5_experiments/       # Output experiments
   └── phase6_evaluation/        # Output evaluation
   ```

### 2.2 Hướng dẫn chi tiết

Xem file: **`kaggle_1m_full_pipeline.ipynb`**

File notebook này chứa đầy đủ code cho tất cả các phase, được viết thành 1 notebook duy nhất với các section:

1. **Setup & Imports** - Cài đặt dependencies
2. **Phase 2: Preprocessing** - Xử lý dữ liệu với 1M mẫu
3. **Phase 3: Class Balance** - Hybrid resampling
4. **Phase 4: Feature Selection** - Chọn 25 features tốt nhất
5. **Phase 5: Model Training** - Huấn luyện 5 experiments
6. **Phase 6: Evaluation** - Đánh giá cuối cùng
7. **Export Results** - Download kết quả về máy

---

## Phần 3: So sánh kết quả

### 3.1 Bảng tổng hợp metrics

| Batch | Accuracy | F1-Macro | F1-Weighted | ROC-AUC | Precision (Macro) | Recall (Macro) | Trạng thái |
|-------|----------|----------|-------------|---------|-------------------|----------------|------------|
| **250K** | **96.93%** | **93.70%** | **96.97%** | **0.9986** | **92.71%** | **94.92%** | ✅ Đã hoàn thành (`batch_250k_results/`) |
| **500K** | **86.93%** | **85.65%** | **87.30%** | **0.9914** | **84.43%** | **88.27%** | ✅ Đã hoàn thành (`batch_500k_results/`) |
| **1M** | *(Kaggle Plan)* | *(Kaggle Plan)* | *(Kaggle Plan)* | *(Kaggle Plan)* | *(Kaggle Plan)* | *(Kaggle Plan)* | 📋 Theo kế hoạch Kaggle |

### 3.2 Biểu đồ so sánh

- Confusion Matrix so sánh 3 batch
- Learning Curve: Accuracy vs Training Size
- F1-Score per Class so sánh 3 batch
- ROC Curves so sánh 3 batch

---

## Phần 4: Timeline thực hiện

| Ngày | Công việc | Nền tảng |
|------|-----------|----------|
| **Ngày 1** | Chạy Batch 250K → Thu thập kết quả | Colab |
| **Ngày 2** | Chạy Batch 500K → Thu thập kết quả | Colab |
| **Ngày 3** | Chạy Batch 1M → Thu thập kết quả | Kaggle |
| **Ngày 4** | Tổng hợp, vẽ biểu đồ so sánh, viết phân tích | Local |

---

## Phần 5: Troubleshooting

### Lỗi thường gặp và cách xử lý

| Lỗi | Nguyên nhân | Giải pháp |
|------|-------------|------------|
| **OOM (Out of Memory)** | Dataset quá lớn cho RAM | Giảm sample size hoặc bật GPU swap |
| **Colab Disconnect** | Timeout khi chạy lâu | Sử dụng Kaggle cho batch 1M |
| **SMOTE Memory Error** | Quá nhiều mẫu cho SMOTE | Giảm `--target-majority` xuống 20000 |
| **File Not Found** | Path sai | Kiểm tra `os.getcwd()` và sử dụng absolute path |
| **Permission Denied** | Google Drive chưa mount đúng | Re-mount Drive và kiểm tra path |

### Công thức tính RAM cần thiết

```
RAM cần = (Số dòng × Số features × 4 bytes × 3-4 multiplier)
Ví dụ: 500K dòng × 60 features × 4 bytes × 3 = ~360 MB
+ SMOTE buffer: × 2-3
+ Model training: × 2-3
```

---

## Checklist trước khi chạy

- [ ] Upload `train.csv` lên Google Drive/Kaggle
- [ ] Cài đặt đầy đủ dependencies
- [ ] Tạo cấu trúc thư mục
- [ ] Backup code gốc
- [ ] Chuẩn bị Google Drive đủ space (cần ~5-10GB)
- [ ] Kiểm tra kết nối internet ổn định

---

## Output artifacts cần thu thập

Sau mỗi batch, cần thu thập:

```
├── batch_xxxk_results/
│   ├── phase5_experiments/
│   │   ├── leaderboard.csv
│   │   ├── leaderboard.json
│   │   └── exp_*/model/model.pkl
│   ├── phase6_evaluation/
│   │   ├── final/final_metrics.json
│   │   └── comparison/evaluation_comparison.xlsx
│   └── models/
│       ├── selected_features.json
│       └── class_weights.json
```

---

## Liên hệ & Support

Nếu gặp vấn đề:
1. Kiểm tra RAM usage: `!free -h`
2. Kiểm tra Disk usage: `!df -h`
3. Restart Runtime và chạy lại từ đầu
4. Giảm batch size nếu cần
