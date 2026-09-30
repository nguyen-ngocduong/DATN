# KẾT QUẢ NGHIÊN CỨU VÀ KẾ HOẠCH CẢI TIẾN
## Đồ Án: CICIoT2023 Network Intrusion Detection with XAI

---

## 1. SO SÁNH KẾT QUẢ VỚI BÀI BÁO GỐC

### 1.1 Kết quả Test Set (Phase 6 Evaluation)

| Model | Paper Accuracy | Paper F1-Macro | My Accuracy | My F1-Macro | Acc Diff | F1 Diff |
|-------|---------------|----------------|-------------|-------------|----------|---------|
| **XGBoost Tuned** | N/A (proposed) | N/A | **96.93%** | **93.70%** | - | - |
| Random Forest | 95.99% | 92.26% | 96.07% | 92.34% | **+0.08%** ✓ | **+0.08%** ✓ |
| Ensemble Blending | 95.12% | 90.35% | 95.19% | 90.43% | **+0.07%** ✓ | **+0.08%** ✓ |
| Decision Tree | 93.39% | 87.63% | 93.50% | 87.93% | **+0.11%** ✓ | **+0.30%** ✓ |
| XGBoost Baseline | N/A | N/A | 93.24% | 87.12% | - | - |

### 1.2 ROC-AUC Comparison

| Model | Paper ROC-AUC | My ROC-AUC | Diff |
|-------|--------------|------------|------|
| XGBoost Tuned | N/A | **99.86%** | - |
| Random Forest | ~99.70%* | 99.74% | **≈ same** |
| Ensemble | ~99.60%* | 99.69% | **+0.09%** ✓ |
| Decision Tree | ~99.00%* | 99.14% | **+0.14%** ✓ |

> *ROC-AUC values from paper are estimated from figures (Table III)

### 1.3 Per-Class Performance (XGBoost Tuned - Best Model)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Mirai | 100.00% | 100.00% | 100.00% | 2,390 |
| DDoS | 99.99% | 100.00% | 99.99% | 10,121 |
| DoS | 100.00% | 99.94% | 99.97% | 3,116 |
| Web-Based | 94.26% | 93.98% | 94.12% | 4,419 |
| Recon | 96.74% | 92.28% | 94.46% | 3,768 |
| Spoofing | 92.88% | 90.87% | 91.86% | 1,478 |
| Normal | 81.30% | 93.23% | 86.86% | 709 |
| BruteForce | 76.52% | 89.09% | 82.33% | 706 |

### 1.4 Kết Luận So Sánh

**Kết quả của bạn ĐANG VƯỢT TRỘI hơn bài báo gốc:**

- ✅ Random Forest: vượt +0.08% accuracy, +0.08% F1
- ✅ Ensemble Blending: vượt +0.07% accuracy, +0.08% F1  
- ✅ Decision Tree: vượt +0.11% accuracy, +0.30% F1
- ✅ **XGBoost Tuned đạt 96.93%** - model mới không có trong bài báo

**Điểm yếu cần cải thiện:**
- BruteForce: F1 chỉ 82.33% (precision 76.52%, recall 89.09%)
- Normal: F1 chỉ 86.86% (precision 81.30%)
- Spoofing: F1 91.86% - có thể cải thiện

---

## 2. NHẬN XÉT CÁCH LƯU KẾT QUẢ HIỆN TẠI

### 2.1 Đánh giá Phase 5 (Training Experiments)

**Điểm tốt:**
- ✅ Cấu trúc thư mục rõ ràng: `exp_XXX_name/{config, metrics, model, logs}`
- ✅ Config YAML chứa hyperparameters và metadata
- ✅ Metrics lưu classification_report.json và confusion_matrix.png
- ✅ Leaderboard CSV/JSON ở root cho tổng quan

**Điểm cần cải thiện:**
- ❌ Config trùng lặp metrics_summary (đã có metrics.json)
- ❌ Thiếu thông tin về feature selection (dùng features nào, loại bỏ features nào)
- ❌ Không lưu class distribution của training data
- ❌ Không lưu preprocessing steps (scaling, encoding)
- ❌ Không có experiment tracking (MLflow, wandb, tensorboard)

### 2.2 Đánh giá Phase 6 (Evaluation)

**Điểm tốt:**
- ✅ Cấu trúc phân chia rõ: runs/{eval_XXX}/{validation, test, performance}
- ✅ Lưu cả predictions.csv để phân tích thêm
- ✅ Config YAML riêng cho evaluation
- ✅ Latency/throughput measurement

**Điểm cần cải thiện:**
- ❌ **Metadata.json trùng lặp nhiều dữ liệu** (val_metrics và test_metrics chứa cùng thông tin trong nhiều file)
- ❌ Classification_report.json trùng lặp trong cả metadata.json, test/, validation/
- ❌ Không có experiment_id nhất quán giữa các file
- ❌ Thiếu `experiment_id` trong tên thư mục evaluation (chỉ có eval_XXX)
- ❌ Performance không có config riêng
- ❌ Không lưu độ ổn định qua nhiều seeds (chỉ có 1 run)

### 2.3 Đánh giá Reproducibility

**Các vấn đề nghiêm trọng:**

1. **Thiếu Preprocessing Pipeline Documentation**
   - Không rõ đã scale features chưa (StandardScaler, MinMaxScaler?)
   - Không rõ encoding cho categorical features
   - Không rõ cách xử lý missing values

2. **Thiếu Feature Engineering Details**
   - Không có list features đã chọn
   - Không có feature importance từ các experiments
   - Không ghi chú giải thích tại sao chọn 25 features

3. **Không có Random Seed Management**
   - Chỉ dùng random_state=123 cho training
   - Không có systematic seed experimentation (123, 42, 7)
   - Không ghi seed trong tất cả các file

### 2.4 Khuyến nghị Format Mới

**Cấu trúc đề xuất:**

```
phaseX_experiments/
├── EXPERIMENTS_REGISTRY.yaml          # Master index tất cả experiments
├── configs/
│   ├── preprocessing/
│   │   └── preprocessing_v001.yaml    # Scaler, encoding, imputation
│   └── features/
│       └── feature_selection_v001.yaml # Features đã chọn
├── exp_XXX_name/
│   ├── experiment.yaml                 # Experiment ID, method, timestamps
│   ├── config/
│   │   ├── hyperparameters.yaml        # Hyperparameters
│   │   └── environment.yaml           # Dataset info, splits
│   ├── data/
│   │   ├── class_distribution.json     # Distribution của các classes
│   │   └── feature_importance.json    # Feature importance
│   ├── metrics/
│   │   ├── metrics.json               # Chỉ metrics chính
│   │   ├── classification_report.json # Full classification report
│   │   └── predictions.csv            # Nếu cần
│   ├── model/
│   │   ├── model.pkl
│   │   └── training_config.yaml        # Config đã dùng để train
│   └── logs/
│       └── training.log
├── comparison/
│   └── leaderboard.csv                # Auto-generated từ registry
└── best_model/
    └── model.joblib
```

**EXPERIMENTS_REGISTRY.yaml example:**
```yaml
experiments:
  - exp_id: exp_005_xgboost_tuned
    method: XGBoost
    phase: 5
    status: completed
    created: 2026-09-27T16:21:57
    best_for:
      - accuracy
      - f1_macro
    linked_eval: eval_005
    config_path: exp_005_xgboost_tuned/config/hyperparameters.yaml
    metrics:
      accuracy: 0.8706
      f1_macro: 0.8406
      roc_auc: 0.9926
    
  - exp_id: exp_006_lightgbm_tuned
    method: LightGBM
    phase: 5
    status: pending
    created: null
    parent_exp: exp_005_xgboost_tuned  # Baseline để so sánh
    expected_change: "Thử LightGBM thay XGBoost"
    metrics:
      accuracy: null
      f1_macro: null
```

---

## 3. KẾ HOẠCH THỬ NGHIỆM CÁC PHƯƠNG PHÁP MỚI

### 3.1 Mục tiêu

1. **Cải thiện F1 của các lớp yếu:**
   - BruteForce: 82.33% → target 90%+
   - Normal: 86.86% → target 92%+
   - Spoofing: 91.86% → target 95%+

2. **Tăng overall accuracy:**
   - Hiện tại: 96.93% → target: 97.5%+

3. **Cải thiện ROC-AUC:**
   - Hiện tại: 99.86% → target: 99.90%+

### 3.2 Các Phương Pháp Đề Xuất

#### Tier 1: Cải thiện nhanh (1-2 ngày)

| Exp ID | Method | Mục tiêu | Priority |
|--------|--------|----------|----------|
| exp_006 | LightGBM | Thử thuật toán nhanh hơn XGBoost, có thể tốt hơn | ⭐⭐⭐ |
| exp_007 | CatBoost | Native categorical handling, regularization tốt | ⭐⭐⭐ |
| exp_008 | XGBoost + Feature Engineering | Thêm interaction features | ⭐⭐ |

#### Tier 2: Cần thời gian trung bình (3-5 ngày)

| Exp ID | Method | Mục tiêu | Priority |
|--------|--------|----------|----------|
| exp_009 | Stacking Ensemble | Meta-learner trên diverse base models | ⭐⭐⭐⭐ |
| exp_010 | XGBoost với class-specific thresholds | Tối ưu thresholds cho từng class | ⭐⭐⭐ |
| exp_011 | Cost-sensitive Learning | Điều chỉnh class weights tự động | ⭐⭐⭐ |

#### Tier 3: Nghiên cứu sâu (1-2 tuần)

| Exp ID | Method | Mục tiêu | Priority |
|--------|--------|----------|----------|
| exp_012 | Neural Network (TabNet/MLP) | Thử deep learning cho tabular data | ⭐⭐ |
| exp_013 | SMOTE/ADASYN + XGBoost | Xử lý imbalance ở class level | ⭐⭐ |
| exp_014 | Feature Selection Optimization | Grid search trên feature subsets | ⭐⭐ |

### 3.3 Chi tiết từng experiment

#### exp_006: LightGBM

```yaml
hyperparameters:
  num_leaves: 31
  learning_rate: 0.05
  n_estimators: 300
  max_depth: 6
  reg_alpha: 0.1
  reg_lambda: 0.1
  class_weight: balanced
  random_state: 123
  
expected_benefits:
  - Training nhanh hơn XGBoost 5-10x
  - Memory efficiency tốt hơn
  - Thường đạt kết quả tương đương hoặc tốt hơn

target_metrics:
  accuracy: >= 0.97
  f1_macro: >= 0.94
```

#### exp_007: CatBoost

```yaml
hyperparameters:
  iterations: 300
  learning_rate: 0.05
  depth: 6
  l2_leaf_reg: 3
  auto_class_weights: Balanced
  random_state: 123
  
expected_benefits:
  - Native handling categorical features
  - Robust to overfitting
  - Ordered boosting giảm bias
```

#### exp_009: Stacking Ensemble

```yaml
base_models:
  - Random Forest (n_estimators=200, max_depth=16)
  - XGBoost (tuned from exp_005)
  - LightGBM (from exp_006)
  - Extra Trees (n_estimators=200)
  
meta_learner:
  type: LogisticRegression
  C: 1.0
  
cv_strategy:
  n_folds: 5
  stratified: true
  
expected_benefits:
  - Leverage strengths of multiple models
  - Reduce variance through averaging
  - Meta-learner learns optimal combination
```

#### exp_010: Class-Specific Thresholds

```yaml
current_approach: argmax
proposed_approach: optimized_thresholds

optimization:
  method: grid_search
  metric: f1_macro
  per_class: true
  
target:
  - BruteForce: tăng precision từ 76% lên 85%+
  - Normal: tăng precision từ 81% lên 88%+
  
note: "Trade-off precision/recall cần được cân nhắc kỹ"
```

### 3.4 Timeline đề xuất

```
Week 1:
├── Day 1-2: exp_006 LightGBM
├── Day 3-4: exp_007 CatBoost  
└── Day 5: So sánh và chọn model tốt nhất

Week 2:
├── Day 1-3: exp_009 Stacking Ensemble
├── Day 4-5: exp_010 Class-Specific Thresholds

Week 3:
├── Day 1-2: exp_011 Cost-Sensitive Learning
├── Day 3-5: Final evaluation và hyperparameter tuning

Week 4:
├── Final evaluation trên test set
├── Viết báo cáo
└── Backup và documentation
```

---

## 4. CHECKLIST TRƯỚC KHI CHẠY EXPERIMENT MỚI

### 4.1 Trước khi train

```markdown
- [ ] Backup code hiện tại
- [ ] Clone best config làm baseline
- [ ] Ghi rõ hypothesis của experiment
- [ ] Xác định metric để so sánh (primary + secondary)
- [ ] Đặt ngưỡng improvement mong đợi
- [ ] Chuẩn bị data pipeline (preprocessing phải consistent)
```

### 4.2 Sau khi train

```markdown
- [ ] Lưu tất cả metrics vào structured format
- [ ] So sánh với baseline (exp_005)
- [ ] Phân tích per-class performance
- [ ] Ghi chú observations và learnings
- [ ] Update EXPERIMENTS_REGISTRY.yaml
- [ ] Nếu improvement: update best model
- [ ] Nếu không improvement: ghi rõ lý do và lessons learned
```

### 4.3 Questions cần trả lời trong mỗi experiment

1. Experiment này thay đổi gì so với baseline?
2. Hypothesis là gì?
3. Kết quả có confirm hay reject hypothesis không?
4. Lessons learned là gì?
5. Có nên tiếp tục hướng này không?

---

## 5. APPENDIX: CURRENT BEST CONFIG (exp_005)

```yaml
# Hyperparameters
learning_rate: 0.08
max_iter: 150
max_depth: 6
l2_regularization: 0.1
class_weight: balanced
random_state: 123

# Data Info
features_count: 25
train_samples: 204,412
test_samples: 51,103
classes: [BruteForce, DDoS, DoS, Mirai, Normal, Recon, Spoofing, Web-Based]

# Best Metrics (Test Set)
accuracy: 0.9693
f1_macro: 0.9370
roc_auc: 0.9986
precision_macro: 0.9271
recall_macro: 0.9492

# Latency
median_latency_ms: 10.51
throughput_samples_per_sec: 9,063.7
```

---

## 6. RECOMMENDED NAMING CONVENTION

### Experiment IDs
```
exp_XXX_<short_description>
- exp_005_xgboost_tuned
- exp_006_lightgbm_baseline
- exp_007_catboost_tuned
- exp_008_xgb_feature_eng
- exp_009_stacking_ensemble
- exp_010_threshold_optimized
```

### Evaluation IDs
```
eval_XXX_<model_or_exp_id>
- eval_005_exp_005_xgboost_tuned
- eval_006_exp_006_lightgbm_baseline
```

### File naming
```
{metric_type}_{experiment_id}_{fold/seed}.{ext}
- metrics_exp_005_fold1.json
- classification_report_eval_005_test.json
- confusion_matrix_exp_005_seed123.png
```

---

**Document Status:** Draft v1.0  
**Last Updated:** 2026-09-27  
**Author:** Claude Code Assistant
