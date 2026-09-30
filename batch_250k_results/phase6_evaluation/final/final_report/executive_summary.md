# BÁO CÁO TỔNG KẾT ĐÁNH GIÁ MÔ HÌNH (PHASE 6 EXECUTIVE SUMMARY)
## Đề tài: Phát hiện và Giải thích Tấn công Mạng IoT (CICIoT2023) với XAI

---

### 1. Mô hình Được Chọn Lựa (Selected Champion Model)
- **Mã thực nghiệm Phase 5**: `exp_005_xgboost_tuned`
- **Mã đợt đánh giá Phase 6**: `eval_005_exp_005_xgboost_tuned`
- **Tên mô hình**: **XGBoost / Gradient Boosting Tuned**
- **Nguyên tắc lựa chọn**: Đạt điểm số **F1-Macro cao nhất trên tập Validation độc lập (93.91%)**, đảm bảo tính khách quan và chặn hoàn toàn hiện tượng rò rỉ dữ liệu (zero leakage).

---

### 2. Kết Quả Kiểm Thử Cuối Cùng Trên Tập Test Độc Lập
| Chỉ số Đánh giá (Metric) | Điểm số trên Tập Test | Chuẩn bài báo IEEE Access | So sánh |
| :--- | :---: | :---: | :---: |
| **Accuracy (Độ chính xác toàn thể)** | **96.93%** | N/A (Mô hình mới) | Vượt RF (+0.86%) |
| **F1-Score (Macro)** | **93.70%** | N/A (Mô hình mới) | Vượt RF (+1.36%) |
| **ROC-AUC (One-vs-Rest Macro)** | **0.9986** | N/A (Mô hình mới) | Vượt RF (+0.12%) |
| **Precision (Macro)** | **92.71%** | N/A | Tỷ lệ báo động chính xác cao |
| **Recall (Macro)** | **94.92%** | N/A | Bao phủ toàn diện các lớp tấn công |
| **Độ trễ Suy Luận (Inference Latency)** | **10.51 ms / mẫu** | Thời gian thực (Real-time) | Phù hợp triển khai Gateway IoT |
| **Throughput (Thông lượng xử lý)** | **9,063.7 mẫu / giây** | Thông lượng cao | Xử lý lưu lượng lớn |

---

### 3. Đánh Giá Độ Ổn Định Qua Nhiều Lần Chạy (Multi-Seed Stability)
Mô hình quán quân được kiểm thử độ bền vững qua các random seed `[42, 123, 2024]` trên các phân vùng kiểm thử con:
- **Accuracy**: `96.93% ± 0.05%`
- **F1_Macro**: `93.70% ± 0.12%`
- **ROC_AUC**: `0.9986 ± 0.0002`

---

### 4. Sẵn Sàng Chuyển Giao Cho Module XAI (Phase 6.5)
Trọng số mô hình tối ưu đã được đồng bộ hóa và lưu trữ tại:
1. Checkpoint cục bộ Phase 6: `phase6_evaluation/final/selected_model/model.joblib`
2. Checkpoint toàn dự án: `models/best_model.joblib`

Mô hình đã sẵn sàng 100% để trích xuất giải thích cục bộ và toàn cục thông qua **SHAP, LIME và Counterfactual Explanations**!
