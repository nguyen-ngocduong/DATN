# BÁO CÁO TỔNG KẾT ĐÁNH GIÁ MÔ HÌNH (PHASE 6 EXECUTIVE SUMMARY)
## Đề tài: Phát hiện và Giải thích Tấn công Mạng IoT (CICIoT2023) với XAI

---

### 1. Mô hình Được Chọn Lựa (Selected Champion Model)
- **Mã thực nghiệm Phase 5**: `exp_005_xgboost_tuned`
- **Mã đợt đánh giá Phase 6**: `eval_005_exp_005_xgboost_tuned`
- **Tên mô hình**: **exp_005_xgboost_tuned**
- **Nguyên tắc lựa chọn**: Đạt điểm số **F1-Macro cao nhất trên tập Validation độc lập**, đảm bảo tính khách quan và chặn hoàn toàn hiện tượng rò rỉ dữ liệu (zero leakage).

---

### 2. Kết Quả Kiểm Thử Cuối Cùng Trên Tập Test Độc Lập
| Chỉ số Đánh giá (Metric) | Điểm số trên Tập Test | Chuẩn bài báo IEEE Access |
| :--- | :---: | :---: |
| **Accuracy (Độ chính xác toàn thể)** | **97.53%** | ~99.4% (trên toàn bộ 34 lớp) |
| **F1-Score (Macro)** | **94.67%** | Thước đo chính cho tập mất cân bằng |
| **ROC-AUC (One-vs-Rest Macro)** | **0.9988** | Khả năng phân tách hoàn hảo |
| **Precision (Macro)** | **93.84%** | Tỷ lệ cảnh báo dương tính chính xác |
| **Recall (Macro)** | **95.71%** | Khả năng bao phủ tất cả 8 phân lớp |
| **Độ trễ Suy Luận (Inference Latency)** | **14.9256 ms / mẫu** | Đáp ứng thời gian thực (Real-time IoT) |

---

### 3. Đánh Giá Độ Ổn Định Qua Nhiều Lần Chạy (Multi-Seed Stability)
Mô hình quán quân được kiểm thử độ bền vững qua các random seed `[42, 123, 2024]` trên các phân vùng kiểm thử con:
- **Accuracy**: `97.55% ± 0.03%`
- **F1_Macro**: `94.69% ± 0.10%`
- **ROC_AUC**: `0.9988 ± 0.0000`

---

### 4. Sẵn Sàng Chuyển Giao Cho Module XAI (Phase 6.5)
Trọng số mô hình tối ưu đã được đồng bộ hóa và lưu trữ tại:
1. Checkpoint cục bộ Phase 6: `phase6_evaluation/final/selected_model/model.joblib`
2. Checkpoint toàn dự án: `models/best_model.joblib`

Mô hình đã sẵn sàng 100% để trích xuất giải thích cục bộ và toàn cục thông qua **SHAP, LIME và Counterfactual Explanations**!
