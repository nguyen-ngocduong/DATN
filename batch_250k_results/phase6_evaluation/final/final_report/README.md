# Phase 6 Final Evaluation Report & Champion Model Delivery
**Đề tài:** Phát hiện và Giải thích Tấn công Mạng IoT (CICIoT2023) với XAI  
**Chuẩn tham chiếu:** IEEE Access 2023 Research Paper Taxonomy (8 Attack Classes)

---

## 1. Mục Đích & Phạm Vi
Báo cáo này tổng kết kết quả đánh giá độc lập mô hình phát hiện xâm nhập mạng IoT (NIDS) trên bộ dữ liệu chuẩn CICIoT2023:
- Tuyệt đối tuân thủ nguyên tắc không rò rỉ dữ liệu (**Zero Data Leakage**): Tuyển chọn mô hình chiến thắng hoàn toàn dựa trên tập **Validation Set** độc lập (25,551 mẫu), sau đó kiểm định khách quan trên tập **Test Set** (25,552 mẫu).
- Toàn bộ kết quả chi tiết được tổng hợp trong [executive_summary.md](file:///home/admin123/Desktop/dataocubuntu/do_an/CICIoT2023/phase6_evaluation/final/final_report/executive_summary.md).

---

## 2. Mô Hình Quán Quân Tuyển Chọn (Champion Model)
- **Tên mô hình:** XGBoost / Gradient Boosting Tuned (`eval_005_exp_005_xgboost_tuned`)
- **Trọng số nhị phân:** [model.joblib](file:///home/admin123/Desktop/dataocubuntu/do_an/CICIoT2023/phase6_evaluation/final/selected_model/model.joblib) & `models/best_model.joblib`
- **Metadata chi tiết:** [model_info.yaml](file:///home/admin123/Desktop/dataocubuntu/do_an/CICIoT2023/phase6_evaluation/final/selected_model/model_info.yaml) & [final_metrics.json](file:///home/admin123/Desktop/dataocubuntu/do_an/CICIoT2023/phase6_evaluation/final/final_metrics.json)

### Chỉ số hiệu năng trên tập Kiểm Thử Độc Lập (Test Set):
| Chỉ số | Điểm số Đạt được | So sánh Bài Báo IEEE Access (Table 3) |
| :--- | :--- | :--- |
| **Accuracy** | **96.93%** | Cao hơn đáng kể (Bài báo: ~92.5%) |
| **Macro F1-Score** | **93.70%** | Vượt trội trên các lớp tấn công hiếm |
| **ROC-AUC (Macro OvR)** | **0.9986** | Khả năng phân tách nhãn gần như tuyệt đối |
| **Median Inference Latency** | **10.51 ms** | Đạt yêu cầu giám sát mạng IoT thời gian thực |
| **Throughput** | **95,148 samples/sec** | Tốc độ xử lý lưu lượng mạng cực nhanh |

---

## 3. Danh Mục Tệp Bàn Giao
1. `final_confusion_matrix.png`: Ma trận nhầm lẫn 8 lớp của mô hình quán quân trên tập kiểm thử độc lập.
2. `executive_summary.md`: Báo cáo tóm tắt điều hành trình bày trước hội đồng bảo vệ đồ án tốt nghiệp.
3. `selected_model/`: Thư mục lưu trữ model binary và cấu hình siêu tham số sẵn sàng kết nối trực tiếp vào module XAI (Phase 7 - LIME & Counterfactuals).
