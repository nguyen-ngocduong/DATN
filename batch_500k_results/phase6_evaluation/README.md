# Phase 6: Model Evaluation & Independent Testing Pipeline
## CICIoT2023 Network Intrusion Detection with XAI

Hệ thống đánh giá mô hình đa đợt (Multi-Run Model Evaluation) tuân thủ nghiêm ngặt nguyên tắc phân tách dữ liệu:
```text
Phase 5 (Train Candidates) -> Phase 6 (Validation-based Selection) -> Phase 6 (Independent Test Verification)
```

### Cấu Trúc Thư Mục Kết Quả (Deliverables Structure):
```text
phase6_evaluation/
│
├── README.md                          # Tài liệu hướng dẫn & tổng quan
│
├── runs/                              # Các đợt đánh giá độc lập
│   ├── eval_001/                      # Đợt đánh giá mô hình Random Forest
│   │   ├── config.yaml                # Cấu hình siêu tham số & liên kết Phase 5
│   │   ├── metadata.json              # Thông số hệ thống & thời gian
│   │   ├── validation/                # Kết quả trên tập Validation (25,551 mẫu)
│   │   │   ├── confusion_matrix.png
│   │   │   ├── classification_report.json
│   │   │   ├── classification_report.txt
│   │   │   ├── roc_auc.json
│   │   │   └── predictions.csv
│   │   ├── test/                      # Kết quả trên tập Test độc lập (25,552 mẫu)
│   │   │   ├── confusion_matrix.png
│   │   │   ├── classification_report.json
│   │   │   ├── classification_report.txt
│   │   │   ├── roc_auc.json
│   │   │   └── predictions.csv
│   │   └── performance/               # Đo kiểm độ trễ suy luận & throughput
│   │       └── latency.json
│   ├── eval_002/                      # Đợt đánh giá Gradient Boosting Baseline
│   ├── eval_003/                      # Đợt đánh giá Decision Tree Baseline
│   ├── eval_004/                      # Đợt đánh giá Ensemble Blending Model
│   └── eval_005/                      # Đợt đánh giá XGBoost / Boosting Tinh chỉnh
│
├── comparison/                        # So sánh liên đợt & bảng xếp hạng
│   ├── evaluation_summary.csv         # Bảng xếp hạng chi tiết tất cả các đợt
│   ├── evaluation_comparison.xlsx     # Báo cáo định dạng Excel nhiều sheet
│   ├── per_class_summary.csv          # Chi tiết Precision/Recall/F1 cho cả 8 lớp
│   ├── stability_summary.csv          # Đánh giá độ ổn định Mean ± Std qua 3 seed
│   └── evaluation_comparison.png      # Biểu đồ so sánh trực quan đa trục
│
└── final/                             # Mô hình quán quân được tuyển chọn
    ├── selected_model/
    │   ├── model.pkl                  # Checkpoint mô hình
    │   ├── model.joblib               # Checkpoint chuẩn nén Joblib
    │   └── model_info.yaml            # Metadata chi tiết về lý do lựa chọn
    ├── final_metrics.json             # Các chỉ số kiểm thử cuối cùng
    └── final_report/
        ├── README.md
        ├── executive_summary.md       # Báo cáo tổng kết toàn diện
        └── final_confusion_matrix.png # Ma trận nhầm lẫn chuẩn Fig. 4a bài báo
```

Mô hình chiến thắng được tự động đồng bộ sang `models/best_model.joblib` để sử dụng trực tiếp trong module **Phase 6.5 (Explainable AI - XAI)**.
