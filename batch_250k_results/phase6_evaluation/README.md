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
├── EVALUATION_REGISTRY.yaml           # Master index tất cả các đợt đánh giá
│
├── runs/                              # Các đợt đánh giá độc lập
│   ├── eval_001_exp_001_rf_baseline/  # Random Forest Baseline
│   │   ├── config.yaml                # Cấu hình siêu tham số, seed & liên kết Phase 5
│   │   ├── metadata.json              # Thông số hệ thống & tóm tắt scalar metrics (không trùng lặp)
│   │   ├── validation/                # Kết quả trên tập Validation (26,706 mẫu)
│   │   │   ├── confusion_matrix.png
│   │   │   ├── classification_report.json
│   │   │   ├── classification_report.txt
│   │   │   ├── roc_auc.json
│   │   │   └── predictions.csv
│   │   ├── test/                      # Kết quả trên tập Test độc lập (26,707 mẫu)
│   │   │   ├── confusion_matrix.png
│   │   │   ├── classification_report.json
│   │   │   ├── classification_report.txt
│   │   │   ├── roc_auc.json
│   │   │   └── predictions.csv
│   │   └── performance/               # Đo kiểm độ trễ suy luận & throughput
│   │       ├── latency.json
│   │       └── config.yaml
│   ├── eval_002_exp_002_xgboost_baseline/
│   ├── eval_003_exp_003_decision_tree_baseline/
│   ├── eval_004_exp_004_ensemble_blending_paper/
│   └── eval_005_exp_005_xgboost_tuned/
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
