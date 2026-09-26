# CICIoT2023 Network Intrusion Detection - Data Science Plan

## 1. Project Overview

| Item | Details |
|------|---------|
| **Dataset** | CICIoT2023 (Canadian Institute for Cybersecurity IoT 2023) |
| **Task** | Multi-class Network Intrusion Detection |
| **Rows** | ~5.5M |
| **Features** | 46 |
| **Target** | `label` (35+ attack types) |
| **Output** | Trained classification model + visualizations |

---

## 2. Dataset Summary

### 2.1 Feature Categories

| Category | Features | Description |
|----------|----------|-------------|
| **Flow Stats** | `flow_duration`, `Rate`, `Srate`, `Drate` | Traffic flow metrics |
| **Packet Flags** | `fin_flag_number`, `syn_flag_number`, `rst_flag_number`, etc. | TCP flag counts |
| **Protocol Ratios** | `HTTP`, `HTTPS`, `DNS`, `Telnet`, `SMTP`, `SSH`, `IRC`, `TCP`, `UDP`, `DHCP`, `ARP`, `ICMP`, `IPv`, `LLC` | Protocol distribution |
| **Statistical** | `Tot sum`, `Min`, `Max`, `AVG`, `Std`, `Tot size`, `IAT`, `Number` | Aggregate statistics |
| **Magnitude** | `Magnitue`, `Radius`, `Covariance`, `Variance`, `Weight` | Distribution metrics |
| **Header** | `Header_Length`, `Protocol Type`, `Duration` | Packet header info |

### 2.2 Attack Types (Target Classes)

| Category | Attack Types |
|----------|-------------|
| **DDoS** | DDoS-SYN_Flood, DDoS-TCP_Flood, DDoS-ICMP_Flood, DDoS-UDP_Flood, DDoS-ACK_Fragmentation, DDoS-PSHACK_Flood, DDoS-RSTFINFlood, DDoS-SynonymousIP_Flood |
| **DoS** | DoS-SYN_Flood, DoS-TCP_Flood, DoS-UDP_Flood |
| **Mirai** | Mirai-greeth_flood, Mirai-greip_flood, Mirai-udpplain |
| **Recon** | Recon-HostDiscovery |
| **Benign** | BenignTraffic (normal traffic) |

---

## 3. Implementation Phases

### Phase 1: Data Loading & EDA
**Goal:** Understand data structure and quality

```
Tasks:
├── 1.1 Load data with optimized dtypes
├── 1.2 Basic info (shape, dtypes, memory)
├── 1.3 Missing values analysis
├── 1.4 Duplicate rows check
├── 1.5 Statistical summary (describe())
├── 1.6 Class distribution visualization
└── 1.7 Feature correlation analysis
```

**Deliverables:**
- `01_eda.py` - EDA script
- `01_eda_report.html` - Visualized report

---

### Phase 2: Data Preprocessing
**Goal:** Clean and prepare data for modeling

```
Tasks:
├── 2.1 Handle missing/infinite values
├── 2.2 Remove constant/near-constant features
├── 2.3 Feature engineering
│   ├── Ratio features (packet size ratios)
│   ├── Aggregate features (flag sums)
│   └── Protocol-based groupings
├── 2.4 Group attacks into categories (optional)
├── 2.5 Encode target labels
└── 2.6 Train/test split (stratified, 80/20)
```

**Deliverables:**
- `02_preprocessing.py` - Preprocessing pipeline
- `X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl`

---

### Phase 3: Class Imbalance Handling
**Goal:** Address imbalanced class distribution

```
Tasks:
├── 3.1 Analyze class imbalance ratio
├── 3.2 Choose strategy:
│   ├── Option A: Class weights
│   ├── Option B: SMOTE oversampling
│   └── Option C: Undersampling
└── 3.3 Apply chosen strategy
```

**Deliverables:**
- `03_balance.py` - Imbalance handling script
- Class distribution visualization

---

### Phase 4: Feature Selection
**Goal:** Identify most predictive features

```
Tasks:
├── 4.1 Variance threshold filtering
├── 4.2 Correlation analysis (remove highly correlated)
├── 4.3 Model-based feature importance
│   ├── Random Forest importance
│   └── XGBoost importance
├── 4.4 Select top N features
└── 4.5 Optional: PCA for dimensionality reduction
```

**Deliverables:**
- `04_feature_selection.py`
- Feature importance plot
- Reduced feature set

---

### Phase 5: Model Training
**Goal:** Train and compare multiple models

```
Tasks:
├── 5.1 Baseline Models
│   ├── Logistic Regression
│   ├── Decision Tree
│   └── Random Forest
├── 5.2 Gradient Boosting Models
│   ├── XGBoost
│   └── LightGBM
├── 5.3 Neural Network (optional)
│   └── MLP Classifier
└── 5.4 Hyperparameter tuning (RandomizedSearchCV)
```

**Deliverables:**
- `05_train_models.py`
- Trained model files (`.joblib`)

---

### Phase 6: Model Evaluation
**Goal:** Evaluate models with comprehensive metrics

```
Tasks:
├── 6.1 Metrics calculation
│   ├── Accuracy
│   ├── Precision, Recall, F1-Score
│   ├── Confusion Matrix
│   └── Classification Report
├── 6.2 ROC-AUC (One-vs-Rest)
├── 6.3 Cross-validation scores
├── 6.4 Training time comparison
└── 6.5 Inference speed test
```

**Deliverables:**
- `06_evaluate.py`
- Evaluation metrics table
- Confusion matrix heatmap
- ROC curves

---

### Phase 6.5: Explainable AI (XAI)
**Goal:** Interpret model decisions for security analysts

> ⭐ **New Phase** - Critical for security use cases

```
Tasks:
├── 6.5.1 Global Feature Importance (SHAP)
│   ├── SHAP summary plot (bar/beeswarm)
│   ├── Mean absolute SHAP values per class
│   └── Top features driving each attack type
├── 6.5.2 Local Explanations (SHAP/LIME)
│   ├── SHAP force plot for individual predictions
│   ├── LIME explanations for edge cases
│   └── Counterfactual explanations
├── 6.5.3 Feature Dependence Analysis
│   ├── Partial Dependence Plots (PDP)
│   ├── SHAP interaction values heatmap
│   └── Individual Conditional Expectation (ICE)
├── 6.5.4 Attack Pattern Explanations
│   ├── Extract top contributing features per attack
│   ├── Create attack signature summaries
│   └── Compare benign vs malicious patterns
└── 6.5.5 XAI Dashboard
    ├── Interactive SHAP visualizations
    ├── Query-based explanation lookup
    └── Confidence + explanation pairing
```

**Deliverables:**
- `06b_xai.py` - XAI analysis script
- `shap_values_*.pkl` - Cached SHAP values
- `plots/shap_summary.png` - Global importance
- `plots/shap_force_*.html` - Interactive local explanations
- `plots/pdp_*.png` - Partial dependence plots
- `attack_signatures.json` - Top features per attack type

**Why XAI matters for Intrusion Detection:**
- Security analysts need to **trust** and **verify** model decisions
- Explains why specific traffic is flagged as malicious
- Helps identify potential model bias or adversarial evasion
- Supports compliance with regulations (GDPR Article 22)
- Enables continuous improvement through pattern discovery

---

### Phase 7: Visualization Dashboard
**Goal:** Create interactive visualizations

```
Tasks:
├── 7.1 Class distribution chart
├── 7.2 Feature importance bar chart
├── 7.3 Correlation heatmap
├── 7.4 Model performance comparison
├── 7.5 Confusion matrix
└── 7.6 Build dashboard (Plotly/Dash)
```

**Deliverables:**
- `dashboard/` - Interactive dashboard
- `plots/` - Static visualizations

---

### Phase 8: Final Model & Pipeline
**Goal:** Create production-ready model

```
Tasks:
├── 8.1 Select best model
├── 8.2 Retrain on full training data
├── 8.3 Save model + preprocessor
├── 8.4 Create inference pipeline
├── 8.5 Write prediction script
└── 8.6 Document API endpoints (if needed)
```

**Deliverables:**
- `best_model.joblib`
- `preprocessor.joblib`
- `predict.py` - Inference script
- `README.md` - Usage documentation

---

## 4. Technical Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **EDA** | Pandas Profiling, SweetViz |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **ML Models** | Scikit-learn, XGBoost, LightGBM |
| **Imbalance** | imbalanced-learn (SMOTE) |
| **XAI** | SHAP, LIME, DALEX |
| **Dashboard** | Plotly Dash / Streamlit |
| **Model Export** | Joblib, Pickle |

---

## 5. Expected Timeline

| Phase | Estimated Time | Priority |
|-------|---------------|----------|
| Phase 1: EDA | 2-3 hours | High |
| Phase 2: Preprocessing | 1-2 hours | High |
| Phase 3: Imbalance | 1 hour | Medium |
| Phase 4: Feature Selection | 2-3 hours | Medium |
| Phase 5: Training | 3-4 hours | High |
| Phase 6: Evaluation | 1-2 hours | High |
| Phase 6.5: XAI ⭐ | 2-3 hours | High |
| Phase 7: Visualization | 2-3 hours | Medium |
| Phase 8: Pipeline | 2-3 hours | Medium |

**Total:** ~16-23 hours

---

## 6. File Structure

```
CICIoT2023/
├── PLAN.md
├── train.csv
├── 01_eda.py
├── 02_preprocessing.py
├── 03_balance.py
├── 04_feature_selection.py
├── 05_train_models.py
├── 06_evaluate.py
├── 06b_xai.py          # NEW: XAI analysis
├── 07_visualize.py
├── 08_finalize.py
├── notebooks/
│   └── EDA_notebook.ipynb
├── models/
│   ├── best_model.joblib
│   └── preprocessor.joblib
├── plots/
│   ├── class_distribution.html
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── shap_summary.png      # NEW
│   ├── shap_force_*.html    # NEW
│   └── pdp_*.png            # NEW
├── xai/                       # NEW: XAI outputs
│   ├── shap_values.pkl
│   └── attack_signatures.json
├── dashboard/
│   └── app.py
└── README.md
```

---

## 7. Success Criteria

- [ ] Model achieves **>95% accuracy** on test set
- [ ] F1-score **>0.90** for minority classes
- [ ] Inference time **<100ms** per sample
- [ ] All attack types correctly identified
- [ ] Interactive dashboard deployed
- [ ] Reproducible pipeline documented
- [ ] ⭐ **XAI:** SHAP explanations generated for all attack types
- [ ] ⭐ **XAI:** Top contributing features identified per attack class
- [ ] ⭐ **XAI:** Interactive explanation dashboard functional

---

## 8. Notes & Considerations

1. **Memory Management:** 5.5M rows requires chunked processing
2. **Class Imbalance:** Some attacks may have <1000 samples
3. **Feature Scaling:** Required for Logistic Regression, optional for tree-based
4. **Time Constraints:** May need to subsample for initial experiments
5. **Multi-class vs Binary:** Consider both approaches (attack vs benign)

---

*Generated for CICIoT2023 Network Intrusion Detection Project*
