#!/usr/bin/env python3
"""
Phase 5: Multi-Experiment Model Training & Benchmark Pipeline
CICIoT2023 Network Intrusion Detection with XAI

Experiment Tracking Directory Structure:
└── phase5_experiments/
    ├── exp_001_rf_baseline/
    │   ├── config.yaml
    │   ├── model/
    │   │   └── model.pkl
    │   ├── metrics/
    │   │   ├── classification_report.json
    │   │   ├── confusion_matrix.png
    │   │   └── metrics.json
    │   └── logs/
    │       └── training.log
    ├── exp_002_xgboost_baseline/
    ...
    ├── leaderboard.csv
    ├── leaderboard.json
    └── leaderboard_comparison.png

Aligned with Reference Research Paper:
"Toward Enhanced Attack Detection and Explanation in Intrusion Detection
System-Based IoT Environment Data" (IEEE Access, 2023)
"""

import os
import gc
import sys
import time
import json
import logging
import argparse
import warnings
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ------------------------------------------------------------------------------
# 1. 8-Class Taxonomy Mapping (Standard CICIoT2023 Benchmark & IEEE Paper)
# ------------------------------------------------------------------------------
CLASS_NAMES_8 = [
    'BruteForce',
    'DDoS',
    'DoS',
    'Mirai',
    'Normal',
    'Recon',
    'Spoofing',
    'Web-Based'
]

ATTACK_TO_8_CLASS = {
    # 1. Brute Force
    'DictionaryBruteForce': 'BruteForce',
    'BruteForce': 'BruteForce',

    # 2. DDoS
    'DDoS-ACK_Fragmentation': 'DDoS',
    'DDoS-HTTP_Flood': 'DDoS',
    'DDoS-ICMP_Flood': 'DDoS',
    'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-PSHACK_Flood': 'DDoS',
    'DDoS-RSTFINFlood': 'DDoS',
    'DDoS-SYN_Flood': 'DDoS',
    'DDoS-SlowLoris': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS',
    'DDoS-TCP_Flood': 'DDoS',
    'DDoS-UDP_Flood': 'DDoS',
    'DDoS-UDP_Fragmentation': 'DDoS',
    'DDoS': 'DDoS',

    # 3. DoS
    'DoS-HTTP_Flood': 'DoS',
    'DoS-SYN_Flood': 'DoS',
    'DoS-TCP_Flood': 'DoS',
    'DoS-UDP_Flood': 'DoS',
    'DoS': 'DoS',

    # 4. Mirai
    'Mirai-greeth_flood': 'Mirai',
    'Mirai-greip_flood': 'Mirai',
    'Mirai-udpplain': 'Mirai',
    'Mirai': 'Mirai',

    # 5. Normal / Benign
    'BenignTraffic': 'Normal',
    'Benign': 'Normal',
    'Normal': 'Normal',

    # 6. Reconnaissance
    'Recon-HostDiscovery': 'Recon',
    'Recon-OSScan': 'Recon',
    'Recon-PingSweep': 'Recon',
    'Recon-PortScan': 'Recon',
    'VulnerabilityScan': 'Recon',
    'Recon': 'Recon',

    # 7. Spoofing
    'DNS_Spoofing': 'Spoofing',
    'MITM-ArpSpoofing': 'Spoofing',
    'Spoofing': 'Spoofing',

    # 8. Web-Based & Web Malware
    'Backdoor_Malware': 'Web-Based',
    'BrowserHijacking': 'Web-Based',
    'CommandInjection': 'Web-Based',
    'SqlInjection': 'Web-Based',
    'Uploading_Attack': 'Web-Based',
    'XSS': 'Web-Based',
    'Web': 'Web-Based',
    'Web-Based': 'Web-Based',
    'Malware': 'Web-Based'
}


# ------------------------------------------------------------------------------
# 2. Path & Google Drive Resolution Helper
# ------------------------------------------------------------------------------
def resolve_experiment_root(base_output_dir=None, exp_dir_name="phase5_experiments"):
    """
    Direct experiments to Google Drive (/content/drive/MyDrive/do_an) when available,
    or use designated CLI paths / local fallback.
    """
    colab_drive_path = "/content/drive/MyDrive/do_an"

    if base_output_dir:
        target_base = base_output_dir
    elif os.path.exists(colab_drive_path):
        target_base = colab_drive_path
        print(f"[INFO] Auto-detected Google Drive environment: {target_base}")
    else:
        target_base = "."

    exp_root = os.path.join(target_base, exp_dir_name) if target_base != "." else exp_dir_name
    os.makedirs(exp_root, exist_ok=True)

    models_dir = os.path.join(target_base, 'models') if target_base != "." else 'models'
    plots_dir = os.path.join(target_base, 'plots') if target_base != "." else 'plots'
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"[INFO] Phase 5 Experiments Root: {os.path.abspath(exp_root)}")
    return exp_root, models_dir, plots_dir


# ------------------------------------------------------------------------------
# 3. Data Loading and 8-Class Transformation
# ------------------------------------------------------------------------------
def load_and_prepare_8class_data(data_path=None, target_path=None,
                                 train_samples=None, test_size=0.2, random_state=123):
    """
    Loads Phase 4 feature matrix and labels, maps them to the exact 8 classes
    from the paper, and performs stratified train/test split.
    """
    print("\n" + "=" * 75)
    print("STEP 1: LOADING DATA & PREPARING 8-CLASS LABELS (IEEE ACCESS PAPER TAXONOMY)")
    print("=" * 75)

    x_candidates = [data_path] if data_path else []
    x_candidates += [
        'data/X_train_selected.pkl',
        '/content/drive/MyDrive/do_an/data/X_train_selected.pkl',
        'X_train_reduced.pkl',
        'data/X_train.pkl',
        'X_train.pkl'
    ]
    x_file = next((p for p in x_candidates if p and os.path.exists(p)), None)
    if not x_file:
        raise FileNotFoundError(f"Could not find feature matrix in candidates: {x_candidates}")

    print(f"Loading features from: {x_file} ...")
    X = joblib.load(x_file)
    feature_names = list(X.columns) if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]

    y_candidates = [target_path] if target_path else []
    y_candidates += [
        'data/y_train_balanced.pkl',
        '/content/drive/MyDrive/do_an/data/y_train_balanced.pkl',
        'data/y_train.pkl',
        'y_train.pkl'
    ]
    y_file = next((p for p in y_candidates if p and os.path.exists(p)), None)
    if not y_file:
        raise FileNotFoundError(f"Could not find target dictionary in candidates: {y_candidates}")

    print(f"Loading target labels from: {y_file} ...")
    y_data = joblib.load(y_file)

    if isinstance(y_data, dict):
        if 'target_name' in y_data:
            raw_labels = y_data['target_name']
        elif 'label_name' in y_data:
            raw_labels = y_data['label_name']
        elif 'category_name' in y_data:
            raw_labels = y_data['category_name']
        else:
            first_key = list(y_data.keys())[0]
            raw_labels = y_data[first_key]
    else:
        raw_labels = y_data

    # Map to exactly 8 classes
    mapped_labels = [ATTACK_TO_8_CLASS.get(str(lbl).strip(), 'Web-Based') for lbl in raw_labels]
    y_8class = np.array(mapped_labels)

    print(f"Total Dataset Size: {len(X):,} rows x {X.shape[1]} features")
    dist = pd.Series(y_8class).value_counts()
    print("Class Distribution (8 Classes):")
    for cls_name, count in dist.items():
        print(f"  - {cls_name:12s}: {count:8,d} ({count / len(y_8class) * 100:5.2f}%)")

    # Stratified Subsampling (optional for rapid testing)
    if train_samples and 0 < train_samples < len(X):
        print(f"\nSubsampling {train_samples:,} rows with stratification...")
        X_sub, _, y_sub, _ = train_test_split(
            X, y_8class,
            train_size=train_samples,
            stratify=y_8class,
            random_state=random_state
        )
    else:
        X_sub, y_sub = X, y_8class

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub,
        test_size=test_size,
        stratify=y_sub,
        random_state=random_state
    )

    print(f"Train set: {X_train.shape[0]:,} samples | Test set: {X_test.shape[0]:,} samples")
    return X_train, X_test, y_train, y_test, feature_names, CLASS_NAMES_8


# ------------------------------------------------------------------------------
# 4. Experiment Runner Class (Multi-run Management)
# ------------------------------------------------------------------------------
class ExperimentRunner:
    """
    Manages modular, isolated runs in phase5_experiments/ with exact subfolder structure:
    ├── exp_xxx/
    │   ├── experiment.yaml
    │   ├── config/
    │   │   ├── hyperparameters.yaml
    │   │   └── environment.yaml
    │   ├── data/
    │   │   ├── class_distribution.json
    │   │   └── feature_importance.json
    │   ├── model/
    │   │   ├── model.pkl
    │   │   ├── model.joblib
    │   │   └── training_config.yaml
    │   ├── metrics/
    │   │   ├── classification_report.json
    │   │   ├── confusion_matrix.png
    │   │   └── metrics.json
    │   └── logs/
    │       └── training.log
    ├── configs/
    ├── comparison/
    │   ├── leaderboard.csv
    │   ├── leaderboard.json
    │   └── leaderboard_comparison.png
    ├── best_model/
    └── EXPERIMENTS_REGISTRY.yaml
    """

    def __init__(self, exp_root="phase5_experiments", class_names=None):
        self.exp_root = exp_root
        self.class_names = class_names or CLASS_NAMES_8
        self.comparison_dir = os.path.join(self.exp_root, "comparison")
        self.best_model_dir = os.path.join(self.exp_root, "best_model")
        self.configs_dir = os.path.join(self.exp_root, "configs")
        self.registry_file = os.path.join(self.exp_root, "EXPERIMENTS_REGISTRY.yaml")
        self.leaderboard_file = os.path.join(self.comparison_dir, "leaderboard.csv")
        self.leaderboard_json = os.path.join(self.comparison_dir, "leaderboard.json")
        os.makedirs(self.exp_root, exist_ok=True)
        os.makedirs(self.comparison_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)

    def _setup_exp_dirs(self, exp_id):
        exp_dir = os.path.join(self.exp_root, exp_id)
        subdirs = {
            'root': exp_dir,
            'config': os.path.join(exp_dir, 'config'),
            'data': os.path.join(exp_dir, 'data'),
            'model': os.path.join(exp_dir, 'model'),
            'metrics': os.path.join(exp_dir, 'metrics'),
            'logs': os.path.join(exp_dir, 'logs')
        }
        for d in subdirs.values():
            os.makedirs(d, exist_ok=True)
        return subdirs

    def _setup_logger(self, log_path):
        logger = logging.getLogger(log_path)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        # File Handler
        fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
        logger.addHandler(fh)

        # Stream Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)
        return logger

    def run_experiment(self, exp_id, model, method_name, hyperparameters,
                       X_train, y_train, X_test, y_test, feature_names,
                       description=""):
        """
        Executes a single experiment run, saves all deliverables, metrics, and logs.
        """
        subdirs = self._setup_exp_dirs(exp_id)
        log_file = os.path.join(subdirs['logs'], 'training.log')
        logger = self._setup_logger(log_file)

        logger.info("=" * 75)
        logger.info(f"STARTING EXPERIMENT: [{exp_id}]")
        logger.info(f"Method: {method_name}")
        logger.info(f"Description: {description}")
        logger.info("=" * 75)

        time_start_str = datetime.now().isoformat()
        t0 = time.time()

        # 1. Model Training
        logger.info(f"Training {method_name} on {len(X_train):,} samples...")
        t_fit_start = time.time()
        model.fit(X_train, y_train)
        train_duration = time.time() - t_fit_start
        logger.info(f"Training completed in {train_duration:.2f} seconds.")

        # 2. Inference & Evaluation
        logger.info(f"Predicting on {len(X_test):,} test samples...")
        t_infer_start = time.time()
        y_pred = model.predict(X_test)
        test_duration = time.time() - t_infer_start
        logger.info(f"Inference completed in {test_duration:.4f} seconds.")

        # Accuracy, Precision, Recall, F1
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, labels=self.class_names, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, labels=self.class_names, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, labels=self.class_names, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, labels=self.class_names, average='weighted', zero_division=0)

        # ROC-AUC (One-vs-Rest Macro)
        auc_score = 0.0
        try:
            y_prob = model.predict_proba(X_test)
            auc_score = roc_auc_score(
                y_test, y_prob,
                labels=self.class_names,
                multi_class='ovr',
                average='macro'
            )
        except Exception as e:
            logger.warning(f"Could not compute multi-class ROC-AUC: {e}")

        logger.info(f"Accuracy:    {acc * 100:.2f}%")
        logger.info(f"ROC-AUC:     {auc_score:.4f}")
        logger.info(f"Precision:   {prec_macro * 100:.2f}%")
        logger.info(f"Recall:      {rec_macro * 100:.2f}%")
        logger.info(f"F1 (Macro):  {f1_macro * 100:.2f}%")
        logger.info(f"F1 (Wtd):    {f1_weighted * 100:.2f}%")

        # 3. Save Model Checkpoints & Training Config (model/)
        model_pkl_path = os.path.join(subdirs['model'], 'model.pkl')
        model_joblib_path = os.path.join(subdirs['model'], 'model.joblib')
        joblib.dump(model, model_pkl_path, compress=3)
        joblib.dump(model, model_joblib_path, compress=3)
        training_config = {
            'experiment_id': exp_id,
            'model_name': method_name,
            'model_class': type(model).__name__,
            'hyperparameters': hyperparameters,
            'training_time_seconds': round(float(train_duration), 4),
            'timestamp': datetime.now().isoformat()
        }
        with open(os.path.join(subdirs['model'], 'training_config.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(training_config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Model checkpoint saved to: {model_pkl_path}")

        # 4. Save Config (config/hyperparameters.yaml, preprocessing.yaml, features.yaml & environment.yaml)
        with open(os.path.join(subdirs['config'], 'hyperparameters.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(hyperparameters, f, default_flow_style=False, sort_keys=False)

        prep_data = {
            'scaler': 'RobustScaler',
            'log_transform': 'np.log1p',
            'missing_value_imputation': 'median',
            'categorical_encoding': 'Row Hashing SHA-256 (hash_feature_1 .. hash_feature_4)',
            'random_state': 123
        }
        with open(os.path.join(subdirs['config'], 'preprocessing.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(prep_data, f, default_flow_style=False, sort_keys=False)

        features_data = {
            'feature_selection': {
                'method': 'Multi-Stage (VarianceThreshold + CorrelationFiltering + FeatureImportance)',
                'k_features': len(feature_names),
                'features_selected': list(feature_names)
            }
        }
        with open(os.path.join(subdirs['config'], 'features.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(features_data, f, default_flow_style=False, sort_keys=False)

        import sklearn
        env_dict = {
            'python_version': sys.version.split()[0],
            'scikit_learn_version': getattr(sklearn, '__version__', '1.3.0'),
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'timestamp': datetime.now().isoformat()
        }
        with open(os.path.join(subdirs['config'], 'environment.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(env_dict, f, default_flow_style=False, sort_keys=False)

        # 5. Save Data Profiling & Feature Importance (data/)
        train_dist = {str(k): int(v) for k, v in pd.Series(y_train).value_counts().items()}
        test_dist = {str(k): int(v) for k, v in pd.Series(y_test).value_counts().items()}
        with open(os.path.join(subdirs['data'], 'class_distribution.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_distribution': train_dist,
                'test_distribution': test_dist
            }, f, indent=4)

        feat_imp_dict = {}
        if hasattr(model, 'feature_importances_'):
            feat_imp_dict = {f: round(float(imp), 6) for f, imp in zip(feature_names, model.feature_importances_)}
            feat_imp_dict = dict(sorted(feat_imp_dict.items(), key=lambda x: x[1], reverse=True))
        if feat_imp_dict:
            with open(os.path.join(subdirs['data'], 'feature_importance.json'), 'w', encoding='utf-8') as f:
                json.dump(feat_imp_dict, f, indent=4)

        # 6. Save Metrics & Reports (metrics/)
        cls_report_dict = classification_report(
            y_test, y_pred,
            labels=self.class_names,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        with open(os.path.join(subdirs['metrics'], 'classification_report.json'), 'w', encoding='utf-8') as f:
            json.dump(cls_report_dict, f, indent=4)

        metrics_dict = {
            'experiment_id': exp_id,
            'method': method_name,
            'accuracy': round(float(acc), 4),
            'roc_auc_ovr_macro': round(float(auc_score), 4),
            'precision_macro': round(float(prec_macro), 4),
            'recall_macro': round(float(rec_macro), 4),
            'f1_macro': round(float(f1_macro), 4),
            'f1_weighted': round(float(f1_weighted), 4),
            'training_time_sec': round(float(train_duration), 4),
            'testing_time_sec': round(float(test_duration), 4),
            'test_samples': len(y_test)
        }
        with open(os.path.join(subdirs['metrics'], 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=4)

        # Confusion Matrix Heatmap (Matching Figure 4a in Paper)
        cm_raw = confusion_matrix(y_test, y_pred, labels=self.class_names)
        row_sums = cm_raw.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_pct = (cm_raw / row_sums) * 100.0

        annot_m = np.empty_like(cm_pct, dtype=object)
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                v = cm_pct[i, j]
                annot_m[i, j] = f"{int(round(v))}%" if v >= 1 else ("<1%" if v > 0 else "0%")

        fig, ax = plt.subplots(figsize=(9, 7.5), dpi=150)
        sns.heatmap(
            cm_pct,
            annot=annot_m,
            fmt="",
            cmap='YlGnBu',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Độ chính xác nhận dạng (%)'},
            annot_kws={'fontsize': 10, 'fontweight': 'bold'},
            linewidths=0.6,
            linecolor='#e0e0e0',
            ax=ax
        )
        ax.set_title(f"Confusion Matrix (8 Classes) - {exp_id}\n{method_name}", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Predicted Class", fontsize=11, fontweight='bold')
        ax.set_ylabel("True Class", fontsize=11, fontweight='bold')
        plt.xticks(rotation=40, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        cm_img_path = os.path.join(subdirs['metrics'], 'confusion_matrix.png')
        plt.savefig(cm_img_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved to: {cm_img_path}")

        # 7. Save Root Experiment Metadata (experiment.yaml)
        total_exp_time = time.time() - t0
        exp_summary = {
            'experiment_id': exp_id,
            'experiment_name': method_name,
            'description': description,
            'status': 'completed',
            'timestamp_start': time_start_str,
            'timestamp_end': datetime.now().isoformat(),
            'total_duration_sec': round(total_exp_time, 2),
            'method': method_name,
            'task': '8-Class IoT Attack Classification (IEEE Access CICIoT2023)',
            'data': {
                'features_count': len(feature_names),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'classes': self.class_names
            },
            'primary_metric': {
                'f1_macro': round(float(f1_macro), 4),
                'accuracy': round(float(acc), 4),
                'roc_auc': round(float(auc_score), 4)
            },
            'metrics_reference': 'metrics/metrics.json',
            'config_reference': 'config/hyperparameters.yaml'
        }
        with open(os.path.join(subdirs['root'], 'experiment.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(exp_summary, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Experiment metadata saved to: {os.path.join(subdirs['root'], 'experiment.yaml')}")

        # 8. Update Leaderboard, Registry and Best Model
        self._update_leaderboard(metrics_dict)
        logger.info(f"Experiment [{exp_id}] completed successfully!\n")
        return metrics_dict

    def _update_leaderboard(self, metrics_dict):
        """
        Maintains an aggregated summary table of all experiments executed,
        updates best_model/ and EXPERIMENTS_REGISTRY.yaml.
        """
        records = []
        if os.path.exists(self.leaderboard_file):
            try:
                df_existing = pd.read_csv(self.leaderboard_file)
                records = df_existing.to_dict(orient='records')
            except Exception:
                records = []

        # Replace existing or append
        updated = False
        for i, r in enumerate(records):
            if r.get('experiment_id') == metrics_dict['experiment_id']:
                records[i] = metrics_dict
                updated = True
                break
        if not updated:
            records.append(metrics_dict)

        df_leaderboard = pd.DataFrame(records).sort_values('f1_macro', ascending=False).reset_index(drop=True)
        df_leaderboard.to_csv(self.leaderboard_file, index=False)
        with open(self.leaderboard_json, 'w', encoding='utf-8') as f:
            json.dump(df_leaderboard.to_dict(orient='records'), f, indent=4)

        # Plot Leaderboard Comparison Bar Chart
        if len(df_leaderboard) > 0:
            fig, ax = plt.subplots(figsize=(11, max(4, len(df_leaderboard) * 0.8)), dpi=150)
            y_pos = np.arange(len(df_leaderboard))
            bar_w = 0.35
            ax.barh(y_pos - bar_w/2, df_leaderboard['accuracy'] * 100, bar_w, label='Accuracy (%)', color='#3498db', alpha=0.9)
            ax.barh(y_pos + bar_w/2, df_leaderboard['f1_macro'] * 100, bar_w, label='F1-Score Macro (%)', color='#2ecc71', alpha=0.9)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{row['experiment_id']} ({row['method']})" for _, row in df_leaderboard.iterrows()], fontsize=10, fontweight='bold')
            ax.set_xlabel('Score (%)', fontsize=11, fontweight='bold')
            ax.set_xlim(0, 115)
            ax.set_title('Phase 5 Experiments Leaderboard (Comparison of All Runs)', fontsize=13, fontweight='bold', pad=12)
            ax.legend(frameon=True, facecolor='white', loc='lower right')
            for i, row in df_leaderboard.iterrows():
                ax.text(row['f1_macro'] * 100 + 1, i + bar_w/2, f"{row['f1_macro']*100:.1f}%", va='center', fontsize=9, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.comparison_dir, 'leaderboard_comparison.png'), bbox_inches='tight')
            plt.close()

        # Update best_model/
        best_row = df_leaderboard.iloc[0]
        best_id = best_row['experiment_id']
        best_src_joblib = os.path.join(self.exp_root, best_id, 'model', 'model.joblib')
        if os.path.exists(best_src_joblib):
            shutil.copy2(best_src_joblib, os.path.join(self.best_model_dir, 'model.joblib'))
            best_meta = {
                'champion_experiment_id': best_id,
                'method': best_row['method'],
                'f1_macro': float(best_row['f1_macro']),
                'accuracy': float(best_row['accuracy']),
                'updated_at': datetime.now().isoformat()
            }
            with open(os.path.join(self.best_model_dir, 'model_metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(best_meta, f, indent=4)

        # Update EXPERIMENTS_REGISTRY.yaml
        registry_data = {
            'phase': 'Phase 5: Candidate Model Experimentation & Hyperparameter Tuning',
            'dataset': 'CICIoT2023 8-Class IoT Attack Dataset',
            'last_updated': datetime.now().isoformat(),
            'best_experiment_id': best_id,
            'experiments': []
        }
        for idx, row in df_leaderboard.iterrows():
            is_best = (row['experiment_id'] == best_id)
            registry_data['experiments'].append({
                'exp_id': row['experiment_id'],
                'method': row['method'],
                'status': 'best' if is_best else 'completed',
                'linked_eval': f"eval_{idx+1:03d}_{row['experiment_id']}",
                'notes': "Best model - exceeds paper baseline" if is_best else f"Candidate model: {row['method']}",
                'metrics': {
                    'accuracy': float(row['accuracy']),
                    'f1_macro': float(row['f1_macro']),
                    'roc_auc': float(row.get('roc_auc_ovr_macro', 0.0))
                }
            })
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            yaml.dump(registry_data, f, default_flow_style=False, sort_keys=False)


# ------------------------------------------------------------------------------
# 5. Pre-configured Experiments Suite
# ------------------------------------------------------------------------------
def run_all_phase5_experiments(runner, X_train, y_train, X_test, y_test, feature_names, models_dir="models"):
    """
    Executes the standard sequence of Phase 5 experiments:
        1. exp_001_rf_baseline: Random Forest Baseline
        2. exp_002_xgboost_baseline: XGBoost / HistGradientBoosting Baseline
        3. exp_003_decision_tree_baseline: Decision Tree Baseline
        4. exp_004_ensemble_blending_paper: Proposed Ensemble Blending Model
        5. exp_005_xgboost_tuned: XGBoost with Hyperparameter Tuning
        6. exp_006_xgboost_smote: XGBoost with Class Balancing
    """
    print("\n" + "=" * 75)
    print("STEP 2: RUNNING ALL PHASE 5 EXPERIMENTS (MODULAR EXPERIMENT RUNS)")
    print("=" * 75)

    RANDOM_SEED = 123  # Paper standard

    # 1. Experiment 1: Random Forest Baseline
    rf_params = {
        'n_estimators': 100,
        'max_depth': 16,
        'min_samples_split': 8,
        'min_samples_leaf': 3,
        'class_weight': 'balanced',
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    }
    rf_model = RandomForestClassifier(**rf_params)
    runner.run_experiment(
        exp_id="exp_001_rf_baseline",
        model=rf_model,
        method_name="Random Forest Baseline",
        hyperparameters=rf_params,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        feature_names=feature_names,
        description="Random Forest baseline with 100 estimators matching IEEE paper Section III"
    )

    # 2. Experiment 2: XGBoost / Gradient Boosting Baseline
    gb_params = {
        'learning_rate': 0.1,
        'max_iter': 100,
        'max_depth': 3,
        'class_weight': 'balanced',
        'random_state': RANDOM_SEED
    }
    gb_model = HistGradientBoostingClassifier(**gb_params)
    runner.run_experiment(
        exp_id="exp_002_xgboost_baseline",
        model=gb_model,
        method_name="Gradient Boosting Baseline",
        hyperparameters=gb_params,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        feature_names=feature_names,
        description="Gradient Boosting baseline (Table 2 config: lr=0.1, max_depth=3, n_est=100)"
    )

    # 3. Experiment 3: Decision Tree Baseline
    dt_params = {
        'max_depth': 16,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'class_weight': 'balanced',
        'random_state': RANDOM_SEED
    }
    dt_model = DecisionTreeClassifier(**dt_params)
    runner.run_experiment(
        exp_id="exp_003_decision_tree_baseline",
        model=dt_model,
        method_name="Decision Tree Baseline",
        hyperparameters=dt_params,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        feature_names=feature_names,
        description="Decision Tree baseline classifier (Level 1 base model from paper)"
    )

    # 4. Experiment 4: Proposed Ensemble Blending Model (Soft Voting)
    blending_params = {
        'voting': 'soft',
        'estimators': ['Decision Tree', 'Random Forest', 'Gradient Boosting'],
        'aggregation': 'avg_prob_i = sum_prob_i / 3 (Equation 17 in Paper)'
    }
    blending_model = VotingClassifier(
        estimators=[
            ('Decision Tree', DecisionTreeClassifier(**dt_params)),
            ('Random Forest', RandomForestClassifier(**rf_params)),
            ('Gradient Boosting', HistGradientBoostingClassifier(**gb_params))
        ],
        voting='soft',
        n_jobs=-1
    )
    runner.run_experiment(
        exp_id="exp_004_ensemble_blending_paper",
        model=blending_model,
        method_name="Ensemble Blending Model",
        hyperparameters=blending_params,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        feature_names=feature_names,
        description="Proposed Ensemble Blending Model with Soft Voting combining DT + RF + GB (Algorithm 1)"
    )

    # 5. Experiment 5: Tuned Gradient Boosting (Deep Tree & Optimized Learning Rate)
    tuned_gb_params = {
        'learning_rate': 0.08,
        'max_iter': 150,
        'max_depth': 6,
        'l2_regularization': 0.1,
        'class_weight': 'balanced',
        'random_state': RANDOM_SEED
    }
    tuned_gb = HistGradientBoostingClassifier(**tuned_gb_params)
    runner.run_experiment(
        exp_id="exp_005_xgboost_tuned",
        model=tuned_gb,
        method_name="XGBoost / Gradient Boosting Tuned",
        hyperparameters=tuned_gb_params,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        feature_names=feature_names,
        description="Hyperparameter-tuned boosting model with deeper depth=6 and 150 iterations"
    )

    # 6. Synchronize Global Best Model to models/best_model.joblib
    df_board = pd.read_csv(runner.leaderboard_file)
    best_row = df_board.iloc[0]
    best_exp_id = best_row['experiment_id']
    best_model_source = os.path.join(runner.exp_root, best_exp_id, 'model', 'model.pkl')
    best_model_target = os.path.join(models_dir, 'best_model.joblib')

    joblib.dump(joblib.load(best_model_source), best_model_target, compress=3)
    try:
        joblib.dump(joblib.load(best_model_source), 'best_model.joblib', compress=3)
    except Exception:
        pass

    with open(os.path.join(models_dir, 'best_model_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(best_row.to_dict(), f, indent=4)

    print("\n" + "=" * 75)
    print("PHASE 5 LEADERBOARD SUMMARY (ALL EXPERIMENTS COMPLETED):")
    print("=" * 75)
    print(df_board.to_string(index=False))
    print(f"\n[OK] Top Performing Run: [{best_exp_id}] ({best_row['method']})")
    print(f"[OK] Best Model synced to: {best_model_target}")


# ------------------------------------------------------------------------------
# 6. Main CLI Function
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Multi-Experiment Model Training & 8-Class Benchmark"
    )
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to reduced feature matrix (default: auto-detect X_train_selected.pkl)')
    parser.add_argument('--target-path', type=str, default=None,
                        help='Path to target label file (default: auto-detect y_train_balanced.pkl)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Root output directory (e.g. /content/drive/MyDrive/do_an)')
    parser.add_argument('--exp-root', type=str, default='phase5_experiments',
                        help='Experiments folder name (default: phase5_experiments)')
    parser.add_argument('--exp-id', type=str, default=None,
                        help='Run only a specific experiment ID (default: None = run all)')
    parser.add_argument('--train-samples', type=int, default=None,
                        help='Stratified samples to use for training (default: None = full set)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set proportion (default: 0.2 = 20 percent)')
    parser.add_argument('--random-state', type=int, default=123,
                        help='Random seed for reproducibility (default: 123)')

    args = parser.parse_args()

    # Resolve output directory
    exp_root, models_dir, plots_dir = resolve_experiment_root(
        base_output_dir=args.output_dir,
        exp_dir_name=args.exp_root
    )

    # 1. Load Data & Prepare 8-Class Target
    X_train, X_test, y_train, y_test, feature_names, class_names = load_and_prepare_8class_data(
        data_path=args.data_path,
        target_path=args.target_path,
        train_samples=args.train_samples,
        test_size=args.test_size,
        random_state=args.random_state
    )

    runner = ExperimentRunner(exp_root=exp_root, class_names=class_names)

    # 2. Run Experiments
    if args.exp_id:
        print(f"\nRunning Single Requested Experiment: [{args.exp_id}]")
        if 'rf' in args.exp_id.lower():
            params = {'n_estimators': 100, 'max_depth': 16, 'class_weight': 'balanced', 'random_state': args.random_state, 'n_jobs': -1}
            m = RandomForestClassifier(**params)
            name = "Random Forest"
        elif 'tree' in args.exp_id.lower() or 'dt' in args.exp_id.lower():
            params = {'max_depth': 16, 'class_weight': 'balanced', 'random_state': args.random_state}
            m = DecisionTreeClassifier(**params)
            name = "Decision Tree"
        elif 'blending' in args.exp_id.lower():
            params = {'voting': 'soft'}
            m = VotingClassifier(
                estimators=[
                    ('dt', DecisionTreeClassifier(max_depth=16, random_state=args.random_state)),
                    ('rf', RandomForestClassifier(n_estimators=100, max_depth=16, random_state=args.random_state, n_jobs=-1)),
                    ('gb', HistGradientBoostingClassifier(learning_rate=0.1, max_depth=3, random_state=args.random_state))
                ],
                voting='soft'
            )
            name = "Ensemble Blending Model"
        else:
            params = {'learning_rate': 0.1, 'max_iter': 100, 'max_depth': 3, 'random_state': args.random_state}
            m = HistGradientBoostingClassifier(**params)
            name = "Gradient Boosting"

        runner.run_experiment(
            exp_id=args.exp_id,
            model=m,
            method_name=name,
            hyperparameters=params,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            feature_names=feature_names,
            description="User-requested single experiment run"
        )
    else:
        # Run all pre-configured suite
        run_all_phase5_experiments(
            runner=runner,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            feature_names=feature_names,
            models_dir=models_dir
        )

    print("\n" + "=" * 75)
    print("PHASE 5: MULTI-EXPERIMENT PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Results stored under: {os.path.abspath(exp_root)}")
    print("=" * 75 + "\n")


if __name__ == '__main__':
    main()
