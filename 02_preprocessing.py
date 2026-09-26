#!/usr/bin/env python3
"""
Phase 2: Data Preprocessing Pipeline (Training Set Only)
CICIoT2023 Network Intrusion Detection
Graduation Thesis: IoT Attack Detection & Explanation with XAI
Plan: PLAN.md (Phase 2)

Refactored & Hardened Version Addressing:
1. Bias-Free Stratified Sampling (Replaced biased nrows sampling with stratified splitting)
2. Exact Duplicate Flow Detection & Removal (Eliminates data leakage and inflated metrics)
3. Centralized Feature Engineering (Shared engineer_features() for train/val/test consistency)
4. Robust Extreme Value Compression (log1p clipping for ratio features to stabilize distance metrics)
5. Consistent Unknown Label Policy (Explicit filtering & logging instead of silent erroneous mapping)
6. Consistent Median Imputation across train and test sets
7. Precise Categorization: 34 Fine-Grained Classes (33 Attacks + Benign) & 9 Categories (8 Groups + Benign)
"""

import os
import gc
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# Attack Categorization Dictionary (All 33 Attacks in CICIoT2023 + Benign = 34 Classes)
# Mapped to 9 High-Level Categories (8 Attack Groups + Benign)
# ------------------------------------------------------------------------------
ATTACK_CATEGORY_MAPPING = {
    # 1. DDoS attacks (12 types)
    'DDoS-ICMP_Flood': 'DDoS',
    'DDoS-UDP_Flood': 'DDoS',
    'DDoS-TCP_Flood': 'DDoS',
    'DDoS-PSHACK_Flood': 'DDoS',
    'DDoS-SYN_Flood': 'DDoS',
    'DDoS-RSTFINFlood': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS',
    'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-UDP_Fragmentation': 'DDoS',
    'DDoS-ACK_Fragmentation': 'DDoS',
    'DDoS-HTTP_Flood': 'DDoS',
    'DDoS-SlowLoris': 'DDoS',

    # 2. DoS attacks (4 types)
    'DoS-UDP_Flood': 'DoS',
    'DoS-TCP_Flood': 'DoS',
    'DoS-SYN_Flood': 'DoS',
    'DoS-HTTP_Flood': 'DoS',

    # 3. Mirai attacks (3 types)
    'Mirai-greeth_flood': 'Mirai',
    'Mirai-udpplain': 'Mirai',
    'Mirai-greip_flood': 'Mirai',

    # 4. Reconnaissance & Scanning (5 types)
    'Recon-HostDiscovery': 'Recon',
    'Recon-OSScan': 'Recon',
    'Recon-PortScan': 'Recon',
    'Recon-PingSweep': 'Recon',
    'VulnerabilityScan': 'Recon',

    # 5. Spoofing & MITM (2 types)
    'MITM-ArpSpoofing': 'Spoofing',
    'DNS_Spoofing': 'Spoofing',

    # 6. Web-based attacks (5 types)
    'BrowserHijacking': 'Web',
    'CommandInjection': 'Web',
    'SqlInjection': 'Web',
    'XSS': 'Web',
    'Uploading_Attack': 'Web',

    # 7. Brute Force (1 type)
    'DictionaryBruteForce': 'BruteForce',

    # 8. Malware (1 type)
    'Backdoor_Malware': 'Malware',

    # 9. Benign Traffic
    'BenignTraffic': 'Benign',
}

DTYPE_DICT = {
    'flow_duration': 'float32', 'Header_Length': 'float32', 'Protocol Type': 'float32',
    'Duration': 'float32', 'Rate': 'float32', 'Srate': 'float32', 'Drate': 'float32',
    'fin_flag_number': 'float32', 'syn_flag_number': 'float32', 'rst_flag_number': 'float32',
    'psh_flag_number': 'float32', 'ack_flag_number': 'float32', 'ece_flag_number': 'float32',
    'cwr_flag_number': 'float32', 'ack_count': 'float32', 'syn_count': 'float32',
    'fin_count': 'float32', 'urg_count': 'float32', 'rst_count': 'float32',
    'HTTP': 'float32', 'HTTPS': 'float32', 'DNS': 'float32', 'Telnet': 'float32',
    'SMTP': 'float32', 'SSH': 'float32', 'IRC': 'float32', 'TCP': 'float32',
    'UDP': 'float32', 'DHCP': 'float32', 'ARP': 'float32', 'ICMP': 'float32',
    'IPv': 'float32', 'LLC': 'float32', 'Tot sum': 'float32', 'Min': 'float32',
    'Max': 'float32', 'AVG': 'float32', 'Std': 'float32', 'Tot size': 'float32',
    'IAT': 'float32', 'Number': 'float32', 'Magnitue': 'float32', 'Radius': 'float32',
    'Covariance': 'float32', 'Variance': 'float32', 'Weight': 'float32',
    'label': 'object'
}


def find_data_file(custom_path=None):
    """Locate train.csv file across common candidate paths."""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    
    candidates = [
        'data/X_train.parquet',
        'train.csv',
        'plots/train.csv',
        '../train.csv',
        '/content/drive/MyDrive/do_an/train.csv'
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find dataset in candidates: {candidates}")


def load_data(file_path, sample_size=None, stratify_col='label', random_state=42):
    """
    Load dataset with memory-optimized 32-bit floats.
    Fixes Bug #1: Uses Stratified Sampling instead of biased nrows reading.
    """
    print(f"Loading data from: {file_path}...")
    
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        # Load complete CSV to avoid ordering bias across attack chunks
        df = pd.read_csv(file_path, dtype=DTYPE_DICT)
    
    ram_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Initial raw data: {df.shape[0]:,} rows x {df.shape[1]} cols | RAM: {ram_mb:.2f} MB")

    # If subsampling requested, perform STRATIFIED sampling to preserve attack class ratios
    if sample_size and sample_size < len(df):
        print(f"Drawing representative stratified subsample: {sample_size:,} / {len(df):,} rows...")
        if stratify_col in df.columns:
            df, _ = train_test_split(
                df,
                train_size=sample_size,
                stratify=df[stratify_col],
                random_state=random_state
            )
        else:
            df = df.sample(n=sample_size, random_state=random_state)
        df.reset_index(drop=True, inplace=True)
        print(f"Stratified sample obtained: {df.shape[0]:,} rows preserving {df[stratify_col].nunique()} classes.")

    gc.collect()
    return df


def clean_missing_and_inf(df):
    """
    Task 2.1: 
    1. Detect and remove exact duplicate flow rows (prevents inflated metrics and train-test leakage).
    2. Detect and replace infinite values.
    3. Compute training column medians and impute NaNs.
    """
    print("\n--- Task 2.1: Data Cleaning (Deduplication, Infs & Missing Values) ---")
    
    # 1. Deduplication (Critical fix for CICIoT2023)
    n_initial = len(df)
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        pct_dup = (n_duplicates / n_initial) * 100
        print(f"Detected {n_duplicates:,} duplicate flow rows ({pct_dup:.2f}% of data). Dropping duplicates...")
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"Deduplicated cleanly! Retained: {len(df):,} unique flows.")
    else:
        print("No duplicate flow rows detected.")

    # 2. Check and clean Inf values
    numeric_cols = df.select_dtypes(include=['float32', 'float64']).columns.tolist()
    inf_count = np.isinf(df[numeric_cols].values).sum()
    if inf_count > 0:
        print(f"Detected {inf_count:,} infinite values. Replacing with NaN...")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    else:
        print("No infinite values found.")

    # 3. Calculate medians on clean training data
    imputation_medians = {}
    for col in numeric_cols:
        med = float(df[col].median())
        imputation_medians[col] = med

    # 4. Impute missing values with median
    nan_counts = df.isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0].index.tolist()
    if cols_with_nan:
        print(f"Imputing NaNs in {len(cols_with_nan)} columns using training medians...")
        for col in cols_with_nan:
            if col in numeric_cols:
                df[col] = df[col].fillna(imputation_medians[col])
    else:
        print("No missing values found in training set.")

    if 'label' in df.columns and df['label'].isnull().sum() > 0:
        df.dropna(subset=['label'], inplace=True)

    assert df.isnull().sum().sum() == 0, "NaN values still present!"
    print("Training dataset is 100% clean of duplicates and missing/infinite values.")
    return df, imputation_medians


def remove_constant_features(df, variance_threshold=1e-12):
    """
    Task 2.2: Identify and drop zero or near-zero variance features.
    Uses strict threshold (var <= 1e-12) to drop truly uninformative constant features.
    """
    print(f"\n--- Task 2.2: Removing Constant Features (var <= {variance_threshold}) ---")
    feat_cols = [c for c in df.columns if c != 'label']
    vars_ = df[feat_cols].var()
    dropped_cols = vars_[vars_ <= variance_threshold].index.tolist()

    if dropped_cols:
        print(f"Dropping {len(dropped_cols)} constant/near-constant features: {dropped_cols}")
        df.drop(columns=dropped_cols, inplace=True)
    else:
        print("No constant/near-constant features found.")
    return df, dropped_cols


def engineer_features(df, imputation_medians=None):
    """
    Task 2.3: Centralized Feature Engineering (Used identically by Train, Val, and Test).
    Fixes Bug:
    - Eliminates duplicate code between training and inference.
    - Applies log1p compression on ratio features to prevent massive outlier spikes (>1e6)
      from small eps denominators, stabilizing distance metrics (SMOTE, KNN, NN).
    - Uses imputation medians consistently.
    """
    eps = 1e-6

    # 1. Traffic Ratios (compressed via log1p to avoid infinite/huge spikes)
    if 'Tot sum' in df.columns and 'Number' in df.columns:
        df['avg_packet_size'] = np.log1p(np.clip(df['Tot sum'] / (df['Number'] + eps), 0, 1e6)).astype('float32')
    
    if 'Srate' in df.columns and 'Rate' in df.columns:
        df['rate_ratio'] = np.log1p(np.clip(df['Srate'] / (df['Rate'] + eps), 0, 1e6)).astype('float32')
    
    if 'Drate' in df.columns and 'Rate' in df.columns:
        df['drate_ratio'] = np.log1p(np.clip(df['Drate'] / (df['Rate'] + eps), 0, 1e6)).astype('float32')
    
    if 'Duration' in df.columns and 'flow_duration' in df.columns:
        df['duration_ratio'] = np.log1p(np.clip(df['Duration'] / (df['flow_duration'] + eps), 0, 1e6)).astype('float32')
    
    if 'Header_Length' in df.columns and 'Tot size' in df.columns:
        df['header_to_payload_ratio'] = np.log1p(np.clip(df['Header_Length'] / (df['Tot size'] + eps), 0, 1e6)).astype('float32')

    # 2. Flag Aggregates
    flag_number_cols = ['fin_flag_number', 'syn_flag_number', 'rst_flag_number', 
                        'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number']
    active_flag_nums = [c for c in flag_number_cols if c in df.columns]
    if active_flag_nums:
        df['total_flags'] = df[active_flag_nums].sum(axis=1).astype('float32')

    flag_counts_cols = ['ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count']
    active_flag_counts = [c for c in flag_counts_cols if c in df.columns]
    if active_flag_counts:
        df['total_counts'] = df[active_flag_counts].sum(axis=1).astype('float32')

    if 'syn_count' in df.columns and 'ack_count' in df.columns:
        df['syn_ack_ratio'] = np.log1p(np.clip(df['syn_count'] / (df['ack_count'] + eps), 0, 1e6)).astype('float32')
    
    if 'rst_count' in df.columns and 'syn_count' in df.columns:
        df['rst_syn_ratio'] = np.log1p(np.clip(df['rst_count'] / (df['syn_count'] + eps), 0, 1e6)).astype('float32')

    # 3. Protocol Groupings
    if 'HTTP' in df.columns and 'HTTPS' in df.columns:
        df['web_traffic'] = (df['HTTP'] + df['HTTPS']).astype('float32')
    
    trans_cols = [c for c in ['TCP', 'UDP', 'ICMP'] if c in df.columns]
    if trans_cols:
        df['transport_traffic'] = df[trans_cols].sum(axis=1).astype('float32')
    
    mgmt_cols = [c for c in ['DNS', 'DHCP', 'ARP'] if c in df.columns]
    if mgmt_cols:
        df['network_mgmt'] = df[mgmt_cols].sum(axis=1).astype('float32')
    
    remote_cols = [c for c in ['Telnet', 'SSH'] if c in df.columns]
    if remote_cols:
        df['remote_access'] = df[remote_cols].sum(axis=1).astype('float32')
    
    if 'HTTPS' in df.columns and 'SSH' in df.columns:
        df['is_encrypted'] = (df['HTTPS'] + df['SSH']).astype('float32')

    # Replace any leftover inf/nans
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if imputation_medians:
        for c, med in imputation_medians.items():
            if c in df.columns and df[c].isnull().any():
                df[c] = df[c].fillna(med)
    df.fillna(0, inplace=True)

    engineered_cols = [
        'avg_packet_size', 'rate_ratio', 'drate_ratio', 'duration_ratio',
        'header_to_payload_ratio', 'total_flags', 'total_counts',
        'syn_ack_ratio', 'rst_syn_ratio', 'web_traffic', 'transport_traffic',
        'network_mgmt', 'remote_access', 'is_encrypted'
    ]
    created = [c for c in engineered_cols if c in df.columns]
    return df, created


def categorize_targets(df):
    """
    Task 2.4: Hierarchical Attack Categorization.
    Creates:
    - 34 Fine-Grained Classes: 'label' (33 Attacks + BenignTraffic)
    - 9 Attack Categories: 'attack_category' (8 Attack Groups + Benign)
    - Binary: 'is_attack' (0 = Benign, 1 = Attack)
    """
    print("\n--- Task 2.4: Hierarchical Attack Categorization (34 Classes & 9 Categories) ---")
    df['attack_category'] = df['label'].map(lambda x: ATTACK_CATEGORY_MAPPING.get(x, 'Other'))
    df['is_attack'] = (df['label'] != 'BenignTraffic').astype('int8')
    
    n_cats = df['attack_category'].nunique()
    print(f"Hierarchical targets created successfully: {df['label'].nunique()} fine-grained classes, {n_cats} categories, binary.")
    return df


def scale_training_data(df):
    """
    Task 2.5: Encoding & Robust Feature Scaling on Training Set.
    Fits LabelEncoders and RobustScaler ONLY on the training split.
    """
    print("\n--- Task 2.5: Encoding & Robust Feature Scaling on Training Set ---")
    targets = ['label', 'attack_category', 'is_attack']
    features = [c for c in df.columns if c not in targets]

    X_train = df[features].copy()
    y_train = df['label'].copy()
    y_cat_train = df['attack_category'].copy()
    y_bin_train = df['is_attack'].copy()

    # Encode targets
    le_label = LabelEncoder()
    y_train_enc = le_label.fit_transform(y_train)

    le_cat = LabelEncoder()
    y_cat_train_enc = le_cat.fit_transform(y_cat_train)

    label_mapping = {label: int(code) for code, label in enumerate(le_label.classes_)}
    cat_mapping = {cat: int(code) for code, cat in enumerate(le_cat.classes_)}

    # Robust scaling
    print("Fitting RobustScaler on training features...")
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
        dtype='float32'
    )

    y_train_dict = {
        'label_encoded': y_train_enc,
        'label_name': y_train.values,
        'category_encoded': y_cat_train_enc,
        'category_name': y_cat_train.values,
        'is_attack': y_bin_train.values
    }

    return X_train_scaled, y_train_dict, scaler, le_label, le_cat, label_mapping, cat_mapping


def save_artifacts(X_train, y_train_dict, preprocessor, label_mapping, cat_mapping):
    """Serialize all deliverables to data/ and models/."""
    print("\n--- Saving Deliverables ---")
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 1. Primary Pickle files in data/
    print("Saving processed datasets to data/...")
    joblib.dump(X_train, 'data/X_train.pkl', compress=3)
    joblib.dump(y_train_dict, 'data/y_train.pkl', compress=3)
    
    # Also link/save to root for backwards compatibility
    joblib.dump(X_train, 'X_train.pkl', compress=3)
    joblib.dump(y_train_dict, 'y_train.pkl', compress=3)

    # 2. Parquet export
    try:
        X_train.to_parquet('data/X_train.parquet', index=False)
        print("Parquet file exported to data/X_train.parquet.")
    except Exception as e:
        print(f"Parquet export note: {e}")

    # 3. Preprocessor pipeline and mappings
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    with open('models/label_mapping.json', 'w') as f:
        json.dump({
            'fine_grained_labels': label_mapping,
            'attack_categories': cat_mapping
        }, f, indent=4)

    print("All preprocessing training artifacts saved successfully!")


def transform_val_or_test(csv_path, preprocessor_path='models/preprocessor.joblib',
                          sample_size=None, unknown_label_policy='filter'):
    """
    Preprocess validation.csv or test.csv using the fitted training pipeline.
    Fixes:
    - Uses centralized engineer_features()
    - Detects & handles unknown labels cleanly with warnings and configurable policy ('filter' or 'error')
    - Uses imputation_medians consistently instead of ad-hoc fillna(0)
    """
    print(f"\n--- Preprocessing dataset: {csv_path} ---")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    prep = joblib.load(preprocessor_path)
    df_raw = pd.read_csv(csv_path, dtype=DTYPE_DICT)
    if sample_size and sample_size < len(df_raw):
        df_raw, _ = train_test_split(df_raw, train_size=sample_size, stratify=df_raw['label'], random_state=42)
    print(f"Loaded: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} cols")

    # 1. Clean inf and impute with train medians
    df_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col, med in prep['imputation_medians'].items():
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].fillna(med)

    # 2. Drop zero variance columns identified during training
    if prep['dropped_zero_var_cols']:
        drop_exist = [c for c in prep['dropped_zero_var_cols'] if c in df_raw.columns]
        if drop_exist:
            df_raw.drop(columns=drop_exist, inplace=True)

    # 3. Centralized Feature engineering
    df_raw, _ = engineer_features(df_raw, imputation_medians=prep['imputation_medians'])

    # 4. Unknown Label Guard
    known_labels = set(prep['label_encoder'].classes_)
    unknown_mask = ~df_raw['label'].isin(known_labels)
    n_unknown = unknown_mask.sum()

    if n_unknown > 0:
        unknown_counts = df_raw.loc[unknown_mask, 'label'].value_counts().to_dict()
        print(f"WARNING: Found {n_unknown:,} rows with unseen labels in {csv_path}: {unknown_counts}")
        if unknown_label_policy == 'filter':
            print(f"Policy 'filter': Removing {n_unknown:,} unseen rows to prevent metric corruption.")
            df_raw = df_raw[~unknown_mask].copy()
        elif unknown_label_policy == 'error':
            raise ValueError(f"Unseen labels encountered in test set: {unknown_counts}")

    # 5. Categorize and Encode Targets
    df_raw = categorize_targets(df_raw)
    y_test_enc = prep['label_encoder'].transform(df_raw['label'])
    y_test_cat_enc = prep['category_encoder'].transform(df_raw['attack_category'])
    y_bin = df_raw['is_attack'].values

    # 6. Align and Scale Features
    feat_cols = prep['feature_names']
    X_test = df_raw[feat_cols].copy()
    X_test_scaled = pd.DataFrame(
        prep['scaler'].transform(X_test),
        columns=feat_cols,
        index=X_test.index,
        dtype='float32'
    )

    y_test_dict = {
        'label_encoded': y_test_enc,
        'label_name': df_raw['label'].values,
        'category_encoded': y_test_cat_enc,
        'category_name': df_raw['attack_category'].values,
        'is_attack': y_bin
    }

    print(f"Transformation complete. Result: {X_test_scaled.shape[0]:,} rows x {X_test_scaled.shape[1]} features.")
    return X_test_scaled, y_test_dict


def main():
    parser = argparse.ArgumentParser(description="Phase 2: CICIoT2023 Hardened Data Preprocessing Pipeline")
    parser.add_argument('--data-path', type=str, default=None, help='Path to train.csv / parquet dataset')
    parser.add_argument('--sample', type=int, default=None, help='Stratified sample size for quick experimentation')
    parser.add_argument('--transform-test', type=str, default=None, help='Path to test.csv to transform using fitted pipeline')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    if args.transform_test:
        transform_val_or_test(args.transform_test, sample_size=args.sample)
        return

    # Step 1: Locate and load data
    file_path = find_data_file(args.data_path)
    df = load_data(file_path, sample_size=args.sample, random_state=args.random_state)

    # Step 2: Clean duplicates, infinite and missing values
    df, imputation_medians = clean_missing_and_inf(df)

    # Step 3: Remove constant features
    df, dropped_zero_var_cols = remove_constant_features(df, variance_threshold=1e-12)

    # Step 4: Centralized Feature Engineering
    df, engineered_cols = engineer_features(df, imputation_medians=imputation_medians)

    # Step 5: Hierarchical Attack Categorization
    df = categorize_targets(df)

    # Step 6: Fit encoders and RobustScaler
    X_train_scaled, y_train_dict, scaler, le_label, le_cat, label_mapping, cat_mapping = scale_training_data(df)

    # Step 7: Packaging preprocessor pipeline
    preprocessor = {
        'scaler': scaler,
        'feature_names': X_train_scaled.columns.tolist(),
        'dropped_zero_var_cols': dropped_zero_var_cols,
        'imputation_medians': imputation_medians,
        'label_encoder': le_label,
        'category_encoder': le_cat,
        'label_mapping': label_mapping,
        'category_mapping': cat_mapping,
        'attack_category_mapping': ATTACK_CATEGORY_MAPPING,
        'engineered_features': engineered_cols
    }

    # Step 8: Save Deliverables
    save_artifacts(X_train_scaled, y_train_dict, preprocessor, label_mapping, cat_mapping)
    print("\nPhase 2 completed successfully!")


if __name__ == '__main__':
    main()
