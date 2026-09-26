#!/usr/bin/env python3
"""
Phase 2: Data Preprocessing Pipeline (Training Set Only)
CICIoT2023 Network Intrusion Detection
Author: Data Science Team
Plan: PLAN.md (Phase 2)

Note:
    plots/train.csv is solely the training set.
    No train/test split is performed on train.csv.
    All transformers are fit on train.csv and exported to models/preprocessor.joblib.
    A dedicated function transform_val_or_test() is provided for future validation/test sets.

Tasks Implemented:
    2.1 Handle missing / infinite values
    2.2 Remove constant / near-constant features
    2.3 Feature engineering (packet ratios, flag sums, protocol groupings)
    2.4 Hierarchical attack categorization (Multi-class, 8-Class, Binary)
    2.5 Target label encoding & feature scaling (RobustScaler)
    2.6 Export preprocessed training data and pipeline
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

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# Attack Categorization Dictionary (All 34 Attacks in CICIoT2023 + Benign)
# ------------------------------------------------------------------------------
ATTACK_CATEGORY_MAPPING = {
    # 1. DDoS attacks
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

    # 2. DoS attacks
    'DoS-UDP_Flood': 'DoS',
    'DoS-TCP_Flood': 'DoS',
    'DoS-SYN_Flood': 'DoS',
    'DoS-HTTP_Flood': 'DoS',

    # 3. Mirai attacks
    'Mirai-greeth_flood': 'Mirai',
    'Mirai-udpplain': 'Mirai',
    'Mirai-greip_flood': 'Mirai',

    # 4. Reconnaissance & Scanning
    'Recon-HostDiscovery': 'Recon',
    'Recon-OSScan': 'Recon',
    'Recon-PortScan': 'Recon',
    'Recon-PingSweep': 'Recon',
    'VulnerabilityScan': 'Recon',

    # 5. Spoofing & MITM
    'MITM-ArpSpoofing': 'Spoofing',
    'DNS_Spoofing': 'Spoofing',

    # 6. Web-based attacks
    'BrowserHijacking': 'Web',
    'CommandInjection': 'Web',
    'SqlInjection': 'Web',
    'XSS': 'Web',
    'Uploading_Attack': 'Web',

    # 7. Brute Force
    'DictionaryBruteForce': 'BruteForce',

    # 8. Malware
    'Backdoor_Malware': 'Malware',

    # 9. Benign
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
        'plots/train.csv',
        'train.csv',
        '../train.csv',
        '/content/drive/MyDrive/do_an/train.csv'
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find dataset in candidates: {candidates}")


def load_data(file_path, sample_size=None):
    """Load dataset with memory-optimized 32-bit floats."""
    print(f"Loading data from: {file_path}...")
    if sample_size:
        print(f"Sampling {sample_size:,} rows for memory efficiency...")
        df = pd.read_csv(file_path, dtype=DTYPE_DICT, nrows=sample_size)
    else:
        df = pd.read_csv(file_path, dtype=DTYPE_DICT)
    
    gc.collect()
    ram_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} cols | RAM: {ram_mb:.2f} MB")
    return df


def clean_missing_and_inf(df):
    """Task 2.1: Detect and clean infinite and missing values, compute medians."""
    print("\n--- Task 2.1: Cleaning Missing & Infinite Values ---")
    numeric_cols = df.select_dtypes(include=['float32', 'float64']).columns.tolist()

    # Check and replace inf
    inf_count = np.isinf(df[numeric_cols].values).sum()
    if inf_count > 0:
        print(f"Detected {inf_count:,} infinite values. Replacing with NaN...")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    else:
        print("No infinite values found.")

    # Calculate medians on training data
    imputation_medians = {}
    for col in numeric_cols:
        med = float(df[col].median())
        imputation_medians[col] = med

    # Impute missing values with median
    nan_counts = df.isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0].index.tolist()
    if cols_with_nan:
        print(f"Imputing NaNs in {len(cols_with_nan)} columns using column medians...")
        for col in cols_with_nan:
            if col in numeric_cols:
                df[col].fillna(imputation_medians[col], inplace=True)
    else:
        print("No missing values found in training set.")

    if df['label'].isnull().sum() > 0:
        df.dropna(subset=['label'], inplace=True)

    assert df.isnull().sum().sum() == 0, "NaN values still present!"
    print("Training dataset is 100% clean of missing/infinite values.")
    return df, imputation_medians


def remove_constant_features(df):
    """Task 2.2: Identify and drop zero-variance features."""
    print("\n--- Task 2.2: Removing Constant Features ---")
    feat_cols = [c for c in df.columns if c != 'label']
    vars_ = df[feat_cols].var()
    zero_var = vars_[vars_ == 0].index.tolist()

    if zero_var:
        print(f"Dropping {len(zero_var)} zero-variance features: {zero_var}")
        df.drop(columns=zero_var, inplace=True)
    else:
        print("No strictly zero-variance features found.")
    return df, zero_var


def engineer_features(df):
    """Task 2.3: Compute domain-specific network traffic ratios and aggregates."""
    print("\n--- Task 2.3: Feature Engineering ---")
    eps = 1e-6

    # 1. Ratios
    if 'Tot sum' in df.columns and 'Number' in df.columns:
        df['avg_packet_size'] = (df['Tot sum'] / (df['Number'] + eps)).astype('float32')
    if 'Srate' in df.columns and 'Rate' in df.columns:
        df['rate_ratio'] = (df['Srate'] / (df['Rate'] + eps)).astype('float32')
    if 'Drate' in df.columns and 'Rate' in df.columns:
        df['drate_ratio'] = (df['Drate'] / (df['Rate'] + eps)).astype('float32')
    if 'Duration' in df.columns and 'flow_duration' in df.columns:
        df['duration_ratio'] = (df['Duration'] / (df['flow_duration'] + eps)).astype('float32')
    if 'Header_Length' in df.columns and 'Tot size' in df.columns:
        df['header_to_payload_ratio'] = (df['Header_Length'] / (df['Tot size'] + eps)).astype('float32')

    # 2. Flag aggregates
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
        df['syn_ack_ratio'] = (df['syn_count'] / (df['ack_count'] + eps)).astype('float32')
    if 'rst_count' in df.columns and 'syn_count' in df.columns:
        df['rst_syn_ratio'] = (df['rst_count'] / (df['syn_count'] + eps)).astype('float32')

    # 3. Protocol groupings
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

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    gc.collect()

    all_eng_features = [
        'avg_packet_size', 'rate_ratio', 'drate_ratio', 'duration_ratio',
        'header_to_payload_ratio', 'total_flags', 'total_counts',
        'syn_ack_ratio', 'rst_syn_ratio', 'web_traffic', 'transport_traffic',
        'network_mgmt', 'remote_access', 'is_encrypted'
    ]
    created = [c for c in all_eng_features if c in df.columns]
    print(f"Engineered {len(created)} features successfully. Total columns now: {df.shape[1]}")
    return df, created


def categorize_targets(df):
    """Task 2.4: Create hierarchical attack categories and binary target."""
    print("\n--- Task 2.4: Hierarchical Attack Categorization ---")
    df['attack_category'] = df['label'].map(lambda x: ATTACK_CATEGORY_MAPPING.get(x, 'Other'))
    df['is_attack'] = (df['label'] != 'BenignTraffic').astype('int8')
    print("Class categories created: DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce, Malware, Benign")
    return df


def scale_training_data(df):
    """Task 2.5: Fit encoders and RobustScaler on training set."""
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
    print("Fitting RobustScaler on X_train...")
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
    """Serialize all deliverables."""
    print("\n--- Saving Deliverables ---")
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 1. Pickle files (Root and data/)
    print("Saving pickle files (X_train.pkl, y_train.pkl)...")
    for base_dir in ['', 'data/']:
        joblib.dump(X_train, os.path.join(base_dir, 'X_train.pkl'), compress=3)
        joblib.dump(y_train_dict, os.path.join(base_dir, 'y_train.pkl'), compress=3)

    # 2. Parquet
    try:
        X_train.to_parquet('data/X_train.parquet', index=False)
        print("Parquet file exported to data/X_train.parquet.")
    except Exception as e:
        print(f"Parquet export skipped: {e}")

    # 3. Preprocessor pipeline and mappings
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    with open('models/label_mapping.json', 'w') as f:
        json.dump({
            'fine_grained_labels': label_mapping,
            'attack_categories': cat_mapping
        }, f, indent=4)

    print("All preprocessing training artifacts saved successfully!")


def transform_val_or_test(csv_path, preprocessor_path='models/preprocessor.joblib', sample_size=None):
    """Preprocess validation.csv or test.csv using the fitted training pipeline."""
    print(f"--- Preprocessing dataset: {csv_path} ---")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    prep = joblib.load(preprocessor_path)
    df_raw = pd.read_csv(csv_path, dtype=DTYPE_DICT, nrows=sample_size)
    print(f"Loaded: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} cols")

    # Clean inf and impute with train medians
    df_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col, med in prep['imputation_medians'].items():
        if col in df_raw.columns:
            df_raw[col].fillna(med, inplace=True)

    # Drop zero variance columns
    if prep['dropped_zero_var_cols']:
        df_raw.drop(columns=[c for c in prep['dropped_zero_var_cols'] if c in df_raw.columns], inplace=True)

    # Identical feature engineering
    eps = 1e-6
    if 'Tot sum' in df_raw.columns and 'Number' in df_raw.columns:
        df_raw['avg_packet_size'] = (df_raw['Tot sum'] / (df_raw['Number'] + eps)).astype('float32')
    if 'Srate' in df_raw.columns and 'Rate' in df_raw.columns:
        df_raw['rate_ratio'] = (df_raw['Srate'] / (df_raw['Rate'] + eps)).astype('float32')
    if 'Drate' in df_raw.columns and 'Rate' in df_raw.columns:
        df_raw['drate_ratio'] = (df_raw['Drate'] / (df_raw['Rate'] + eps)).astype('float32')
    if 'Duration' in df_raw.columns and 'flow_duration' in df_raw.columns:
        df_raw['duration_ratio'] = (df_raw['Duration'] / (df_raw['flow_duration'] + eps)).astype('float32')
    if 'Header_Length' in df_raw.columns and 'Tot size' in df_raw.columns:
        df_raw['header_to_payload_ratio'] = (df_raw['Header_Length'] / (df_raw['Tot size'] + eps)).astype('float32')

    flag_number_cols = ['fin_flag_number', 'syn_flag_number', 'rst_flag_number', 
                        'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number']
    active_flags = [c for c in flag_number_cols if c in df_raw.columns]
    if active_flags:
        df_raw['total_flags'] = df_raw[active_flags].sum(axis=1).astype('float32')

    flag_counts_cols = ['ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count']
    active_counts = [c for c in flag_counts_cols if c in df_raw.columns]
    if active_counts:
        df_raw['total_counts'] = df_raw[active_counts].sum(axis=1).astype('float32')

    if 'syn_count' in df_raw.columns and 'ack_count' in df_raw.columns:
        df_raw['syn_ack_ratio'] = (df_raw['syn_count'] / (df_raw['ack_count'] + eps)).astype('float32')
    if 'rst_count' in df_raw.columns and 'syn_count' in df_raw.columns:
        df_raw['rst_syn_ratio'] = (df_raw['rst_count'] / (df_raw['syn_count'] + eps)).astype('float32')

    if 'HTTP' in df_raw.columns and 'HTTPS' in df_raw.columns:
        df_raw['web_traffic'] = (df_raw['HTTP'] + df_raw['HTTPS']).astype('float32')
    trans_cols = [c for c in ['TCP', 'UDP', 'ICMP'] if c in df_raw.columns]
    if trans_cols:
        df_raw['transport_traffic'] = df_raw[trans_cols].sum(axis=1).astype('float32')
    mgmt_cols = [c for c in ['DNS', 'DHCP', 'ARP'] if c in df_raw.columns]
    if mgmt_cols:
        df_raw['network_mgmt'] = df_raw[mgmt_cols].sum(axis=1).astype('float32')
    remote_cols = [c for c in ['Telnet', 'SSH'] if c in df_raw.columns]
    if remote_cols:
        df_raw['remote_access'] = df_raw[remote_cols].sum(axis=1).astype('float32')
    if 'HTTPS' in df_raw.columns and 'SSH' in df_raw.columns:
        df_raw['is_encrypted'] = (df_raw['HTTPS'] + df_raw['SSH']).astype('float32')

    df_raw.replace([np.inf, -np.inf], 0, inplace=True)
    df_raw.fillna(0, inplace=True)

    expected_features = prep['feature_names']
    for f in expected_features:
        if f not in df_raw.columns:
            df_raw[f] = 0.0

    X_unscaled = df_raw[expected_features].copy()
    X_scaled = pd.DataFrame(
        prep['scaler'].transform(X_unscaled),
        columns=expected_features,
        index=X_unscaled.index,
        dtype='float32'
    )

    y_dict = None
    if 'label' in df_raw.columns:
        cat_series = df_raw['label'].map(lambda x: prep['attack_category_mapping'].get(x, 'Other'))
        bin_series = (df_raw['label'] != 'BenignTraffic').astype('int8')
        known_labels = set(prep['label_encoder'].classes_)
        safe_labels = df_raw['label'].map(lambda x: x if x in known_labels else prep['label_encoder'].classes_[0])
        y_dict = {
            'label_encoded': prep['label_encoder'].transform(safe_labels),
            'label_name': df_raw['label'].values,
            'category_encoded': prep['category_encoder'].transform(cat_series),
            'category_name': cat_series.values,
            'is_attack': bin_series.values
        }

    return X_scaled, y_dict


def main():
    parser = argparse.ArgumentParser(description="Phase 2: CICIoT2023 Preprocessing Pipeline (Training Set Only)")
    parser.add_argument('--data-path', type=str, default=None, help='Path to train.csv')
    parser.add_argument('--sample', type=int, default=None, help='Sample N rows (None for full dataset)')
    args = parser.parse_args()

    data_file = find_data_file(args.data_path)
    df = load_data(data_file, sample_size=args.sample)
    df, imputation_medians = clean_missing_and_inf(df)
    df, dropped_zero_var = remove_constant_features(df)
    df, created_eng = engineer_features(df)
    df = categorize_targets(df)

    X_train_scaled, y_train_dict, scaler, le_label, le_cat, lbl_map, cat_map = scale_training_data(df)

    preprocessor = {
        'scaler': scaler,
        'feature_names': list(X_train_scaled.columns),
        'dropped_zero_var_cols': dropped_zero_var,
        'imputation_medians': imputation_medians,
        'label_encoder': le_label,
        'category_encoder': le_cat,
        'label_mapping': lbl_map,
        'category_mapping': cat_map,
        'attack_category_mapping': ATTACK_CATEGORY_MAPPING,
        'engineered_features': created_eng
    }

    save_artifacts(X_train_scaled, y_train_dict, preprocessor, lbl_map, cat_map)
    print("\nPhase 2 Preprocessing Pipeline (Training Set) completed successfully!")


if __name__ == '__main__':
    main()
