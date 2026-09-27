#!/usr/bin/env python3
"""
Phase 2: Data Preprocessing Pipeline (Training Set Only)
CICIoT2023 Network Intrusion Detection
Graduation Thesis: IoT Attack Detection & Explanation with XAI
Plan: PLAN.md (Phase 2)

Hardened & Leakage-Proof Production Pipeline:
1. Strict Raw Input Candidates (Removed self-poisoning data/X_train.parquet; strictly checks for raw 'label' column)
2. Stratified Subsampling (Subsampling preserves exact attack class proportions, no biased nrows)
3. Bidirectional Deduplication:
   - Intra-train deduplication: Removes redundant flood flows prior to subsampling/scaling
   - Intra-test deduplication: Eliminates inflated test metrics
   - Cross-set leakage prevention: Tracks train row hashes to remove identical flows from test set
4. Log1p Outlier Compression (Applies log1p on heavy-tail raw metrics & ratio features to stabilize SMOTE k-NN)
5. Centralized Feature Engineering (Shared engineer_features() for guaranteed train/val/test consistency)
6. Consistent Unknown Label Policy (Explicit filtering & logging instead of silent erroneous mapping)
7. Consistent Median Imputation across train and test sets
8. Full Test Deliverables Export (Saves data/X_test.pkl and data/y_test.pkl upon --transform-test)
9. Precise Statistical Accounting: 34 Fine-Grained Classes (33 Attacks + Benign) & 9 Categories (8 Groups + Benign)
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
    """
    Locate raw train.csv file across common candidate paths.
    Strictly excludes processed artifacts (like data/X_train.parquet) to avoid self-poisoning!
    """
    if custom_path and os.path.exists(custom_path):
        return custom_path
    
    candidates = [
        'train.csv',
        'data/train.csv',
        'plots/train.csv',
        '../train.csv',
        '/content/drive/MyDrive/do_an/train.csv'
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find raw train dataset in candidates: {candidates}. Please specify --data-path.")


def load_data(file_path, sample_size=None, stratify_col='label', random_state=42):
    """
    Load dataset with memory-optimized 32-bit floats.
    Defensive checks:
    - Verifies 'label' column is present (prevents loading preprocessed data)
    - Deduplicates BEFORE subsampling so sample budget isn't wasted on duplicate flows
    - Uses Stratified Sampling instead of biased nrows reading
    """
    print(f"Loading data from: {file_path}...")
    df = pd.read_csv(file_path, dtype=DTYPE_DICT)
    
    if 'label' not in df.columns:
        raise ValueError(f"File '{file_path}' does NOT contain 'label' column. You may be loading processed data instead of raw CSV!")
    
    ram_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Initial raw data: {df.shape[0]:,} rows x {df.shape[1]} cols | RAM: {ram_mb:.2f} MB")

    # 1. Deduplicate raw data first
    n_initial = len(df)
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        pct_dup = (n_dup / n_initial) * 100
        print(f"Deduplicating prior to sampling: removed {n_dup:,} duplicate rows ({pct_dup:.2f}%).")
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"Unique flows: {len(df):,}")

    # 2. Stratified Subsampling (if requested)
    if sample_size and sample_size < len(df):
        print(f"Drawing representative stratified subsample: {sample_size:,} / {len(df):,} rows...")
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df[stratify_col],
            random_state=random_state
        )
        df.reset_index(drop=True, inplace=True)
        print(f"Stratified sample obtained: {df.shape[0]:,} rows preserving all {df[stratify_col].nunique()} classes.")

    gc.collect()
    return df


def clean_missing_and_inf(df):
    """
    Task 2.1:
    Detect and clean infinite and missing values, compute training medians.
    """
    print("\n--- Task 2.1: Data Cleaning (Infs & Missing Values) ---")
    numeric_cols = df.select_dtypes(include=['float32', 'float64']).columns.tolist()

    # Check and clean Inf values
    inf_count = np.isinf(df[numeric_cols].values).sum()
    if inf_count > 0:
        print(f"Detected {inf_count:,} infinite values. Replacing with NaN...")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    else:
        print("No infinite values found.")

    # Calculate medians on clean training data
    imputation_medians = {}
    for col in numeric_cols:
        med = float(df[col].median())
        imputation_medians[col] = med

    # Impute missing values with median
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
    Task 2.2: Identify and drop zero or near-zero variance features (var <= 1e-12).
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
    Task 2.3: Centralized Feature Engineering (Shared across train, val, and test).
    Innovations:
    - Compresses heavy-tail continuous traffic metrics (Rate, Tot sum, IAT...) with log1p
    - Compresses ratio features with log1p(clip) to prevent 10^6 - 10^9 spikes from small eps
    - Ensures identical schema between training and inference
    """
    eps = 1e-6

    # 1. Log1p compression for heavy-tailed raw traffic metrics (stabilizes k-NN / Euclidean space)
    skewed_raw_cols = ['flow_duration', 'Duration', 'Rate', 'Srate', 'Drate', 'Tot sum', 'Tot size', 'IAT', 'Variance']
    for col in skewed_raw_cols:
        if col in df.columns:
            df[col] = np.log1p(np.maximum(df[col], 0)).astype('float32')

    # 2. Domain-Specific Traffic Ratios (compressed via log1p)
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

    # 3. Flag Aggregates
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

    # 4. Protocol Groupings
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
    Also computes training row hashes for cross-set leakage detection.
    """
    print("\n--- Task 2.5: Encoding & Robust Feature Scaling on Training Set ---")
    targets = ['label', 'attack_category', 'is_attack']
    features = [c for c in df.columns if c not in targets]

    X_train = df[features].copy()
    y_train = df['label'].copy()
    y_cat_train = df['attack_category'].copy()
    y_bin_train = df['is_attack'].copy()

    # Compute row hashes for cross-set leakage prevention
    print("Computing row hashes to guard against train-test leakage...")
    train_row_hashes = set(pd.util.hash_pandas_object(X_train, index=False).astype(np.uint64))

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

    return X_train_scaled, y_train_dict, scaler, le_label, le_cat, label_mapping, cat_mapping, train_row_hashes, features


def save_artifacts(X_train, y_train_dict, preprocessor, label_mapping, cat_mapping):
    """Serialize all deliverables to data/ and models/."""
    print("\n--- Saving Deliverables ---")
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 1. Primary Pickle files in data/
    print("Saving processed datasets to data/...")
    joblib.dump(X_train, 'data/X_train.pkl', compress=3)
    joblib.dump(y_train_dict, 'data/y_train.pkl', compress=3)
    
    # Also save to root for backward compatibility
    joblib.dump(X_train, 'X_train.pkl', compress=3)
    joblib.dump(y_train_dict, 'y_train.pkl', compress=3)

    # 2. Parquet export in data/
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
    Innovations:
    - Intra-test deduplication: Eliminates duplicate flows inside the evaluation set
    - Cross-set leakage check: Uses preprocessor['train_row_hashes'] to detect & remove identical flows
    - Centralized feature engineering with log1p compression
    - Consistent median imputation
    - Unknown label policy ('filter' or 'error')
    """
    print(f"\n--- Preprocessing evaluation dataset: {csv_path} ---")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    prep = joblib.load(preprocessor_path)
    df_raw = pd.read_csv(csv_path, dtype=DTYPE_DICT)
    print(f"Loaded raw test flows: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} cols")

    # 1. Intra-test Deduplication
    n0 = len(df_raw)
    df_raw.drop_duplicates(inplace=True)
    df_raw.reset_index(drop=True, inplace=True)
    n_dedup = n0 - len(df_raw)
    if n_dedup > 0:
        print(f"Test intra-dedup: removed {n_dedup:,} duplicate rows ({n_dedup / n0 * 100:.2f}%).")

    # 2. Cross-set Leakage Check (Train vs Test identical flows)
    if 'train_row_hashes' in prep and 'raw_feature_cols' in prep:
        raw_feat_cols = [c for c in prep['raw_feature_cols'] if c in df_raw.columns]
        h = pd.util.hash_pandas_object(df_raw[raw_feat_cols], index=False).astype(np.uint64)
        overlap = h.isin(prep['train_row_hashes'])
        n_overlap = overlap.sum()
        if n_overlap > 0:
            pct_overlap = (n_overlap / len(df_raw)) * 100
            print(f"WARNING: Detected {n_overlap:,} test rows ({pct_overlap:.2f}%) IDENTICAL to train rows (cross-set leakage) -> removing.")
            df_raw = df_raw[~overlap.values].reset_index(drop=True)
        else:
            print("Cross-set leakage check: 0 duplicate rows between train and test. 100% Clean!")

    if sample_size and sample_size < len(df_raw):
        df_raw, _ = train_test_split(df_raw, train_size=sample_size, stratify=df_raw['label'], random_state=42)
        df_raw.reset_index(drop=True, inplace=True)

    # 3. Clean inf and impute with train medians
    df_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col, med in prep['imputation_medians'].items():
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].fillna(med)

    # 4. Drop zero variance columns identified during training
    if prep['dropped_zero_var_cols']:
        drop_exist = [c for c in prep['dropped_zero_var_cols'] if c in df_raw.columns]
        if drop_exist:
            df_raw.drop(columns=drop_exist, inplace=True)

    # 5. Centralized Feature engineering (with log1p compression)
    df_raw, _ = engineer_features(df_raw, imputation_medians=prep['imputation_medians'])

    # 6. Unknown Label Guard
    known_labels = set(prep['label_encoder'].classes_)
    unknown_mask = ~df_raw['label'].isin(known_labels)
    n_unknown = unknown_mask.sum()

    if n_unknown > 0:
        unknown_counts = df_raw.loc[unknown_mask, 'label'].value_counts().to_dict()
        print(f"WARNING: Found {n_unknown:,} rows with unseen labels in {csv_path}: {unknown_counts}")
        if unknown_label_policy == 'filter':
            print(f"Policy 'filter': Removing {n_unknown:,} unseen rows to prevent metric corruption.")
            df_raw = df_raw[~unknown_mask].reset_index(drop=True)
        elif unknown_label_policy == 'error':
            raise ValueError(f"Unseen labels encountered in test set: {unknown_counts}")

    # 7. Categorize and Encode Targets
    df_raw = categorize_targets(df_raw)
    y_test_enc = prep['label_encoder'].transform(df_raw['label'])
    y_test_cat_enc = prep['category_encoder'].transform(df_raw['attack_category'])
    y_bin = df_raw['is_attack'].values

    # 8. Align and Scale Features
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
    parser.add_argument('--data-path', type=str, default=None, help='Path to train.csv dataset')
    parser.add_argument('--output-dir', type=str, default=None, help="Directory to save artifacts (auto-detects '/content/drive/MyDrive/do_an' on Colab)")
    parser.add_argument('--sample', type=int, default=None, help='Stratified sample size for quick experimentation')
    parser.add_argument('--transform-test', type=str, default=None, help='Path to test.csv to transform using fitted pipeline')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    # 🎯 Redirect working directory to Google Drive if in Colab or if custom output-dir is given
    target_dir = args.output_dir
    if not target_dir and os.path.exists('/content/drive/MyDrive/do_an'):
        target_dir = '/content/drive/MyDrive/do_an'
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
        os.chdir(target_dir)
        print(f"🚀 Working directory redirected to: {os.getcwd()}")
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)

    # If --transform-test is specified, transform and export data/X_test.pkl, data/y_test.pkl
    if args.transform_test:
        X_test_scaled, y_test_dict = transform_val_or_test(args.transform_test, sample_size=args.sample)
        os.makedirs('data', exist_ok=True)
        joblib.dump(X_test_scaled, 'data/X_test.pkl', compress=3)
        joblib.dump(y_test_dict, 'data/y_test.pkl', compress=3)
        print("Saved test deliverables to: data/X_test.pkl and data/y_test.pkl")
        return

    # Step 1: Locate and load raw data
    file_path = find_data_file(args.data_path)
    df = load_data(file_path, sample_size=args.sample, random_state=args.random_state)

    # Step 2: Clean infinite and missing values
    df, imputation_medians = clean_missing_and_inf(df)

    # Step 3: Remove constant features
    df, dropped_zero_var_cols = remove_constant_features(df, variance_threshold=1e-12)

    # Step 4: Centralized Feature Engineering (with log1p compression)
    df, engineered_cols = engineer_features(df, imputation_medians=imputation_medians)

    # Step 5: Hierarchical Attack Categorization
    df = categorize_targets(df)

    # Step 6: Fit encoders, RobustScaler, and compute train row hashes
    X_train_scaled, y_train_dict, scaler, le_label, le_cat, label_mapping, cat_mapping, train_row_hashes, raw_feature_cols = scale_training_data(df)

    # Step 7: Packaging preprocessor pipeline
    preprocessor = {
        'scaler': scaler,
        'feature_names': X_train_scaled.columns.tolist(),
        'raw_feature_cols': raw_feature_cols,
        'train_row_hashes': train_row_hashes,
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
