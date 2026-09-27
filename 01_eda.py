#!/usr/bin/env python3
"""
Phase 1: Exploratory Data Analysis (EDA) Pipeline
CICIoT2023 Network Intrusion Detection
Plan: PLAN.md (Phase 1)

Tasks:
    1.1 Load raw data with memory-optimized 32-bit floats
    1.2 Basic data inspection (shape, dtypes, memory footprint)
    1.3 Missing and infinite values analysis
    1.4 Duplicate rows inspection
    1.5 Statistical summary (describe)
    1.6 Class distribution analysis (34 Fine-Grained Classes & 9 Categories)
    1.7 Feature correlation & protocol distribution analysis

Deliverables:
    - 01_eda.py (CLI script)
    - plots/01_class_distribution.png
    - plots/02_category_distribution.png
    - plots/03_correlation_heatmap.png
    - plots/04_feature_distributions.png
    - plots/05_protocol_distribution.png
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Schema definitions
ATTACK_CATEGORY_MAPPING = {
    'DDoS-ICMP_Flood': 'DDoS', 'DDoS-UDP_Flood': 'DDoS', 'DDoS-TCP_Flood': 'DDoS',
    'DDoS-PSHACK_Flood': 'DDoS', 'DDoS-SYN_Flood': 'DDoS', 'DDoS-RSTFINFlood': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS', 'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-UDP_Fragmentation': 'DDoS', 'DDoS-ACK_Fragmentation': 'DDoS',
    'DDoS-HTTP_Flood': 'DDoS', 'DDoS-SlowLoris': 'DDoS',
    'DoS-UDP_Flood': 'DoS', 'DoS-TCP_Flood': 'DoS', 'DoS-SYN_Flood': 'DoS', 'DoS-HTTP_Flood': 'DoS',
    'Mirai-greeth_flood': 'Mirai', 'Mirai-udpplain': 'Mirai', 'Mirai-greip_flood': 'Mirai',
    'Recon-HostDiscovery': 'Recon', 'Recon-OSScan': 'Recon', 'Recon-PortScan': 'Recon',
    'Recon-PingSweep': 'Recon', 'VulnerabilityScan': 'Recon',
    'MITM-ArpSpoofing': 'Spoofing', 'DNS_Spoofing': 'Spoofing',
    'BrowserHijacking': 'Web', 'CommandInjection': 'Web', 'SqlInjection': 'Web',
    'XSS': 'Web', 'Uploading_Attack': 'Web',
    'DictionaryBruteForce': 'BruteForce',
    'Backdoor_Malware': 'Malware',
    'BenignTraffic': 'Benign'
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
    if custom_path and os.path.exists(custom_path):
        return custom_path
    candidates = [
        'train.csv',
        '/content/drive/MyDrive/do_an/train.csv',
        'data/train.csv',
        'plots/train.csv',
        '../train.csv'
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find raw train dataset. Checked candidates: {candidates}")


def load_eda_data(file_path, sample_size=None, random_state=42):
    print("=" * 75)
    print(f"LOADING DATASET FOR EDA: {file_path}")
    print("=" * 75)
    df = pd.read_csv(file_path, dtype=DTYPE_DICT)

    if 'label' not in df.columns:
        raise ValueError(f"File '{file_path}' does not contain 'label' column. Please supply raw train.csv.")

    print(f"Full dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Memory footprint:   {mem_mb:.2f} MB")

    if sample_size and sample_size < len(df):
        print(f"Subsampling {sample_size:,} rows with stratification for fast EDA...")
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(df, train_size=sample_size, stratify=df['label'], random_state=random_state)
        df.reset_index(drop=True, inplace=True)
        print(f"Sampled shape:      {df.shape[0]:,} rows x {df.shape[1]} columns")

    return df


def inspect_data_quality(df):
    print("\n" + "=" * 75)
    print("DATA QUALITY & INTEGRITY CHECK")
    print("=" * 75)

    # Missing values
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    print(f"Missing values (NaN): {total_missing:,}")

    # Infinite values
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[num_cols]).sum().sum()
    print(f"Infinite values (Inf): {inf_counts:,}")

    # Duplicate rows
    dup_count = df.duplicated().sum()
    dup_pct = (dup_count / len(df)) * 100
    print(f"Duplicate rows:       {dup_count:,} ({dup_pct:.2f}%)")

    # Map attack category
    df['attack_category'] = df['label'].map(ATTACK_CATEGORY_MAPPING).fillna('Unknown')
    return df


def generate_eda_plots(df, plots_dir='plots'):
    os.makedirs(plots_dir, exist_ok=True)
    print("\n" + "=" * 75)
    print(f"GENERATING EDA VISUALIZATION PLOTS IN {plots_dir}/")
    print("=" * 75)

    # 1. Plot 01: 34 Fine-Grained Class Distribution
    fig, ax = plt.subplots(figsize=(14, 7))
    class_counts = df['label'].value_counts()
    sns.barplot(x=class_counts.values, y=class_counts.index, palette='viridis', ax=ax, edgecolor='black', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_title('Phân phối 34 Lớp Tấn công CICIoT2023 (Thang đo Logarit)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Số lượng mẫu (Log Scale)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Nhãn Tấn công (33 Attacks + Benign)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    p01 = os.path.join(plots_dir, '01_class_distribution.png')
    plt.savefig(p01, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p01}")

    # 2. Plot 02: 9 Category Distribution
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cat_counts = df['attack_category'].value_counts()
    colors = sns.color_palette('tab10', len(cat_counts))
    bars = ax.bar(cat_counts.index, cat_counts.values, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_yscale('log')
    ax.set_title('Phân phối 9 Nhóm Tấn công (8 Attack Groups + Benign)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Nhóm Tấn công', fontsize=11, fontweight='bold')
    ax.set_ylabel('Số lượng mẫu (Log Scale)', fontsize=11, fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=10, fontweight='bold')
    # Annotate counts
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:,}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    plt.tight_layout()
    p02 = os.path.join(plots_dir, '02_category_distribution.png')
    plt.savefig(p02, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p02}")

    # 3. Plot 03: Feature Correlation Heatmap (Sample of key features)
    fig, ax = plt.subplots(figsize=(12, 10))
    key_features = [
        'flow_duration', 'Header_Length', 'Duration', 'Rate', 'Srate',
        'syn_flag_number', 'ack_flag_number', 'rst_flag_number', 'psh_flag_number',
        'HTTP', 'HTTPS', 'DNS', 'TCP', 'UDP', 'ICMP',
        'Tot sum', 'Tot size', 'Min', 'Max', 'AVG', 'Std'
    ]
    avail_feats = [f for f in key_features if f in df.columns]
    corr_matrix = df[avail_feats].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmin=-1.0, vmax=1.0, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Ma trận Tương quan Các Đặc trưng Lưu lượng Trọng yếu (EDA)', fontsize=13, fontweight='bold', pad=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    p03 = os.path.join(plots_dir, '03_correlation_heatmap.png')
    plt.savefig(p03, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p03}")

    # 4. Plot 04: Feature Distributions
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    dist_cols = ['flow_duration', 'Rate', 'Tot size', 'Header_Length']
    for idx, col in enumerate(dist_cols):
        r, c = idx // 2, idx % 2
        ax = axes[r, c]
        if col in df.columns:
            # log1p for heavy tails
            vals = np.log1p(np.clip(df[col].dropna(), 0, 1e7))
            sns.histplot(vals, kde=True, ax=ax, color='#2980b9', bins=40, edgecolor='black', linewidth=0.5)
            ax.set_title(f'Phân phối log(1 + {col})', fontsize=11, fontweight='bold')
            ax.set_xlabel('Giá trị sau chuẩn hóa Log', fontsize=9.5)
            ax.set_ylabel('Tần suất', fontsize=9.5)
    plt.tight_layout()
    p04 = os.path.join(plots_dir, '04_feature_distributions.png')
    plt.savefig(p04, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p04}")

    # 5. Plot 05: Protocol Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    proto_cols = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SSH', 'ARP']
    avail_proto = [p for p in proto_cols if p in df.columns]
    proto_sums = df[avail_proto].sum().sort_values(ascending=False)
    ax.bar(proto_sums.index, proto_sums.values, color='#8e44ad', edgecolor='black', linewidth=0.6)
    ax.set_title('Phân bố Giao thức Mạng trong Dữ liệu Lưu lượng IoT', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Giao thức / Ứng dụng', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tổng số gói / Tần suất kích hoạt', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    plt.xticks(rotation=25, ha='right', fontsize=10, fontweight='bold')
    plt.tight_layout()
    p05 = os.path.join(plots_dir, '05_protocol_distribution.png')
    plt.savefig(p05, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p05}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: CICIoT2023 Exploratory Data Analysis (EDA)")
    parser.add_argument('--data-path', type=str, default=None, help='Path to train.csv dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Thư mục làm việc/lưu trữ kết quả (Mặc định: auto-detect '/content/drive/MyDrive/do_an' nếu trên Colab, hoặc '.')")
    parser.add_argument('--plots-dir', type=str, default='plots', help='Directory to save EDA plots')
    parser.add_argument('--sample', type=int, default=100000, help='Sample size for EDA analysis (default: 100,000)')
    parser.add_argument('--skip-plots', action='store_true', default=False, help='Skip saving plots')
    args = parser.parse_args()

    # 🎯 Redirect working directory to Google Drive if in Colab or if custom output-dir is given
    target_dir = args.output_dir
    if not target_dir and os.path.exists('/content/drive/MyDrive/do_an'):
        target_dir = '/content/drive/MyDrive/do_an'
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
        os.chdir(target_dir)
        print(f"🚀 Working directory redirected to: {os.getcwd()}")
        os.makedirs('plots', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)

    # 1. Load Data
    file_path = find_data_file(args.data_path)
    df = load_eda_data(file_path, sample_size=args.sample)

    # 2. Inspect Quality
    df = inspect_data_quality(df)

    # 3. Generate Plots
    if not args.skip_plots:
        generate_eda_plots(df, plots_dir=args.plots_dir)

    print("\n" + "=" * 75)
    print("PHASE 1 (EDA) COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == '__main__':
    main()
