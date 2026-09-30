#!/usr/bin/env python3
"""
Phase 3: Class Imbalance Handling Pipeline (Paper-Aligned SMOTETomek & Hybrid Resampling)
CICIoT2023 Network Intrusion Detection
Graduation Thesis: IoT Attack Detection & Explanation with XAI
Plan: PLAN.md (Phase 3)

Hardened & Bulletproof Version Addressing:
1. Proportionally Scaled Guaranteed Minority Preservation:
   - Uses ratio-scaled threshold (counts * ratio < min_subsample_keep) so rare classes (Malware, Web, BruteForce)
     retain 100% of their real samples even in large subsampling ratios, avoiding synthetic extrapolation on tiny counts.
2. Adaptive Fallback for SMOTE (Pre-amplifies classes with n <= k using ROS to guarantee SMOTE k-NN stability)
3. Dual Class Weight Archiving (Saves both raw inverse-frequency and smoothed weights for rigorous ablation studies)
4. Synchronized Target Tiers (Supports Fine-Grained: 34 classes & Category: 9 classes with unambiguous y_train_balanced.pkl interface)
5. Memory-Safe SMOTEENN (Passes explicit sampling_strategy to avoid memory blowup)
6. Added BorderlineSMOTE Strategy (Targets borderline support vectors in noisy IoT networks)
7. Matplotlib Clean Formatting (Fixes fixed tick warning, uses rng-seeded sampling, tab20/continuous colormaps, dynamic titles)
8. Exact Statistical Accounting (34 Fine-Grained Classes: 33 Attacks + Benign; 9 Categories: 8 Groups + Benign)
"""

import os
import gc
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN

warnings.filterwarnings('ignore')


def load_phase2_data():
    """Load preprocessed feature matrix, target dictionary, and pipeline from Phase 2."""
    print("=" * 80)
    print("STEP 1: LOADING PREPROCESSED DATA (PHASE 2)")
    print("=" * 80)

    # Load X_train
    x_paths = ['data/X_train.pkl', 'X_train.pkl']
    for p in x_paths:
        if os.path.exists(p):
            print(f"Loading features from: {p}")
            X_train = joblib.load(p)
            break
    else:
        raise FileNotFoundError("Could not find X_train.pkl in data/ or root directory!")

    # Load y_train
    y_paths = ['data/y_train.pkl', 'y_train.pkl']
    for p in y_paths:
        if os.path.exists(p):
            print(f"Loading labels from: {p}")
            y_train_dict = joblib.load(p)
            break
    else:
        raise FileNotFoundError("Could not find y_train.pkl in data/ or root directory!")

    # Load preprocessor
    prep_path = 'models/preprocessor.joblib'
    if not os.path.exists(prep_path):
        raise FileNotFoundError(f"Missing {prep_path}. Please run Phase 2 first!")
    preprocessor = joblib.load(prep_path)

    print(f"Successfully loaded X_train: {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
    print(f"Target dictionary keys: {list(y_train_dict.keys())}")
    return X_train, y_train_dict, preprocessor


def analyze_imbalance(y_names, y_cat_names, label_mapping, category_mapping):
    """
    Task 3.1: Quantitative Imbalance Analysis
    Calculates Imbalance Ratio (IR), Shannon Entropy, and 4-Tier Severity breakdown.
    34 Fine-Grained Classes: 33 Attacks + Benign
    9 Attack Categories: 8 Attack Groups + Benign
    """
    print("\n" + "=" * 80)
    print("TASK 3.1: IMBALANCE QUANTITATIVE ANALYSIS")
    print("=" * 80)

    total_samples = len(y_names)
    label_counts = pd.Series(y_names).value_counts()
    label_pct = (label_counts / total_samples) * 100

    cat_counts = pd.Series(y_cat_names).value_counts()
    cat_pct = (cat_counts / total_samples) * 100

    # Imbalance Ratio (IR)
    ir_fine = label_counts.max() / label_counts.min()
    ir_cat = cat_counts.max() / cat_counts.min()

    # Shannon Entropy Ratio (Normalized [0, 1])
    probs_fine = label_counts.values / total_samples
    actual_entropy_fine = -np.sum(probs_fine * np.log2(probs_fine + 1e-12))
    max_entropy_fine = np.log2(len(label_counts))
    entropy_ratio_fine = actual_entropy_fine / max_entropy_fine

    probs_cat = cat_counts.values / total_samples
    actual_entropy_cat = -np.sum(probs_cat * np.log2(probs_cat + 1e-12))
    max_entropy_cat = np.log2(len(cat_counts))
    entropy_ratio_cat = actual_entropy_cat / max_entropy_cat

    # 4-Tier Severity breakdown for 34 classes
    majority = label_pct[label_pct >= 5.0].index.tolist()
    medium = label_pct[(label_pct >= 1.0) & (label_pct < 5.0)].index.tolist()
    minority = label_pct[(label_pct >= 0.1) & (label_pct < 1.0)].index.tolist()
    extreme = label_pct[label_pct < 0.1].index.tolist()

    print(f"Total Traffic Samples: {total_samples:,}")
    print(f"Fine-Grained Classes: {len(label_counts)} (33 Attacks + Benign) | Categories: {len(cat_counts)} (8 Attack Groups + Benign)")
    print(f"Fine-Grained IR:       {ir_fine:,.1f}:1 (Max: {label_counts.idxmax()}={label_counts.max():,}, Min: {label_counts.idxmin()}={label_counts.min():,})")
    print(f"Category IR:           {ir_cat:,.1f}:1 (Max: {cat_counts.idxmax()}={cat_counts.max():,}, Min: {cat_counts.idxmin()}={cat_counts.min():,})")
    print(f"Shannon Entropy Ratio: {entropy_ratio_fine:.4f} (Fine-Grained) | {entropy_ratio_cat:.4f} (Category)")
    print("-" * 80)
    print(f"Severity Tiers (34 Classes):")
    print(f"  [1] Majority (>= 5.0%):     {len(majority):2d} classes ({sum(label_pct[c] for c in majority):.2f}% of traffic)")
    print(f"  [2] Medium (1.0% - 5.0%):   {len(medium):2d} classes ({sum(label_pct[c] for c in medium):.2f}% of traffic)")
    print(f"  [3] Minority (0.1% - 1.0%): {len(minority):2d} classes ({sum(label_pct[c] for c in minority):.2f}% of traffic)")
    print(f"  [4] Extreme (< 0.1%):       {len(extreme):2d} classes ({sum(label_pct[c] for c in extreme):.2f}% of traffic)")

    return {
        'label_counts': label_counts,
        'cat_counts': cat_counts,
        'ir_fine': ir_fine,
        'ir_cat': ir_cat,
        'entropy_ratio_fine': entropy_ratio_fine,
        'entropy_ratio_cat': entropy_ratio_cat,
        'majority': majority,
        'medium': medium,
        'minority': minority,
        'extreme': extreme
    }


def compute_all_weights(y_encoded, y_cat_encoded, y_binary, reverse_label_map, reverse_cat_map,
                        smoothing='none'):
    """
    Task 3.2A: Cost-Sensitive Learning - Inverse Frequency Class Weights
    Computes and returns BOTH raw inverse-frequency weights and smoothed weights across 3 tiers.
    smoothing: 'none', 'sqrt', 'log', 'cap' (cap max weight at 100)
    """
    print("\n" + "=" * 80)
    print("TASK 3.2A: COST-SENSITIVE LEARNING (INVERSE FREQUENCY WEIGHTS)")
    print("=" * 80)

    def apply_smoothing(weights_dict, mode):
        if mode == 'sqrt':
            return {k: round(float(np.sqrt(v)), 6) for k, v in weights_dict.items()}
        elif mode == 'log':
            return {k: round(float(np.log1p(v)), 6) for k, v in weights_dict.items()}
        elif mode == 'cap':
            return {k: round(float(min(v, 100.0)), 6) for k, v in weights_dict.items()}
        return {k: round(float(v), 6) for k, v in weights_dict.items()}

    # 1. Fine-grained (34 classes)
    unique_labels = np.unique(y_encoded)
    cw_arr = compute_class_weight('balanced', classes=unique_labels, y=y_encoded)
    raw_cw_encoded = dict(zip(unique_labels.tolist(), [round(float(w), 6) for w in cw_arr]))
    raw_cw_named = {reverse_label_map[c]: raw_cw_encoded[c] for c in unique_labels}
    cw_encoded = apply_smoothing(raw_cw_encoded, smoothing)
    cw_named = {reverse_label_map[c]: cw_encoded[c] for c in unique_labels}

    # 2. Categories (9 groups)
    unique_cats = np.unique(y_cat_encoded)
    cat_arr = compute_class_weight('balanced', classes=unique_cats, y=y_cat_encoded)
    raw_cat_encoded = dict(zip(unique_cats.tolist(), [round(float(w), 6) for w in cat_arr]))
    raw_cat_named = {reverse_cat_map[c]: raw_cat_encoded[c] for c in unique_cats}
    cat_encoded = apply_smoothing(raw_cat_encoded, smoothing)
    cat_named = {reverse_cat_map[c]: cat_encoded[c] for c in unique_cats}

    # 3. Binary (Benign vs Attack)
    unique_bin = np.unique(y_binary)
    bin_arr = compute_class_weight('balanced', classes=unique_bin, y=y_binary)
    bin_weights = dict(zip(unique_bin.tolist(), [round(float(w), 6) for w in bin_arr]))

    if smoothing != 'none':
        print(f"Applied weight smoothing: '{smoothing}' (prevents gradient explosion in neural nets/logistic).")

    print(f"Computed weights for {len(cw_encoded)} fine-grained classes.")
    print(f"  Highest weight: {max(cw_named.items(), key=lambda x: x[1])}")
    print(f"  Lowest weight:  {min(cw_named.items(), key=lambda x: x[1])}")
    print(f"Computed weights for {len(cat_encoded)} categories (8 Attack Groups + Benign).")
    for cat_name, w in sorted(cat_named.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat_name:<15}: weight = {w:8.4f}")
    print(f"Binary weights: Benign (0) = {bin_weights.get(0, 0):.4f}, Attack (1) = {bin_weights.get(1, 0):.4f}")

    return cw_encoded, cw_named, raw_cw_named, cat_encoded, cat_named, raw_cat_named, bin_weights


def guaranteed_minority_subsample(X, y, sample_size, min_subsample_keep=100, random_state=42):
    """
    Proportionally Scaled Guaranteed Minority Preservation.
    Fixes Bug #2:
    - Scales threshold according to ratio = sample_size / total_rows.
    - Any class whose expected count in the subsample is < min_subsample_keep has 100% of its real instances preserved!
    - Remaining budget is drawn via stratified sampling from larger classes.
    - Explicit assertion guarantees ZERO classes are dropped.
    """
    total_rows = len(y)
    if sample_size is None or sample_size >= total_rows:
        return X.copy(), np.array(y)

    counts = pd.Series(y).value_counts()
    ratio = sample_size / total_rows
    # Identify classes where stratified sampling would result in fewer than min_subsample_keep real samples
    rare_classes = counts[(counts * ratio) < min_subsample_keep].index.tolist()
    
    if rare_classes:
        print(f"  [Safety Guard] Preserving 100% of samples for {len(rare_classes)} rare classes (expected < {min_subsample_keep} in {sample_size:,} subsample)...")
        y_series = pd.Series(y)
        rare_indices = y_series[y_series.isin(rare_classes)].index.values
        common_indices = y_series[~y_series.isin(rare_classes)].index.values

        X_rare = X.iloc[rare_indices] if hasattr(X, 'iloc') else X[rare_indices]
        y_rare = y[rare_indices]

        X_common = X.iloc[common_indices] if hasattr(X, 'iloc') else X[common_indices]
        y_common = y[common_indices]

        remaining_budget = max(len(counts) * 20, sample_size - len(y_rare))
        if remaining_budget < len(y_common):
            _, X_sub_com, _, y_sub_com = train_test_split(
                X_common, y_common,
                test_size=remaining_budget,
                stratify=y_common,
                random_state=random_state
            )
        else:
            X_sub_com, y_sub_com = X_common, y_common

        feat_cols = X.columns if hasattr(X, 'columns') else None
        if hasattr(X, 'iloc'):
            X_sub = pd.concat([X_rare, X_sub_com], axis=0).reset_index(drop=True)
        else:
            X_sub = pd.DataFrame(np.vstack([X_rare, X_sub_com]), columns=feat_cols)
        y_sub = np.concatenate([y_rare, y_sub_com])
    else:
        _, X_sub, _, y_sub = train_test_split(
            X, y,
            test_size=sample_size,
            stratify=y,
            random_state=random_state
        )
        X_sub = pd.DataFrame(X_sub, columns=X.columns) if hasattr(X, 'columns') else pd.DataFrame(X_sub)
        y_sub = np.array(y_sub)

    # Assert 100% class preservation
    n_orig = pd.Series(y).nunique()
    n_sub = pd.Series(y_sub).nunique()
    assert n_sub == n_orig, f"CRITICAL BUG: Classes were lost in subsampling! (Expected {n_orig}, got {n_sub})"
    print(f"  Subsampling complete: {len(y_sub):,} samples preserving all {n_sub} classes perfectly.")
    return X_sub, y_sub


def adaptive_smote_pre_ros(X_work, y_work, strategy_over, k_neighbors=5, random_state=42):
    """
    Adaptive ROS fallback when a class has n <= k_neighbors.
    SMOTE requires at least k+1 samples to form k nearest neighbors.
    Pre-amplifies ultra-rare classes with RandomOverSampler up to k+2 samples so SMOTE never crashes!
    """
    counts = pd.Series(y_work).value_counts()
    ros_fix = {}
    for cls, target in strategy_over.items():
        if counts.get(cls, 0) <= k_neighbors:
            ros_fix[cls] = k_neighbors + 2

    if ros_fix:
        print(f"  [Adaptive ROS Fallback] Pre-amplifying {len(ros_fix)} classes (n <= {k_neighbors}) up to {k_neighbors+2} samples for SMOTE stability...")
        ros = RandomOverSampler(sampling_strategy=ros_fix, random_state=random_state)
        X_work, y_work = ros.fit_resample(X_work, y_work)
    
    return X_work, y_work


def apply_resampling_pipeline(X_train, y_target, reverse_map,
                              strategy='hybrid', sample_size=150000,
                              min_subsample_keep=100,
                              target_majority=30000, target_minority=8000,
                              k_neighbors=5, random_state=42):
    """
    Main Resampling Controller supporting:
    - 'hybrid': Scalable SMOTE + Tomek Links for Big Data IoT
    - 'smotetomek': Pure Paper-Standard SMOTE + Tomek Links
    - 'borderlinesmote': Borderline-SMOTE + Tomek Links
    - 'smote': SMOTE only
    - 'smoteenn': Memory-Safe SMOTE + ENN
    """
    print("\n" + "=" * 80)
    print(f"TASK 3.2B: RESAMPLING PIPELINE [Strategy: {strategy.upper()}]")
    print("=" * 80)

    # Proportionally scaled guaranteed minority preservation during subsampling
    X_sub, y_sub = guaranteed_minority_subsample(
        X_train, y_target, sample_size=sample_size, min_subsample_keep=min_subsample_keep, random_state=random_state
    )

    counts_before = pd.Series(y_sub).value_counts()
    print(f"Class count range before: min={counts_before.min():,}, max={counts_before.max():,}")

    t0 = time.time()
    
    # 1. Determine oversampling targets
    strategy_over = {c: target_minority for c, cnt in counts_before.items() if cnt < target_minority}
    # 2. Determine undersampling targets
    strategy_under = {c: target_majority for c, cnt in counts_before.items() if cnt > target_majority}

    if strategy == 'hybrid':
        # Stage 1: Smart Majority Pruning
        if strategy_under:
            print(f"  [Stage 1: Majority Pruning] Pruning {len(strategy_under)} majority classes down to {target_majority:,}...")
            rus = RandomUnderSampler(sampling_strategy=strategy_under, random_state=random_state)
            X_work, y_work = rus.fit_resample(X_sub, y_sub)
        else:
            X_work, y_work = X_sub.copy(), y_sub.copy()

        # Stage 2: SMOTE Synthesis with Adaptive ROS Fallback
        if strategy_over:
            X_work, y_work = adaptive_smote_pre_ros(X_work, y_work, strategy_over, k_neighbors=k_neighbors, random_state=random_state)
            print(f"  [Stage 2: SMOTE Synthesis] Interpolating {len(strategy_over)} minority classes up to {target_minority:,}...")
            smote = SMOTE(sampling_strategy=strategy_over, k_neighbors=k_neighbors, random_state=random_state)
            X_work, y_work = smote.fit_resample(X_work, y_work)

        # Stage 3: Tomek Links Boundary Cleaning
        print("  [Stage 3: Tomek Links Cleaning] Detecting and removing borderline ambiguous pairs...")
        tomek = TomekLinks(sampling_strategy='all', n_jobs=-1)
        X_bal, y_bal = tomek.fit_resample(X_work, y_work)

    elif strategy == 'borderlinesmote':
        if strategy_under:
            rus = RandomUnderSampler(sampling_strategy=strategy_under, random_state=random_state)
            X_work, y_work = rus.fit_resample(X_sub, y_sub)
        else:
            X_work, y_work = X_sub.copy(), y_sub.copy()

        if strategy_over:
            X_work, y_work = adaptive_smote_pre_ros(X_work, y_work, strategy_over, k_neighbors=k_neighbors, random_state=random_state)
            print(f"  [BorderlineSMOTE] Synthesizing along decision boundaries...")
            bsmote = BorderlineSMOTE(sampling_strategy=strategy_over, k_neighbors=k_neighbors, random_state=random_state)
            X_work, y_work = bsmote.fit_resample(X_work, y_work)

        tomek = TomekLinks(sampling_strategy='all', n_jobs=-1)
        X_bal, y_bal = tomek.fit_resample(X_work, y_work)

    elif strategy == 'smotetomek':
        X_work, y_work = X_sub.copy(), y_sub.copy()
        if strategy_over:
            X_work, y_work = adaptive_smote_pre_ros(X_work, y_work, strategy_over, k_neighbors=k_neighbors, random_state=random_state)
            smote = SMOTE(sampling_strategy=strategy_over, k_neighbors=k_neighbors, random_state=random_state)
        else:
            smote = 'passthrough'
        
        tomek = TomekLinks(sampling_strategy='all', n_jobs=-1)
        if strategy_over:
            smt = SMOTETomek(smote=smote, tomek=tomek, random_state=random_state)
            X_bal, y_bal = smt.fit_resample(X_work, y_work)
        else:
            X_bal, y_bal = tomek.fit_resample(X_work, y_work)

    elif strategy == 'smote':
        X_work, y_work = X_sub.copy(), y_sub.copy()
        if strategy_over:
            X_work, y_work = adaptive_smote_pre_ros(X_work, y_work, strategy_over, k_neighbors=k_neighbors, random_state=random_state)
            smote = SMOTE(sampling_strategy=strategy_over, k_neighbors=k_neighbors, random_state=random_state)
            X_bal, y_bal = smote.fit_resample(X_work, y_work)
        else:
            X_bal, y_bal = X_work, y_work

    elif strategy == 'smoteenn':
        # Memory-safe SMOTEENN with explicit sampling strategy
        X_work, y_work = X_sub.copy(), y_sub.copy()
        if strategy_over:
            X_work, y_work = adaptive_smote_pre_ros(X_work, y_work, strategy_over, k_neighbors=k_neighbors, random_state=random_state)
            smote = SMOTE(sampling_strategy=strategy_over, k_neighbors=k_neighbors, random_state=random_state)
            sme = SMOTEENN(smote=smote, random_state=random_state, n_jobs=-1)
            X_bal, y_bal = sme.fit_resample(X_work, y_work)
        else:
            X_bal, y_bal = X_work, y_work
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    elapsed = time.time() - t0
    counts_after = pd.Series(y_bal).value_counts()

    print(f"\nResampling complete in {elapsed:.2f} seconds!")
    print(f"Sample progression: {len(y_sub):,} -> {len(y_bal):,} rows")
    print(f"New class count range: min={counts_after.min():,}, max={counts_after.max():,}")
    print(f"New Imbalance Ratio: {counts_after.max() / counts_after.min():.2f}:1 (previously {counts_before.max() / counts_before.min():.2f}:1)")

    return X_bal, y_bal, counts_before, counts_after, elapsed


def generate_visualizations(analysis_results, class_weights_named, cat_weights_named,
                            strategy_name='Hybrid', target_tier='label',
                            counts_before=None, counts_after=None,
                            X_pca_sample=None, y_pca_sample=None, reverse_map=None):
    """
    Generate all Phase 3 publication-ready plots.
    Fixes:
    - Sets explicit ticks before set_xticklabels (fixes matplotlib warning)
    - Dynamic titles reflecting strategy and tier
    - Seed-controlled PCA sampling with rng
    - Tab20/continuous colormaps for fine-grained 34 classes
    """
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    os.makedirs('plots', exist_ok=True)

    label_counts = analysis_results['label_counts']
    majority = analysis_results['majority']
    medium = analysis_results['medium']
    minority = analysis_results['minority']
    extreme = analysis_results['extreme']

    # --------------------------------------------------------------------------
    # Plot 1: 06_imbalance_analysis.png
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    ax1 = axes[0]
    colors_bar = plt.cm.plasma(np.linspace(0.15, 0.85, len(label_counts)))[::-1]
    ax1.barh(range(len(label_counts)), label_counts.values, color=colors_bar, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(label_counts)))
    ax1.set_yticklabels(label_counts.index, fontsize=7.5, fontweight='medium')
    ax1.set_xscale('log')
    ax1.set_xlabel('Sample Count (Log Scale)', fontsize=11, fontweight='bold')
    ax1.set_title(f'CICIoT2023 Class Distribution (33 Attacks + Benign = 34 Classes)\nImbalance Ratio = {analysis_results["ir_fine"]:,.1f}:1',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5, axis='x')
    ax1.invert_yaxis()

    ax2 = axes[1]
    tier_labels = ['Majority\n(>=5%)', 'Medium\n(1%-5%)', 'Minority\n(0.1%-1%)', 'Extreme\n(<0.1%)']
    tier_counts = [len(majority), len(medium), len(minority), len(extreme)]
    tier_colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    bars = ax2.bar(tier_labels, tier_counts, color=tier_colors, edgecolor='black', linewidth=0.8, width=0.55)
    ax2.set_ylabel('Number of Attack Classes', fontsize=11, fontweight='bold')
    ax2.set_title('Class Imbalance Severity Breakdown', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    for bar, count in zip(bars, tier_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{count} classes", ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('plots/06_imbalance_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  [Saved] plots/06_imbalance_analysis.png")

    # --------------------------------------------------------------------------
    # Plot 2: 07_class_weights.png
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(20, 11))
    
    # Fine-grained weights
    ax1 = axes[0]
    sorted_cw = sorted(class_weights_named.items(), key=lambda x: x[1])
    cw_names = [x[0] for x in sorted_cw]
    cw_vals = [x[1] for x in sorted_cw]
    colors_w = plt.cm.viridis(np.linspace(0.1, 0.95, len(cw_vals)))
    ax1.barh(range(len(cw_names)), cw_vals, color=colors_w, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(cw_names)))
    ax1.set_yticklabels(cw_names, fontsize=7)
    ax1.set_xscale('log')
    ax1.set_xlabel('Class Weight Value (Log Scale)', fontsize=11, fontweight='bold')
    ax1.set_title('Inverse Frequency Class Weights (34 Classes)', fontsize=12, fontweight='bold')
    ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=1.2, label='Neutral Weight = 1.0')
    ax1.grid(True, linestyle='--', alpha=0.5, axis='x')
    ax1.legend()

    # Category weights
    ax2 = axes[1]
    sorted_cat = sorted(cat_weights_named.items(), key=lambda x: x[1])
    cat_names = [x[0] for x in sorted_cat]
    cat_vals = [x[1] for x in sorted_cat]
    bars_cat = ax2.bar(cat_names, cat_vals, color='#3498db', edgecolor='black', linewidth=0.8, width=0.55)
    ax2.set_ylabel('Category Weight Value', fontsize=11, fontweight='bold')
    ax2.set_title('Inverse Frequency Category Weights (8 Attack Groups + Benign = 9 Classes)', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(cat_names)))  # Fix: explicit ticks before set_xticklabels
    ax2.set_xticklabels(cat_names, rotation=35, ha='right', fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    for bar, val in zip(bars_cat, cat_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{val:.2f}", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('plots/07_class_weights.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  [Saved] plots/07_class_weights.png")

    # --------------------------------------------------------------------------
    # Plot 3: 08_resampling_comparison.png (Before vs After)
    # --------------------------------------------------------------------------
    if counts_before is not None and counts_after is not None:
        fig, ax = plt.subplots(figsize=(16, 8))
        common_indices = counts_before.index
        x_indices = np.arange(len(common_indices))
        bar_width = 0.4

        counts_b_aligned = [counts_before.get(k, 0) for k in common_indices]
        counts_a_aligned = [counts_after.get(k, 0) for k in common_indices]
        tick_labels = [reverse_map.get(k, str(k)) for k in common_indices] if reverse_map else [str(k) for k in common_indices]

        ax.bar(x_indices - bar_width/2, counts_b_aligned, width=bar_width,
               label='Original (Before Resampling)', color='#e74c3c', alpha=0.85, edgecolor='black')
        ax.bar(x_indices + bar_width/2, counts_a_aligned, width=bar_width,
               label=f'Resampled ({strategy_name})', color='#2ecc71', alpha=0.85, edgecolor='black')

        ax.set_ylabel('Sample Count (Log Scale)', fontsize=11, fontweight='bold')
        ax.set_yscale('log')
        ax.set_title(f'Comparison: Class Distribution Before vs. After [{strategy_name.upper()}] ({target_tier.title()} Level)',
                     fontsize=13, fontweight='bold')
        ax.set_xticks(x_indices)
        ax.set_xticklabels(tick_labels, rotation=35, ha='right', fontsize=9 if len(tick_labels) > 15 else 10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        ax.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig('plots/08_resampling_comparison.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("  [Saved] plots/08_resampling_comparison.png")

    # --------------------------------------------------------------------------
    # Plot 4: 08b_smote_tomek_boundary.png (PCA 2D Decision Space Visualization)
    # --------------------------------------------------------------------------
    if X_pca_sample is not None and y_pca_sample is not None:
        try:
            print("  Generating 2D PCA decision space visualization with seed control...")
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X_pca_sample)
            
            fig, ax = plt.subplots(figsize=(13, 8))
            unique_classes = np.unique(y_pca_sample)
            n_classes = len(unique_classes)
            
            if n_classes <= 10:
                palette = plt.cm.tab10(np.linspace(0, 1, n_classes))
            elif n_classes <= 20:
                palette = plt.cm.tab20(np.linspace(0, 1, n_classes))
            else:
                palette = plt.cm.gist_rainbow(np.linspace(0, 1, n_classes))
            
            for cls_idx, color in zip(unique_classes, palette):
                mask = (y_pca_sample == cls_idx)
                label_txt = reverse_map.get(cls_idx, f"Class {cls_idx}") if reverse_map else f"Class {cls_idx}"
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                           c=[color], label=label_txt, alpha=0.65, s=16, edgecolor='none')

            ax.set_title(f'PCA 2D Decision Space After [{strategy_name.upper()}] ({target_tier.title()} Level)\nExplained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%',
                         fontsize=12, fontweight='bold')
            ax.set_xlabel('Principal Component 1', fontweight='bold')
            ax.set_ylabel('Principal Component 2', fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8.5 if n_classes > 15 else 9.5)

            plt.tight_layout()
            plt.savefig('plots/08b_smote_tomek_boundary.png', dpi=200, bbox_inches='tight')
            plt.close()
            print("  [Saved] plots/08b_smote_tomek_boundary.png")
        except Exception as e:
            print(f"  Warning: Could not generate PCA plot: {e}")


def save_artifacts(cw_encoded, cw_named, raw_cw_named,
                   cat_encoded, cat_named, raw_cat_named, bin_weights,
                   preprocessor, X_balanced=None, y_balanced_dict=None,
                   resampling_summary=None, smoothing='none', batch_suffix=None):
    """
    Task 3.3: Save all deliverables into models/ and data/
    Saves BOTH raw and smoothed weights for rigorous ablation study.
    """
    print("\n" + "=" * 80)
    print("TASK 3.3: SERIALIZING DELIVERABLES")
    print("=" * 80)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    suffix = f"_{batch_suffix}" if batch_suffix else ""

    # 1. Save Class Weights dictionaries (both smoothed and raw)
    all_weights = {
        'fine_grained_encoded': cw_encoded,
        'fine_grained_named': cw_named,
        'fine_grained_raw': raw_cw_named,
        'category_encoded': cat_encoded,
        'category_named': cat_named,
        'category_raw': raw_cat_named,
        'binary': bin_weights,
        'smoothing_mode': smoothing
    }
    joblib.dump(all_weights, f'data/class_weights{suffix}.pkl')
    joblib.dump(all_weights, 'data/class_weights.pkl')
    print(f"  [Saved] data/class_weights{suffix}.pkl")

    # 2. Save JSON format for human readability
    weights_json = {
        'fine_grained_smoothed': cw_named,
        'fine_grained_raw': raw_cw_named,
        'categories_smoothed': cat_named,
        'categories_raw': raw_cat_named,
        'binary': {str(k): v for k, v in bin_weights.items()},
        'smoothing_mode': smoothing
    }
    with open(f'models/class_weights{suffix}.json', 'w') as f:
        json.dump(weights_json, f, indent=4)
    print(f"  [Saved] models/class_weights{suffix}.json")

    # 3. Update preprocessor joblib
    preprocessor['class_weights_encoded'] = cw_encoded
    preprocessor['class_weights_named'] = cw_named
    preprocessor['category_weights_encoded'] = cat_encoded
    preprocessor['binary_weights'] = bin_weights
    preprocessor['class_weights_raw'] = raw_cw_named
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    print("  [Updated] models/preprocessor.joblib")

    # 4. Save balanced dataset if generated
    if X_balanced is not None and y_balanced_dict is not None:
        x_bal_path = f'data/X_train_balanced{suffix}.pkl'
        y_bal_path = f'data/y_train_balanced{suffix}.pkl'
        joblib.dump(X_balanced, x_bal_path, compress=3)
        joblib.dump(y_balanced_dict, y_bal_path, compress=3)
        print(f"  [Saved] {x_bal_path} & {y_bal_path}")

    # 5. Save Resampling Summary JSON
    if resampling_summary:
        with open(f'models/resampling_summary{suffix}.json', 'w') as f:
            json.dump(resampling_summary, f, indent=4)
        print(f"  [Saved] models/resampling_summary{suffix}.json")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Hardened Imbalance Handling Pipeline")
    parser.add_argument('--strategy', type=str, default='hybrid',
                        choices=['hybrid', 'smotetomek', 'borderlinesmote', 'smote', 'smoteenn', 'weights_only'],
                        help="Resampling strategy: 'hybrid' (Scalable IoT), 'smotetomek', 'borderlinesmote', 'smote', 'smoteenn', 'weights_only'")
    parser.add_argument('--target-tier', type=str, default='label',
                        choices=['label', 'category'],
                        help="Target tier: 'label' (34 fine-grained classes) or 'category' (9 classes: 8 attack groups + Benign)")
    parser.add_argument('--sample-size', type=int, default=150000,
                        help="Sample size before resampling with guaranteed minority preservation (default: 150000)")
    parser.add_argument('--min-keep', type=int, default=100,
                        help="Minimum expected samples threshold to guarantee full (100%%) preservation (default: 100)")
    parser.add_argument('--target-majority', type=int, default=30000,
                        help="Max samples per majority class after pruning (default: 30000)")
    parser.add_argument('--target-minority', type=int, default=8000,
                        help="Min samples per minority class after SMOTE (default: 8000)")
    parser.add_argument('--k-neighbors', type=int, default=5,
                        help="Number of nearest neighbors for SMOTE (default: 5)")
    parser.add_argument('--weight-smoothing', type=str, default='none',
                        choices=['none', 'sqrt', 'log', 'cap'],
                        help="Class weight smoothing mode: 'none', 'sqrt', 'log', 'cap' (cap at 100)")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Thư mục làm việc/lưu trữ kết quả (Mặc định: auto-detect '/content/drive/MyDrive/do_an' nếu trên Colab, hoặc '.')")
    parser.add_argument('--random-state', type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument('--skip-plots', action='store_true',
                        help="Skip generating visualization plots")
    parser.add_argument('--batch-suffix', type=str, default=None,
                        help="Suffix for output dataset files (e.g. '500k' -> X_train_balanced_500k.pkl)")
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
        os.makedirs('plots', exist_ok=True)

    # Step 1: Load Data
    X_train, y_train_dict, preprocessor = load_phase2_data()

    y_encoded = y_train_dict['label_encoded']
    y_names = y_train_dict['label_name']
    y_cat_encoded = y_train_dict['category_encoded']
    y_cat_names = y_train_dict['category_name']
    y_binary = y_train_dict['is_attack']

    label_mapping = preprocessor['label_mapping']
    category_mapping = preprocessor['category_mapping']
    reverse_label = {v: k for k, v in label_mapping.items()}
    reverse_cat = {v: k for k, v in category_mapping.items()}

    # Task 3.1: Quantitative Imbalance Analysis
    analysis_results = analyze_imbalance(y_names, y_cat_names, label_mapping, category_mapping)

    # Task 3.2A: Compute Class Weights (Always executed)
    cw_enc, cw_named, raw_cw_named, cat_enc, cat_named, raw_cat_named, bin_w = compute_all_weights(
        y_encoded, y_cat_encoded, y_binary, reverse_label, reverse_cat,
        smoothing=args.weight_smoothing
    )

    # Task 3.2B: Resampling Pipeline
    X_bal, y_bal = None, None
    y_balanced_dict = None
    counts_before, counts_after = None, None
    resampling_summary = None

    if args.strategy != 'weights_only':
        target_series = y_encoded if args.target_tier == 'label' else y_cat_encoded
        target_rev_map = reverse_label if args.target_tier == 'label' else reverse_cat

        X_bal, y_bal, counts_before, counts_after, elapsed = apply_resampling_pipeline(
            X_train, target_series, target_rev_map,
            strategy=args.strategy,
            sample_size=args.sample_size,
            min_subsample_keep=args.min_keep,
            target_majority=args.target_majority,
            target_minority=args.target_minority,
            k_neighbors=args.k_neighbors,
            random_state=args.random_state
        )

        # Structured y_balanced dictionary to avoid interface mismatch
        y_balanced_dict = {
            'target_encoded': y_bal,
            'target_tier': args.target_tier,
            'target_name': np.array([target_rev_map[i] for i in y_bal]),
            'label_mapping': label_mapping,
            'category_mapping': category_mapping
        }

        resampling_summary = {
            'strategy': args.strategy,
            'target_tier': args.target_tier,
            'initial_sample_size': args.sample_size,
            'min_keep_threshold': args.min_keep,
            'final_sample_size': len(y_bal),
            'execution_time_seconds': round(elapsed, 2),
            'target_majority': args.target_majority,
            'target_minority': args.target_minority,
            'k_neighbors': args.k_neighbors,
            'weight_smoothing': args.weight_smoothing,
            'class_distribution_before': {str(k): int(v) for k, v in counts_before.items()},
            'class_distribution_after': {str(k): int(v) for k, v in counts_after.items()},
            'warning': "Do NOT apply class_weights when training models on X_train_balanced.pkl to avoid double-correction bias."
        }

    # Generate Visualizations
    if not args.skip_plots:
        X_pca, y_pca = None, None
        if X_bal is not None:
            rng = np.random.default_rng(args.random_state)
            n_pca = min(5000, len(y_bal))
            idx_pca = rng.choice(len(y_bal), size=n_pca, replace=False)
            X_pca = X_bal.iloc[idx_pca] if hasattr(X_bal, 'iloc') else X_bal[idx_pca]
            y_pca = y_bal[idx_pca]

        generate_visualizations(
            analysis_results, cw_named, cat_named,
            strategy_name=args.strategy,
            target_tier=args.target_tier,
            counts_before=counts_before, counts_after=counts_after,
            X_pca_sample=X_pca, y_pca_sample=y_pca,
            reverse_map=reverse_label if args.target_tier == 'label' else reverse_cat
        )

    # Task 3.3: Save deliverables
    save_artifacts(cw_enc, cw_named, raw_cw_named,
                   cat_enc, cat_named, raw_cat_named, bin_w,
                   preprocessor, X_bal, y_balanced_dict, resampling_summary,
                   smoothing=args.weight_smoothing,
                   batch_suffix=args.batch_suffix)

    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()
