#!/usr/bin/env python3
"""
Phase 3: Class Imbalance Handling Pipeline (Paper-Aligned SMOTETomek & Hybrid Resampling)
CICIoT2023 Network Intrusion Detection
Graduation Thesis: IoT Attack Detection & Explanation with XAI
Plan: PLAN.md (Phase 3)

Key Innovations & Methodologies:
1. Imbalance Analysis:
   - Imbalance Ratio (IR) & Shannon Entropy
   - 4-Tier Severity Stratification: Majority, Medium, Minority, Extreme Minority
2. Cost-Sensitive Learning (Inverse Frequency Class Weights):
   - Computes balanced weights for 3 tiers: 34 Attacks, 8 Categories, Binary
   - Preserves natural traffic distribution with O(1) memory overhead
3. Paper-Aligned Resampling (SMOTE + Tomek Links):
   - SMOTE: Generates synthetic interpolated samples for minority attacks
   - Tomek Links: Detects and removes ambiguous borderline noise samples
   - Two-Stage Scalable Hybrid: Majority Pruning + SMOTE Synthesis + Tomek Cleaning
4. Rich Visualizations:
   - plots/06_imbalance_analysis.png: Log-scale distribution & severity breakdown
   - plots/07_class_weights.png: Inverse-frequency class weights
   - plots/08_resampling_comparison.png: Before vs After class distribution
   - plots/08b_smote_tomek_boundary.png: 2D PCA boundary cleaning visualization
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
    x_paths = ['X_train.pkl', 'data/X_train.pkl']
    for p in x_paths:
        if os.path.exists(p):
            print(f"Loading features from: {p}")
            X_train = joblib.load(p)
            break
    else:
        raise FileNotFoundError("Could not find X_train.pkl in root or data/ directory!")

    # Load y_train
    y_paths = ['y_train.pkl', 'data/y_train.pkl']
    for p in y_paths:
        if os.path.exists(p):
            print(f"Loading labels from: {p}")
            y_train_dict = joblib.load(p)
            break
    else:
        raise FileNotFoundError("Could not find y_train.pkl in root or data/ directory!")

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

    # 4-Tier Severity breakdown for 34 attack classes
    majority = label_pct[label_pct >= 5.0].index.tolist()
    medium = label_pct[(label_pct >= 1.0) & (label_pct < 5.0)].index.tolist()
    minority = label_pct[(label_pct >= 0.1) & (label_pct < 1.0)].index.tolist()
    extreme = label_pct[label_pct < 0.1].index.tolist()

    print(f"Total Traffic Samples: {total_samples:,}")
    print(f"Fine-Grained Classes: {len(label_counts)} | Categories: {len(cat_counts)}")
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


def compute_all_weights(y_encoded, y_cat_encoded, y_binary, reverse_label_map, reverse_cat_map):
    """
    Task 3.2A: Cost-Sensitive Learning - Inverse Frequency Class Weights
    Computes exact weights across 3 classification tiers.
    """
    print("\n" + "=" * 80)
    print("TASK 3.2A: COST-SENSITIVE LEARNING (INVERSE FREQUENCY WEIGHTS)")
    print("=" * 80)

    # 1. Fine-grained (34 classes)
    unique_labels = np.unique(y_encoded)
    cw_arr = compute_class_weight('balanced', classes=unique_labels, y=y_encoded)
    cw_encoded = dict(zip(unique_labels.tolist(), cw_arr.tolist()))
    cw_named = {reverse_label_map[c]: round(w, 6) for c, w in cw_encoded.items()}

    # 2. Categories (8 groups)
    unique_cats = np.unique(y_cat_encoded)
    cat_arr = compute_class_weight('balanced', classes=unique_cats, y=y_cat_encoded)
    cat_encoded = dict(zip(unique_cats.tolist(), cat_arr.tolist()))
    cat_named = {reverse_cat_map[c]: round(w, 6) for c, w in cat_encoded.items()}

    # 3. Binary (Benign vs Attack)
    unique_bin = np.unique(y_binary)
    bin_arr = compute_class_weight('balanced', classes=unique_bin, y=y_binary)
    bin_weights = dict(zip(unique_bin.tolist(), bin_arr.tolist()))

    print(f"Computed weights for {len(cw_encoded)} fine-grained classes.")
    print(f"  Highest weight: {max(cw_named.items(), key=lambda x: x[1])}")
    print(f"  Lowest weight:  {min(cw_named.items(), key=lambda x: x[1])}")
    print(f"Computed weights for {len(cat_encoded)} categories.")
    for cat_name, w in sorted(cat_named.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat_name:<15}: weight = {w:8.4f}")
    print(f"Binary weights: Benign (0) = {bin_weights.get(0, 0):.4f}, Attack (1) = {bin_weights.get(1, 0):.4f}")

    return cw_encoded, cw_named, cat_encoded, cat_named, bin_weights


def paper_smotetomek_resample(X_sub, y_sub, target_minority=5000, k_neighbors=5, random_state=42):
    """
    Paper-Standard Resampling: SMOTE + Tomek Links (SMOTETomek).
    1. SMOTE creates synthetic samples for minority classes.
    2. Tomek Links identifies & removes overlapping borderline noise samples.
    """
    counts = pd.Series(y_sub).value_counts()
    min_count = counts.min()
    effective_k = max(1, min(k_neighbors, min_count - 1)) if min_count > 1 else 1

    strategy_over = {}
    for cls, cnt in counts.items():
        if cnt < target_minority:
            strategy_over[cls] = target_minority

    print(f"  [SMOTE] Classes targeted for synthesis: {len(strategy_over)} (target={target_minority:,}, k={effective_k})")
    
    if strategy_over:
        smote = SMOTE(sampling_strategy=strategy_over, k_neighbors=effective_k, random_state=random_state)
    else:
        smote = 'passthrough'

    tomek = TomekLinks(sampling_strategy='all', n_jobs=-1)

    if strategy_over:
        smt = SMOTETomek(smote=smote, tomek=tomek, random_state=random_state)
        X_res, y_res = smt.fit_resample(X_sub, y_sub)
    else:
        X_res, y_res = tomek.fit_resample(X_sub, y_sub)

    return X_res, y_res


def scalable_hybrid_resample(X_sub, y_sub, target_majority=50000, target_minority=8000,
                            k_neighbors=5, random_state=42):
    """
    Paper-Enhanced Scalable Two-Stage Hybrid Strategy for Big Data IoT:
    Stage 1: Smart Pruning of massive majority classes (DDoS, DoS) via RandomUnderSampler.
    Stage 2: Synthetic Interpolation of minority classes via SMOTE (Borderline/Standard).
    Stage 3: Tomek Links Boundary Cleaning to eliminate overlapping synthetic & real noise.
    """
    counts_init = pd.Series(y_sub).value_counts()
    
    # 1. Under-sampling strategy
    strategy_under = {}
    for cls, cnt in counts_init.items():
        if cnt > target_majority:
            strategy_under[cls] = target_majority

    if strategy_under:
        print(f"  [Stage 1: Majority Pruning] {len(strategy_under)} majority classes pruned to {target_majority:,}")
        rus = RandomUnderSampler(sampling_strategy=strategy_under, random_state=random_state)
        X_work, y_work = rus.fit_resample(X_sub, y_sub)
    else:
        X_work, y_work = X_sub.copy(), y_sub.copy()

    # 2. Over-sampling strategy (SMOTE)
    counts_mid = pd.Series(y_work).value_counts()
    strategy_over = {}
    for cls, cnt in counts_mid.items():
        if cnt < target_minority:
            strategy_over[cls] = target_minority

    if strategy_over:
        min_cnt = min(counts_mid[cls] for cls in strategy_over.keys())
        eff_k = max(1, min(k_neighbors, min_cnt - 1)) if min_cnt > 1 else 1
        print(f"  [Stage 2: SMOTE Synthesis] {len(strategy_over)} minority classes synthesized to {target_minority:,} (k={eff_k})")
        smote = SMOTE(sampling_strategy=strategy_over, k_neighbors=eff_k, random_state=random_state)
        X_work, y_work = smote.fit_resample(X_work, y_work)
    
    # 3. Boundary Cleaning (Tomek Links)
    print("  [Stage 3: Tomek Links Cleaning] Detecting and removing borderline ambiguous pairs...")
    t0_tomek = time.time()
    tomek = TomekLinks(sampling_strategy='all', n_jobs=-1)
    X_work, y_work = tomek.fit_resample(X_work, y_work)
    print(f"  [Stage 3 Done] Tomek Links boundary cleaning completed in {time.time()-t0_tomek:.2f}s")

    return X_work, y_work


def apply_resampling_pipeline(X_train, y_target, reverse_map,
                              strategy='hybrid', sample_size=200000,
                              target_majority=35000, target_minority=8000,
                              k_neighbors=5, random_state=42):
    """
    Main Resampling Controller supporting:
    - 'smotetomek' (Paper Method)
    - 'hybrid' (Scalable Two-Stage SMOTE + Tomek Links for IoT)
    - 'smoteenn' (SMOTE + Edited Nearest Neighbours)
    - 'smote' (SMOTE only)
    """
    print("\n" + "=" * 80)
    print(f"TASK 3.2B: RESAMPLING PIPELINE [Strategy: {strategy.upper()}]")
    print("=" * 80)

    total_rows = len(y_target)
    # Stratified Subsampling if dataset exceeds sample_size
    if sample_size and sample_size < total_rows:
        print(f"Drawing stratified representative sample: {sample_size:,} / {total_rows:,} rows...")
        from sklearn.model_selection import train_test_split
        _, X_sub, _, y_sub = train_test_split(
            X_train, y_target,
            test_size=sample_size,
            stratify=y_target,
            random_state=random_state
        )
        # Convert to numpy/dataframe clean
        X_sub = pd.DataFrame(X_sub, columns=X_train.columns) if hasattr(X_train, 'columns') else pd.DataFrame(X_sub)
        y_sub = np.array(y_sub)
    else:
        X_sub = X_train.copy()
        y_sub = np.array(y_target)

    counts_before = pd.Series(y_sub).value_counts()
    print(f"Sample size before resampling: {len(y_sub):,} rows across {len(counts_before)} classes")
    print(f"Class count range: min={counts_before.min():,}, max={counts_before.max():,}")

    t0 = time.time()
    if strategy == 'smotetomek':
        X_bal, y_bal = paper_smotetomek_resample(
            X_sub, y_sub,
            target_minority=target_minority,
            k_neighbors=k_neighbors,
            random_state=random_state
        )
    elif strategy == 'hybrid':
        X_bal, y_bal = scalable_hybrid_resample(
            X_sub, y_sub,
            target_majority=target_majority,
            target_minority=target_minority,
            k_neighbors=k_neighbors,
            random_state=random_state
        )
    elif strategy == 'smote':
        counts = pd.Series(y_sub).value_counts()
        strat_over = {c: target_minority for c, cnt in counts.items() if cnt < target_minority}
        min_cnt = min(counts[c] for c in strat_over.keys()) if strat_over else 2
        eff_k = max(1, min(k_neighbors, min_cnt - 1)) if min_cnt > 1 else 1
        smote = SMOTE(sampling_strategy=strat_over, k_neighbors=eff_k, random_state=random_state)
        X_bal, y_bal = smote.fit_resample(X_sub, y_sub)
    elif strategy == 'smoteenn':
        sme = SMOTEENN(random_state=random_state, n_jobs=-1)
        X_bal, y_bal = sme.fit_resample(X_sub, y_sub)
    else:
        raise ValueError(f"Unknown resampling strategy: {strategy}")

    elapsed = time.time() - t0
    counts_after = pd.Series(y_bal).value_counts()

    print(f"\nResampling complete in {elapsed:.2f} seconds!")
    print(f"Rows progression: {len(y_sub):,} -> {len(y_bal):,} rows")
    print(f"New class count range: min={counts_after.min():,}, max={counts_after.max():,}")
    print(f"New Imbalance Ratio: {counts_after.max() / counts_after.min():.2f}:1 (previously {counts_before.max() / counts_before.min():.2f}:1)")

    return X_bal, y_bal, counts_before, counts_after, elapsed


def generate_visualizations(analysis_results, class_weights_named, cat_weights_named,
                            counts_before=None, counts_after=None,
                            X_pca_sample=None, y_pca_sample=None, reverse_cat_map=None):
    """
    Generate all Phase 3 publication-ready plots.
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
    ax1.set_yticklabels(label_counts.index, fontsize=7, fontweight='medium')
    ax1.set_xscale('log')
    ax1.set_xlabel('Sample Count (Log Scale)', fontsize=11, fontweight='bold')
    ax1.set_title(f'CICIoT2023 Class Distribution (34 Attacks + Benign)\nImbalance Ratio = {analysis_results["ir_fine"]:,.1f}:1',
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
    ax1.set_title('Inverse Frequency Class Weights (34 Attacks)', fontsize=12, fontweight='bold')
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
    ax2.set_title('Inverse Frequency Category Weights (8 Attack Groups + Benign)', fontsize=12, fontweight='bold')
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

        # Use category labels if mapping provided
        if reverse_cat_map:
            tick_labels = [reverse_cat_map.get(k, str(k)) for k in common_indices]
        else:
            tick_labels = [str(k) for k in common_indices]

        ax.bar(x_indices - bar_width/2, counts_b_aligned, width=bar_width,
               label='Original (Before Resampling)', color='#e74c3c', alpha=0.85, edgecolor='black')
        ax.bar(x_indices + bar_width/2, counts_a_aligned, width=bar_width,
               label='Resampled (SMOTE + Tomek Links)', color='#2ecc71', alpha=0.85, edgecolor='black')

        ax.set_ylabel('Sample Count (Log Scale)', fontsize=11, fontweight='bold')
        ax.set_yscale('log')
        ax.set_title('Comparison: Class Distribution Before vs. After Paper-Aligned Resampling',
                     fontsize=13, fontweight='bold')
        ax.set_xticks(x_indices)
        ax.set_xticklabels(tick_labels, rotation=35, ha='right', fontsize=10, fontweight='bold')
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
            print("  Generating 2D PCA boundary cleaning visualization...")
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X_pca_sample)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            unique_classes = np.unique(y_pca_sample)
            palette = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
            
            for cls_idx, color in zip(unique_classes, palette):
                mask = (y_pca_sample == cls_idx)
                label_txt = reverse_cat_map.get(cls_idx, f"Class {cls_idx}") if reverse_cat_map else f"Class {cls_idx}"
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                           c=[color], label=label_txt, alpha=0.6, s=15, edgecolor='none')

            ax.set_title(f'PCA 2D Decision Space After SMOTETomek Boundary Cleaning\nExplained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%',
                         fontsize=12, fontweight='bold')
            ax.set_xlabel('Principal Component 1', fontweight='bold')
            ax.set_ylabel('Principal Component 2', fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

            plt.tight_layout()
            plt.savefig('plots/08b_smote_tomek_boundary.png', dpi=200, bbox_inches='tight')
            plt.close()
            print("  [Saved] plots/08b_smote_tomek_boundary.png")
        except Exception as e:
            print(f"  Warning: Could not generate PCA plot: {e}")


def save_artifacts(cw_encoded, cw_named, cat_encoded, cat_named, bin_weights,
                   preprocessor, X_balanced=None, y_balanced=None,
                   resampling_summary=None):
    """
    Task 3.3: Save all deliverables into models/ and data/
    """
    print("\n" + "=" * 80)
    print("TASK 3.3: SERIALIZING DELIVERABLES")
    print("=" * 80)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 1. Save Class Weights dictionaries
    all_weights = {
        'fine_grained_encoded': cw_encoded,
        'fine_grained_named': cw_named,
        'category_encoded': cat_encoded,
        'category_named': cat_named,
        'binary': bin_weights
    }
    joblib.dump(all_weights, 'class_weights.pkl')
    joblib.dump(all_weights, 'data/class_weights.pkl')
    print("  [Saved] class_weights.pkl & data/class_weights.pkl")

    # 2. Save JSON format for human readability
    weights_json = {
        'fine_grained': cw_named,
        'categories': cat_named,
        'binary': {str(k): v for k, v in bin_weights.items()}
    }
    with open('models/class_weights.json', 'w') as f:
        json.dump(weights_json, f, indent=4)
    print("  [Saved] models/class_weights.json")

    # 3. Update preprocessor joblib
    preprocessor['class_weights_encoded'] = cw_encoded
    preprocessor['class_weights_named'] = cw_named
    preprocessor['category_weights_encoded'] = cat_encoded
    preprocessor['binary_weights'] = bin_weights
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    print("  [Updated] models/preprocessor.joblib")

    # 4. Save balanced dataset if generated
    if X_balanced is not None and y_balanced is not None:
        joblib.dump(X_balanced, 'data/X_train_balanced.pkl', compress=3)
        joblib.dump(y_balanced, 'data/y_train_balanced.pkl', compress=3)
        print("  [Saved] data/X_train_balanced.pkl & data/y_train_balanced.pkl")

    # 5. Save Resampling Summary JSON
    if resampling_summary:
        with open('models/resampling_summary.json', 'w') as f:
            json.dump(resampling_summary, f, indent=4)
        print("  [Saved] models/resampling_summary.json")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Imbalance Handling with Paper-Aligned SMOTETomek & Hybrid Resampling")
    parser.add_argument('--strategy', type=str, default='hybrid',
                        choices=['hybrid', 'smotetomek', 'smote', 'smoteenn', 'weights_only'],
                        help="Resampling strategy: 'hybrid' (Scalable SMOTE+Tomek for IoT), 'smotetomek' (Paper), 'smote', 'smoteenn', 'weights_only'")
    parser.add_argument('--target-tier', type=str, default='category',
                        choices=['category', 'label'],
                        help="Target level to resample: 'category' (8 attack groups) or 'label' (34 attacks)")
    parser.add_argument('--sample-size', type=int, default=150000,
                        help="Stratified sample size before resampling (default: 150000 rows)")
    parser.add_argument('--target-majority', type=int, default=30000,
                        help="Max samples per majority class after pruning (default: 30000)")
    parser.add_argument('--target-minority', type=int, default=8000,
                        help="Min samples per minority class after SMOTE (default: 8000)")
    parser.add_argument('--k-neighbors', type=int, default=5,
                        help="Number of nearest neighbors for SMOTE (default: 5)")
    parser.add_argument('--skip-plots', action='store_true',
                        help="Skip generating visualization plots")
    args = parser.parse_args()

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

    # Task 3.1: Imbalance Analysis
    analysis_results = analyze_imbalance(y_names, y_cat_names, label_mapping, category_mapping)

    # Task 3.2A: Compute Class Weights (Always executed)
    cw_enc, cw_named, cat_enc, cat_named, bin_w = compute_all_weights(
        y_encoded, y_cat_encoded, y_binary, reverse_label, reverse_cat
    )

    # Task 3.2B: Resampling
    X_bal, y_bal = None, None
    counts_before, counts_after = None, None
    resampling_summary = None

    if args.strategy != 'weights_only':
        target_series = y_cat_encoded if args.target_tier == 'category' else y_encoded
        target_rev_map = reverse_cat if args.target_tier == 'category' else reverse_label

        X_bal, y_bal, counts_before, counts_after, elapsed = apply_resampling_pipeline(
            X_train, target_series, target_rev_map,
            strategy=args.strategy,
            sample_size=args.sample_size,
            target_majority=args.target_majority,
            target_minority=args.target_minority,
            k_neighbors=args.k_neighbors
        )

        resampling_summary = {
            'strategy': args.strategy,
            'target_tier': args.target_tier,
            'initial_sample_size': args.sample_size,
            'final_sample_size': len(y_bal),
            'execution_time_seconds': round(elapsed, 2),
            'target_majority': args.target_majority,
            'target_minority': args.target_minority,
            'k_neighbors': args.k_neighbors,
            'class_distribution_before': {str(k): int(v) for k, v in counts_before.items()},
            'class_distribution_after': {str(k): int(v) for k, v in counts_after.items()}
        }

    # Generate Visualizations
    if not args.skip_plots:
        # Sample 5000 rows for PCA visualization if resampled
        X_pca, y_pca = None, None
        if X_bal is not None:
            n_pca = min(5000, len(y_bal))
            idx_pca = np.random.choice(len(y_bal), size=n_pca, replace=False)
            X_pca = X_bal.iloc[idx_pca] if hasattr(X_bal, 'iloc') else X_bal[idx_pca]
            y_pca = y_bal[idx_pca]

        generate_visualizations(
            analysis_results, cw_named, cat_named,
            counts_before, counts_after,
            X_pca, y_pca,
            reverse_cat if args.target_tier == 'category' else reverse_label
        )

    # Task 3.3: Save deliverables
    save_artifacts(cw_enc, cw_named, cat_enc, cat_named, bin_w,
                   preprocessor, X_bal, y_bal, resampling_summary)

    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()
