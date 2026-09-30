#!/usr/bin/env python3
"""
Phase 4: Feature Selection Pipeline
CICIoT2023 Network Intrusion Detection
Plan: PLAN.md (Phase 4)

Tasks Implemented:
    4.1 Variance threshold filtering (remove low-variance / quasi-constant features)
    4.2 Correlation analysis (identify and prune collinear feature pairs |r| >= threshold)
    4.3 Model-based feature importance (Random Forest + XGBoost / GBDT consensus)
    4.4 Select Top N predictive features & export reduced dataset
    4.5 Optional: PCA dimensionality reduction & scree analysis

Deliverables:
    - 04_feature_selection.py (CLI script)
    - data/X_train_selected.pkl, X_train_reduced.pkl
    - models/selected_features.json, models/feature_selector.joblib
    - plots/09_variance_distribution.png
    - plots/10_correlation_clustermap.png
    - plots/11_feature_importance.png
    - plots/12_cumulative_importance.png
    - plots/13_pca_explained_variance.png
"""

import os
import gc
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.decomposition import PCA

# Try importing XGBoost or LightGBM; fallback gracefully if not installed
XGB_AVAILABLE = False
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    pass

LGBM_AVAILABLE = False
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    pass

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')


# ------------------------------------------------------------------------------
# 1. Data Loading Helper
# ------------------------------------------------------------------------------
def load_phase_data(data_dir=None, use_balanced=True, batch_suffix=None):
    """
    Load training data from Phase 3 balanced set (or Phase 2 preprocessed set).
    """
    print("=" * 70)
    print("LOADING TRAINING DATA")
    print("=" * 70)

    suffix = f"_{batch_suffix}" if batch_suffix else ""

    # Search paths for X_train
    if use_balanced:
        x_candidates = [
            f'data/X_train_balanced{suffix}.pkl', f'X_train_balanced{suffix}.pkl',
            'data/X_train_balanced.pkl', 'X_train_balanced.pkl',
            'data/X_train.pkl', 'X_train.pkl'
        ]
        y_candidates = [
            f'data/y_train_balanced{suffix}.pkl', f'y_train_balanced{suffix}.pkl',
            'data/y_train_balanced.pkl', 'y_train_balanced.pkl',
            'data/y_train.pkl', 'y_train.pkl'
        ]
    else:
        x_candidates = ['data/X_train.pkl', 'X_train.pkl', 'data/X_train_balanced.pkl', 'X_train_balanced.pkl']
        y_candidates = ['data/y_train.pkl', 'y_train.pkl', 'data/y_train_balanced.pkl', 'y_train_balanced.pkl']

    if data_dir:
        x_candidates = [
            os.path.join(data_dir, f'X_train_balanced{suffix}.pkl'),
            os.path.join(data_dir, 'X_train_balanced.pkl'),
            os.path.join(data_dir, 'X_train.pkl')
        ] + x_candidates
        y_candidates = [
            os.path.join(data_dir, f'y_train_balanced{suffix}.pkl'),
            os.path.join(data_dir, 'y_train_balanced.pkl'),
            os.path.join(data_dir, 'y_train.pkl')
        ] + y_candidates

    x_path = next((p for p in x_candidates if p and os.path.exists(p)), None)
    y_path = next((p for p in y_candidates if p and os.path.exists(p)), None)

    if not x_path or not y_path:
        raise FileNotFoundError(
            "Cannot find X_train.pkl or y_train.pkl! Please run Phase 2 (02_preprocessing.py) or Phase 3 first."
        )

    print(f"Loading features from: {x_path}")
    X_train = joblib.load(x_path)
    print(f"Loading targets from:  {y_path}")
    y_train = joblib.load(y_path)

    # If y_train is saved as dictionary from Phase 2/3
    if isinstance(y_train, dict):
        y_dict = y_train
    else:
        y_dict = {'target': y_train}

    # Load preprocessor if available
    preprocessor = None
    if os.path.exists('models/preprocessor.joblib'):
        preprocessor = joblib.load('models/preprocessor.joblib')
        print("Loaded preprocessor from models/preprocessor.joblib")

    # Ensure X_train is a DataFrame
    if not isinstance(X_train, pd.DataFrame):
        feature_names = preprocessor.get('feature_names', [f'feat_{i}' for i in range(X_train.shape[1])])
        X_train = pd.DataFrame(X_train, columns=feature_names, dtype='float32')

    print(f"Dataset Shape: {X_train.shape[0]:,} samples x {X_train.shape[1]} features")
    mem_mb = X_train.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Memory Usage:  {mem_mb:.2f} MB")

    return X_train, y_dict, preprocessor


def resolve_target(y_dict, requested_tier='auto'):
    """
    Intelligently resolve target array and class metadata from target dictionary,
    supporting both Phase 2 (y_train.pkl) and Phase 3 (y_train_balanced.pkl).
    """
    if isinstance(y_dict, np.ndarray):
        return y_dict, 'target', len(np.unique(y_dict))
    if not isinstance(y_dict, dict):
        arr = np.array(y_dict)
        return arr, 'target', len(np.unique(arr))

    # Priority 1: Exact key match if requested specifically
    if requested_tier and requested_tier != 'auto' and requested_tier in y_dict:
        y_arr = y_dict[requested_tier]
        return y_arr, requested_tier, len(np.unique(y_arr))

    # Priority 2: Phase 3 balanced target
    if 'target_encoded' in y_dict:
        tier_name = y_dict.get('target_tier', 'label')
        y_arr = y_dict['target_encoded']
        print(f"Detected Phase 3 balanced target: '{tier_name}' ({len(np.unique(y_arr))} classes)")
        return y_arr, f"balanced_{tier_name}", len(np.unique(y_arr))

    # Priority 3: Phase 2 label_encoded (34 fine-grained classes)
    if requested_tier in ('label', 'label_encoded') and 'label_encoded' in y_dict:
        return y_dict['label_encoded'], 'label_encoded', len(np.unique(y_dict['label_encoded']))

    # Priority 4: Phase 2 category_encoded (9 attack categories)
    if requested_tier in ('category', 'category_encoded') and 'category_encoded' in y_dict:
        return y_dict['category_encoded'], 'category_encoded', len(np.unique(y_dict['category_encoded']))

    # Fallback order
    for k in ['label_encoded', 'category_encoded', 'target_encoded', 'target', 'is_attack']:
        if k in y_dict:
            y_arr = y_dict[k]
            return y_arr, k, len(np.unique(y_arr))

    first_k = list(y_dict.keys())[0]
    y_arr = y_dict[first_k]
    return y_arr, first_k, len(np.unique(y_arr))


# ------------------------------------------------------------------------------
# 2. Task 4.1: Variance Threshold Filtering
# ------------------------------------------------------------------------------
def task_4_1_variance_filtering(X, threshold=0.001):
    """
    Identify and filter out features with variance below the specified threshold.
    """
    print("\n" + "=" * 70)
    print(f"TASK 4.1: VARIANCE THRESHOLD FILTERING (threshold = {threshold})")
    print("=" * 70)

    variances = X.var(axis=0)
    var_df = pd.DataFrame({
        'feature': X.columns,
        'variance': variances.values
    }).sort_values('variance', ascending=True)

    low_var_features = var_df[var_df['variance'] <= threshold]['feature'].tolist()
    kept_features = var_df[var_df['variance'] > threshold]['feature'].tolist()

    print(f"Total features analyzed:        {len(X.columns)}")
    print(f"Low-variance features (<= {threshold}): {len(low_var_features)}")
    print(f"Retained features (> {threshold}):      {len(kept_features)}")

    if low_var_features:
        print("\nFeatures with variance <= threshold:")
        for feat in low_var_features[:10]:
            print(f"  - {feat:30s} (var = {variances[feat]:.8f})")
        if len(low_var_features) > 10:
            print(f"  ... and {len(low_var_features) - 10} more.")
    else:
        print("All features exceed variance threshold. No features dropped by variance filter.")

    return kept_features, low_var_features, var_df


# ------------------------------------------------------------------------------
# 3. Task 4.2: Correlation Analysis & Collinearity Pruning
# ------------------------------------------------------------------------------
def task_4_2_correlation_analysis(X, feature_list, threshold=0.90, max_sample=100000):
    """
    Identify and prune collinear feature pairs with |Pearson r| >= threshold.
    Greedy pruning: drops the feature with higher average correlation to other features.
    """
    print("\n" + "=" * 70)
    print(f"TASK 4.2: CORRELATION ANALYSIS (Collinearity threshold |r| >= {threshold})")
    print("=" * 70)

    X_sub = X[feature_list]
    if len(X_sub) > max_sample:
        print(f"Subsampling {max_sample:,} rows for fast Pearson correlation computation...")
        X_sub = X_sub.sample(n=max_sample, random_state=42)

    corr_matrix = X_sub.corr(method='pearson')
    abs_corr = corr_matrix.abs()

    # Find upper triangle pairs
    upper_tri = abs_corr.where(np.triu(np.ones(abs_corr.shape), k=1).astype(bool))
    collinear_pairs = []

    for col in upper_tri.columns:
        high_corr = upper_tri[col][upper_tri[col] >= threshold]
        for row, r_val in high_corr.items():
            collinear_pairs.append({
                'feature_1': row,
                'feature_2': col,
                'correlation': r_val
            })

    collinear_df = pd.DataFrame(collinear_pairs)
    print(f"Detected {len(collinear_df)} collinear feature pairs with |r| >= {threshold}")

    # Greedy removal: features with highest mean absolute correlation
    features_to_drop = set()
    mean_corr = abs_corr.mean()

    for pair in collinear_pairs:
        f1, f2 = pair['feature_1'], pair['feature_2']
        if f1 not in features_to_drop and f2 not in features_to_drop:
            # Drop the one with higher average correlation across the board
            drop_feat = f1 if mean_corr[f1] >= mean_corr[f2] else f2
            keep_feat = f2 if drop_feat == f1 else f1
            features_to_drop.add(drop_feat)
            print(f"  Collinear pair: {f1:25s} <-> {f2:25s} (r={pair['correlation']:.4f}) -> Drop '{drop_feat}'")

    dropped_corr_features = sorted(list(features_to_drop))
    kept_features = [f for f in feature_list if f not in features_to_drop]

    print(f"\nCollinear features dropped: {len(dropped_corr_features)}")
    print(f"Features remaining after correlation pruning: {len(kept_features)}")

    return kept_features, dropped_corr_features, corr_matrix, collinear_df


# ------------------------------------------------------------------------------
# 4. Task 4.3: Model-based Feature Importance (RF + XGBoost/GBDT)
# ------------------------------------------------------------------------------
def task_4_3_model_importance(X, y, feature_list, sample_size=100000, random_state=42):
    """
    Train Random Forest and XGBoost/LightGBM on a stratified sample to compute
    Mean Decrease in Impurity / Gain feature importance.
    """
    print("\n" + "=" * 70)
    print("TASK 4.3: MODEL-BASED FEATURE IMPORTANCE")
    print("=" * 70)

    X_filtered = X[feature_list]

    # Stratified subsampling
    if sample_size and len(X_filtered) > sample_size:
        print(f"Drawing stratified sample of {sample_size:,} samples across classes...")
        from sklearn.model_selection import train_test_split
        try:
            X_sample, _, y_sample, _ = train_test_split(
                X_filtered, y,
                train_size=sample_size,
                stratify=y,
                random_state=random_state
            )
        except Exception:
            # If extreme minority classes have too few samples for stratify
            sample_idx = np.random.RandomState(random_state).choice(len(X_filtered), size=sample_size, replace=False)
            X_sample = X_filtered.iloc[sample_idx]
            y_sample = y[sample_idx] if isinstance(y, np.ndarray) else y.iloc[sample_idx]
    else:
        X_sample = X_filtered
        y_sample = y

    print(f"Training on sample: {X_sample.shape[0]:,} rows x {X_sample.shape[1]} features")

    # 1. Random Forest
    print("\n[1/2] Training Random Forest Classifier (n_estimators=100, max_depth=14)...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        min_samples_split=10,
        max_features='sqrt',
        n_jobs=-1,
        random_state=random_state
    )
    rf.fit(X_sample, y_sample)
    rf_imp = rf.feature_importances_
    rf_norm = rf_imp / rf_imp.sum()
    print("Random Forest training completed.")

    # 2. XGBoost or LightGBM or ExtraTrees
    xgb_imp = None
    model_name = "Gradient Boosting"

    if XGB_AVAILABLE:
        print("[2/2] Training XGBoost Classifier (n_estimators=100, max_depth=6)...")
        model_name = "XGBoost"
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=random_state,
            eval_metric='mlogloss',
            tree_method='hist'
        )
        xgb_clf.fit(X_sample, y_sample)
        xgb_imp = xgb_clf.feature_importances_
    elif LGBM_AVAILABLE:
        print("[2/2] Training LightGBM Classifier (n_estimators=100, max_depth=6)...")
        model_name = "LightGBM"
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=random_state,
            verbosity=-1
        )
        lgb_clf.fit(X_sample, y_sample)
        xgb_imp = lgb_clf.feature_importances_
    else:
        print("[2/2] XGBoost/LightGBM not installed. Training ExtraTreesClassifier...")
        model_name = "ExtraTrees"
        et_clf = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=14,
            max_features='sqrt',
            n_jobs=-1,
            random_state=random_state
        )
        et_clf.fit(X_sample, y_sample)
        xgb_imp = et_clf.feature_importances_

    xgb_norm = xgb_imp / xgb_imp.sum()
    print(f"{model_name} training completed.")

    # 3. Consensus Importance (Ensemble Average)
    consensus_imp = (rf_norm + xgb_norm) / 2.0

    importance_df = pd.DataFrame({
        'feature': feature_list,
        'rf_importance': rf_norm,
        f'{model_name.lower()}_importance': xgb_norm,
        'consensus_importance': consensus_imp
    }).sort_values('consensus_importance', ascending=False).reset_index(drop=True)

    importance_df['rank'] = range(1, len(importance_df) + 1)
    importance_df['cumulative_importance'] = importance_df['consensus_importance'].cumsum()

    print("\nTop 15 Most Predictive Features (Consensus Score):")
    for idx, row in importance_df.head(15).iterrows():
        print(f"  #{row['rank']:2d}: {row['feature']:28s} "
              f"Score: {row['consensus_importance']*100:5.2f}% "
              f"(Cumulative: {row['cumulative_importance']*100:5.2f}%)")

    return importance_df, model_name


# ------------------------------------------------------------------------------
# 5. Task 4.4: Select Top N Features & Build Reduced Dataset
# ------------------------------------------------------------------------------
def task_4_4_select_features(X, importance_df, top_k=25, cumulative_threshold=0.95):
    """
    Select top features by rank (top_k) or by cumulative importance threshold.
    """
    print("\n" + "=" * 70)
    print(f"TASK 4.4: SELECTING TOP {top_k} FEATURES (or cumulative >= {cumulative_threshold*100:.0f}%)")
    print("=" * 70)

    # Condition 1: Top K
    top_k_features = importance_df.head(top_k)['feature'].tolist()

    # Condition 2: Cumulative threshold
    cum_features = importance_df[importance_df['cumulative_importance'] <= cumulative_threshold]['feature'].tolist()
    # Ensure at least 1 feature above threshold is included to reach the cutoff
    if len(cum_features) < len(importance_df):
        cum_features.append(importance_df.iloc[len(cum_features)]['feature'])

    # Final selection: use top_k by default, but ensure at least 95% coverage if specified
    selected_features = top_k_features
    selected_cum_imp = importance_df[importance_df['feature'].isin(selected_features)]['consensus_importance'].sum()

    print(f"Selected {len(selected_features)} features.")
    print(f"Total Cumulative Importance Captured: {selected_cum_imp * 100:.2f}%")

    dropped_features = [f for f in X.columns if f not in selected_features]
    print(f"Features pruned/eliminated:          {len(dropped_features)}")

    # Create reduced dataset
    X_selected = X[selected_features].copy()

    return X_selected, selected_features, dropped_features


# ------------------------------------------------------------------------------
# 6. Task 4.5: Optional PCA for Dimensionality Reduction
# ------------------------------------------------------------------------------
def task_4_5_pca_analysis(X_selected, variance_target=0.95):
    """
    Perform PCA dimensionality reduction analysis on selected features.
    """
    print("\n" + "=" * 70)
    print(f"TASK 4.5: PCA DIMENSIONALITY REDUCTION ANALYSIS (Target variance = {variance_target*100:.0f}%)")
    print("=" * 70)

    # Subsample for PCA if large
    n_samples = min(200000, len(X_selected))
    if len(X_selected) > n_samples:
        X_pca_sample = X_selected.sample(n=n_samples, random_state=42)
    else:
        X_pca_sample = X_selected

    max_components = min(X_selected.shape[1], 35)
    pca = PCA(n_components=max_components, random_state=42)
    pca.fit(X_pca_sample)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # Calculate required components for thresholds
    n_80 = int(np.argmax(cumulative_var >= 0.80) + 1)
    n_90 = int(np.argmax(cumulative_var >= 0.90) + 1)
    n_95 = int(np.argmax(cumulative_var >= 0.95) + 1)
    n_99 = int(np.argmax(cumulative_var >= 0.99) + 1) if cumulative_var[-1] >= 0.99 else len(cumulative_var)

    print(f"PCA Total Components Evaluated: {max_components}")
    print(f"Components needed for  80% variance: {n_80:2d} / {X_selected.shape[1]}")
    print(f"Components needed for  90% variance: {n_90:2d} / {X_selected.shape[1]}")
    print(f"Components needed for  95% variance: {n_95:2d} / {X_selected.shape[1]}")
    print(f"Components needed for  99% variance: {n_99:2d} / {X_selected.shape[1]}")

    pca_summary = {
        'n_components_evaluated': max_components,
        'components_for_80_pct': n_80,
        'components_for_90_pct': n_90,
        'components_for_95_pct': n_95,
        'components_for_99_pct': n_99,
        'explained_variance_ratio': explained_var.tolist(),
        'cumulative_variance_ratio': cumulative_var.tolist()
    }

    return pca, pca_summary


# ------------------------------------------------------------------------------
# 7. Visualization Generator
# ------------------------------------------------------------------------------
def create_visualizations(var_df, var_threshold, corr_matrix, collinear_features,
                          importance_df, second_model_name, pca_summary=None, plots_dir='plots'):
    """
    Generate and save all static visualization plots for Phase 4.
    """
    print("\n" + "=" * 70)
    print(f"GENERATING PHASE 4 VISUALIZATION PLOTS IN {plots_dir}/")
    print("=" * 70)
    os.makedirs(plots_dir, exist_ok=True)

    # --- Plot 09: Variance Distribution ---
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_vars = var_df.sort_values('variance', ascending=False)
    ax.bar(range(len(sorted_vars)), sorted_vars['variance'], color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(var_threshold, color='#e74c3c', linestyle='--', linewidth=2, label=f'Threshold ({var_threshold})')
    ax.set_yscale('log')
    ax.set_title('Feature Variance Distribution (Log Scale)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Feature Index (Sorted by Variance)', fontsize=11)
    ax.set_ylabel('Variance (Log Scale)', fontsize=11)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    p09 = os.path.join(plots_dir, '09_variance_distribution.png')
    plt.savefig(p09, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p09}")

    # --- Plot 10: Correlation Heatmap / Clustermap ---
    fig, ax = plt.subplots(figsize=(14, 11))
    # Pick top 25 features for clear visibility in correlation matrix
    sample_cols = corr_matrix.columns[:25]
    sub_corr = corr_matrix.loc[sample_cols, sample_cols]
    mask = np.triu(np.ones_like(sub_corr, dtype=bool))
    sns.heatmap(
        sub_corr, mask=mask, cmap='coolwarm', vmin=-1.0, vmax=1.0,
        center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
        ax=ax, annot=False
    )
    ax.set_title('Feature Correlation Heatmap (Sample of 25 Features)', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    p10 = os.path.join(plots_dir, '10_correlation_clustermap.png')
    plt.savefig(p10, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p10}")

    # --- Plot 11: Feature Importance (RF vs Second Model) ---
    top_plot = importance_df.head(20).sort_values('consensus_importance', ascending=True)
    y_pos = np.arange(len(top_plot))
    height = 0.35

    fig, ax = plt.subplots(figsize=(13, 8))
    sec_col = f'{second_model_name.lower()}_importance'
    ax.barh(y_pos - height/2, top_plot['rf_importance'] * 100, height, label='Random Forest', color='#2980b9', alpha=0.9)
    ax.barh(y_pos + height/2, top_plot[sec_col] * 100, height, label=second_model_name, color='#e67e22', alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_plot['feature'], fontsize=10)
    ax.set_xlabel('Relative Feature Importance (%)', fontsize=11)
    ax.set_title(f'Top 20 Feature Importance Comparison: Random Forest vs {second_model_name}', fontsize=14, fontweight='bold', pad=15)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
    plt.tight_layout()
    p11 = os.path.join(plots_dir, '11_feature_importance.png')
    plt.savefig(p11, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p11}")

    # --- Plot 12: Cumulative Feature Importance ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    x_ranks = importance_df['rank'].values
    y_cum = importance_df['cumulative_importance'].values * 100
    y_ind = importance_df['consensus_importance'].values * 100

    ax1.bar(x_ranks, y_ind, color='#3498db', alpha=0.6, label='Individual Score (%)', edgecolor='black', linewidth=0.3)
    ax1.set_xlabel('Feature Rank', fontsize=11)
    ax1.set_ylabel('Individual Importance (%)', color='#2980b9', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='#2980b9')

    ax2 = ax1.twinx()
    ax2.plot(x_ranks, y_cum, color='#e74c3c', linewidth=2.5, marker='o', markersize=4, label='Cumulative Score (%)')
    ax2.axhline(95, color='#27ae60', linestyle='--', linewidth=1.8, label='95% Threshold')
    ax2.set_ylabel('Cumulative Importance (%)', color='#c0392b', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='#c0392b')
    ax2.set_ylim(0, 105)

    plt.title('Pareto Chart: Individual & Cumulative Feature Importance', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    p12 = os.path.join(plots_dir, '12_cumulative_importance.png')
    plt.savefig(p12, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p12}")

    # --- Plot 13: PCA Explained Variance ---
    if pca_summary:
        fig, ax = plt.subplots(figsize=(11, 6))
        cum_pca = np.array(pca_summary['cumulative_variance_ratio']) * 100
        ind_pca = np.array(pca_summary['explained_variance_ratio']) * 100
        comp_idx = range(1, len(cum_pca) + 1)

        ax.bar(comp_idx, ind_pca, color='#9b59b6', alpha=0.6, label='Individual Explained Var (%)', edgecolor='black', linewidth=0.3)
        ax.plot(comp_idx, cum_pca, color='#e67e22', marker='s', markersize=4, linewidth=2, label='Cumulative Explained Var (%)')
        ax.axhline(95, color='#27ae60', linestyle='--', linewidth=1.5, label='95% Variance Cutoff')

        ax.set_xlabel('Principal Component Index', fontsize=11)
        ax.set_ylabel('Explained Variance (%)', fontsize=11)
        ax.set_title('PCA Scree Plot & Cumulative Explained Variance', fontsize=14, fontweight='bold', pad=15)
        ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='center right')
        plt.tight_layout()
        p13 = os.path.join(plots_dir, '13_pca_explained_variance.png')
        plt.savefig(p13, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {p13}")


# ------------------------------------------------------------------------------
# 8. Save Artifacts Helper
# ------------------------------------------------------------------------------
def save_feature_selection_artifacts(X_selected, selected_features, dropped_features,
                                     dropped_var_features, dropped_corr_features,
                                     importance_df, pca_model=None, pca_summary=None,
                                     output_dir='data', models_dir='models', batch_suffix=None):
    """
    Serialize all deliverables for Phase 4.
    """
    print("\n" + "=" * 70)
    print("SAVING DELIVERABLES")
    print("=" * 70)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    suffix = f"_{batch_suffix}" if batch_suffix else ""

    # 1. Save reduced dataset
    p_pkl = os.path.join(output_dir, f'X_train_selected{suffix}.pkl')
    print(f"Saving selected feature matrix to: {p_pkl}")
    joblib.dump(X_selected, p_pkl, compress=3)

    # Also save to root if needed for direct loading
    if not batch_suffix:
        joblib.dump(X_selected, 'X_train_reduced.pkl', compress=3)
        print("Saved mirror: X_train_reduced.pkl")

    try:
        p_parquet = os.path.join(output_dir, f'X_train_selected{suffix}.parquet')
        X_selected.to_parquet(p_parquet, index=False)
        print(f"Exported Parquet: {p_parquet}")
    except Exception as e:
        print(f"Parquet export skipped: {e}")

    # 2. Save selected_features.json
    meta_json = {
        'total_features_original': X_selected.shape[1] + len(dropped_features),
        'total_features_selected': len(selected_features),
        'selected_features': selected_features,
        'dropped_features': dropped_features,
        'dropped_variance_features': dropped_var_features,
        'dropped_correlation_features': dropped_corr_features,
        'cumulative_importance_captured': float(importance_df[importance_df['feature'].isin(selected_features)]['consensus_importance'].sum()),
        'top_15_features': importance_df.head(15)[['rank', 'feature', 'consensus_importance']].to_dict(orient='records')
    }
    if pca_summary:
        meta_json['pca_analysis'] = {
            'components_for_90_pct': pca_summary['components_for_90_pct'],
            'components_for_95_pct': pca_summary['components_for_95_pct']
        }

    p_json = os.path.join(models_dir, f'selected_features{suffix}.json')
    with open(p_json, 'w', encoding='utf-8') as f:
        json.dump(meta_json, f, indent=4)
    print(f"Saved metadata: {p_json}")

    # 3. Save feature selector artifact
    selector_artifact = {
        'selected_features': selected_features,
        'dropped_features': dropped_features,
        'importance_df': importance_df,
        'pca_model': pca_model
    }
    p_sel = os.path.join(models_dir, 'feature_selector.joblib')
    joblib.dump(selector_artifact, p_sel)
    print(f"Saved feature selector pipeline: {p_sel}")

    # 4. Save PCA model if computed
    if pca_model:
        p_pca = os.path.join(models_dir, 'pca_transformer.joblib')
        joblib.dump(pca_model, p_pca)
        print(f"Saved PCA transformer: {p_pca}")

    # 5. Transform and save test set if present
    for p_test in ['data/X_test.pkl', 'X_test.pkl']:
        if os.path.exists(p_test):
            try:
                X_test = joblib.load(p_test)
                if hasattr(X_test, 'columns'):
                    avail_cols = [c for c in selected_features if c in X_test.columns]
                    X_test_sel = X_test[avail_cols].copy()
                    out_test_p = os.path.join(output_dir, 'X_test_selected.pkl')
                    joblib.dump(X_test_sel, out_test_p, compress=3)
                    print(f"Applied feature selection to test set -> saved: {out_test_p} ({X_test_sel.shape[1]} features)")
                break
            except Exception as e:
                print(f"Note: Could not transform test set: {e}")

    print("\nAll Phase 4 deliverables saved successfully!")


# ------------------------------------------------------------------------------
# 9. Main Pipeline
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 4: CICIoT2023 Feature Selection Pipeline")
    parser.add_argument('--data-dir', type=str, default=None, help='Directory containing X_train.pkl')
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Thư mục làm việc/lưu trữ kết quả (Mặc định: auto-detect '/content/drive/MyDrive/do_an' nếu trên Colab, hoặc '.')")
    parser.add_argument('--models-dir', type=str, default='models', help='Directory to save model artifacts')
    parser.add_argument('--plots-dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--target-tier', type=str, default='auto',
                        help="Target classification tier: 'auto', 'label', 'category', 'label_encoded', 'category_encoded'")
    parser.add_argument('--sample-size', type=int, default=100000,
                        help='Stratified sample size for tree importance models (default: 100,000)')
    parser.add_argument('--var-threshold', type=float, default=0.001,
                        help='Variance threshold for quasi-constant feature filtering (default: 0.001)')
    parser.add_argument('--corr-threshold', type=float, default=0.90,
                        help='Collinearity correlation threshold (default: 0.90)')
    parser.add_argument('--top-k', type=int, default=25,
                        help='Number of top features to select (default: 25)')
    parser.add_argument('--cumulative-threshold', type=float, default=0.95,
                        help='Cumulative importance threshold (default: 0.95)')
    parser.add_argument('--run-pca', action='store_true', default=True,
                        help='Perform PCA dimensionality reduction analysis')
    parser.add_argument('--use-balanced', action='store_true', default=True,
                        help='Use balanced training set from Phase 3 if available (default: True)')
    parser.add_argument('--no-balanced', dest='use_balanced', action='store_false',
                        help='Do not use balanced dataset, force using raw Phase 2 train set')
    parser.add_argument('--skip-plots', action='store_true', default=False,
                        help='Skip saving visualization plots')
    parser.add_argument('--batch-suffix', type=str, default=None,
                        help="Suffix for dataset files (e.g. '500k' -> loads X_train_balanced_500k.pkl, saves X_train_selected_500k.pkl)")
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

    # 1. Load Data
    X_train, y_dict, preprocessor = load_phase_data(data_dir=args.data_dir, use_balanced=args.use_balanced, batch_suffix=args.batch_suffix)

    # Resolve target
    y_target, tier_name, n_classes = resolve_target(y_dict, requested_tier=args.target_tier)
    print(f"Using target tier: '{tier_name}' ({n_classes} distinct classes)")

    # 2. Task 4.1: Variance Threshold Filtering
    features_after_var, dropped_var, var_df = task_4_1_variance_filtering(
        X_train, threshold=args.var_threshold
    )

    # 3. Task 4.2: Correlation Analysis & Collinearity Pruning
    features_after_corr, dropped_corr, corr_matrix, collinear_df = task_4_2_correlation_analysis(
        X_train, feature_list=features_after_var, threshold=args.corr_threshold
    )

    # 4. Task 4.3: Model-based Feature Importance
    importance_df, second_model_name = task_4_3_model_importance(
        X_train, y_target, feature_list=features_after_corr, sample_size=args.sample_size
    )

    # 5. Task 4.4: Select Top N Features
    X_selected, selected_features, dropped_features = task_4_4_select_features(
        X_train, importance_df, top_k=args.top_k, cumulative_threshold=args.cumulative_threshold
    )

    # 6. Task 4.5: PCA Analysis (Optional)
    pca_model, pca_summary = None, None
    if args.run_pca:
        pca_model, pca_summary = task_4_5_pca_analysis(X_selected, variance_target=0.95)

    # 7. Generate Visualizations
    if not args.skip_plots:
        create_visualizations(
            var_df=var_df,
            var_threshold=args.var_threshold,
            corr_matrix=corr_matrix,
            collinear_features=dropped_corr,
            importance_df=importance_df,
            second_model_name=second_model_name,
            pca_summary=pca_summary,
            plots_dir=args.plots_dir
        )

    # 8. Save Artifacts
    save_feature_selection_artifacts(
        X_selected=X_selected,
        selected_features=selected_features,
        dropped_features=dropped_features,
        dropped_var_features=dropped_var,
        dropped_corr_features=dropped_corr,
        importance_df=importance_df,
        pca_model=pca_model,
        pca_summary=pca_summary,
        output_dir='data',
        models_dir=args.models_dir,
        batch_suffix=args.batch_suffix
    )

    print("\n" + "=" * 70)
    print("PHASE 4: FEATURE SELECTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == '__main__':
    main()
