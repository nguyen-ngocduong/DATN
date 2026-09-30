#!/usr/bin/env python3
"""
Automated Pipeline Runner for Batch 500K (Scalability Ablation Study)
Graduation Thesis: CICIoT2023 Network Intrusion Detection
Usage:
  python3 run_batch_500k.py [--minority-target 16000] [--sample-size 500000]
"""

import os
import sys
import time
import shutil
import argparse
import subprocess

def run_step(step_name, cmd):
    print("\n" + "=" * 80)
    print(f">> RUNNING {step_name.upper()}:")
    print(f"   Command: {' '.join(cmd)}")
    print("=" * 80 + "\n")
    t0 = time.time()
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"\n[ERROR] Step {step_name} failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    elapsed = time.time() - t0
    print(f"\n[OK] Step {step_name} completed in {elapsed:.2f}s ({elapsed/60:.2f} mins)")
    return elapsed

def main():
    parser = argparse.ArgumentParser(description="Automated 500K Pipeline Runner")
    parser.add_argument('--sample-size', type=int, default=500000, help='Subsample size for Phase 3 (default: 500,000)')
    parser.add_argument('--minority-target', type=int, default=16000, help='SMOTE target minority per class (default: 16,000)')
    parser.add_argument('--majority-target', type=int, default=30000, help='RUS target majority per class (default: 30,000)')
    parser.add_argument('--skip-phase3', action='store_true', help='Skip Phase 3 if data/X_train_balanced_500k.pkl exists')
    parser.add_argument('--skip-phase4', action='store_true', help='Skip Phase 4 if data/X_train_selected_500k.pkl exists')
    args = parser.parse_args()

    total_t0 = time.time()
    print("=" * 80)
    print("STARTING BATCH 500K AUTOMATED PIPELINE (SCALABILITY ABLATION STUDY)")
    print(f"Sample Size: {args.sample_size:,} | Minority Target: {args.minority_target:,} | Majority Target: {args.majority_target:,}")
    print("=" * 80)

    # 1. Pre-flight check
    x_paths = ['data/X_train.pkl', 'X_train.pkl', '/content/data/X_train.pkl']
    raw_x = next((p for p in x_paths if os.path.exists(p)), None)
    if not raw_x:
        print("[ERROR] Cannot locate raw preprocessed data/X_train.pkl! Please ensure Phase 2 data is available.")
        sys.exit(1)
    print(f"[Pre-flight] Found base dataset: {raw_x}")

    # 2. Phase 3: Class Balancing for 500k
    bal_pkl = 'data/X_train_balanced_500k.pkl'
    if not (args.skip_phase3 and os.path.exists(bal_pkl)):
        cmd_p3 = [
            sys.executable, '03_balance.py',
            '--strategy', 'hybrid',
            '--sample-size', str(args.sample_size),
            '--target-minority', str(args.minority_target),
            '--target-majority', str(args.majority_target),
            '--batch-suffix', '500k',
            '--skip-plots'
        ]
        run_step("Phase 3: Hybrid Resampling (500K)", cmd_p3)
    else:
        print(f"[Skip] Phase 3 skipped: {bal_pkl} already exists.")

    # 3. Phase 4: Feature Selection for 500k
    sel_pkl = 'data/X_train_selected_500k.pkl'
    if not (args.skip_phase4 and os.path.exists(sel_pkl)):
        cmd_p4 = [
            sys.executable, '04_feature_selection.py',
            '--top-k', '25',
            '--use-balanced',
            '--batch-suffix', '500k',
            '--skip-plots'
        ]
        run_step("Phase 4: Feature Selection (500K)", cmd_p4)
    else:
        print(f"[Skip] Phase 4 skipped: {sel_pkl} already exists.")

    # 4. Phase 5: Multi-Model Training on 500k
    cmd_p5 = [
        sys.executable, '05_train_models.py',
        '--data-path', 'data/X_train_selected_500k.pkl',
        '--target-path', 'data/y_train_balanced_500k.pkl',
        '--exp-root', 'phase5_experiments_500k',
        '--random-state', '123'
    ]
    run_step("Phase 5: Candidate Model Training (500K)", cmd_p5)

    # 5. Phase 6: Model Evaluation on 500k
    cmd_p6 = [
        sys.executable, '06_evaluate.py',
        '--data-path', 'data/X_train_selected_500k.pkl',
        '--target-path', 'data/y_train_balanced_500k.pkl',
        '--exp-root', 'phase5_experiments_500k',
        '--eval-root', 'phase6_evaluation_500k',
        '--random-state', '123'
    ]
    run_step("Phase 6: Independent Evaluation (500K)", cmd_p6)

    # 6. Archive Deliverables into batch_500k_results/
    print("\n" + "=" * 80)
    print("ARCHIVING BATCH 500K ARTIFACTS INTO batch_500k_results/")
    print("=" * 80)
    dest_500k = 'batch_500k_results'
    os.makedirs(dest_500k, exist_ok=True)

    if os.path.exists('phase5_experiments_500k'):
        p5_dest = os.path.join(dest_500k, 'phase5_experiments')
        if os.path.exists(p5_dest):
            shutil.rmtree(p5_dest)
        shutil.copytree('phase5_experiments_500k', p5_dest)
        print(f"[OK] Archived phase5_experiments_500k -> {p5_dest}")

    if os.path.exists('phase6_evaluation_500k'):
        p6_dest = os.path.join(dest_500k, 'phase6_evaluation')
        if os.path.exists(p6_dest):
            shutil.rmtree(p6_dest)
        shutil.copytree('phase6_evaluation_500k', p6_dest)
        print(f"[OK] Archived phase6_evaluation_500k -> {p6_dest}")

    m_dest = os.path.join(dest_500k, 'models')
    os.makedirs(m_dest, exist_ok=True)
    for f in ['selected_features_500k.json', 'class_weights_500k.json', 'resampling_summary_500k.json']:
        src = os.path.join('models', f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(m_dest, f.replace('_500k', '')))
    print(f"[OK] Archived models metadata -> {m_dest}")

    # 7. Run Comparison: 250k vs 500k
    cmd_comp = [sys.executable, 'compare_scaling.py']
    run_step("Scalability Comparison (250K vs 500K)", cmd_comp)

    total_elapsed = time.time() - total_t0
    print("\n" + "=" * 80)
    print(f"🎉 BATCH 500K PIPELINE FULLY COMPLETED IN {total_elapsed/60:.2f} MINUTES!")
    print("=" * 80)

if __name__ == '__main__':
    main()
