#!/usr/bin/env python3
"""
Compare Data Scaling Results: Batch 250K vs Batch 500K (vs Batch 1M)
Graduation Thesis: CICIoT2023 Network Intrusion Detection
"""

import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_metrics_for_batch(batch_name, eval_dir=None, p5_dir=None):
    """Load primary metrics from phase6_evaluation or phase5_experiments."""
    candidates_eval = [
        os.path.join(eval_dir, 'final', 'final_metrics.json') if eval_dir else None,
        f'batch_{batch_name}_results/phase6_evaluation/final/final_metrics.json',
        f'phase6_evaluation_{batch_name}/final/final_metrics.json',
        'phase6_evaluation/final/final_metrics.json' if batch_name == '250k' else None
    ]
    eval_file = next((p for p in candidates_eval if p and os.path.exists(p)), None)

    candidates_p5 = [
        os.path.join(p5_dir, 'leaderboard.csv') if p5_dir else None,
        f'batch_{batch_name}_results/phase5_experiments/leaderboard.csv',
        f'phase5_experiments_{batch_name}/leaderboard.csv',
        'phase5_experiments/leaderboard.csv' if batch_name == '250k' else None
    ]
    p5_file = next((p for p in candidates_p5 if p and os.path.exists(p)), None)

    res = {
        'batch': batch_name.upper(),
        'accuracy': None,
        'f1_macro': None,
        'f1_weighted': None,
        'roc_auc': None,
        'precision_macro': None,
        'recall_macro': None,
        'latency_p50_ms': None,
        'throughput_samples_sec': None,
        'status': 'Pending'
    }

    if eval_file:
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                d = json.load(f)
            t_m = d.get('final_test_metrics', d.get('test_metrics', {}))
            res['accuracy'] = round(float(t_m.get('accuracy', 0)), 4)
            res['f1_macro'] = round(float(t_m.get('f1_macro', 0)), 4)
            res['roc_auc'] = round(float(t_m.get('roc_auc', 0)), 4)
            res['f1_weighted'] = round(float(t_m.get('f1_weighted', 0)), 4)

            # Look for champion run metadata for detailed precision/recall/latency
            eval_dir_actual = os.path.dirname(os.path.dirname(eval_file))
            champ_id = d.get('champion_eval_id', '')
            meta_candidates = glob.glob(os.path.join(eval_dir_actual, 'runs', f"{champ_id}*", 'metadata.json')) if champ_id else []
            if not meta_candidates:
                meta_candidates = glob.glob(os.path.join(eval_dir_actual, 'runs', 'eval_*', 'metadata.json'))
            if meta_candidates:
                with open(meta_candidates[0], 'r', encoding='utf-8') as mf:
                    meta = json.load(mf)
                ts = meta.get('test_metrics', meta.get('test_summary', {}))
                lat = meta.get('latency_metrics', meta.get('latency_performance', {}))
                res['precision_macro'] = round(float(ts.get('precision_macro', 0)), 4)
                res['recall_macro'] = round(float(ts.get('recall_macro', 0)), 4)
                res['f1_weighted'] = round(float(ts.get('f1_weighted', res['f1_weighted'])), 4)
                res['latency_p50_ms'] = round(float(lat.get('single_sample_latency_ms', {}).get('median_p50', 0)), 2)
                res['throughput_samples_sec'] = round(float(lat.get('batch_performance', {}).get('throughput_samples_per_sec', 0)), 1)
            
            res['status'] = 'Completed'
        except Exception as e:
            print(f"Warning reading {eval_file}: {e}")

    elif p5_file:
        try:
            df = pd.read_csv(p5_file)
            best = df.sort_values(by=['f1_macro', 'accuracy'], ascending=False).iloc[0]
            res['accuracy'] = round(float(best.get('accuracy', 0)), 4)
            res['f1_macro'] = round(float(best.get('f1_macro', 0)), 4)
            res['f1_weighted'] = round(float(best.get('f1_weighted', 0)), 4)
            res['roc_auc'] = round(float(best.get('roc_auc', best.get('roc_auc_ovr_macro', 0))), 4)
            res['status'] = 'Completed (P5)'
        except Exception as e:
            print(f"Warning reading {p5_file}: {e}")

    return res

def main():
    print("=" * 75)
    print("SCALABILITY COMPARISON: BATCH 250K vs BATCH 500K vs BATCH 1M")
    print("=" * 75)

    batches = ['250k', '500k', '1m']
    rows = []
    for b in batches:
        r = load_metrics_for_batch(b)
        rows.append(r)

    df_comp = pd.DataFrame(rows)
    print("\n--- Current Scalability Comparison Table ---")
    print(df_comp.to_string(index=False))

    os.makedirs('scaling_comparison', exist_ok=True)
    csv_path = 'scaling_comparison/data_scaling_comparison.csv'
    df_comp.to_csv(csv_path, index=False)
    print(f"\n[Saved] CSV summary -> {csv_path}")

    try:
        xlsx_path = 'scaling_comparison/data_scaling_comparison.xlsx'
        df_comp.to_excel(xlsx_path, index=False)
        print(f"[Saved] Excel summary -> {xlsx_path}")
    except Exception as e:
        print(f"Excel export skipped: {e}")

    # Plot comparison if both 250k and 500k are completed
    completed = [r for r in rows if r['accuracy'] is not None]
    if len(completed) >= 2:
        df_plot = pd.DataFrame(completed)
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(df_plot))
        w = 0.35
        ax.bar(x - w/2, df_plot['accuracy'] * 100, w, label='Accuracy (%)', color='#2b5c8f')
        ax.bar(x + w/2, df_plot['f1_macro'] * 100, w, label='Macro F1 (%)', color='#d95f02')
        ax.set_xticks(x)
        ax.set_xticklabels(df_plot['batch'])
        ax.set_ylabel('Score (%)')
        ax.set_ylim(85, 100)
        ax.set_title('Data Scalability Ablation: Performance by Training Scale')
        ax.legend()
        for i, val in enumerate(df_plot['accuracy']):
            ax.text(i - w/2, val * 100 + 0.3, f"{val*100:.2f}%", ha='center', fontsize=9, fontweight='bold')
        for i, val in enumerate(df_plot['f1_macro']):
            ax.text(i + w/2, val * 100 + 0.3, f"{val*100:.2f}%", ha='center', fontsize=9, fontweight='bold')
        plt.tight_layout()
        chart_p = 'scaling_comparison/scalability_chart.png'
        plt.savefig(chart_p, dpi=300)
        plt.close()
        print(f"[Saved] Comparison chart -> {chart_p}")

if __name__ == '__main__':
    main()
