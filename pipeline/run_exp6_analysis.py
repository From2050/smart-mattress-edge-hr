#!/usr/bin/env python3
"""
Exp 6 — 分析: 可靠通道比例 vs MAE 相關性 + 人口統計分析

基於 Exp 1 結果，分析:
  1. 每受試者可靠通道比例 vs MAE (Pearson + Spearman)
  2. 體重 vs MAE
  3. 性別 vs MAE
  4. 每受試者 Tier 分布
  5. 離群受試者分析
"""

import os, sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from common import load_all_data, NUM_CHANNELS, OUTPUT_ROOT, DATA_DIR

OUTPUT_DIR = os.path.join(OUTPUT_ROOT, 'exp6_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_demographics():
    """Load demographics with subject_weights.csv as primary source.

    Priority:
      1) subject_weights.csv for weight/sex (authoritative)
      2) Utility.txt for age and missing fallback values

    Utility parser supports both separators used in the dataset:
      - Key:Value
      - Key,Value
    """
    demo = {}

    # Primary source: subject_weights.csv
    sw_path = os.path.join(DATA_DIR, 'subject_weights.csv')
    if os.path.exists(sw_path):
        sw = pd.read_csv(sw_path)
        sw.columns = [c.strip() for c in sw.columns]
        for _, row in sw.iterrows():
            try:
                subj = int(row['Subject'])
            except Exception:
                continue
            demo[subj] = {
                'weight': float(row['Weight']) if pd.notna(row.get('Weight')) else None,
                'sex': str(row['Sex']).strip().upper() if pd.notna(row.get('Sex')) else None,
                'age': None,
            }

    for subj_dir in os.listdir(DATA_DIR):
        try:
            subj = int(subj_dir)
        except ValueError:
            continue

        util_path = os.path.join(DATA_DIR, subj_dir, 'Utility.txt')
        if not os.path.exists(util_path):
            continue

        weight, sex, age = None, None, None
        with open(util_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Accept both "key:value" and "key,value" formats.
                if ':' in line:
                    parts = line.split(':', 1)
                elif ',' in line:
                    parts = line.split(',', 1)
                else:
                    continue

                if len(parts) < 2:
                    continue

                key = parts[0].strip().lower()
                val = parts[1].strip()

                if '體重' in key or 'weight' in key:
                    try:
                        weight = float(val.replace('kg', '').strip())
                    except ValueError:
                        pass
                elif '性別' in key or 'sex' in key or 'gender' in key:
                    sex = val.upper()
                elif '年齡' in key or 'age' in key:
                    try:
                        age = float(val)
                    except ValueError:
                        pass

        if subj not in demo:
            demo[subj] = {'weight': None, 'sex': None, 'age': None}

        # Keep csv as authoritative for weight/sex; only fill if missing.
        if demo[subj].get('weight') is None and weight is not None:
            demo[subj]['weight'] = weight
        if not demo[subj].get('sex') and sex:
            demo[subj]['sex'] = sex

        # Age is usually only available in Utility.txt.
        if age is not None:
            demo[subj]['age'] = age

    return demo


def main():
    print("=" * 72)
    print("  Exp 6 — Correlation & Demographic Analysis")
    print("=" * 72)

    # ── Load Exp 1 results ──
    exp1_dir = os.path.join(OUTPUT_ROOT, 'exp1_main')
    results_path = os.path.join(exp1_dir, 'full_results.csv')
    if not os.path.exists(results_path):
        print(f"  ERROR: Exp 1 results not found at {results_path}")
        print("  Please run run_exp1_main.py first.")
        return

    df = pd.read_csv(results_path)
    df_nn = df[df['experiment'] == 'CNN_MLP']

    # ── Load reliability stats ──
    rel_path = os.path.join(exp1_dir, 'reliability_stats.csv')
    if os.path.exists(rel_path):
        rel_df = pd.read_csv(rel_path)
    else:
        # Compute from raw data
        all_items = load_all_data()
        rel_rows = []
        for sub in sorted(set(it['subject'] for it in all_items)):
            sub_items = [x for x in all_items if x['subject'] == sub]
            ratios = [it['labels'].sum() / NUM_CHANNELS for it in sub_items]
            rel_rows.append({
                'subject': sub,
                'mean_reliable_ratio': np.mean(ratios),
                'n_windows': len(sub_items),
            })
        rel_df = pd.DataFrame(rel_rows)

    # ── Per-subject MAE ──
    sub_mae = (df_nn.groupby('subject')
               .agg(MAE_viterbi=('abs_err_viterbi', 'mean'),
                    MAE_raw=('abs_err_raw', 'mean'),
                    Acc5=('abs_err_viterbi', lambda x: (x < 5).mean() * 100),
                    N=('abs_err_raw', 'count'))
               .reset_index())

    # Merge reliability
    merged = sub_mae.merge(rel_df[['subject', 'mean_reliable_ratio']],
                           on='subject', how='left')

    # ── 1. Reliability vs MAE ──
    from scipy.stats import pearsonr, spearmanr
    valid = merged.dropna(subset=['mean_reliable_ratio'])
    r_vals = valid['mean_reliable_ratio'].values
    m_vals = valid['MAE_viterbi'].values

    if len(valid) >= 5:
        pearson_r, pearson_p = pearsonr(r_vals, m_vals)
        spearman_r, spearman_p = spearmanr(r_vals, m_vals)
    else:
        pearson_r = spearman_r = pearson_p = spearman_p = float('nan')

    print(f"\n  Reliability Ratio vs MAE (N={len(valid)}):")
    print(f"    Pearson  r={pearson_r:.3f}  p={pearson_p:.4f}")
    print(f"    Spearman r={spearman_r:.3f}  p={spearman_p:.4f}")

    corr_df = pd.DataFrame([{
        'metric': 'reliability_ratio_vs_MAE',
        'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'spearman_r': spearman_r, 'spearman_p': spearman_p,
        'N': len(valid),
    }])

    # ── 2. Demographics ──
    demo = load_demographics()
    merged['weight'] = merged['subject'].map(
        lambda s: demo.get(s, {}).get('weight'))
    merged['sex'] = merged['subject'].map(
        lambda s: demo.get(s, {}).get('sex'))
    merged['age'] = merged['subject'].map(
        lambda s: demo.get(s, {}).get('age'))

    # Weight vs MAE
    w_valid = merged.dropna(subset=['weight'])
    if len(w_valid) >= 5:
        w_pearson_r, w_pearson_p = pearsonr(
            w_valid['weight'].values, w_valid['MAE_viterbi'].values)
        w_spearman_r, w_spearman_p = spearmanr(
            w_valid['weight'].values, w_valid['MAE_viterbi'].values)
        print(f"\n  Weight vs MAE (N={len(w_valid)}):")
        print(f"    Pearson  r={w_pearson_r:.3f}  p={w_pearson_p:.4f}")
        print(f"    Spearman r={w_spearman_r:.3f}  p={w_spearman_p:.4f}")
        corr_df = pd.concat([corr_df, pd.DataFrame([{
            'metric': 'weight_vs_MAE',
            'pearson_r': w_pearson_r, 'pearson_p': w_pearson_p,
            'spearman_r': w_spearman_r, 'spearman_p': w_spearman_p,
            'N': len(w_valid),
        }])], ignore_index=True)

    # Sex vs MAE
    sex_groups = merged.dropna(subset=['sex']).groupby('sex')['MAE_viterbi']
    if len(sex_groups) >= 2:
        print(f"\n  Sex vs MAE:")
        for sex, group in sex_groups:
            print(f"    {sex}: MAE={group.mean():.2f} ± {group.std():.2f}  "
                  f"(N={len(group)})")

    # ── Save ──
    corr_df.to_csv(os.path.join(OUTPUT_DIR, 'correlations.csv'), index=False)
    merged.to_csv(os.path.join(OUTPUT_DIR, 'subject_analysis.csv'), index=False)

    # ── 3. Tier analysis ──
    def tier(mae):
        if mae < 5:
            return 'A (<5)'
        elif mae < 10:
            return 'B (5-10)'
        else:
            return 'C (≥10)'
    merged['tier'] = merged['MAE_viterbi'].apply(tier)
    tier_stats = merged.groupby('tier').agg(
        N=('subject', 'count'),
        mean_MAE=('MAE_viterbi', 'mean'),
        mean_reliable_ratio=('mean_reliable_ratio', 'mean'),
    ).reset_index()
    tier_stats.to_csv(os.path.join(OUTPUT_DIR, 'tier_summary.csv'), index=False)
    print(f"\n  Tier summary:")
    print(tier_stats.to_string(index=False))

    # ── 4. Outlier subjects (MAE > 15) ──
    outliers = merged[merged['MAE_viterbi'] > 15]
    if len(outliers) > 0:
        print(f"\n  Outlier subjects (MAE > 15):")
        for _, row in outliers.iterrows():
            print(f"    S{int(row['subject'])}  MAE={row['MAE_viterbi']:.2f}  "
                  f"reliable={row['mean_reliable_ratio']:.3f}  "
                  f"N={int(row['N'])}")

    print(f"\n{'=' * 72}")
    print(f"  Exp 6 COMPLETE")
    print(f"  Results saved to {OUTPUT_DIR}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
