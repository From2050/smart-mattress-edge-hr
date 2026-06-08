#!/usr/bin/env python3
"""
Exp 1 — 主模型 LOSO 整體效能 + 傳統基線

以 CNN+MLP (NoAttentionCNN_Large, ~353K) 為主模型，
33 subjects LOSO cross-validation。

輸出:
  - 整體 MAE / Std / Acc5 (Raw + Viterbi)
  - 每姿勢 MAE
  - 每受試者 MAE + Tier 分類
  - Viterbi 改善分析
  - 傳統基線 (avg, max) 對照
  - 資料統計 (windows 數, 可靠通道比例)
"""

import os, sys, time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from common import (
    NoAttentionCNN_Large, count_parameters,
    load_all_data, set_seed, train_reliability, evaluate_reliability,
    evaluate_conventional, make_fuser, summarize_results,
    DEVICE, SEED, EPOCHS, BATCH_SIZE, LR, OUTPUT_ROOT, NUM_CHANNELS,
)
import torch

OUTPUT_DIR = os.path.join(OUTPUT_ROOT, 'exp1_main')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    t0 = time.time()
    set_seed(SEED)

    print("=" * 72)
    print("  Exp 1 — Main LOSO: CNN+MLP (NoAttentionCNN_Large)")
    print(f"  Device: {DEVICE}  |  Seed: {SEED}")
    print(f"  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
    print("=" * 72)

    n_params = count_parameters(NoAttentionCNN_Large())
    print(f"  Model params: {n_params:,}")

    # ── Load data ──
    all_items = load_all_data()
    subjects = sorted(set(it['subject'] for it in all_items))
    print(f"  Loaded {len(all_items)} windows from {len(subjects)} subjects\n")

    # ── Data statistics ──
    total_reliable = sum(it['labels'].sum() for it in all_items)
    total_channels = len(all_items) * NUM_CHANNELS
    print(f"  Reliable channel ratio: {total_reliable / total_channels:.3f}")
    print(f"  Total windows: {len(all_items)}")
    posture_counts = {}
    for it in all_items:
        posture_counts[it['posture']] = posture_counts.get(it['posture'], 0) + 1
    for p in sorted(posture_counts):
        print(f"    {p:15s}: {posture_counts[p]:5d} windows")
    print()

    fuser = make_fuser()
    results_nn = []

    # ── CNN+MLP LOSO ──
    for si, test_sub in enumerate(subjects):
        set_seed(SEED)
        train_items = [x for x in all_items if x['subject'] != test_sub]
        test_items  = [x for x in all_items if x['subject'] == test_sub]
        if not train_items or not test_items:
            continue

        model = NoAttentionCNN_Large().to(DEVICE)
        train_reliability(model, train_items)
        rows = evaluate_reliability(model, test_items, fuser)
        for r in rows:
            r['experiment'] = 'CNN_MLP'
        results_nn.extend(rows)

        sub_df = pd.DataFrame([r for r in results_nn if r['subject'] == test_sub])
        mae_v = sub_df['abs_err_viterbi'].mean()
        mae_r = sub_df['abs_err_raw'].mean()
        print(f"  [{si+1:2d}/{len(subjects)}] S{test_sub:3d}  "
              f"Raw={mae_r:.1f}  Viterbi={mae_v:.1f}")
        del model
        torch.cuda.empty_cache()

    # ── Conventional baselines ──
    results_conv = []
    for method in ['avg', 'max']:
        print(f"\n  Conventional baseline: {method}")
        for si, test_sub in enumerate(subjects):
            test_items = [x for x in all_items if x['subject'] == test_sub]
            rows = evaluate_conventional(method, test_items, make_fuser())
            for r in rows:
                r['experiment'] = f'conv_{method}'
            results_conv.extend(rows)

    # ── Merge and save ──
    all_results = results_nn + results_conv
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'full_results.csv'), index=False)

    # ── Global summary ──
    summary = summarize_results(df)
    summary.to_csv(os.path.join(OUTPUT_DIR, 'summary.csv'), index=False)
    print(f"\n{'=' * 72}")
    print("  Global Summary")
    print("=" * 72)
    print(summary.to_string(index=False))

    # ── CNN+MLP: Per-posture ──
    df_nn = df[df['experiment'] == 'CNN_MLP']
    posture_summary = (df_nn.groupby('posture')
                       .agg(MAE_raw=('abs_err_raw', 'mean'),
                            MAE_viterbi=('abs_err_viterbi', 'mean'),
                            Acc5=('abs_err_viterbi',
                                  lambda x: (x < 5).mean() * 100),
                            N=('abs_err_raw', 'count'))
                       .reset_index())
    posture_summary.to_csv(os.path.join(OUTPUT_DIR, 'per_posture.csv'),
                           index=False)
    print(f"\n  Per-posture (CNN+MLP Viterbi):")
    print(posture_summary.to_string(index=False))

    # ── CNN+MLP: Per-subject + Tier ──
    subject_summary = (df_nn.groupby('subject')
                       .agg(MAE_raw=('abs_err_raw', 'mean'),
                            MAE_viterbi=('abs_err_viterbi', 'mean'),
                            Acc5=('abs_err_viterbi',
                                  lambda x: (x < 5).mean() * 100),
                            N=('abs_err_raw', 'count'))
                       .reset_index())

    def tier(mae):
        if mae < 5:
            return 'A (<5)'
        elif mae < 10:
            return 'B (5-10)'
        else:
            return 'C (≥10)'

    subject_summary['tier'] = subject_summary['MAE_viterbi'].apply(tier)
    subject_summary.to_csv(os.path.join(OUTPUT_DIR, 'per_subject.csv'),
                           index=False)

    tier_counts = subject_summary['tier'].value_counts()
    print(f"\n  Subject tiers (Viterbi MAE):")
    for t in ['A (<5)', 'B (5-10)', 'C (≥10)']:
        print(f"    {t}: {tier_counts.get(t, 0)} subjects")

    # ── Viterbi improvement ──
    mae_raw = df_nn['abs_err_raw'].mean()
    mae_vit = df_nn['abs_err_viterbi'].mean()
    improve = (mae_raw - mae_vit) / mae_raw * 100
    print(f"\n  Viterbi improvement: {mae_raw:.2f} → {mae_vit:.2f} "
          f"(Δ={mae_raw - mae_vit:.2f}, {improve:.1f}%)")

    # ── Reliable channel statistics per subject ──
    rel_stats = []
    for sub in subjects:
        sub_items = [x for x in all_items if x['subject'] == sub]
        ratios = [it['labels'].sum() / NUM_CHANNELS for it in sub_items]
        rel_stats.append({
            'subject': sub,
            'mean_reliable_ratio': np.mean(ratios),
            'median_reliable_ratio': np.median(ratios),
            'n_windows': len(sub_items),
        })
    rel_df = pd.DataFrame(rel_stats)
    rel_df.to_csv(os.path.join(OUTPUT_DIR, 'reliability_stats.csv'),
                  index=False)

    elapsed = (time.time() - t0) / 60.0
    print(f"\n{'=' * 72}")
    print(f"  Exp 1 COMPLETE — {elapsed:.1f} min")
    print(f"  Results saved to {OUTPUT_DIR}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
