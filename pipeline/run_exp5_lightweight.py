#!/usr/bin/env python3
"""
Exp 5 — 輕量骨幹比較 (33 subjects LOSO)

以論文 11 層全卷積 CNN (~16K params) 搭配我們的可靠度融合管線，
對比我們的 CNN+MLP (~353K params)。
後處理管線完全相同，僅骨幹不同。
"""

import os, sys, time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from common import (
    PaperCNN_Reliability, NoAttentionCNN_Large, count_parameters,
    load_all_data, set_seed, train_reliability, evaluate_reliability,
    make_fuser, summarize_results,
    DEVICE, SEED, EPOCHS, BATCH_SIZE, LR, OUTPUT_ROOT,
)
import torch

OUTPUT_DIR = os.path.join(OUTPUT_ROOT, 'exp5_lightweight')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    t0 = time.time()
    set_seed(SEED)

    n_small = count_parameters(PaperCNN_Reliability())
    n_large = count_parameters(NoAttentionCNN_Large())

    print("=" * 72)
    print("  Exp 5 — Lightweight Backbone Comparison")
    print(f"  Device: {DEVICE}  |  Seed: {SEED}")
    print(f"  PaperCNN_Reliability  {n_small:>10,} params  (11-layer AllConv)")
    print(f"  NoAttentionCNN_Large  {n_large:>10,} params  (CNN+MLP)")
    print(f"  Size ratio: {n_large / n_small:.1f}x")
    print("=" * 72)

    all_items = load_all_data()
    subjects = sorted(set(it['subject'] for it in all_items))
    print(f"  Loaded {len(all_items)} windows from {len(subjects)} subjects\n")

    fuser = make_fuser()
    results_small = []

    # ── Load Exp 1 canonical CNN+MLP results (avoid retraining → CUDA non-determinism) ──
    exp1_path = os.path.join(OUTPUT_ROOT, 'exp1_main', 'full_results.csv')
    assert os.path.exists(exp1_path), \
        f"Exp 1 results not found: {exp1_path}\n  Run run_exp1_main.py first."
    df_exp1 = pd.read_csv(exp1_path)
    df_exp1 = df_exp1[df_exp1['experiment'] == 'CNN_MLP'].copy()
    print(f"  Loaded Exp 1 CNN+MLP results: {len(df_exp1)} rows")

    for si, test_sub in enumerate(subjects):
        train_items = [x for x in all_items if x['subject'] != test_sub]
        test_items  = [x for x in all_items if x['subject'] == test_sub]
        if not train_items or not test_items:
            continue

        # ── PaperCNN (small) ──
        set_seed(SEED)
        model_small = PaperCNN_Reliability().to(DEVICE)
        train_reliability(model_small, train_items)
        res_s = evaluate_reliability(model_small, test_items, fuser)
        results_small.extend(res_s)
        mae_s = np.mean([r['abs_err_viterbi'] for r in res_s]) if res_s else 0
        del model_small
        torch.cuda.empty_cache()

        # ── OursCNN (large) — from Exp 1 ──
        sub_exp1 = df_exp1[df_exp1['subject'] == test_sub]
        mae_l = sub_exp1['abs_err_viterbi'].mean() if len(sub_exp1) > 0 else 0

        print(f"  [{si+1:2d}/{len(subjects)}] S{test_sub:3d}  "
              f"PaperCNN={mae_s:.1f}  OursCNN={mae_l:.1f} (from Exp1)")

    # ── Save ──
    df_small = pd.DataFrame(results_small)
    df_small['method'] = 'PaperCNN_Reliability'
    df_large = df_exp1.rename(columns={'experiment': '_exp'}).copy()
    df_large['method'] = 'OursCNN_Reliability'
    df_all = pd.concat([df_small, df_large], ignore_index=True)
    df_all.to_csv(os.path.join(OUTPUT_DIR, 'full_results.csv'), index=False)

    # ── Summary ──
    print(f"\n{'=' * 72}")
    print("  Summary")
    print("=" * 72)
    summary_rows = []
    for name, df in [('PaperCNN_Reliability', df_small),
                     ('OursCNN_Reliability', df_large)]:
        mae_v = df['abs_err_viterbi'].mean()
        std_v = df['abs_err_viterbi'].std()
        acc5  = (df['abs_err_viterbi'] < 5).mean() * 100
        mae_r = df['abs_err_raw'].mean()
        summary_rows.append({
            'method': name, 'MAE_viterbi': mae_v, 'Std_viterbi': std_v,
            'MAE_raw': mae_r, 'Acc5_viterbi': acc5, 'N': len(df),
        })
        print(f"  {name:25s}  MAE={mae_v:.2f} ± {std_v:.2f}  "
              f"Acc5={acc5:.1f}%  (raw={mae_r:.2f})")
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'summary.csv'), index=False)

    # ── Per-subject ──
    sub_comp = []
    for name, df in [('PaperCNN_Reliability', df_small),
                     ('OursCNN_Reliability', df_large)]:
        for sub in subjects:
            sdf = df[df['subject'] == sub]
            if len(sdf) == 0:
                continue
            sub_comp.append({
                'method': name, 'subject': sub,
                'MAE_viterbi': sdf['abs_err_viterbi'].mean(),
                'Acc5': (sdf['abs_err_viterbi'] < 5).mean() * 100,
                'N': len(sdf),
            })
    df_sub = pd.DataFrame(sub_comp)
    df_sub.to_csv(os.path.join(OUTPUT_DIR, 'per_subject.csv'), index=False)

    # ── Per-posture ──
    posture_comp = []
    for name, df in [('PaperCNN_Reliability', df_small),
                     ('OursCNN_Reliability', df_large)]:
        for pos in sorted(df['posture'].unique()):
            pdf = df[df['posture'] == pos]
            posture_comp.append({
                'method': name, 'posture': pos,
                'MAE_viterbi': pdf['abs_err_viterbi'].mean(),
                'N': len(pdf),
            })
    pd.DataFrame(posture_comp).to_csv(
        os.path.join(OUTPUT_DIR, 'per_posture.csv'), index=False)

    # ── Wilcoxon ──
    from scipy.stats import wilcoxon
    maes_s, maes_l = [], []
    for sub in subjects:
        ms = df_sub[(df_sub['method'] == 'PaperCNN_Reliability')
                    & (df_sub['subject'] == sub)]
        ml = df_sub[(df_sub['method'] == 'OursCNN_Reliability')
                    & (df_sub['subject'] == sub)]
        if len(ms) > 0 and len(ml) > 0:
            maes_s.append(ms.iloc[0]['MAE_viterbi'])
            maes_l.append(ml.iloc[0]['MAE_viterbi'])
    maes_s, maes_l = np.array(maes_s), np.array(maes_l)
    diff = maes_s - maes_l
    try:
        stat, pval = wilcoxon(maes_s, maes_l)
    except ValueError:
        stat, pval = 0.0, 1.0
    print(f"\n  Wilcoxon: PaperCNN {maes_s.mean():.2f} vs OursCNN {maes_l.mean():.2f}  "
          f"Δ={diff.mean():+.2f}  p={pval:.4f}")
    n_paper_better = int((diff < -0.1).sum())
    n_tie = int(((diff >= -0.1) & (diff <= 0.1)).sum())
    n_ours_better = int((diff > 0.1).sum())
    print(f"    PaperCNN better: {n_paper_better}  "
          f"Tie: {n_tie}  OursCNN better: {n_ours_better}")
    pd.DataFrame([{
        'method_A': 'PaperCNN_Reliability', 'method_B': 'OursCNN_Reliability',
        'mean_A': maes_s.mean(), 'mean_B': maes_l.mean(),
        'delta': diff.mean(), 'p_value': pval,
        'A_better': n_paper_better, 'tie': n_tie, 'B_better': n_ours_better,
    }]).to_csv(os.path.join(OUTPUT_DIR, 'statistical_test.csv'), index=False)

    elapsed = (time.time() - t0) / 60.0
    print(f"\n{'=' * 72}")
    print(f"  Exp 5 COMPLETE — {elapsed:.1f} min")
    print(f"  Results saved to {OUTPUT_DIR}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
