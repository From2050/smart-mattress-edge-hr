#!/usr/bin/env python3
"""
Exp 3 — 消融實驗 + 多種子穩定性驗證

Part A — 消融 (LOSO, 33 subjects):
  NN-based (±SQI):
        no_attention_large_sqi / _nosqi — NoAttentionCNN_Large (~353K, ProposeCNN)
        full_sqi / full_nosqi        — SpatialAttentionCNN (~355K, Transformer variant)
        no_cnn_sqi / _nosqi          — NoCNN_MLP_Large (~354K, MLP-only variant)
  Conventional (±SQI):
    conv_avg_sqi / _nosqi
    conv_max_sqi / _nosqi
    PCA baselines (no SQI):
        pca_1 / pca_3 / pca_weighted

Part B — 多種子 (Seeds {42, 123, 2025}):
    Full vs NoAttentionCNN_Large, 3 seeds × 33 folds
"""

import os, sys, time, argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from common import (
    SpatialAttentionCNN, NoCNN_MLP_Large, NoAttentionCNN_Large,
    count_parameters, load_all_data, set_seed,
    train_reliability, evaluate_reliability,
    evaluate_conventional, make_fuser, summarize_results,
    DEVICE, SEED, EPOCHS, BATCH_SIZE, LR, OUTPUT_ROOT,
)
from run_exp8_pca_baseline import evaluate_pca_method
import torch

OUTPUT_DIR = os.path.join(OUTPUT_ROOT, 'exp3_ablation')
os.makedirs(OUTPUT_DIR, exist_ok=True)

NN_MODELS = {
    'no_attention_large': NoAttentionCNN_Large,
    'full':               SpatialAttentionCNN,
    'no_cnn':             NoCNN_MLP_Large,
}

MULTI_SEED_MODELS = {
    'full':               SpatialAttentionCNN,
    'no_attention_large': NoAttentionCNN_Large,
}

SEEDS = [42, 123, 2025]


def run_ablation(all_items, subjects, exp1_canonical_df=None):
    """Part A: ablation study with SQI on/off + PCA baselines.

    If exp1_canonical_df is provided, no_attention_large_nosqi is taken from
    Exp1 canonical CNN_MLP results to keep cross-experiment consistency.
    """
    print("\n" + "=" * 72)
    print("  Part A — Ablation Study")
    print("=" * 72)
    all_results = []

    # ── NN models ──
    for model_name, model_cls in NN_MODELS.items():
        n_params = count_parameters(model_cls())
        print(f"\n{'─' * 60}")
        print(f"  Model: {model_name}  ({n_params:,} params)")
        print(f"{'─' * 60}")

        for si, test_sub in enumerate(subjects):
            set_seed(SEED)
            train_items = [x for x in all_items if x['subject'] != test_sub]
            test_items  = [x for x in all_items if x['subject'] == test_sub]
            if not train_items or not test_items:
                continue

            # Keep no_attention_large no-SQI exactly aligned with Exp1 canonical results.
            if model_name == 'no_attention_large' and exp1_canonical_df is not None:
                can_sub = exp1_canonical_df[exp1_canonical_df['subject'] == test_sub].copy()
                if len(can_sub) > 0:
                    rows_nosqi = can_sub.to_dict('records')
                    for r in rows_nosqi:
                        r['experiment'] = 'no_attention_large_nosqi'
                    all_results.extend(rows_nosqi)

                model = model_cls().to(DEVICE)
                train_reliability(model, train_items)
                rows_sqi = evaluate_reliability(model, test_items, make_fuser(), use_sqi=True)
                for r in rows_sqi:
                    r['experiment'] = 'no_attention_large_sqi'
                all_results.extend(rows_sqi)
            else:
                model = model_cls().to(DEVICE)
                train_reliability(model, train_items)

                for use_sqi in [True, False]:
                    tag = f"{model_name}_{'sqi' if use_sqi else 'nosqi'}"
                    rows = evaluate_reliability(model, test_items, make_fuser(),
                                                use_sqi=use_sqi)
                    for r in rows:
                        r['experiment'] = tag
                    all_results.extend(rows)

            sub_df = pd.DataFrame([r for r in all_results
                                   if r['subject'] == test_sub
                                   and r['experiment'] == f"{model_name}_nosqi"])
            if not sub_df.empty:
                print(f"  [{si+1:2d}/{len(subjects)}] S{test_sub}  "
                      f"Viterbi={sub_df['abs_err_viterbi'].mean():.1f}")
            del model
            torch.cuda.empty_cache()

    # ── Conventional baselines ──
    for method in ['avg', 'max']:
        print(f"\n  Conventional: {method}")
        for si, test_sub in enumerate(subjects):
            test_items = [x for x in all_items if x['subject'] == test_sub]
            for use_sqi in [True, False]:
                tag = f"conv_{method}_{'sqi' if use_sqi else 'nosqi'}"
                rows = evaluate_conventional(method, test_items, make_fuser(),
                                             use_sqi=use_sqi)
                for r in rows:
                    r['experiment'] = tag
                all_results.extend(rows)

    # ── PCA baselines (no SQI) ──
    pca_methods = ['pca_1', 'pca_3', 'pca_weighted']
    for method in pca_methods:
        print(f"\n  PCA baseline: {method}")
        for si, test_sub in enumerate(subjects):
            test_items = [x for x in all_items if x['subject'] == test_sub]
            rows = evaluate_pca_method(method, test_items, make_fuser())
            tag = method
            for r in rows:
                r['experiment'] = tag
            all_results.extend(rows)

            sub_df = pd.DataFrame(rows)
            if not sub_df.empty:
                print(f"    [{si+1:2d}/{len(subjects)}] S{test_sub}  "
                      f"Viterbi={sub_df['abs_err_viterbi'].mean():.1f}")

    return pd.DataFrame(all_results)


def run_multi_seed(all_items, subjects):
    """Part B: multi-seed validation."""
    print("\n" + "=" * 72)
    print(f"  Part B — Multi-Seed Validation (Seeds: {SEEDS})")
    print("=" * 72)
    all_results = []

    for model_name, model_cls in MULTI_SEED_MODELS.items():
        n_params = count_parameters(model_cls())
        for seed in SEEDS:
            print(f"\n{'─' * 60}")
            print(f"  {model_name} (seed={seed}, {n_params:,} params)")
            print(f"{'─' * 60}")

            for si, test_sub in enumerate(subjects):
                set_seed(seed + si)  # per-fold seed
                train_items = [x for x in all_items if x['subject'] != test_sub]
                test_items  = [x for x in all_items if x['subject'] == test_sub]
                if not train_items or not test_items:
                    continue

                model = model_cls().to(DEVICE)
                train_reliability(model, train_items)
                rows = evaluate_reliability(model, test_items, make_fuser())
                for r in rows:
                    r['experiment'] = f"{model_name}_seed{seed}"
                    r['seed'] = seed
                all_results.extend(rows)

                sub_df = pd.DataFrame([r for r in all_results
                                       if r['subject'] == test_sub
                                       and r['experiment'] == f"{model_name}_seed{seed}"])
                if not sub_df.empty:
                    print(f"    [{si+1:2d}/{len(subjects)}] S{test_sub}  "
                          f"{sub_df['abs_err_viterbi'].mean():.1f}")
                del model
                torch.cuda.empty_cache()

    return pd.DataFrame(all_results)


def run_sqi_only(all_items, subjects, exp1_canonical_df=None):
    """Run only no_attention_large SQI ablation and refresh summary rows."""
    print("\n" + "=" * 72)
    print("  SQI-only Ablation (no_attention_large)")
    print("=" * 72)

    results = []
    model_name = 'no_attention_large'

    for si, test_sub in enumerate(subjects):
        set_seed(SEED)
        train_items = [x for x in all_items if x['subject'] != test_sub]
        test_items = [x for x in all_items if x['subject'] == test_sub]
        if not train_items or not test_items:
            continue

        # no-SQI row aligned with Exp1 canonical
        if exp1_canonical_df is not None:
            can_sub = exp1_canonical_df[exp1_canonical_df['subject'] == test_sub].copy()
            if len(can_sub) > 0:
                rows_nosqi = can_sub.to_dict('records')
                for r in rows_nosqi:
                    r['experiment'] = f'{model_name}_nosqi'
                results.extend(rows_nosqi)

        # SQI row from current rerun
        model = NoAttentionCNN_Large().to(DEVICE)
        train_reliability(model, train_items)
        rows_sqi = evaluate_reliability(model, test_items, make_fuser(), use_sqi=True)
        for r in rows_sqi:
            r['experiment'] = f'{model_name}_sqi'
        results.extend(rows_sqi)

        sub_df = pd.DataFrame(rows_sqi)
        if not sub_df.empty:
            print(f"  [{si+1:2d}/{len(subjects)}] S{test_sub}  SQI-Viterbi={sub_df['abs_err_viterbi'].mean():.1f}")

        del model
        torch.cuda.empty_cache()

    df_sqi = pd.DataFrame(results)
    df_sqi.to_csv(os.path.join(OUTPUT_DIR, 'ablation_sqi_only_results.csv'), index=False)

    sqi_summary = summarize_results(df_sqi)
    sqi_summary.to_csv(os.path.join(OUTPUT_DIR, 'ablation_sqi_only_summary.csv'), index=False)

    # Patch main ablation summary rows (keep other rows as-is)
    main_summary_path = os.path.join(OUTPUT_DIR, 'ablation_summary.csv')
    if os.path.exists(main_summary_path):
        main_sum = pd.read_csv(main_summary_path)
        main_sum = main_sum[~main_sum['experiment'].isin([
            'no_attention_large_nosqi',
            'no_attention_large_sqi',
        ])]
        main_sum = pd.concat([main_sum, sqi_summary], ignore_index=True)
        main_sum = main_sum.sort_values('experiment').reset_index(drop=True)
        main_sum.to_csv(main_summary_path, index=False)
        print(f"  Updated: {main_summary_path}")

    print("\n  SQI-only summary:")
    print(sqi_summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'sqi_only'], default='full',
                        help='full: original Exp3 flow; sqi_only: rerun only no_attention_large SQI ablation')
    args = parser.parse_args()

    t0 = time.time()
    set_seed(SEED)

    print("=" * 72)
    print("  Exp 3 — Ablation + Multi-Seed Validation")
    print(f"  Device: {DEVICE}")
    print("=" * 72)

    # Print model sizes
    for name, cls in NN_MODELS.items():
        print(f"    {name:25s}  {count_parameters(cls()):>8,} params")

    all_items = load_all_data()
    subjects = sorted(set(it['subject'] for it in all_items))
    print(f"\n  Loaded {len(all_items)} windows from {len(subjects)} subjects")

    # ── Load Exp1 canonical for alignment ──
    exp1_path = os.path.join(OUTPUT_ROOT, 'exp1_main', 'full_results.csv')
    exp1_canonical_df = None
    if os.path.exists(exp1_path):
        exp1_df = pd.read_csv(exp1_path)
        exp1_canonical_df = exp1_df[exp1_df['experiment'] == 'CNN_MLP'].copy()
        print(f"  Loaded Exp1 canonical rows for alignment: {len(exp1_canonical_df)}")
    else:
        print("  Exp1 canonical not found; no_attention_large_nosqi will be retrained in Exp3")

    if args.mode == 'sqi_only':
        run_sqi_only(all_items, subjects, exp1_canonical_df=exp1_canonical_df)
        elapsed = (time.time() - t0) / 60.0
        print(f"\n{'=' * 72}")
        print(f"  Exp 3 SQI-only COMPLETE — {elapsed:.1f} min")
        print(f"  Results saved to {OUTPUT_DIR}")
        print(f"{'=' * 72}")
        return

    # ── Part A ──
    df_ablation = run_ablation(all_items, subjects, exp1_canonical_df=exp1_canonical_df)
    df_ablation.to_csv(os.path.join(OUTPUT_DIR, 'ablation_full_results.csv'),
                       index=False)

    abl_summary = summarize_results(df_ablation)
    abl_summary.to_csv(os.path.join(OUTPUT_DIR, 'ablation_summary.csv'),
                       index=False)
    print(f"\n{'=' * 72}")
    print("  Ablation Summary")
    print("=" * 72)
    print(abl_summary.to_string(index=False))

    # Per-subject for ablation
    abl_sub = (df_ablation.groupby(['experiment', 'subject'])
               .agg(MAE_viterbi=('abs_err_viterbi', 'mean'),
                    N=('abs_err_raw', 'count'))
               .reset_index())
    abl_sub.to_csv(os.path.join(OUTPUT_DIR, 'ablation_per_subject.csv'),
                   index=False)

    # ── Part B ──
    df_seed = run_multi_seed(all_items, subjects)
    df_seed.to_csv(os.path.join(OUTPUT_DIR, 'multiseed_full_results.csv'),
                   index=False)

    seed_summary = summarize_results(df_seed)
    seed_summary.to_csv(os.path.join(OUTPUT_DIR, 'multiseed_summary.csv'),
                        index=False)
    print(f"\n{'=' * 72}")
    print("  Multi-Seed Summary")
    print("=" * 72)
    print(seed_summary.to_string(index=False))

    # Cross-seed stability
    print("\n  Cross-seed stability (MAE Viterbi):")
    for model_name in MULTI_SEED_MODELS:
        maes = []
        for seed in SEEDS:
            tag = f"{model_name}_seed{seed}"
            sdf = df_seed[df_seed['experiment'] == tag]
            if len(sdf) > 0:
                maes.append(sdf['abs_err_viterbi'].mean())
        if maes:
            print(f"    {model_name}: {' / '.join(f'{m:.2f}' for m in maes)}  "
                  f"± {np.std(maes):.2f}")

    elapsed = (time.time() - t0) / 60.0
    print(f"\n{'=' * 72}")
    print(f"  Exp 3 COMPLETE — {elapsed:.1f} min")
    print(f"  Results saved to {OUTPUT_DIR}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
