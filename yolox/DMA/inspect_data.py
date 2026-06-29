"""
Inspect DMA training data to verify quality before training.

Usage:
  python3 yolox/DMA/inspect_data.py --data-dir datasets/dma_train/

Checks:
  1. Label distribution (pos/neg ratio)
  2. Feature statistics & NaN/Inf
  3. Separability: do positive pairs actually have lower costs than negatives?
  4. Appearance availability (has_appearance flag)
  5. Model sanity: random-init weights vs trained weights output
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FEATURE_NAMES = [
    "motion_iou",        # 0
    "motion_cost",       # 1
    "mahalanobis_norm",  # 2
    "cov_trace_log",     # 3
    "cov_mean_log",      # 4
    "vel_magnitude",     # 5
    "time_since_update", # 6
    "cosine_dist",       # 7
    "feat_variance",     # 8
    "det_score",         # 9
    "bbox_area_log",     # 10
    "bbox_aspect",       # 11
    "track_age_norm",    # 12
    "tracklet_len_norm", # 13
    "has_appearance",    # 14
]


def sep(title=""):
    print(f"\n{'─'*60}")
    if title:
        print(f"  {title}")
        print('─'*60)


def load_all(data_dir: str):
    paths = sorted(Path(data_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files in {data_dir}")

    all_feat, all_labels, all_mc, all_ac = [], [], [], []
    for p in paths:
        d = np.load(p)
        all_feat.append(d["features"])
        all_labels.append(d["labels"])
        if "motion_costs" in d:
            all_mc.append(d["motion_costs"])
        if "appearance_costs" in d:
            all_ac.append(d["appearance_costs"])

    features = np.concatenate(all_feat, axis=0).astype(np.float32)
    labels   = np.concatenate(all_labels, axis=0).astype(np.float32)
    mc = np.concatenate(all_mc, axis=0) if all_mc else features[:, 1]
    ac = np.concatenate(all_ac, axis=0) if all_ac else features[:, 7]
    return features, labels, mc, ac


def check_labels(labels):
    sep("1. LABEL DISTRIBUTION")
    n = len(labels)
    pos = int(labels.sum())
    neg = n - pos
    print(f"  Total samples : {n:,}")
    print(f"  Positive (match) : {pos:,}  ({100*pos/n:.1f}%)")
    print(f"  Negative (no-match): {neg:,}  ({100*neg/n:.1f}%)")
    print(f"  Imbalance ratio  : 1 : {neg//max(pos,1)}")


def check_features(features, labels):
    sep("2. FEATURE STATISTICS")
    pos = labels == 1
    neg = labels == 0
    print(f"  {'Feature':<20} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} "
          f"{'NaN':>5} {'pos_mean':>10} {'neg_mean':>10} {'sep?':>6}")
    for i, name in enumerate(FEATURE_NAMES):
        col = features[:, i]
        nan_cnt = int(np.isnan(col).sum() + np.isinf(col).sum())
        pm = col[pos].mean() if pos.sum() > 0 else float('nan')
        nm = col[neg].mean() if neg.sum() > 0 else float('nan')
        # Simple separability: are means different enough?
        diff = abs(pm - nm) / (col.std() + 1e-8)
        sep_mark = "✓" if diff > 0.1 else " "
        print(f"  {name:<20} {col.mean():>8.4f} {col.std():>8.4f} "
              f"{col.min():>8.4f} {col.max():>8.4f} {nan_cnt:>5} "
              f"{pm:>10.4f} {nm:>10.4f} {sep_mark:>6}")


def check_costs(mc, ac, labels):
    sep("3. COST SEPARABILITY  (key: do positives have LOWER cost?)")
    pos = labels == 1
    neg = labels == 0

    mc_pos, mc_neg = mc[pos], mc[neg]
    ac_pos, ac_neg = ac[pos], ac[neg]

    print(f"\n  motion_cost   pos: mean={mc_pos.mean():.4f}  neg: mean={mc_neg.mean():.4f}  "
          f"diff={mc_neg.mean()-mc_pos.mean():.4f}  {'✓ separable' if mc_neg.mean() > mc_pos.mean() else '✗ NOT separable'}")
    print(f"  appear_cost   pos: mean={ac_pos.mean():.4f}  neg: mean={ac_neg.mean():.4f}  "
          f"diff={ac_neg.mean()-ac_pos.mean():.4f}  {'✓ separable' if ac_neg.mean() > ac_pos.mean() else '✗ NOT separable'}")

    # Baseline: what if we use only motion_cost at threshold=0.5?
    pred_mc = (mc < 0.5).astype(float)
    acc_mc = (pred_mc == labels).mean()
    tp = ((pred_mc == 1) & (labels == 1)).sum()
    fp = ((pred_mc == 1) & (labels == 0)).sum()
    fn = ((pred_mc == 0) & (labels == 1)).sum()
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1_mc = 2 * prec * rec / max(prec + rec, 1e-8)

    pred_ac = (ac < 0.5).astype(float)
    acc_ac = (pred_ac == labels).mean()
    tp2 = ((pred_ac == 1) & (labels == 1)).sum()
    fp2 = ((pred_ac == 1) & (labels == 0)).sum()
    fn2 = ((pred_ac == 0) & (labels == 1)).sum()
    prec2 = tp2 / max(tp2 + fp2, 1)
    rec2  = tp2 / max(tp2 + fn2, 1)
    f1_ac = 2 * prec2 * rec2 / max(prec2 + rec2, 1e-8)

    print(f"\n  Baseline threshold=0.5:")
    print(f"    motion_cost alone  → acc={acc_mc:.4f}  f1={f1_mc:.4f}  prec={prec:.4f}  rec={rec:.4f}")
    print(f"    appear_cost alone  → acc={acc_ac:.4f}  f1={f1_ac:.4f}  prec={prec2:.4f}  rec={rec2:.4f}")

    # Optimal threshold for motion_cost
    best_f1, best_thr = 0, 0.5
    for thr in np.arange(0.1, 1.0, 0.05):
        pred = (mc < thr).astype(float)
        tp_ = ((pred == 1) & (labels == 1)).sum()
        fp_ = ((pred == 1) & (labels == 0)).sum()
        fn_ = ((pred == 0) & (labels == 1)).sum()
        p_ = tp_ / max(tp_ + fp_, 1)
        r_ = tp_ / max(tp_ + fn_, 1)
        f1_ = 2*p_*r_ / max(p_+r_, 1e-8)
        if f1_ > best_f1:
            best_f1, best_thr = f1_, thr
    print(f"\n    Best threshold for motion_cost → thr={best_thr:.2f}  f1={best_f1:.4f}")
    print(f"    (This is the UPPER BOUND the DMA model should try to beat)")


def check_appearance(features, labels):
    sep("4. APPEARANCE AVAILABILITY")
    has_app = features[:, 14]
    print(f"  Pairs WITH  appearance: {int(has_app.sum()):,}  ({100*has_app.mean():.1f}%)")
    print(f"  Pairs WITHOUT appearance: {int((1-has_app).sum()):,}  ({100*(1-has_app).mean():.1f}%)")
    if has_app.mean() < 0.01:
        print("\n  ⚠ WARNING: Almost NO pairs have ReID features!")
        print("    → appearance_cost is always 0.5 (neutral placeholder)")
        print("    → Model can only learn from motion features")
        print("    → To fix: re-run generate_data.py with --reid-weights")


def check_model_output(features, labels, mc, ac, ckpt_path=None):
    sep("5. MODEL OUTPUT CHECK")
    try:
        import torch
        from yolox.DMA.model import DynamicWeightNet

        # Random-init model
        model = DynamicWeightNet()
        x = torch.from_numpy(features[:1000]).float()
        with torch.no_grad():
            w = model(x)
        w_m = w[:, 0].numpy()
        w_r = w[:, 1].numpy()
        print(f"  Random-init model:")
        print(f"    w_motion: mean={w_m.mean():.4f}  std={w_m.std():.4f}")
        print(f"    w_reid:   mean={w_r.mean():.4f}  std={w_r.std():.4f}")

        if ckpt_path and Path(ckpt_path).exists():
            model_t, stats = DynamicWeightNet.load(ckpt_path)
            # normalize features with saved stats
            mean = np.array(stats["mean"])
            std  = np.array(stats["std"])
            feat_norm = (features[:1000] - mean) / std
            x_n = torch.from_numpy(feat_norm.astype(np.float32))
            with torch.no_grad():
                w_t = model_t(x_n)
            wm_t = w_t[:, 0].numpy()
            wr_t = w_t[:, 1].numpy()
            print(f"\n  Trained model ({Path(ckpt_path).name}):")
            print(f"    w_motion: mean={wm_t.mean():.4f}  std={wm_t.std():.4f}")
            print(f"    w_reid:   mean={wr_t.mean():.4f}  std={wr_t.std():.4f}")
            if wm_t.std() < 0.01 and wr_t.std() < 0.01:
                print("    ⚠ WARNING: weights are nearly constant → model collapsed!")
            else:
                print("    ✓ Weights are varying across pairs (model is active)")

            # Check correlation: do positive pairs get lower w_motion * mc?
            fused = wm_t * mc[:1000] + wr_t * ac[:1000]
            lbl_s = labels[:1000]
            pos_fused = fused[lbl_s == 1].mean() if (lbl_s == 1).sum() > 0 else float('nan')
            neg_fused = fused[lbl_s == 0].mean() if (lbl_s == 0).sum() > 0 else float('nan')
            print(f"\n    Fused cost: pos pairs={pos_fused:.4f}  neg pairs={neg_fused:.4f}")
            print(f"    {'✓ Correct direction (pos < neg)' if pos_fused < neg_fused else '✗ Wrong direction'}")

    except Exception as e:
        print(f"  [SKIP] torch not available or error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--ckpt", default=None, help="Path to trained checkpoint for output check")
    args = parser.parse_args()

    print(f"Loading data from: {args.data_dir}")
    features, labels, mc, ac = load_all(args.data_dir)
    print(f"Total: {len(labels):,} pairs  |  feature_dim={features.shape[1]}")

    check_labels(labels)
    check_features(features, labels)
    check_costs(mc, ac, labels)
    check_appearance(features, labels)
    check_model_output(features, labels, mc, ac, ckpt_path=args.ckpt)

    sep("DONE")


if __name__ == "__main__":
    main()
