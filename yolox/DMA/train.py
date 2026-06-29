"""
Train DynamicWeightNet.

Usage:
  python -m yolox.DMA.train \\
    --data-dir  data/dma_train \\
    --out-dir   weights/dma \\
    --epochs    50 \\
    --loss      bce          # or 'ranking'

Loss options:
  bce     - binary cross-entropy on the fused cost after soft weighting
  ranking - margin ranking: fused_cost(pos) + margin < fused_cost(neg)
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from yolox.DMA.model import DynamicWeightNet
from yolox.DMA.dataset import DMADataset


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

class BCEWeightedLoss(nn.Module):
    """
    Compute fused_cost = w_motion * motion_cost + w_reid * appearance_cost
    and minimise BCE(fused_cost, 1 - label):
      label=1 (correct match)  → want fused_cost LOW  → target=0
      label=0 (wrong match)    → want fused_cost HIGH → target=1

    motion_cost / appearance_cost must be raw [0,1] values, NOT normalized.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, weights, motion_cost, appearance_cost, labels):
        w_m = weights[:, 0]
        w_r = weights[:, 1]
        fused = w_m * motion_cost + w_r * appearance_cost
        fused = torch.clamp(fused, 1e-6, 1.0 - 1e-6)   # avoid log(0)
        target = 1.0 - labels
        return self.bce(fused, target)


class RankingLoss(nn.Module):
    """
    Margin ranking: fused_cost(pos) + margin < fused_cost(neg)
    """
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, weights, motion_cost, appearance_cost, labels):
        w_m = weights[:, 0]
        w_r = weights[:, 1]
        fused = torch.clamp(w_m * motion_cost + w_r * appearance_cost, 1e-6, 1.0 - 1e-6)

        pos_mask = labels > 0.5
        neg_mask = ~pos_mask
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=weights.device, requires_grad=True)

        pos_costs = fused[pos_mask]
        neg_costs = fused[neg_mask]
        diff = pos_costs.unsqueeze(1) - neg_costs.unsqueeze(0) + self.margin
        return torch.clamp(diff, min=0.0).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for features, labels, motion_cost, app_cost in loader:
        features    = features.to(device)
        labels      = labels.to(device)
        motion_cost = motion_cost.float().to(device)
        app_cost    = app_cost.float().to(device)

        weights = model(features)
        loss = loss_fn(weights, motion_cost, app_cost, labels)
        total_loss += loss.item()

        fused = (weights[:, 0] * motion_cost + weights[:, 1] * app_cost).clamp(1e-6, 1-1e-6)
        # threshold at 0.5: low cost → positive match
        preds = (fused < 0.5).float()
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = (all_preds == all_labels).float().mean().item()

    # F1 on the positive class (correct matches)
    tp = ((all_preds == 1) & (all_labels == 1)).sum().float()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().float()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().float()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return (
        total_loss / max(len(loader), 1),
        acc,
        f1.item(),
        precision.item(),
        recall.item(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_paths = sorted(Path(args.data_dir).glob("*.npz"))
    if not data_paths:
        raise FileNotFoundError(f"No .npz files found in {args.data_dir}")
    print(f"Found {len(data_paths)} sequences")

    train_ds, val_ds = DMADataset.split(
        [str(p) for p in data_paths],
        val_ratio=args.val_ratio,
        normalize=True,
        pos_neg_ratio=args.pos_neg_ratio,
    )
    pos, neg = train_ds.class_balance()
    print(f"Train: {len(train_ds)} samples  pos={pos}  neg={neg}  ratio=1:{neg//max(pos,1)}")
    print(f"Val:   {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 4, shuffle=False,
        num_workers=args.num_workers,
    )

    model = DynamicWeightNet(input_dim=15, hidden_dims=(64, 32)).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    loss_fn = RankingLoss(margin=args.margin) if args.loss == "ranking" else BCEWeightedLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "mean": train_ds.mean.tolist(),
        "std": train_ds.std.tolist(),
    }

    best_val_f1 = 0.0
    best_ckpt = str(out_dir / "dma_best.pth")

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        model.train()
        epoch_loss = 0.0

        for features, labels, motion_cost, app_cost in train_loader:
            features    = features.to(device)
            labels      = labels.to(device)
            motion_cost = motion_cost.float().to(device)
            app_cost    = app_cost.float().to(device)

            weights = model(features)
            loss = loss_fn(weights, motion_cost, app_cost, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(
                model, val_loader, loss_fn, device
            )
            train_loss_avg = epoch_loss / len(train_loader)
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch:3d}/{args.epochs}  "
                f"loss={train_loss_avg:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"acc={val_acc:.4f}  f1={val_f1:.4f}  "
                f"prec={val_prec:.4f}  rec={val_rec:.4f}  "
                f"lr={lr_now:.2e}"
            )
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                model.save(best_ckpt, stats=stats)
                print(f"  ✓ Saved best checkpoint (val_f1={val_f1:.4f})")

    last_ckpt = str(out_dir / "dma_last.pth")
    model.save(last_ckpt, stats=stats)

    with open(out_dir / "normalization_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nTraining complete. Best val_f1: {best_val_f1:.4f}")
    print(f"Best checkpoint: {best_ckpt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", choices=["bce", "ranking"], default="bce")
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--pos-neg-ratio", type=float, default=5.0,
                        help="Downsample negatives to pos:neg = 1:ratio")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
