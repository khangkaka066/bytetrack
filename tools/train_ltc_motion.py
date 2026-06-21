import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(FILE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.train_xlstm_motion import (
    build_lr_scheduler,
    build_samples_for_track,
    denormalize_target,
    evaluate,
    filter_hard_samples,
    filter_tracks_by_frame,
    get_target_stats,
    load_sequences,
    motion_loss,
    normalize_target,
    samples_to_tensors,
    unwrap_model,
)
from yolox.tracker.ltc_motion import LtcMotionResidual


def make_parser():
    parser = argparse.ArgumentParser("Train LTC/CfC motion residual model")
    parser.add_argument("--data-root", type=str, default="datasets/mot/train")
    parser.add_argument("--sequence-glob", type=str, default="*", help="sequence folder glob under data root")
    parser.add_argument("--output", type=str, default="outputs/ltc_motion.pth")
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-parallel", action="store_true", help="use all visible CUDA GPUs with DataParallel")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10, help="early-stop epochs without validation improvement")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="minimum validation loss improvement")
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_mae",
        choices=("val_loss", "val_nll", "val_res", "val_mae"),
        help="metric used for best checkpoint and early stopping",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        choices=("none", "cosine"),
        help="learning-rate schedule; cosine uses epoch-level warmup and annealing",
    )
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--residual-loss-weight", type=float, default=0.5)
    parser.add_argument("--residual-loss-beta", type=float, default=1.0)
    parser.add_argument("--log-var-reg-weight", type=float, default=1e-4)
    parser.add_argument("--hard-sample-min-error", type=float, default=0.0)
    parser.add_argument(
        "--target-normalization",
        type=str,
        default="standard",
        choices=("none", "standard"),
    )
    parser.add_argument("--target-std-eps", type=float, default=1e-6)
    parser.add_argument(
        "--split-mode",
        type=str,
        default="half",
        choices=("sequence", "half"),
        help="sequence: hold out sequences; half: train first half and validate second half",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--class-id", type=int, default=1, help="MOT pedestrian class id")
    parser.add_argument("--min-vis", type=float, default=0.0)
    return parser


def save_checkpoint(
    model,
    args,
    output,
    train_sequences,
    val_sequences,
    epoch,
    best_score,
    best_val_loss,
    best_val_res,
    best_val_mae,
    target_mean,
    target_std,
):
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "history_len": args.history_len,
            "input_dim": args.input_dim,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "epoch": epoch,
            "monitor": args.monitor,
            "best_score": best_score,
            "best_val_loss": best_val_loss,
            "best_val_res": best_val_res,
            "best_val_mae": best_val_mae,
            "residual_loss_weight": args.residual_loss_weight,
            "residual_loss_beta": args.residual_loss_beta,
            "log_var_reg_weight": args.log_var_reg_weight,
            "lr_scheduler": args.lr_scheduler,
            "warmup_epochs": args.warmup_epochs,
            "min_lr": args.min_lr,
            "hard_sample_min_error": args.hard_sample_min_error,
            "target_normalization": args.target_normalization,
            "target_mean": target_mean.cpu().tolist(),
            "target_std": target_std.cpu().tolist(),
            "split_mode": args.split_mode,
            "train_sequences": [name for name, _, _ in train_sequences],
            "val_sequences": [name for name, _, _ in val_sequences],
        },
        output,
    )


def build_split_samples(split_sequences, history_len):
    split_samples = []
    for sequence_name, tracks, _ in split_sequences:
        sequence_samples = []
        for rows in tracks.values():
            sequence_samples.extend(build_samples_for_track(rows, history_len))
        print("{}: {} samples".format(sequence_name, len(sequence_samples)))
        split_samples.extend(sequence_samples)
    return split_samples


def main():
    args = make_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sequences = load_sequences(args.data_root, args.sequence_glob, args.class_id, args.min_vis)

    if args.split_mode == "sequence":
        random.Random(args.seed).shuffle(sequences)
        val_count = int(round(len(sequences) * args.val_ratio))
        if len(sequences) > 1:
            val_count = min(max(val_count, 1), len(sequences) - 1)
        else:
            val_count = 0
        val_sequences = sequences[:val_count]
        train_sequences = sequences[val_count:]
    else:
        train_sequences = []
        val_sequences = []
        for sequence_name, tracks, num_frames in sequences:
            midpoint = num_frames // 2
            train_sequences.append(
                ("{}:first_half".format(sequence_name), filter_tracks_by_frame(tracks, 1, midpoint), midpoint)
            )
            val_sequences.append(
                (
                    "{}:second_half".format(sequence_name),
                    filter_tracks_by_frame(tracks, midpoint + 1, num_frames),
                    num_frames - midpoint,
                )
            )

    print("Split mode:", args.split_mode)
    print("Train sequences:", ", ".join(name for name, _, _ in train_sequences))
    print("Val sequences:", ", ".join(name for name, _, _ in val_sequences) or "none")

    train_samples = build_split_samples(train_sequences, args.history_len)
    val_samples = build_split_samples(val_sequences, args.history_len) if val_sequences else []
    if args.hard_sample_min_error > 0.0:
        original_count = len(train_samples)
        train_samples = filter_hard_samples(train_samples, args.hard_sample_min_error)
        print(
            "Hard sample mining: kept {} / {} samples".format(
                len(train_samples),
                original_count,
            )
        )
    if not train_samples:
        raise RuntimeError("No training samples were created. Check history length and gt files.")

    train_history, train_target_raw = samples_to_tensors(train_samples)
    target_mean, target_std = get_target_stats(
        train_target_raw,
        args.target_normalization,
        args.target_std_eps,
    )
    train_target = normalize_target(train_target_raw, target_mean, target_std)
    print(
        "Target normalization: {} mean {} std {}".format(
            args.target_normalization,
            ["{:.6f}".format(value) for value in target_mean.tolist()],
            ["{:.6f}".format(value) for value in target_std.tolist()],
        )
    )

    train_loader = DataLoader(
        TensorDataset(train_history, train_target),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = None
    if val_samples:
        val_history, val_target_raw = samples_to_tensors(val_samples)
        val_target = normalize_target(val_target_raw, target_mean, target_std)
        val_loader = DataLoader(
            TensorDataset(val_history, val_target),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    model = LtcMotionResidual(
        input_dim=args.input_dim,
        history_len=args.history_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)

    if args.data_parallel:
        if device.type != "cuda":
            raise ValueError("--data-parallel requires a CUDA device")
        gpu_count = torch.cuda.device_count()
        if gpu_count >= 2:
            model = torch.nn.DataParallel(model)
            print("Using DataParallel on {} CUDA GPUs".format(gpu_count))
        else:
            print("--data-parallel requested, but only {} CUDA GPU is visible".format(gpu_count))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_lr_scheduler(optimizer, args)

    best_score = float("inf")
    best_val_loss = float("inf")
    best_val_res = float("inf")
    best_val_mae = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for history, target in train_loader:
            history = history.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            pred_residual, pred_log_var = model(history)
            loss, _, _ = motion_loss(
                pred_residual,
                pred_log_var,
                target,
                args.residual_loss_weight,
                args.residual_loss_beta,
                args.log_var_reg_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_size = history.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

        train_loss = total_loss / max(total_count, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        if val_loader is not None:
            val_loss, val_nll, val_residual_loss, val_mae, val_mae_per_dim = evaluate(
                model,
                val_loader,
                device,
                args.residual_loss_weight,
                args.residual_loss_beta,
                args.log_var_reg_weight,
                target_mean,
                target_std,
            )
            if args.monitor == "val_mae":
                monitor_score = val_mae
            elif args.monitor == "val_res":
                monitor_score = val_residual_loss
            elif args.monitor == "val_nll":
                monitor_score = val_nll
            else:
                monitor_score = val_loss
            mae_dim_text = " ".join(
                "{}:{:.6f}".format(name, value)
                for name, value in zip(("cx", "cy", "a", "h"), val_mae_per_dim)
            )
            print(
                (
                    "epoch {:03d} lr {:.8f} train_loss {:.6f} val_loss {:.6f} "
                    "val_nll {:.6f} val_res {:.6f} val_mae {:.6f} "
                    "mae_dim [{}] monitor {} {:.6f}"
                ).format(
                    epoch,
                    current_lr,
                    train_loss,
                    val_loss,
                    val_nll,
                    val_residual_loss,
                    val_mae,
                    mae_dim_text,
                    args.monitor,
                    monitor_score,
                )
            )
            if monitor_score < best_score - args.min_delta:
                best_score = monitor_score
                best_val_loss = val_loss
                best_val_res = val_residual_loss
                best_val_mae = val_mae
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu()
                    for key, value in unwrap_model(model).state_dict().items()
                }
                epochs_without_improvement = 0
                save_checkpoint(
                    model,
                    args,
                    args.output,
                    train_sequences,
                    val_sequences,
                    epoch,
                    best_score,
                    best_val_loss,
                    best_val_res,
                    best_val_mae,
                    target_mean,
                    target_std,
                )
                print("saved best checkpoint to {} at epoch {}".format(args.output, epoch))
            else:
                epochs_without_improvement += 1
                if args.patience > 0 and epochs_without_improvement >= args.patience:
                    print(
                        "early stopping after {} epochs without validation improvement".format(
                            epochs_without_improvement
                        )
                    )
                    break
        else:
            print("epoch {:03d} lr {:.8f} train_loss {:.6f}".format(epoch, current_lr, train_loss))

        if scheduler is not None:
            scheduler.step()

    if best_state is not None:
        unwrap_model(model).load_state_dict(best_state)

    save_checkpoint(
        model,
        args,
        args.output,
        train_sequences,
        val_sequences,
        best_epoch if best_epoch > 0 else epoch,
        best_score,
        best_val_loss,
        best_val_res,
        best_val_mae,
        target_mean,
        target_std,
    )
    print("Saved checkpoint to {}".format(args.output))


if __name__ == "__main__":
    main()
