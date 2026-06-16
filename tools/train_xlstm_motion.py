import argparse
import glob
import os
import sys
import random
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(FILE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from yolox.tracker.kalman_filter import KalmanFilter
from yolox.tracker.xlstm_motion import XlstmMotionResidual


def make_parser():
    parser = argparse.ArgumentParser("Train xLSTM motion residual model")
    parser.add_argument("--data-root", type=str, default="datasets/mot_frcnn/train")
    parser.add_argument("--sequence-glob", type=str, default="*", help="sequence folder glob under data root")
    parser.add_argument("--output", type=str, default="outputs/xlstm_motion.pth")
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=12)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--backend", type=str, default="cuda")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-parallel", action="store_true", help="use all visible CUDA GPUs with DataParallel")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20, help="early-stop epochs without validation improvement")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="minimum validation loss improvement")
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_loss",
        choices=("val_loss", "val_nll", "val_mae"),
        help="metric used for best checkpoint and early stopping",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--residual-loss-weight",
        type=float,
        default=0.0,
        help="extra SmoothL1 residual loss weight; use 0.5 or 1.0 to emphasize residual accuracy",
    )
    parser.add_argument("--residual-loss-beta", type=float, default=1.0, help="SmoothL1 beta for residual loss")
    parser.add_argument("--log-var-reg-weight", type=float, default=1e-4, help="L2 regularization weight for predicted log variance")
    parser.add_argument(
        "--target-normalization",
        type=str,
        default="none",
        choices=("none", "standard"),
        help="normalize target residuals before training; checkpoint stores stats for inference unnormalization",
    )
    parser.add_argument("--target-std-eps", type=float, default=1e-6)
    parser.add_argument(
        "--split-mode",
        type=str,
        default="sequence",
        choices=("sequence", "half"),
        help="sequence: hold out whole sequences; half: train first half and validate second half of every sequence",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--class-id", type=int, default=1, help="MOT pedestrian class id")
    parser.add_argument("--min-vis", type=float, default=0.0)
    return parser


def tlwh_to_xyah(tlwh):
    x, y, w, h = tlwh
    h = max(float(h), 1e-6)
    return np.asarray([x + w / 2.0, y + h / 2.0, w / h, h], dtype=np.float32)


def make_history_feature(mean, frame_delta, is_missing, missing_count, confidence):
    return np.asarray(
        [
            mean[0],
            mean[1],
            mean[2],
            mean[3],
            mean[4],
            mean[5],
            mean[6],
            mean[7],
            float(frame_delta),
            1.0 if is_missing else 0.0,
            float(missing_count),
            float(confidence),
        ],
        dtype=np.float32,
    )


def read_mot_gt(gt_path, class_id=1, min_vis=0.0):
    tracks = defaultdict(list)
    with open(gt_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue

            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x, y, w, h = [float(value) for value in parts[2:6]]
            mark = int(float(parts[6]))
            obj_class = int(float(parts[7])) if len(parts) > 7 else class_id
            visibility = float(parts[8]) if len(parts) > 8 else 1.0

            if mark != 1 or obj_class != class_id or visibility < min_vis:
                continue
            if w <= 0.0 or h <= 0.0:
                continue

            tracks[track_id].append(
                {
                    "frame": frame,
                    "tlwh": np.asarray([x, y, w, h], dtype=np.float32),
                    "score": 1.0,
                }
            )

    for rows in tracks.values():
        rows.sort(key=lambda item: item["frame"])
    return tracks


def count_sequence_frames(sequence_path):
    img_path = os.path.join(sequence_path, "img1")
    images = glob.glob(os.path.join(img_path, "*.jpg"))
    if images:
        return len(images)
    return 0


def filter_tracks_by_frame(tracks, start_frame, end_frame):
    filtered = {}
    for track_id, rows in tracks.items():
        selected = [
            row
            for row in rows
            if row["frame"] >= start_frame and row["frame"] <= end_frame
        ]
        if selected:
            filtered[track_id] = selected
    return filtered


def build_samples_for_track(track_rows, history_len):
    if len(track_rows) < history_len + 1:
        return []

    kf = KalmanFilter()
    history = deque(maxlen=history_len)
    samples = []

    first = track_rows[0]
    mean, covariance = kf.initiate(tlwh_to_xyah(first["tlwh"]))
    history.append(
        make_history_feature(
            mean=mean,
            frame_delta=1.0,
            is_missing=False,
            missing_count=0,
            confidence=first["score"],
        )
    )

    last_frame = first["frame"]
    missing_count = 0

    for row in track_rows[1:]:
        # Reproduce inference chronology for gaps: predict each missing frame,
        # append a missing history step, then predict once for the matched frame.
        while last_frame + 1 < row["frame"]:
            mean, covariance = kf.predict(mean.copy(), covariance.copy())
            missing_count += 1
            last_frame += 1
            history.append(
                make_history_feature(
                    mean=mean,
                    frame_delta=1.0,
                    is_missing=True,
                    missing_count=missing_count,
                    confidence=0.0,
                )
            )

        mean_pred, cov_pred = kf.predict(mean.copy(), covariance.copy())
        gt_xyah = tlwh_to_xyah(row["tlwh"])
        target_residual = gt_xyah - mean_pred[:4]

        if len(history) == history_len:
            samples.append((np.stack(history).astype(np.float32), target_residual.astype(np.float32)))

        mean, covariance = kf.update(mean_pred, cov_pred, gt_xyah)
        missing_count = 0
        history.append(
            make_history_feature(
                mean=mean,
                frame_delta=1.0,
                is_missing=False,
                missing_count=missing_count,
                confidence=row["score"],
            )
        )
        last_frame = row["frame"]

    return samples


def load_sequences(data_root, sequence_glob, class_id, min_vis):
    sequence_paths = sorted(
        path
        for path in glob.glob(os.path.join(data_root, sequence_glob))
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "gt", "gt.txt"))
    )
    if not sequence_paths:
        raise FileNotFoundError(
            "No MOT gt files found under {} with sequence glob {}".format(data_root, sequence_glob)
        )

    sequences = []
    for sequence_path in sequence_paths:
        sequence_name = os.path.basename(sequence_path.rstrip(os.sep))
        tracks = read_mot_gt(
            os.path.join(sequence_path, "gt", "gt.txt"),
            class_id=class_id,
            min_vis=min_vis,
        )
        num_frames = count_sequence_frames(sequence_path)
        if num_frames == 0:
            max_frame = max((row["frame"] for rows in tracks.values() for row in rows), default=0)
            num_frames = max_frame
        sequences.append((sequence_name, tracks, num_frames))
    return sequences


def samples_to_tensors(samples):
    histories = np.stack([sample[0] for sample in samples]).astype(np.float32)
    targets = np.stack([sample[1] for sample in samples]).astype(np.float32)
    return torch.from_numpy(histories), torch.from_numpy(targets)


def get_target_stats(target_tensor, mode, std_eps):
    if mode == "standard":
        target_mean = target_tensor.mean(dim=0)
        target_std = target_tensor.std(dim=0).clamp_min(std_eps)
    else:
        target_mean = torch.zeros(target_tensor.shape[1], dtype=target_tensor.dtype)
        target_std = torch.ones(target_tensor.shape[1], dtype=target_tensor.dtype)
    return target_mean, target_std


def normalize_target(target_tensor, target_mean, target_std):
    return (target_tensor - target_mean) / target_std


def denormalize_target(target_tensor, target_mean, target_std):
    return target_tensor * target_std.to(target_tensor.device) + target_mean.to(target_tensor.device)


def residual_nll_loss(pred_residual, pred_log_var, target_residual, log_var_reg_weight):
    pred_log_var = torch.clamp(pred_log_var, min=-10.0, max=10.0)
    diff = target_residual - pred_residual
    loss = 0.5 * torch.exp(-pred_log_var) * diff.pow(2) + 0.5 * pred_log_var
    return loss.mean() + log_var_reg_weight * pred_log_var.pow(2).mean()


def motion_loss(
    pred_residual,
    pred_log_var,
    target_residual,
    residual_loss_weight,
    residual_loss_beta,
    log_var_reg_weight,
):
    nll_loss = residual_nll_loss(
        pred_residual,
        pred_log_var,
        target_residual,
        log_var_reg_weight,
    )
    residual_loss = F.smooth_l1_loss(
        pred_residual,
        target_residual,
        beta=residual_loss_beta,
    )
    total_loss = nll_loss + residual_loss_weight * residual_loss
    return total_loss, nll_loss, residual_loss


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    residual_loss_weight,
    residual_loss_beta,
    log_var_reg_weight,
    target_mean,
    target_std,
):
    model.eval()
    total_loss = 0.0
    total_nll = 0.0
    total_residual_loss = 0.0
    total_mae = 0.0
    total_mae_per_dim = torch.zeros(4, dtype=torch.float64)
    total_count = 0
    for history, target in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        pred_residual, pred_log_var = model(history)
        loss, nll_loss, residual_loss = motion_loss(
            pred_residual,
            pred_log_var,
            target,
            residual_loss_weight,
            residual_loss_beta,
            log_var_reg_weight,
        )
        pred_residual_raw = denormalize_target(pred_residual, target_mean, target_std)
        target_raw = denormalize_target(target, target_mean, target_std)
        abs_error = torch.abs(pred_residual_raw - target_raw)
        mae = abs_error.mean()
        mae_per_dim = abs_error.mean(dim=0).detach().cpu().double()
        batch_size = history.size(0)
        total_loss += loss.item() * batch_size
        total_nll += nll_loss.item() * batch_size
        total_residual_loss += residual_loss.item() * batch_size
        total_mae += mae.item() * batch_size
        total_mae_per_dim += mae_per_dim * batch_size
        total_count += batch_size
    total_count = max(total_count, 1)
    return (
        total_loss / total_count,
        total_nll / total_count,
        total_residual_loss / total_count,
        total_mae / total_count,
        (total_mae_per_dim / total_count).tolist(),
    )


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def save_checkpoint(
    model,
    args,
    output,
    train_sequences,
    val_sequences,
    epoch,
    best_score,
    best_val_loss,
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
            "embedding_dim": args.embedding_dim,
            "num_blocks": args.num_blocks,
            "num_heads": args.num_heads,
            "backend": args.backend,
            "epoch": epoch,
            "monitor": args.monitor,
            "best_score": best_score,
            "best_val_loss": best_val_loss,
            "best_val_mae": best_val_mae,
            "residual_loss_weight": args.residual_loss_weight,
            "residual_loss_beta": args.residual_loss_beta,
            "log_var_reg_weight": args.log_var_reg_weight,
            "target_normalization": args.target_normalization,
            "target_mean": target_mean.cpu().tolist(),
            "target_std": target_std.cpu().tolist(),
            "split_mode": args.split_mode,
            "train_sequences": [name for name, _, _ in train_sequences],
            "val_sequences": [name for name, _, _ in val_sequences],
        },
        output,
    )


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
            train_tracks = filter_tracks_by_frame(tracks, 1, midpoint)
            val_tracks = filter_tracks_by_frame(tracks, midpoint + 1, num_frames)
            train_sequences.append(
                ("{}:first_half".format(sequence_name), train_tracks, midpoint)
            )
            val_sequences.append(
                ("{}:second_half".format(sequence_name), val_tracks, num_frames - midpoint)
            )

    def build_split(split_sequences):
        split_samples = []
        for sequence_name, tracks, _ in split_sequences:
            sequence_samples = []
            for rows in tracks.values():
                sequence_samples.extend(build_samples_for_track(rows, args.history_len))
            print("{}: {} samples".format(sequence_name, len(sequence_samples)))
            split_samples.extend(sequence_samples)
        return split_samples

    print("Split mode:", args.split_mode)
    print("Train sequences:", ", ".join(name for name, _, _ in train_sequences))
    print("Val sequences:", ", ".join(name for name, _, _ in val_sequences) or "none")
    train_samples = build_split(train_sequences)
    val_samples = build_split(val_sequences) if val_sequences else []
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

    model = XlstmMotionResidual(
        input_dim=args.input_dim,
        history_len=args.history_len,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        backend=args.backend,
    ).to(device)

    if args.data_parallel:
        if device.type != "cuda":
            raise ValueError("--data-parallel requires a CUDA device")
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            print("--data-parallel requested, but only {} CUDA GPU is visible".format(gpu_count))
        else:
            model = torch.nn.DataParallel(model)
            print("Using DataParallel on {} CUDA GPUs".format(gpu_count))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = float("inf")
    best_val_loss = float("inf")
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
                    "epoch {:03d} train_loss {:.6f} val_loss {:.6f} "
                    "val_nll {:.6f} val_res {:.6f} val_mae {:.6f} "
                    "mae_dim [{}] monitor {} {:.6f}"
                ).format(
                    epoch,
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
                    best_val_mae,
                    target_mean,
                    target_std,
                )
                print(
                    "saved best checkpoint to {} at epoch {} ({} {:.6f})".format(
                        args.output, epoch, args.monitor, best_score
                    )
                )
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
            print("epoch {:03d} train_loss {:.6f}".format(epoch, train_loss))

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
        best_val_mae,
        target_mean,
        target_std,
    )
    print("Saved checkpoint to {}".format(args.output))


if __name__ == "__main__":
    main()
