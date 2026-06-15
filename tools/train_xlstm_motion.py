import argparse
import glob
import os
import sys
import random
from collections import defaultdict, deque

import numpy as np
import torch
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
        sequences.append((sequence_name, tracks))
    return sequences


def samples_to_tensors(samples):
    histories = np.stack([sample[0] for sample in samples]).astype(np.float32)
    targets = np.stack([sample[1] for sample in samples]).astype(np.float32)
    return torch.from_numpy(histories), torch.from_numpy(targets)


def residual_nll_loss(pred_residual, pred_log_var, target_residual):
    pred_log_var = torch.clamp(pred_log_var, min=-10.0, max=10.0)
    diff = target_residual - pred_residual
    loss = 0.5 * torch.exp(-pred_log_var) * diff.pow(2) + 0.5 * pred_log_var
    return loss.mean() + 1e-4 * pred_log_var.pow(2).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0
    for history, target in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        pred_residual, pred_log_var = model(history)
        loss = residual_nll_loss(pred_residual, pred_log_var, target)
        mae = torch.abs(pred_residual - target).mean()
        batch_size = history.size(0)
        total_loss += loss.item() * batch_size
        total_mae += mae.item() * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1), total_mae / max(total_count, 1)


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def save_checkpoint(model, args, output, train_sequences, val_sequences, epoch, best_val_loss):
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
            "best_val_loss": best_val_loss,
            "train_sequences": [name for name, _ in train_sequences],
            "val_sequences": [name for name, _ in val_sequences],
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
    random.Random(args.seed).shuffle(sequences)

    val_count = int(round(len(sequences) * args.val_ratio))
    if len(sequences) > 1:
        val_count = min(max(val_count, 1), len(sequences) - 1)
    else:
        val_count = 0

    val_sequences = sequences[:val_count]
    train_sequences = sequences[val_count:]

    def build_split(split_sequences):
        split_samples = []
        for sequence_name, tracks in split_sequences:
            sequence_samples = []
            for rows in tracks.values():
                sequence_samples.extend(build_samples_for_track(rows, args.history_len))
            print("{}: {} samples".format(sequence_name, len(sequence_samples)))
            split_samples.extend(sequence_samples)
        return split_samples

    print("Train sequences:", ", ".join(name for name, _ in train_sequences))
    print("Val sequences:", ", ".join(name for name, _ in val_sequences) or "none")
    train_samples = build_split(train_sequences)
    val_samples = build_split(val_sequences) if val_sequences else []
    if not train_samples:
        raise RuntimeError("No training samples were created. Check history length and gt files.")

    train_history, train_target = samples_to_tensors(train_samples)
    train_loader = DataLoader(
        TensorDataset(train_history, train_target),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = None
    if val_samples:
        val_history, val_target = samples_to_tensors(val_samples)
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

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for history, target in train_loader:
            history = history.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            pred_residual, pred_log_var = model(history)
            loss = residual_nll_loss(pred_residual, pred_log_var, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_size = history.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

        train_loss = total_loss / max(total_count, 1)
        if val_loader is not None:
            val_loss, val_mae = evaluate(model, val_loader, device)
            print(
                "epoch {:03d} train_loss {:.6f} val_loss {:.6f} val_mae {:.6f}".format(
                    epoch, train_loss, val_loss, val_mae
                )
            )
            if val_loss < best_val_loss - args.min_delta:
                best_val_loss = val_loss
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
                    best_val_loss,
                )
                print("saved best checkpoint to {}".format(args.output))
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
        epoch,
        best_val_loss,
    )
    print("Saved checkpoint to {}".format(args.output))


if __name__ == "__main__":
    main()
