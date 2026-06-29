"""
Generate training data for DynamicWeightNet from MOT-format sequences.

Usage:
  python -m yolox.DMA.generate_data \\
    --seq-dirs  data/MOT17/train/MOT17-02 data/MOT17/train/MOT17-04 \\
    --out-dir   data/dma_train \\
    --reid-model osnet_x1_0 \\
    --reid-weights weights/osnet_x1_0.pth \\
    --det-file  det/det.txt          # relative to seq-dir; omit to use GT boxes

Each output .npz contains:
  features:         (N, FEAT_DIM)
  labels:           (N,)   1 = correct match, 0 = wrong match
  motion_costs:     (N,)
  appearance_costs: (N,)
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Allow running as a standalone script from repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from yolox.tracker.kalman_filter import KalmanFilter
from yolox.DMA.features import extract_pair_features, extract_batch_features, FEAT_DIM


# ─────────────────────────────────────────────────────────────────────────────
# Minimal track state — mirrors STrack but driven by GT associations
# ─────────────────────────────────────────────────────────────────────────────

class _GTTrack:
    """Lightweight track driven by ground-truth associations."""

    _kf = KalmanFilter()

    def __init__(self, gt_id: int, tlwh: np.ndarray, frame_id: int, feat=None):
        self.gt_id = gt_id
        self.track_id = gt_id          # alias so feature extractor works
        self.start_frame = frame_id
        self.frame_id = frame_id
        self.tracklet_len = 0

        self._tlwh = np.asarray(tlwh, dtype=float)
        xyah = self._tlwh_to_xyah(tlwh)
        self.mean, self.covariance = self._kf.initiate(xyah)

        # Appearance
        self.smooth_feat = None
        self.curr_feat = None
        from collections import deque
        self.features = deque([], maxlen=50)
        if feat is not None:
            self._update_feat(feat)

    # ── geometry helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh, dtype=float).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= max(ret[3], 1e-6)
        return ret

    @property
    def tlwh(self) -> np.ndarray:
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self) -> np.ndarray:
        return self._tlwh_to_xyah(self.tlwh)

    # score is used by feature extractor
    @property
    def score(self) -> float:
        return 1.0

    # ── Kalman ───────────────────────────────────────────────────────────────
    def predict(self):
        mean_state = self.mean.copy()
        self.mean, self.covariance = self._kf.predict(mean_state, self.covariance)

    def update_kalman(self, tlwh: np.ndarray):
        xyah = self._tlwh_to_xyah(tlwh)
        self.mean, self.covariance = self._kf.update(self.mean, self.covariance, xyah)

    # ── appearance ───────────────────────────────────────────────────────────
    def _update_feat(self, feat: np.ndarray, alpha: float = 0.9):
        feat = np.asarray(feat, dtype=np.float32)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        self.curr_feat = feat
        self.features.append(feat)
        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            self.smooth_feat = alpha * self.smooth_feat + (1 - alpha) * feat
            sn = np.linalg.norm(self.smooth_feat)
            if sn > 0:
                self.smooth_feat /= sn


class _GTDetection:
    """Minimal detection wrapper so feature extractor can treat it like STrack."""

    def __init__(self, tlwh: np.ndarray, score: float, gt_id: int, feat=None):
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.score = score
        self.gt_id = gt_id
        self.tracklet_len = 0
        self.start_frame = 0
        self.frame_id = 0
        self.mean = None          # not yet activated
        self.covariance = None
        from collections import deque
        self.features = deque([], maxlen=50)
        if feat is not None:
            feat = np.asarray(feat, dtype=np.float32)
            norm = np.linalg.norm(feat)
            self.curr_feat = feat / norm if norm > 0 else feat
        else:
            self.curr_feat = None
        self.smooth_feat = self.curr_feat

    @property
    def tlwh(self) -> np.ndarray:
        return self._tlwh.copy()

    @property
    def tlbr(self) -> np.ndarray:
        ret = self._tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self) -> np.ndarray:
        ret = self._tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= max(ret[3], 1e-6)
        return ret


# ─────────────────────────────────────────────────────────────────────────────
# GT / detection file readers
# ─────────────────────────────────────────────────────────────────────────────

def _load_gt(gt_file: str):
    """
    Returns dict: frame_id -> list of (gt_id, tlwh, conf, cls, vis).
    Filters to class=1 (pedestrian) and visible detections.
    """
    gt = defaultdict(list)
    with open(gt_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            fid = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            cls = int(float(parts[7])) if len(parts) > 7 else 1
            vis = float(parts[8]) if len(parts) > 8 else 1.0
            if cls != 1 or conf < 0.5 or vis < 0.1:
                continue
            gt[fid].append((tid, np.array([x, y, w, h], dtype=np.float32), conf))
    return gt


def _load_detections(det_file: str):
    """
    Returns dict: frame_id -> list of (tlwh, score).
    Standard MOT det.txt: frame, -1, x, y, w, h, score, -1, -1, -1
    """
    dets = defaultdict(list)
    with open(det_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            fid = int(parts[0])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            score = float(parts[6]) if len(parts) > 6 else 1.0
            if score < 0.3:
                continue
            dets[fid].append((np.array([x, y, w, h], dtype=np.float32), score))
    return dets


def _match_dets_to_gt(det_tlwhs, gt_entries, iou_thresh=0.5):
    """
    Greedy IoU matching of detections to GT boxes.
    Returns list of gt_id (or -1) per detection.
    """
    if not gt_entries or not det_tlwhs:
        return [-1] * len(det_tlwhs)

    gt_tlwhs = [e[1] for e in gt_entries]
    gt_ids = [e[0] for e in gt_entries]

    def tlwh_to_tlbr(tlwh):
        ret = tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    assigned = [-1] * len(det_tlwhs)
    used_gt = set()
    # compute IoU matrix
    n_d, n_g = len(det_tlwhs), len(gt_tlwhs)
    iou_mat = np.zeros((n_d, n_g), dtype=np.float32)
    for i, dtlwh in enumerate(det_tlwhs):
        da = tlwh_to_tlbr(dtlwh)
        for j, gtlwh in enumerate(gt_tlwhs):
            ga = tlwh_to_tlbr(gtlwh)
            ix1, iy1 = max(da[0], ga[0]), max(da[1], ga[1])
            ix2, iy2 = min(da[2], ga[2]), min(da[3], ga[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (
                (da[2]-da[0])*(da[3]-da[1])
                + (ga[2]-ga[0])*(ga[3]-ga[1])
                - inter
            )
            iou_mat[i, j] = inter / max(union, 1e-6)

    # greedy assignment
    pairs = sorted(
        [(i, j) for i in range(n_d) for j in range(n_g)],
        key=lambda ij: -iou_mat[ij[0], ij[1]]
    )
    assigned_d = set()
    for i, j in pairs:
        if iou_mat[i, j] < iou_thresh:
            break
        if i in assigned_d or j in used_gt:
            continue
        assigned[i] = gt_ids[j]
        assigned_d.add(i)
        used_gt.add(j)
    return assigned


# ─────────────────────────────────────────────────────────────────────────────
# ReID feature extractor (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _build_reid(args):
    if not args.reid_weights:
        return None
    try:
        import argparse as _ap
        fast_reid_config = getattr(args, "fast_reid_config", None)
        if fast_reid_config:
            reid_args = _ap.Namespace(
                reid_backend="fast",
                fast_reid_config=fast_reid_config,
                fast_reid_weights=args.reid_weights,
                reid_model_path=args.reid_weights,
                reid_device="cuda",
                fast_reid_batch_size=16,
            )
        else:
            reid_args = _ap.Namespace(
                reid_backend="deep",
                reid_model=args.reid_model,
                reid_model_path=args.reid_weights,
                reid_device="cuda",
            )
        from yolox.tracker.reid import build_reid_extractor
        extractor = build_reid_extractor(reid_args)
        print(f"[ReID] Loaded: {args.reid_weights}")
        return extractor
    except Exception as e:
        print(f"[WARN] ReID not available: {e}")
        return None


def _extract_feats(reid, frame_path: str, tlwhs):
    if reid is None or not tlwhs:
        return [None] * len(tlwhs)
    import cv2
    frame = cv2.imread(frame_path)
    if frame is None:
        return [None] * len(tlwhs)
    tlbrs = []
    for tlwh in tlwhs:
        x, y, w, h = tlwh
        tlbrs.append([x, y, x + w, y + h])
    return reid.extract(frame, tlbrs)


# ─────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ─────────────────────────────────────────────────────────────────────────────

def generate_sequence(seq_dir: str, out_path: str, args) -> int:
    seq_dir = Path(seq_dir)
    gt_file = seq_dir / "gt" / "gt.txt"
    img_dir = seq_dir / "img1"

    if not gt_file.exists():
        print(f"[SKIP] No GT file at {gt_file}")
        return 0

    gt = _load_gt(str(gt_file))
    all_frames = sorted(gt.keys())

    # Detections: either from det.txt or fall back to GT boxes
    det_file = seq_dir / "det" / "det.txt"
    if not args.use_gt_dets and det_file.exists():
        dets_by_frame = _load_detections(str(det_file))
        use_gt_as_det = False
    else:
        use_gt_as_det = True
        dets_by_frame = None

    reid = _build_reid(args)
    kf = KalmanFilter()

    # Active GT tracks: gt_id -> _GTTrack
    active_tracks: dict = {}

    all_features, all_labels, all_m_costs, all_a_costs = [], [], [], []

    for frame_id in all_frames:
        gt_this_frame = gt.get(frame_id, [])
        gt_id_to_tlwh = {e[0]: e[1] for e in gt_this_frame}

        # ── Build detections for this frame ──────────────────────────────
        # Try common naming conventions: 8-digit (DanceTrack), 6-digit (MOT17)
        for fmt, ext in [("%08d.jpg", ""), ("%06d.jpg", ""), ("%08d.png", ""), ("%06d.png", "")]:
            frame_path = str(img_dir / (fmt % frame_id))
            if os.path.exists(frame_path):
                break
        else:
            frame_path = ""

        if use_gt_as_det:
            det_tlwhs = [e[1] for e in gt_this_frame]
            det_scores = [e[2] for e in gt_this_frame]
            det_gt_ids = [e[0] for e in gt_this_frame]
        else:
            raw_dets = dets_by_frame.get(frame_id, [])
            det_tlwhs = [d[0] for d in raw_dets]
            det_scores = [d[1] for d in raw_dets]
            det_gt_ids = _match_dets_to_gt(det_tlwhs, gt_this_frame)

        # Extract ReID features for detections
        det_feats = _extract_feats(reid, frame_path, det_tlwhs)

        # Build detection objects
        detections = [
            _GTDetection(tlwh, score, gid, feat)
            for tlwh, score, gid, feat in zip(
                det_tlwhs, det_scores, det_gt_ids, det_feats
            )
        ]

        # ── Predict existing tracks ──────────────────────────────────────
        for trk in active_tracks.values():
            trk.predict()

        if len(active_tracks) == 0 or len(detections) == 0:
            # Update tracks with GT observations then continue
            for gt_id, tlwh in gt_id_to_tlwh.items():
                if gt_id in active_tracks:
                    t = active_tracks[gt_id]
                    t.update_kalman(tlwh)
                    t.frame_id = frame_id
                    t.tracklet_len += 1
                else:
                    feats_for_trk = [
                        det_feats[k]
                        for k, d in enumerate(detections)
                        if d.gt_id == gt_id
                    ]
                    f = feats_for_trk[0] if feats_for_trk else None
                    active_tracks[gt_id] = _GTTrack(gt_id, tlwh, frame_id, feat=f)
            continue

        # ── Generate (track, detection) pairs ───────────────────────────
        track_list = list(active_tracks.values())
        feat_matrix = extract_batch_features(track_list, detections, kf, frame_id)
        # feat_matrix: (n_tracks, n_dets, FEAT_DIM)

        for i, trk in enumerate(track_list):
            for j, det in enumerate(detections):
                label = float(trk.gt_id == det.gt_id and det.gt_id != -1)
                feat_vec = feat_matrix[i, j]
                all_features.append(feat_vec)
                all_labels.append(label)
                all_m_costs.append(feat_vec[1])   # motion_cost = features[1]
                all_a_costs.append(feat_vec[7])   # cosine_dist  = features[7]

        # ── Update active tracks with GT associations ─────────────────────
        seen_ids = set()
        for gt_id, tlwh in gt_id_to_tlwh.items():
            seen_ids.add(gt_id)
            # find matching detection feat
            det_feat = None
            for k, det in enumerate(detections):
                if det.gt_id == gt_id:
                    det_feat = det_feats[k]
                    break
            if gt_id in active_tracks:
                t = active_tracks[gt_id]
                t.update_kalman(tlwh)
                t.frame_id = frame_id
                t.tracklet_len += 1
                if det_feat is not None:
                    t._update_feat(det_feat)
            else:
                active_tracks[gt_id] = _GTTrack(gt_id, tlwh, frame_id, feat=det_feat)

        # Remove tracks not seen for > max_age frames
        to_remove = [
            gt_id for gt_id, trk in active_tracks.items()
            if frame_id - trk.frame_id > args.max_age
        ]
        for gt_id in to_remove:
            del active_tracks[gt_id]

    if not all_features:
        return 0

    np.savez_compressed(
        out_path,
        features=np.stack(all_features).astype(np.float32),
        labels=np.array(all_labels, dtype=np.float32),
        motion_costs=np.array(all_m_costs, dtype=np.float32),
        appearance_costs=np.array(all_a_costs, dtype=np.float32),
    )
    n = len(all_labels)
    pos = int(sum(all_labels))
    print(f"  {Path(seq_dir).name}: {n} pairs  pos={pos}  neg={n-pos}")
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-dirs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reid-model", default="osnet_x1_0")
    parser.add_argument("--reid-weights", default="")
    parser.add_argument("--fast-reid-config", default=None,
                        help="FastReID config .yml (use instead of --reid-model for fast_reid backend)")
    parser.add_argument("--use-gt-dets", action="store_true",
                        help="Use GT boxes as detections instead of det.txt")
    parser.add_argument("--max-age", type=int, default=30,
                        help="Max frames to keep a lost GT track alive")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Expand any directory that has no gt/gt.txt but contains sub-sequences
    seq_dirs = []
    for d in args.seq_dirs:
        d = Path(d)
        if (d / "gt" / "gt.txt").exists():
            seq_dirs.append(d)
        else:
            children = sorted(p for p in d.iterdir() if p.is_dir() and (p / "gt" / "gt.txt").exists())
            if children:
                seq_dirs.extend(children)
            else:
                print(f"[WARN] No sequences found in {d}")

    total = 0
    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        out_path = out_dir / f"{seq_name}.npz"
        print(f"Processing {seq_name} ...")
        n = generate_sequence(str(seq_dir), str(out_path), args)
        total += n

    print(f"\nDone. Total pairs: {total}")
    print(f"Files saved to: {out_dir}")


if __name__ == "__main__":
    main()
