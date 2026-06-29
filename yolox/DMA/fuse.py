"""
DMAFusion — drop-in replacement for the fixed-weight fusion in BYTETracker.

How to integrate into byte_tracker.py:

    from yolox.DMA import DMAFusion

    class BYTETracker:
        def __init__(self, args, frame_rate=30):
            ...
            self.dma = None
            if getattr(args, "with_dma", False):
                self.dma = DMAFusion.from_checkpoint(args.dma_weights)

        def _fuse_reid(self, iou_dists, tracks, detections):
            if self.dma is not None:
                emb_dists = matching.embedding_distance(tracks, detections)
                emb_dists[emb_dists > self.reid_thresh] = 1.0
                return self.dma.fuse(
                    tracks, detections,
                    iou_dists, emb_dists,
                    self.kalman_filter, self.frame_id,
                )
            # original fixed-weight fallback
            ...
"""

import numpy as np
import torch

from .model import DynamicWeightNet
from .features import extract_batch_features, FEAT_DIM


class DMAFusion:
    """
    Wraps DynamicWeightNet for inference inside the tracker's association step.
    """

    def __init__(
        self,
        model: DynamicWeightNet,
        stats: dict,
        device: str = "cpu",
        fallback_w_motion: float = 0.65,
    ):
        """
        Args:
            model:            trained DynamicWeightNet
            stats:            dict with 'mean' and 'std' lists (from training)
            device:           'cuda' or 'cpu'
            fallback_w_motion: weight used when model cannot run (empty matrix)
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.fallback_w_motion = fallback_w_motion

        self.mean = np.array(stats["mean"], dtype=np.float32)
        self.std = np.array(stats["std"], dtype=np.float32)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: str = "cpu") -> "DMAFusion":
        model, stats = DynamicWeightNet.load(ckpt_path)
        if stats is None:
            raise ValueError(
                f"Checkpoint {ckpt_path} has no normalisation stats. "
                "Re-train or pass stats manually."
            )
        return cls(model, stats, device=device)

    def fuse(
        self,
        tracks: list,
        detections: list,
        motion_cost: np.ndarray,
        appearance_cost: np.ndarray,
        kf,
        frame_id: int,
    ) -> np.ndarray:
        """
        Compute per-pair adaptive fusion of motion and appearance costs.

        Args:
            tracks:          list[STrack]   — predicted tracks
            detections:      list[STrack]   — current detections
            motion_cost:     (n_tracks, n_dets)  IoU-based cost
            appearance_cost: (n_tracks, n_dets)  cosine distance
            kf:              KalmanFilter instance
            frame_id:        current frame number

        Returns:
            fused_cost: (n_tracks, n_dets)  np.ndarray
        """
        n_t, n_d = len(tracks), len(detections)
        if n_t == 0 or n_d == 0:
            return motion_cost

        # Check whether appearance is available for at least some pairs
        has_any_app = any(
            t.smooth_feat is not None for t in tracks
        ) and any(
            d.curr_feat is not None for d in detections
        )
        if not has_any_app:
            return motion_cost

        # Extract features for all pairs: (n_t, n_d, FEAT_DIM)
        feat_matrix = extract_batch_features(tracks, detections, kf, frame_id)

        # Normalise and flatten for batch inference
        flat = feat_matrix.reshape(-1, FEAT_DIM)
        flat = (flat - self.mean) / self.std
        flat = np.clip(flat, -10.0, 10.0)   # guard against extreme values

        # Run model
        weights = self.model.predict_numpy(flat)   # (n_t*n_d, 2)
        w_motion = weights[:, 0].reshape(n_t, n_d)
        w_reid = weights[:, 1].reshape(n_t, n_d)

        # Zero out reid weight where appearance is missing
        for i, trk in enumerate(tracks):
            if trk.smooth_feat is None:
                w_motion[i, :] = 1.0
                w_reid[i, :] = 0.0
        for j, det in enumerate(detections):
            if det.curr_feat is None:
                w_motion[:, j] = 1.0
                w_reid[:, j] = 0.0

        fused = w_motion * motion_cost + w_reid * appearance_cost
        return fused

    def fuse_scalar(
        self,
        track,
        detection,
        motion_cost_val: float,
        appearance_cost_val: float,
        kf,
        frame_id: int,
    ) -> float:
        """Convenience method for a single (track, detection) pair."""
        return float(
            self.fuse(
                [track], [detection],
                np.array([[motion_cost_val]], dtype=np.float32),
                np.array([[appearance_cost_val]], dtype=np.float32),
                kf, frame_id,
            )[0, 0]
        )

    def get_weights(
        self,
        tracks: list,
        detections: list,
        kf,
        frame_id: int,
    ) -> tuple:
        """
        Return (w_motion, w_reid) matrices for analysis / visualisation.
        """
        n_t, n_d = len(tracks), len(detections)
        if n_t == 0 or n_d == 0:
            return np.ones((n_t, n_d)), np.zeros((n_t, n_d))

        feat_matrix = extract_batch_features(tracks, detections, kf, frame_id)
        flat = feat_matrix.reshape(-1, FEAT_DIM)
        flat = (flat - self.mean) / self.std
        flat = np.clip(flat, -10.0, 10.0)

        weights = self.model.predict_numpy(flat)
        w_motion = weights[:, 0].reshape(n_t, n_d)
        w_reid = weights[:, 1].reshape(n_t, n_d)
        return w_motion, w_reid
