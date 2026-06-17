"""
BYTETrackerSLSTM — ByteTrack with an sLSTM (xLSTM) trajectory corrector.

Pipeline per frame
──────────────────
1. Detector produces detections D_t.
2. All active tracks: Kalman predict → sLSTM batch-predict next bbox → blend.
3. Cost matrix: IoU on sLSTM-corrected positions + Mahalanobis gating.
4. Two-stage Hungarian matching (high-conf then low-conf) — same as ByteTrack.
5. Matched:   KF update + append real detection tokens to track buffer.
6. Unmatched: KF stays + append sLSTM-predicted tokens to track buffer.
7. New detections: spawn STrackSLSTM + append first observation tokens.

The original BYTETracker code is NOT modified; this is a standalone class.
"""

import os

import numpy as np
import torch

from yolox.tracker.kalman_filter import KalmanFilter
from yolox.tracker.basetrack import TrackState

from .strack_slstm import STrackSLSTM
from .slstm_predictor import SLSTMPredictor
from .matching import (
    linear_assignment,
    iou_distance,
    gate_cost_matrix,
    fuse_score,
)

iou_distance_lstm = iou_distance


def _joint(a: list, b: list) -> list:
    seen = {}
    out = []
    for t in a + b:
        if t.track_id not in seen:
            seen[t.track_id] = 1
            out.append(t)
    return out


def _sub(a: list, b: list) -> list:
    ids = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids]


def _remove_duplicates(
    a: list[STrackSLSTM], b: list[STrackSLSTM]
) -> tuple[list, list]:
    if not a or not b:
        return a, b
    atlbrs = np.array([t.pred_tlbr for t in a])
    btlbrs = np.array([t.pred_tlbr for t in b])
    from cython_bbox import bbox_overlaps as bbox_ious
    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64),
    )
    dup_a, dup_b = set(), set()
    for i in range(len(a)):
        for j in range(len(b)):
            if ious[i, j] > 0.15:
                if a[i].tracklet_len >= b[j].tracklet_len:
                    dup_b.add(j)
                else:
                    dup_a.add(i)
    return (
        [t for i, t in enumerate(a) if i not in dup_a],
        [t for j, t in enumerate(b) if j not in dup_b],
    )


class BYTETrackerSLSTM:
    """
    Parameters
    ----------
    args             : namespace with track_thresh, track_buffer, match_thresh, mot20.
    frame_rate       : video frame rate (used to compute max_time_lost).
    slstm_ckpt       : path to sLSTM .pt checkpoint from Kaggle training.
    vocab_size       : token vocabulary size (must match training, default 256).
    context_length   : token context window length (must match training, default 256).
    alpha0           : max sLSTM blend weight (0.0 = pure Kalman, 1.0 = pure sLSTM).
    beta             : decay rate for blend weight when track is missing.
    device           : torch device string; defaults to CUDA if available.
    """

    def __init__(
        self,
        args,
        frame_rate: int = 30,
        slstm_ckpt: str | None = None,
        vocab_size: int = 256,
        context_length: int = 256,
        alpha0: float = 0.5,
        beta: float = 0.3,
        device: str | None = None,
    ):
        self.tracked: list[STrackSLSTM] = []
        self.lost: list[STrackSLSTM] = []
        self.removed: list[STrackSLSTM] = []

        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh + 0.1
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = KalmanFilter()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.alpha0 = alpha0
        self.beta = beta

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # sLSTM predictor (optional — falls back to pure Kalman if not provided)
        self.slstm: SLSTMPredictor | None = None
        if slstm_ckpt and os.path.isfile(slstm_ckpt):
            self.slstm = SLSTMPredictor(
                ckpt_path=slstm_ckpt,
                vocab_size=vocab_size,
                context_length=context_length,
                device=str(self.device),
            )
        else:
            print("[BYTETrackerSLSTM] No valid slstm_ckpt — running pure Kalman fallback.")

        # Image dims updated each frame (for token encoding/decoding)
        self._img_w = 1920.0
        self._img_h = 1080.0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        output_results,
        img_info: tuple,
        img_size: tuple,
    ) -> list[STrackSLSTM]:
        """
        Process one frame.

        Parameters
        ----------
        output_results : torch.Tensor or ndarray [N, 5] (x1,y1,x2,y2,score)
                         or [N, 6] with class score column.
        img_info : (orig_h, orig_w)
        img_size : (model_h, model_w)
        """
        self.frame_id += 1
        activated, refound, lost_new, removed_new = [], [], [], []

        # ── Parse detections ─────────────────────────────────────────────
        if isinstance(output_results, torch.Tensor):
            output_results = output_results.cpu().numpy()
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]

        img_h, img_w = img_info[0], img_info[1]
        self._img_h, self._img_w = float(img_h), float(img_w)
        scale = min(img_size[0] / img_h, img_size[1] / img_w)
        bboxes /= scale

        high_mask = scores > self.args.track_thresh
        low_mask = (scores > 0.1) & ~high_mask

        dets_high = [
            STrackSLSTM(STrackSLSTM.tlbr_to_tlwh(b), s, self.context_length)
            for b, s in zip(bboxes[high_mask], scores[high_mask])
        ]
        dets_low = [
            STrackSLSTM(STrackSLSTM.tlbr_to_tlwh(b), s, self.context_length)
            for b, s in zip(bboxes[low_mask], scores[low_mask])
        ]

        # ── Partition active tracks ───────────────────────────────────────
        unconfirmed: list[STrackSLSTM] = []
        tracked: list[STrackSLSTM] = []
        for t in self.tracked:
            (tracked if t.is_activated else unconfirmed).append(t)

        # ── Kalman predict → sLSTM batch correct ─────────────────────────
        pool = _joint(tracked, self.lost)
        STrackSLSTM.multi_predict(pool)
        self._slstm_batch_predict(pool)

        # ── Stage 1: high-conf dets ↔ all active+lost tracks ─────────────
        cost1 = iou_distance_lstm(pool, dets_high)
        cost1 = gate_cost_matrix(self.kalman_filter, cost1, pool, dets_high)
        if not self.args.mot20:
            cost1 = fuse_score(cost1, dets_high)
        matches1, u_track, u_det = linear_assignment(cost1, thresh=self.args.match_thresh)

        for ti, di in matches1:
            track, det = pool[ti], dets_high[di]
            det_cxcywh = STrackSLSTM._tlwh_to_cxcywh(det.tlwh)
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refound.append(track)
            track.apply_slstm_matched(det_cxcywh, det.score)
            track.append_tokens(self._make_tokens(det_cxcywh))

        # ── Stage 2: low-conf dets ↔ remaining tracked tracks ────────────
        r_tracked = [pool[i] for i in u_track if pool[i].state == TrackState.Tracked]
        cost2 = iou_distance_lstm(r_tracked, dets_low)
        matches2, u_track2, _ = linear_assignment(cost2, thresh=0.5)

        for ti, di in matches2:
            track, det = r_tracked[ti], dets_low[di]
            det_cxcywh = STrackSLSTM._tlwh_to_cxcywh(det.tlwh)
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refound.append(track)
            track.apply_slstm_matched(det_cxcywh, det.score)
            track.append_tokens(self._make_tokens(det_cxcywh))

        # Tracks still unmatched after stage 2 → mark lost
        unmatched_after_s2 = {r_tracked[i].track_id for i in u_track2}
        for i in u_track:
            t = pool[i]
            if t.state != TrackState.Lost:
                if t.track_id not in unmatched_after_s2:
                    t.mark_lost()
                    lost_new.append(t)
            t.apply_slstm_missing()

        for i in u_track2:
            t = r_tracked[i]
            if t.state != TrackState.Lost:
                t.mark_lost()
                lost_new.append(t)
            t.apply_slstm_missing()

        # ── Unconfirmed tracks ↔ remaining high-conf dets ─────────────────
        rem_dets = [dets_high[i] for i in u_det]
        cost3 = iou_distance_lstm(unconfirmed, rem_dets)
        if not self.args.mot20:
            cost3 = fuse_score(cost3, rem_dets)
        matches3, u_unc, u_det2 = linear_assignment(cost3, thresh=0.7)

        for ti, di in matches3:
            det = rem_dets[di]
            det_cxcywh = STrackSLSTM._tlwh_to_cxcywh(det.tlwh)
            unconfirmed[ti].update(det, self.frame_id)
            unconfirmed[ti].apply_slstm_matched(det_cxcywh, det.score)
            unconfirmed[ti].append_tokens(self._make_tokens(det_cxcywh))
            activated.append(unconfirmed[ti])

        for i in u_unc:
            unconfirmed[i].mark_removed()
            removed_new.append(unconfirmed[i])

        # ── New tracks from unmatched high-conf dets ──────────────────────
        for i in u_det2:
            det = rem_dets[i]
            if det.score < self.det_thresh:
                continue
            det.activate(self.kalman_filter, self.frame_id)
            det_cxcywh = STrackSLSTM._tlwh_to_cxcywh(det.tlwh)
            det.append_tokens(self._make_tokens(det_cxcywh))
            activated.append(det)

        # ── Expire lost tracks ────────────────────────────────────────────
        for t in self.lost:
            if self.frame_id - t.end_frame > self.max_time_lost:
                t.mark_removed()
                removed_new.append(t)

        # ── State bookkeeping ─────────────────────────────────────────────
        self.tracked = [t for t in self.tracked if t.state == TrackState.Tracked]
        self.tracked = _joint(self.tracked, activated)
        self.tracked = _joint(self.tracked, refound)
        self.lost = _sub(self.lost, self.tracked)
        self.lost.extend(lost_new)
        self.lost = _sub(self.lost, self.removed)
        self.removed.extend(removed_new)
        self.tracked, self.lost = _remove_duplicates(self.tracked, self.lost)

        return [t for t in self.tracked if t.is_activated]

    # ── sLSTM helpers ─────────────────────────────────────────────────────────

    def _make_tokens(self, cxcywh: np.ndarray) -> np.ndarray:
        """Quantize a cxcywh bbox to 4 tokens. Returns zeros if no predictor."""
        if self.slstm is None:
            return np.zeros(4, dtype=np.int64)
        return self.slstm.bbox_to_tokens(
            *cxcywh, self._img_w, self._img_h
        )

    def _slstm_batch_predict(self, tracks: list[STrackSLSTM]):
        """
        Run sLSTM on all tracks and update pred_bbox_cxcywh with blended result.

        Tracks with fewer than TOKENS_PER_FRAME tokens get pure Kalman prediction.
        """
        if not tracks:
            return

        # Pure Kalman when no predictor is loaded
        if self.slstm is None:
            for t in tracks:
                t.pred_bbox_cxcywh = t.get_kf_bbox_cxcywh()
            return

        min_tokens = SLSTMPredictor.TOKENS_PER_FRAME
        valid_idx = [i for i, t in enumerate(tracks) if len(t.token_buffer) >= min_tokens]

        # Fall back to Kalman for tracks with insufficient history
        for i in range(len(tracks)):
            if i not in set(valid_idx):
                tracks[i].pred_bbox_cxcywh = tracks[i].get_kf_bbox_cxcywh()

        if not valid_idx:
            return

        valid_tracks = [tracks[i] for i in valid_idx]
        slstm_bboxes, pred_token_list = self.slstm.predict_batch(
            [t.token_buffer for t in valid_tracks],
            self._img_w,
            self._img_h,
        )

        for k, i in enumerate(valid_idx):
            track = tracks[i]
            # Decay sLSTM influence when track has been missing for many frames
            alpha = self.alpha0 * np.exp(-self.beta * track.missing_count)
            kf_bbox = track.get_kf_bbox_cxcywh()
            corrected = kf_bbox + alpha * (slstm_bboxes[k] - kf_bbox)
            corrected = np.clip(
                corrected,
                [0, 0, 1, 1],
                [self._img_w, self._img_h, self._img_w, self._img_h],
            )
            track.pred_bbox_cxcywh = corrected.astype(np.float32)
            track._pred_tokens = pred_token_list[k]
