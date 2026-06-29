"""
Feature extraction for a single (track, detection) candidate pair.

Feature vector layout (FEAT_DIM = 15):
  idx  name                 source
  ---  -------------------  -------
  0    motion_iou           Kalman predicted bbox  ∩  detection bbox
  1    motion_cost          1 - iou
  2    mahalanobis_norm     Mahalanobis distance normalised by chi2inv95[4]
  3    cov_trace_log        log1p(trace of 4×4 position covariance)
  4    cov_mean_log         log1p(mean diagonal of position covariance)
  5    vel_magnitude        tanh-normalised speed from Kalman velocity state
  6    time_since_update    frames since last match, clipped to [0,1]
  7    cosine_dist          cosine dist between track smooth_feat & det feat
  8    feat_variance        mean variance of track's embedding history
  9    det_score            raw detector confidence
  10   bbox_area_log        log1p(w*h) / 15  (approx normalised for HD video)
  11   bbox_aspect          w / h  (clipped to [0.1, 10])
  12   track_age_norm       (current_frame - start_frame) / 300
  13   tracklet_len_norm    consecutive matches / 30
  14   has_appearance       1 if both track and detection have valid embeddings
"""

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance

FEAT_DIM = 15

# chi2inv95 for 4 degrees of freedom (used for Mahalanobis normalisation)
_CHI2_4DOF = 9.4877
_MAX_AGE = 300      # frames, for track age normalisation
_MAX_LEN = 30       # frames, for tracklet length normalisation
_VEL_SCALE = 10.0   # pixels/frame, for velocity normalisation


def _iou(tlbr_a: np.ndarray, tlbr_b: np.ndarray) -> float:
    ix1 = max(tlbr_a[0], tlbr_b[0])
    iy1 = max(tlbr_a[1], tlbr_b[1])
    ix2 = min(tlbr_a[2], tlbr_b[2])
    iy2 = min(tlbr_a[3], tlbr_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a_area = max(1e-6, (tlbr_a[2] - tlbr_a[0]) * (tlbr_a[3] - tlbr_a[1]))
    b_area = max(1e-6, (tlbr_b[2] - tlbr_b[0]) * (tlbr_b[3] - tlbr_b[1]))
    union = a_area + b_area - inter
    return inter / max(union, 1e-6)


def extract_pair_features(
    track,
    detection,
    kf,
    current_frame_id: int,
) -> np.ndarray:
    """
    Extract a fixed-size feature vector for one (track, detection) pair.

    Args:
        track:            STrack (active or lost, has .mean / .covariance)
        detection:        STrack (new detection, .mean may be None)
        kf:               KalmanFilter instance (used for Mahalanobis distance)
        current_frame_id: int

    Returns:
        np.ndarray of shape (FEAT_DIM,) with dtype float32
    """
    # ── Motion ──────────────────────────────────────────────────────────────
    t_tlbr = track.tlbr           # Kalman-predicted
    d_tlbr = detection.tlbr       # raw detection

    iou = _iou(t_tlbr, d_tlbr)
    motion_cost = 1.0 - iou

    try:
        det_xyah = detection.to_xyah().reshape(1, -1)
        maha = kf.gating_distance(
            track.mean, track.covariance, det_xyah, metric="maha"
        )[0]
        maha_norm = float(np.clip(maha / _CHI2_4DOF, 0.0, 1.0))
    except Exception:
        maha_norm = 1.0

    pos_cov = track.covariance[:4, :4]
    cov_diag = np.diag(pos_cov)
    cov_trace_log = float(np.log1p(np.trace(pos_cov)))
    cov_mean_log = float(np.log1p(np.mean(cov_diag)))

    vx, vy = float(track.mean[4]), float(track.mean[5])
    vel_mag = float(np.tanh(np.sqrt(vx ** 2 + vy ** 2) / _VEL_SCALE))

    time_since = max(0, current_frame_id - track.frame_id)
    time_since_norm = float(min(time_since / _MAX_AGE, 1.0))

    # ── Appearance ──────────────────────────────────────────────────────────
    has_app = (
        track.smooth_feat is not None
        and detection.curr_feat is not None
    )

    if has_app:
        try:
            cosine_dist = float(np.clip(
                cosine_distance(track.smooth_feat, detection.curr_feat),
                0.0, 1.0
            ))
        except Exception:
            cosine_dist = 0.5
        if len(track.features) >= 2:
            feat_stack = np.stack(list(track.features))
            feat_var = float(np.mean(np.var(feat_stack, axis=0)))
        else:
            feat_var = 0.0
    else:
        cosine_dist = 0.5   # neutral — no appearance signal
        feat_var = 0.0

    # ── Detection ───────────────────────────────────────────────────────────
    det_score = float(detection.score)
    tlwh = detection.tlwh
    w = max(float(tlwh[2]), 1.0)
    h = max(float(tlwh[3]), 1.0)
    bbox_area_log = float(np.log1p(w * h) / 15.0)
    bbox_aspect = float(np.clip(w / h, 0.1, 10.0))

    # ── Track history ────────────────────────────────────────────────────────
    track_age = max(0, current_frame_id - track.start_frame)
    track_age_norm = float(min(track_age / _MAX_AGE, 1.0))
    tracklet_len_norm = float(min(track.tracklet_len / _MAX_LEN, 1.0))
    has_app_f = float(has_app)

    return np.array([
        iou,               # 0
        motion_cost,       # 1
        maha_norm,         # 2
        cov_trace_log,     # 3
        cov_mean_log,      # 4
        vel_mag,           # 5
        time_since_norm,   # 6
        cosine_dist,       # 7
        feat_var,          # 8
        det_score,         # 9
        bbox_area_log,     # 10
        bbox_aspect,       # 11
        track_age_norm,    # 12
        tracklet_len_norm, # 13
        has_app_f,         # 14
    ], dtype=np.float32)


def extract_batch_features(
    tracks: list,
    detections: list,
    kf,
    current_frame_id: int,
) -> np.ndarray:
    """
    Extract features for all (track, detection) pairs in a cost matrix.

    Returns:
        np.ndarray of shape (len(tracks), len(detections), FEAT_DIM)
    """
    n_t, n_d = len(tracks), len(detections)
    out = np.zeros((n_t, n_d, FEAT_DIM), dtype=np.float32)
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            out[i, j] = extract_pair_features(t, d, kf, current_frame_id)
    return out
