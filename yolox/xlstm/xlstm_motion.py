import numpy as np
import torch
import torch.nn as nn


class XlstmMotionResidual(nn.Module):
    """Predict Kalman-state residuals from a track motion history."""

    def __init__(
        self,
        input_dim=12,
        history_len=16,
        embedding_dim=128,
        num_blocks=4,
        num_heads=4,
        output_dim=4,
        backend="cuda",
    ):
        super().__init__()
        try:
            from xlstm import (
                FeedForwardConfig,
                mLSTMBlockConfig,
                mLSTMLayerConfig,
                sLSTMBlockConfig,
                sLSTMLayerConfig,
                xLSTMBlockStack,
                xLSTMBlockStackConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "xLSTM motion prediction requires NX-AI/xlstm. "
                "Install it with `pip install xlstm` before loading an xLSTM checkpoint."
            ) from exc

        self.input_proj = nn.Linear(input_dim, embedding_dim)
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=num_heads,
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=backend,
                    num_heads=num_heads,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=history_len,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            slstm_at=[],
        )
        self.backbone = xLSTMBlockStack(cfg)
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, output_dim * 2),
        )

    def forward(self, history):
        features = self.input_proj(history)
        features = self.backbone(features)
        output = self.head(features[:, -1])
        residual, log_var = output.chunk(2, dim=-1)
        return residual, log_var


class XlstmMotionPredictor:
    """Optional xLSTM residual corrector for ByteTrack Kalman predictions."""

    def __init__(self, args=None):
        checkpoint = getattr(args, "xlstm_motion_ckpt", None)
        self.enabled = checkpoint is not None
        self.history_len = int(getattr(args, "xlstm_history_len", 16))
        self.input_dim = int(getattr(args, "xlstm_input_dim", 12))
        self.min_history = int(getattr(args, "xlstm_min_history", self.history_len))
        self.covariance_scale = float(getattr(args, "xlstm_covariance_scale", 1.0))
        self.max_abs_residual = float(getattr(args, "xlstm_max_abs_residual", 256.0))

        if not self.enabled:
            self.model = None
            self.device = None
            return

        requested_device = getattr(args, "xlstm_device", None)
        if requested_device is None:
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)

        backend = getattr(args, "xlstm_backend", "cuda")
        embedding_dim = int(getattr(args, "xlstm_embedding_dim", 128))
        num_blocks = int(getattr(args, "xlstm_num_blocks", 4))
        num_heads = int(getattr(args, "xlstm_num_heads", 4))

        self.model = XlstmMotionResidual(
            input_dim=self.input_dim,
            history_len=self.history_len,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            backend=backend,
        ).to(self.device)

        checkpoint_data = torch.load(checkpoint, map_location=self.device)
        state_dict = checkpoint_data.get("model", checkpoint_data)
        target_mean = checkpoint_data.get("target_mean", None)
        target_std = checkpoint_data.get("target_std", None)
        if target_mean is not None and target_std is not None:
            self.target_mean = np.asarray(target_mean, dtype=np.float32)
            self.target_std = np.asarray(target_std, dtype=np.float32)
        else:
            self.target_mean = np.zeros(4, dtype=np.float32)
            self.target_std = np.ones(4, dtype=np.float32)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def refine(self, stracks, means, covariances):
        if not self.enabled or len(stracks) == 0:
            return means, covariances

        ready_indices = []
        histories = []
        for index, track in enumerate(stracks):
            history = getattr(track, "motion_history", None)
            if history is None or len(history) < self.min_history:
                continue
            ready_indices.append(index)
            histories.append(self._history_tensor(history))

        if len(ready_indices) == 0:
            return means, covariances

        history_tensor = torch.from_numpy(np.stack(histories)).to(self.device)
        with torch.no_grad():
            residual, log_var = self.model(history_tensor)

        residual = residual.cpu().numpy()
        log_var = log_var.cpu().numpy()
        residual = residual * self.target_std + self.target_mean
        residual = np.clip(residual, -self.max_abs_residual, self.max_abs_residual)
        variance = np.exp(np.clip(log_var, -10.0, 10.0)) * np.square(self.target_std)

        for batch_index, track_index in enumerate(ready_indices):
            means[track_index, :4] += residual[batch_index]
            means[track_index, 2] = max(means[track_index, 2], 1e-4)
            means[track_index, 3] = max(means[track_index, 3], 1.0)
            covariances[track_index, range(4), range(4)] += (
                variance[batch_index] * self.covariance_scale
            )

        return means, covariances

    def _history_tensor(self, history):
        values = np.asarray(list(history), dtype=np.float32)
        if values.shape[-1] != self.input_dim:
            raise ValueError(
                "xLSTM motion history has input_dim {}, expected {}".format(
                    values.shape[-1], self.input_dim
                )
            )
        values = values[-self.history_len:]
        if len(values) < self.history_len:
            pad = np.repeat(values[:1], self.history_len - len(values), axis=0)
            values = np.concatenate([pad, values], axis=0)
        return values
