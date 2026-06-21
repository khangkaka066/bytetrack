import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CfCCell(nn.Module):
    """Small closed-form continuous-time cell for motion histories."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.hidden_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.zeros(hidden_size))
        self.tau = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h, dt):
        candidate = torch.tanh(self.input_proj(x) + self.hidden_proj(h))
        dt = dt.clamp_min(1e-3).unsqueeze(-1)
        alpha = F.softplus(self.alpha).unsqueeze(0) + 1e-3
        tau = F.softplus(self.tau).unsqueeze(0) + 1e-3
        gate = torch.exp(-alpha * dt / tau).clamp(0.0, 1.0)
        return gate * h + (1.0 - gate) * candidate


class LtcMotionResidual(nn.Module):
    """Predict Kalman-state residuals with native frame-delta dynamics."""

    def __init__(
        self,
        input_dim=12,
        history_len=16,
        hidden_size=128,
        num_layers=2,
        output_dim=4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.history_len = history_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_norm = nn.LayerNorm(input_dim)
        self.cells = nn.ModuleList(
            [
                CfCCell(input_dim if layer_index == 0 else hidden_size, hidden_size)
                for layer_index in range(num_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_dim * 2),
        )

    def forward(self, history):
        # history[:, :, 8] is the actual frame delta collected by ByteTrack.
        dts = history[:, :, 8].clamp_min(1e-3)
        features = self.input_norm(history)
        batch_size = history.shape[0]
        states = [
            history.new_zeros(batch_size, self.hidden_size)
            for _ in range(self.num_layers)
        ]

        for step_index in range(history.shape[1]):
            layer_input = features[:, step_index]
            dt = dts[:, step_index]
            for layer_index, cell in enumerate(self.cells):
                states[layer_index] = cell(layer_input, states[layer_index], dt)
                layer_input = states[layer_index]

        output = self.head(states[-1])
        residual, log_var = output.chunk(2, dim=-1)
        return residual, log_var


class LtcMotionPredictor:
    """Optional LTC/CfC residual corrector for ByteTrack Kalman predictions."""

    def __init__(self, args=None):
        checkpoint = getattr(args, "ltc_motion_ckpt", None)
        self.enabled = checkpoint is not None
        self.history_len = int(getattr(args, "ltc_history_len", 16))
        self.input_dim = int(getattr(args, "ltc_input_dim", 12))
        self.min_history = int(getattr(args, "ltc_min_history", self.history_len))
        self.covariance_scale = float(getattr(args, "ltc_covariance_scale", 1.0))
        self.max_abs_residual = float(getattr(args, "ltc_max_abs_residual", 256.0))

        if not self.enabled:
            self.model = None
            self.device = None
            return

        requested_device = getattr(args, "ltc_device", None)
        if requested_device is None:
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)

        checkpoint_data = torch.load(checkpoint, map_location=self.device)
        state_dict = checkpoint_data.get("model", checkpoint_data)

        self.history_len = int(checkpoint_data.get("history_len", self.history_len))
        self.input_dim = int(checkpoint_data.get("input_dim", self.input_dim))
        hidden_size = int(checkpoint_data.get("hidden_size", getattr(args, "ltc_hidden_size", 128)))
        num_layers = int(checkpoint_data.get("num_layers", getattr(args, "ltc_num_layers", 2)))

        self.model = LtcMotionResidual(
            input_dim=self.input_dim,
            history_len=self.history_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
        ).to(self.device)

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
                "LTC motion history has input_dim {}, expected {}".format(
                    values.shape[-1], self.input_dim
                )
            )
        values = values[-self.history_len:]
        if len(values) < self.history_len:
            pad = np.repeat(values[:1], self.history_len - len(values), axis=0)
            values = np.concatenate([pad, values], axis=0)
        return values
