import torch
import torch.nn as nn
import numpy as np


class DynamicWeightNet(nn.Module):
    """
    Per-pair reliability weight estimator for motion-appearance cost fusion.

    Input:  feature vector describing one (track, detection) candidate pair.
    Output: [w_motion, w_reid] that sum to 1 via softmax.
    """

    def __init__(self, input_dim: int = 15, hidden_dims: tuple = (64, 32)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, input_dim)
        Returns:
            weights: (N, 2)  [w_motion, w_reid], each row sums to 1
        """
        return torch.softmax(self.net(x), dim=-1)

    @torch.no_grad()
    def predict_numpy(self, x_np: np.ndarray) -> np.ndarray:
        """Convenience wrapper for numpy input during tracker inference."""
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(x_np).float().to(device)
        return self(x).cpu().numpy()

    def save(self, path: str, stats: dict = None):
        torch.save({"state_dict": self.state_dict(), "stats": stats}, path)

    @classmethod
    def load(cls, path: str, input_dim: int = 15, hidden_dims: tuple = (64, 32)):
        ckpt = torch.load(path, map_location="cpu")
        model = cls(input_dim=input_dim, hidden_dims=hidden_dims)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        stats = ckpt.get("stats", None)
        return model, stats
