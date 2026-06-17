"""
SLSTMPredictor — xLSTM-based trajectory predictor for multi-object tracking.

Each track maintains a token history buffer. Each bbox is quantized into
4 tokens [x_c, y_c, w, h] in [0, vocab_size-1].

At each frame, sLSTM greedily generates the next 4 tokens (= next bbox)
given the per-track token history, then decodes back to pixel coordinates.
"""

import os

import numpy as np
import torch

try:
    from dacite import from_dict
    from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
    _XLSTM_AVAILABLE = True
except ImportError:
    _XLSTM_AVAILABLE = False


class SLSTMPredictor:
    """Wraps xLSTMLMModel to predict next bbox from per-track token history."""

    TOKENS_PER_FRAME = 4  # [x_c, y_c, w, h]

    def __init__(
        self,
        ckpt_path: str,
        vocab_size: int = 256,
        context_length: int = 256,
        device: str | None = None,
    ):
        if not _XLSTM_AVAILABLE:
            raise ImportError(
                "xlstm is not installed. "
                "Install it with: pip install -e path/to/xlstm"
            )

        self.vocab_size = int(vocab_size)
        self.context_length = int(context_length)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        ckpt = torch.load(ckpt_path, map_location=self.device)

        # Reconstruct model config from checkpoint or use training default
        if "config" in ckpt and "model" in ckpt["config"]:
            model_cfg = dict(ckpt["config"]["model"])
            ckpt_vocab_size = model_cfg.get("vocab_size")
            ckpt_context_length = model_cfg.get("context_length")
            if ckpt_vocab_size is not None:
                self.vocab_size = int(ckpt_vocab_size)
            if ckpt_context_length is not None:
                self.context_length = int(ckpt_context_length)
        else:
            model_cfg = {
                "num_blocks": 2,
                "embedding_dim": 64,
                "mlstm_block": {"mlstm": {"num_heads": 1}},
                "slstm_block": {"slstm": {"backend": "vanilla", "num_heads": 1}},
                "slstm_at": [0, 1],
                "context_length": self.context_length,
                "vocab_size": self.vocab_size,
            }

        model_cfg["context_length"] = self.context_length
        model_cfg["vocab_size"] = self.vocab_size

        # Force vanilla backend for cross-GPU compatibility (CC < 8.0 safe)
        if "slstm_block" in model_cfg:
            model_cfg["slstm_block"]["slstm"]["backend"] = "vanilla"

        self.model = xLSTMLMModel(
            from_dict(xLSTMLMModelConfig, model_cfg)
        ).to(self.device)

        state_key = "model_state_dict" if "model_state_dict" in ckpt else "model"
        state_dict = ckpt[state_key]
        token_emb = state_dict.get("token_embedding.weight")
        if token_emb is not None and token_emb.shape[0] != self.vocab_size:
            raise ValueError(
                "sLSTM checkpoint vocab_size mismatch: "
                f"checkpoint embedding has {token_emb.shape[0]} tokens, "
                f"but predictor was configured with {self.vocab_size}."
            )
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(
            f"[SLSTMPredictor] loaded checkpoint: {ckpt_path} "
            f"(vocab_size={self.vocab_size}, context_length={self.context_length})"
        )

    # ── Token ↔ bbox conversion ──────────────────────────────────────────────

    def bbox_to_tokens(
        self,
        cx: float, cy: float, w: float, h: float,
        img_w: float, img_h: float,
    ) -> np.ndarray:
        """Convert bbox [cx, cy, w, h] (pixels) → 4 quantized tokens."""
        V = self.vocab_size - 1
        return np.array([
            int(np.clip(cx / img_w * V, 0, V)),
            int(np.clip(cy / img_h * V, 0, V)),
            int(np.clip(w  / img_w * V, 0, V)),
            int(np.clip(h  / img_h * V, 0, V)),
        ], dtype=np.int64)

    def tokens_to_bbox(
        self,
        tokens: np.ndarray,
        img_w: float, img_h: float,
    ) -> np.ndarray:
        """Convert 4 tokens → bbox [cx, cy, w, h] in pixels."""
        V = self.vocab_size - 1
        return np.array([
            tokens[0] / V * img_w,
            tokens[1] / V * img_h,
            tokens[2] / V * img_w,
            tokens[3] / V * img_h,
        ], dtype=np.float32)

    # ── Batch prediction ─────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_batch(
        self,
        token_buffers: list,
        img_w: float,
        img_h: float,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Predict next bbox for a batch of tracks.

        Parameters
        ----------
        token_buffers : list of deques / lists of int tokens (per track).
        img_w, img_h  : image dimensions for decoding.

        Returns
        -------
        bboxes      : np.ndarray [B, 4] predicted [cx, cy, w, h] in pixels.
        pred_tokens : list of [4] int arrays, the 4 generated tokens per track.
        """
        B = len(token_buffers)
        if B == 0:
            return np.empty((0, 4), dtype=np.float32), []

        # Left-pad each track's history to context_length
        ctx = torch.zeros(B, self.context_length, dtype=torch.long, device=self.device)
        for i, buf in enumerate(token_buffers):
            toks = list(buf)[-self.context_length:]
            n = len(toks)
            if n > 0:
                toks = np.asarray(toks, dtype=np.int64)
                invalid = (toks < 0) | (toks >= self.vocab_size)
                if invalid.any():
                    bad = toks[invalid]
                    print(
                        "[SLSTMPredictor] clipped out-of-range tokens "
                        f"min={bad.min()} max={bad.max()} "
                        f"to [0, {self.vocab_size - 1}]"
                    )
                    toks = np.clip(toks, 0, self.vocab_size - 1)
                ctx[i, -n:] = torch.as_tensor(toks, dtype=torch.long, device=self.device)

        # Auto-regressively generate TOKENS_PER_FRAME tokens (greedy decoding)
        generated = []
        for _ in range(self.TOKENS_PER_FRAME):
            out = self.model(ctx)                        # [B, L, vocab_size]
            next_tok = out[:, -1].argmax(-1).clamp_(0, self.vocab_size - 1)  # [B]
            generated.append(next_tok)
            ctx = torch.cat([ctx[:, 1:], next_tok.unsqueeze(1)], dim=1)

        pred_tokens_t = torch.stack(generated, dim=1).cpu().numpy()  # [B, 4]

        bboxes = np.stack([
            self.tokens_to_bbox(pred_tokens_t[i], img_w, img_h)
            for i in range(B)
        ], axis=0)
        return bboxes, [pred_tokens_t[i] for i in range(B)]
