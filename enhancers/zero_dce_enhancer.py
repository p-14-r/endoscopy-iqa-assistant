from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from models.interfaces import EnhancementResult, Enhancer


class ZeroDCEEnhancer(Enhancer):
    """Optional Zero-DCE enhancer wrapper.

    Requires:
    - torch
    - a Zero-DCE checkpoint path compatible with DCE net below.

    If requirements are unavailable and `strict=False`, falls back to identity transform.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None, strict: bool = False):
        self.checkpoint_path = checkpoint_path
        self.strict = strict
        self.ready = False
        self.warning: Optional[str] = None

        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            if strict:
                raise ImportError("torch is required for Zero-DCE enhancer") from exc
            self.warning = "torch not installed; enhancer will return input image."
            return

        class _DCE(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU(inplace=True)
                self.e_conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
                self.e_conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
                self.e_conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
                self.e_conv4 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
                self.e_conv5 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
                self.e_conv6 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
                self.e_conv7 = nn.Conv2d(64, 24, 3, 1, 1, bias=True)

            def forward(self, x):
                x1 = self.relu(self.e_conv1(x))
                x2 = self.relu(self.e_conv2(x1))
                x3 = self.relu(self.e_conv3(x2))
                x4 = self.relu(self.e_conv4(x3))
                x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
                x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
                x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

                r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
                x = x + r1 * (x * x - x)
                x = x + r2 * (x * x - x)
                x = x + r3 * (x * x - x)
                x = x + r4 * (x * x - x)
                x = x + r5 * (x * x - x)
                x = x + r6 * (x * x - x)
                x = x + r7 * (x * x - x)
                x = x + r8 * (x * x - x)
                return torch.clamp(x, 0.0, 1.0)

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _DCE().to(self.device).eval()

        if checkpoint_path:
            ckpt = Path(checkpoint_path)
            if not ckpt.exists():
                msg = f"Zero-DCE checkpoint not found at {checkpoint_path}; enhancer will return input image."
                if strict:
                    raise FileNotFoundError(msg)
                self.warning = msg
                return

            state = torch.load(str(ckpt), map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            # Strip DataParallel prefix if present.
            clean_state = {k.replace("module.", ""): v for k, v in state.items()}
            self.model.load_state_dict(clean_state, strict=False)
            self.ready = True
        else:
            msg = "Zero-DCE checkpoint not provided; enhancer will return input image."
            if strict:
                raise ValueError(msg)
            self.warning = msg

    def enhance(self, image: np.ndarray) -> EnhancementResult:
        if not self.ready:
            return EnhancementResult(
                name="zero_dce_optional",
                enhanced_image=image.copy(),
                metadata={"ready": False, "warning": self.warning},
            )

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = self._torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with self._torch.no_grad():
            out = self.model(tensor)

        out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_np = (out_np * 255.0).clip(0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

        return EnhancementResult(
            name="zero_dce_optional",
            enhanced_image=out_bgr,
            metadata={"ready": True, "device": self.device, "checkpoint_path": self.checkpoint_path},
        )
