"""
dataset.py  –  PyTorch Dataset + Video Swin Transformer model wrapper
"""

import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


# ─── Augmentations ───────────────────────────────────────────────────────────

class VideoAugment:
    """
    Clip-consistent spatial augmentations applied IDENTICALLY to every frame.
    This is critical: random crop / flip must be the same across frames.
    """
    def __init__(self, train: bool = True, img_size: int = 224):
        self.train = train
        self.img_size = img_size
        self.crop_size = int(img_size * 0.9)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: (T, 3, H, W) float32 [0, 1]
        if self.train:
            # Random crop (same params for all frames)
            i, j, h, w = transforms.RandomCrop.get_params(
                clip[0], output_size=(self.crop_size, self.crop_size)
            )
            clip = torch.stack([TF.crop(f, i, j, h, w) for f in clip])
            clip = torch.stack([TF.resize(f, [self.img_size, self.img_size]) for f in clip])

            # Random horizontal flip
            if torch.rand(1) > 0.5:
                clip = torch.stack([TF.hflip(f) for f in clip])

            # Color jitter (applied per-frame independently is fine)
            jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
            clip = torch.stack([jitter(f) for f in clip])
        else:
            # Centre crop for val/test
            clip = torch.stack([TF.center_crop(f, self.img_size) for f in clip])

        return clip


# ─── Dataset ─────────────────────────────────────────────────────────────────

class ViolenceDataset(Dataset):
    """
    Loads preprocessed .npy clips produced by preprocess.py.
    Each .npy file has shape (T, H, W, 3) uint8.

    Returns:
        clip  – (3, T, H, W) float32  normalised to ImageNet stats
        label – scalar int64
    """

    # ImageNet mean/std (used by all Kinetics-pretrained models)
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    def __init__(self, csv_path: str, train: bool = True, img_size: int = 224):
        self.records: list[dict] = []
        with open(csv_path) as f:
            self.records = list(csv.DictReader(f))

        self.train = train
        self.augment = VideoAugment(train=train, img_size=img_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        clip_np = np.load(rec["clip_path"])          # (T, H, W, 3) uint8
        label   = int(rec["label"])

        # → (T, 3, H, W) float32 [0, 1]
        clip = torch.from_numpy(clip_np).permute(0, 3, 1, 2).float() / 255.0

        clip = self.augment(clip)                    # (T, 3, H, W)
        clip = clip.permute(1, 0, 2, 3)              # (3, T, H, W)

        # Normalise
        clip = (clip - self.MEAN) / self.STD

        return clip, torch.tensor(label, dtype=torch.long)


# ─── Model ───────────────────────────────────────────────────────────────────

class ViolenceClassifier(nn.Module):
    """
    Video Swin Transformer (Swin-B) pretrained on Kinetics-400.
    The classification head is replaced with a 2-class head.

    Requires:  pip install mmaction2   OR   pip install pytorchvideo
    We use pytorchvideo here (lighter dependency).
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True, drop_path_rate: float = 0.2):
        super().__init__()

        # ── backbone ────────────────────────────────────────────────────────
        try:
            from pytorchvideo.models.hub import mvit_base_16x4  # fallback
            # Primary: Video Swin via torchvision (torch >= 2.1)
            import torchvision.models.video as vm
            # Use Swin-B weights pretrained on Kinetics-400
            weights = vm.Swin3D_B_Weights.KINETICS400_V1 if pretrained else None
            self.backbone = vm.swin3d_b(weights=weights, dropout=0.2)
            in_features = self.backbone.head[1].in_features
            self.backbone.head[1] = nn.Linear(in_features, num_classes)
            self._backbone_type = "swin3d_b"

        except Exception:
            # Fallback: MViT-B (also excellent for this task)
            from pytorchvideo.models.hub import mvit_base_16x4
            self.backbone = mvit_base_16x4(pretrained=pretrained)
            in_features = self.backbone.head.proj.in_features
            self.backbone.head.proj = nn.Linear(in_features, num_classes)
            self._backbone_type = "mvit_b"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, T, H, W)"""
        return self.backbone(x)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: str = "cpu") -> "ViolenceClassifier":
        ckpt = torch.load(ckpt_path, map_location=device)
        model = cls(num_classes=2, pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        return model


# ─── Real-time inference helper ───────────────────────────────────────────────

class RealTimeDetector:
    """
    Sliding-window inference on a live camera or video file.
    Accumulates `window_size` frames, runs one forward pass, then slides
    by `stride` frames.
    """
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model: nn.Module,
        device: str,
        window_size: int = 16,
        stride: int = 8,
        img_size: int = 224,
        confidence_threshold: float = 0.6,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.window_size = window_size
        self.stride = stride
        self.img_size = img_size
        self.threshold = confidence_threshold
        self._buffer: list[np.ndarray] = []
        self._step = 0

    def _letterbox(self, frame: np.ndarray) -> np.ndarray:
        """Same letterbox as preprocess.py for consistency."""
        import cv2
        h, w = frame.shape[:2]
        scale = self.img_size / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        y = (self.img_size - nh) // 2
        x = (self.img_size - nw) // 2
        canvas[y:y+nh, x:x+nw] = resized
        return canvas

    @torch.inference_mode()
    def push_frame(self, bgr_frame: np.ndarray) -> dict | None:
        """
        Push one BGR frame. Returns a prediction dict when a window is ready,
        otherwise returns None.
        """
        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        processed = self._letterbox(rgb)
        self._buffer.append(processed)
        self._step += 1

        if len(self._buffer) < self.window_size:
            return None
        if (self._step - self.window_size) % self.stride != 0 and self._step != self.window_size:
            # slide: keep only last window_size frames
            if len(self._buffer) > self.window_size:
                self._buffer.pop(0)
            return None

        clip = np.stack(self._buffer[-self.window_size:], axis=0).astype(np.float32) / 255.0
        clip = (clip - self.MEAN) / self.STD
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)
        # (1, 3, T, H, W)

        logits = self.model(clip)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = int(probs.argmax())
        conf   = float(probs[pred])

        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

        return {
            "prediction": "violence" if pred == 1 else "nonviolence",
            "confidence": conf,
            "alert": pred == 1 and conf >= self.threshold,
            "probs": {"nonviolence": float(probs[0]), "violence": float(probs[1])},
        }
