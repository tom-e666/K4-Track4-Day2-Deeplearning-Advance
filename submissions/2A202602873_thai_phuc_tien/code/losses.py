"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: 'ce', 'ls' (label smoothing), 'focal', 'ce_weighted'."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    elif kind == "ls":
        smoothing = kw.get("smoothing", 0.1)
        return LabelSmoothingCE(smoothing=smoothing)
    elif kind == "focal":
        gamma = kw.get("gamma", 2.0)
        alpha = kw.get("alpha", None)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        weight = kw.get("weight", None)
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"Không hỗ trợ kind={kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, target)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        log_pt = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)

        focal_term = (1.0 - pt) ** self.gamma
        loss = -focal_term * log_pt

        if self.alpha is not None:
            alpha_tensor = self.alpha.to(logits.device)
            at = alpha_tensor.gather(0, target)
            loss = loss * at

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def class_weights(counts, beta: float = 0.0) -> torch.Tensor:
    """Tính trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN."""
    if isinstance(counts, dict):
        counts = [counts[k] for k in sorted(counts.keys())]
    counts_arr = np.array(counts, dtype=np.float32)

    if beta <= 0.0:
        weights = 1.0 / counts_arr
        weights = weights / weights.mean()
    else:
        effective_num = 1.0 - np.power(beta, counts_arr)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(counts_arr)

    return torch.tensor(weights, dtype=torch.float32)


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix") -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, float]]:
    """Trộn một batch ảnh và nhãn bằng Mixup hoặc CutMix."""
    if alpha <= 0.0:
        return x, (y, y, 1.0)

    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0))
    y_a = y
    y_b = y[perm]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[perm]
        return x_mixed, (y_a, y_b, lam)

    elif mode == "cutmix":
        b, c, h, w = x.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)

        x_mixed = x.clone()
        x_mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]

        real_area = float((x2 - x1) * (y2 - y1))
        lam_real = 1.0 - (real_area / (h * w))
        return x_mixed, (y_a, y_b, lam_real)

    return x, (y, y, 1.0)


def mixed_loss(criterion, logits: torch.Tensor, targets: tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    """Tính loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
