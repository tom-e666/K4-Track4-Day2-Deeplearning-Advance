"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md)."""
from __future__ import annotations

from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn


def predict_logits(model: nn.Module, loader, device, view=None) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    all_filenames = []
    all_targets = []
    all_logits = []

    with torch.inference_mode():
        for images, targets, filenames in loader:
            images = images.to(dev, non_blocking=True)
            if view is not None:
                images = view(images)
            outputs = model(images)

            all_filenames.extend(filenames)
            all_targets.append(targets.cpu().numpy() if isinstance(targets, torch.Tensor) else np.array(targets))
            all_logits.append(outputs.cpu().numpy())

    return all_filenames, np.concatenate(all_targets, axis=0), np.concatenate(all_logits, axis=0)


def view_identity(x: torch.Tensor) -> torch.Tensor:
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int) -> list[torch.Tensor]:
    _, _, h, w = x.shape
    if h < crop or w < crop:
        x = nn.functional.interpolate(x, size=(max(h, crop), max(w, crop)), mode="bilinear", align_corners=False)
        _, _, h, w = x.shape

    return [
        x[:, :, :crop, :crop],
        x[:, :, :crop, w - crop:],
        x[:, :, h - crop:, :crop],
        x[:, :, h - crop:, w - crop:],
        x[:, :, (h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2],
    ]


def views_multiscale(x: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
    return [nn.functional.interpolate(x, size=(s, s), mode="bilinear", align_corners=False) for s in sizes]


def aggregate_views(logits_per_view: list[np.ndarray], space: str = "prob") -> np.ndarray:
    def _softmax(z: np.ndarray) -> np.ndarray:
        e = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    if space == "prob":
        probs_list = [_softmax(v) for v in logits_per_view]
        return np.mean(probs_list, axis=0)
    elif space == "logit":
        avg_logits = np.mean(logits_per_view, axis=0)
        return _softmax(avg_logits)
    else:
        raise ValueError(f"Không hỗ trợ space='{space}'. Chỉ dùng 'prob' hoặc 'logit'.")


def ensemble_probs(list_of_probs: list[np.ndarray]) -> np.ndarray:
    return np.mean(list_of_probs, axis=0)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    def nll(T: float) -> float:
        if T <= 0.0:
            return 1e9
        scaled = val_logits / T
        e = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        probs = e / np.sum(e, axis=-1, keepdims=True)
        eps = 1e-12
        correct = np.clip(probs[np.arange(len(val_labels)), val_labels], eps, 1.0)
        return float(-np.mean(np.log(correct)))

    try:
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
        return float(res.x)
    except Exception:
        candidates = np.linspace(0.1, 5.0, 500)
        losses = [nll(c) for c in candidates]
        return float(candidates[np.argmin(losses)])


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    scaled = logits / max(1e-6, float(T))
    e = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    model.eval()
    fused = deepcopy(model)

    def _fuse_module(m: nn.Module):
        for child in m.children():
            _fuse_module(child)

        children = list(m.named_children())
        for i in range(len(children) - 1):
            n1, c1 = children[i]
            n2, c2 = children[i + 1]
            if isinstance(c1, nn.Conv2d) and isinstance(c2, nn.modules.batchnorm._BatchNorm):
                fused_conv = torch.nn.utils.fusion.fuse_conv_bn_eval(c1, c2)

                setattr(m, n1, fused_conv)
                setattr(m, n2, nn.Identity())

    _fuse_module(fused)
    return fused
