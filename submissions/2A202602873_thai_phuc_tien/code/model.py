"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC."""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp từ timm."""
    is_pretrained = pretrained and (init != "scratch")
    model = timm.create_model(
        name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    if init == "frozen":
        freeze_backbone(model)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng backbone, chỉ giữ train classifier/head và giữ BatchNorm ở eval mode."""
    classifier = model.get_classifier()
    head_params = set(classifier.parameters()) if hasattr(classifier, "parameters") else set()

    for p in model.parameters():
        p.requires_grad = p in head_params

    # Khóa BatchNorm của backbone ở eval mode khi model.train() được gọi
    orig_train = model.train

    def custom_train(mode: bool = True):
        orig_train(mode)
        if mode:
            head_modules = set(classifier.modules()) if hasattr(classifier, "modules") else set()
            for m in model.modules():
                if m not in head_modules and isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return model

    model.train = custom_train


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
    """Chia tham số thành 3 nhóm: weights backbone (wd), norm/bias backbone (wd=0), head mới (10x LR, wd)."""
    classifier = model.get_classifier()
    head_param_ids = {id(p) for p in classifier.parameters()} if hasattr(classifier, "parameters") else set()

    backbone_weights = []
    backbone_norm_bias = []
    head_params = []
    head_norm_bias = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_param_ids:
            if p.ndim <= 1 or name.endswith(".bias"):
                head_norm_bias.append(p)
            else:
                head_params.append(p)
        elif p.ndim <= 1 or name.endswith(".bias"):
            backbone_norm_bias.append(p)
        else:
            backbone_weights.append(p)

    groups = []
    if backbone_weights:
        groups.append({"params": backbone_weights, "lr": lr_backbone, "weight_decay": weight_decay})
    if backbone_norm_bias:
        groups.append({"params": backbone_norm_bias, "lr": lr_backbone, "weight_decay": 0.0})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})
    if head_norm_bias:
        groups.append({"params": head_norm_bias, "lr": lr_head, "weight_decay": 0.0})

    return groups


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm toàn bộ tham số của mô hình."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """Count convolution/matmul MACs, including attention; 2 FLOPs = 1 MAC.

    Requires PyTorch's operator-level counter. Fail explicitly when unavailable
    rather than publish incomplete transformer counts.
    """
    from torch.utils.flop_counter import FlopCounterMode

    parameter = next(model.parameters())
    x = torch.randn(1, 3, img_size, img_size,
                    device=parameter.device, dtype=parameter.dtype)
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            model(x)
        flops = counter.get_total_flops()
        if flops <= 0:
            raise RuntimeError("No supported convolution/matmul operations were counted")
        return float(flops) / 2e9
    finally:
        for module, training in modes.items():
            module.training = training
