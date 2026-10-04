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

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_param_ids:
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

    return groups


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm toàn bộ tham số của mô hình."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """Tính GMAC cho một ảnh 3 x img_size x img_size."""
    try:
        from torchprofile import profile_macs
        dev = next(model.parameters()).device
        x = torch.randn(1, 3, img_size, img_size, device=dev)
        return float(profile_macs(model, x)) / 1e9
    except Exception:
        pass

    try:
        from thop import profile
        dev = next(model.parameters()).device
        x = torch.randn(1, 3, img_size, img_size, device=dev)
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return float(macs) / 1e9
    except Exception:
        pass

    # Đếm MACs chuẩn xác bằng hook cho Conv2d và Linear
    total_macs = 0
    hooks = []

    def hook_fn(module, inp, out):
        nonlocal total_macs
        if isinstance(module, nn.Conv2d):
            b, c_out, h_out, w_out = out.shape
            k_h, k_w = module.kernel_size
            c_in = module.in_channels // module.groups
            total_macs += b * c_out * h_out * w_out * (c_in * k_h * k_w)
        elif isinstance(module, nn.Linear):
            b = out.shape[0]
            total_macs += b * module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook_fn))

    dev = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, img_size, img_size, device=dev)
        try:
            model(x)
        except Exception:
            pass

    for h in hooks:
        h.remove()
    if was_training:
        model.train()

    return total_macs / 1e9
