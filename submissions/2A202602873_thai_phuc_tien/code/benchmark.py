"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1)."""
from __future__ import annotations

import time
from contextlib import nullcontext
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian thực thi một hàm fn() (mili-giây), trả về p50, p95, p99."""
    if warmup < 0 or iters < 1:
        raise ValueError("warmup must be >= 0 and iters must be >= 1")

    # Warmup
    for _ in range(warmup):
        fn()
    if sync:
        sync()

    # Lượt đo chính thức
    times = []
    for _ in range(iters):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times_arr = np.array(times)
    return {
        "p50": float(np.percentile(times_arr, 50)),
        "p95": float(np.percentile(times_arr, 95)),
        "p99": float(np.percentile(times_arr, 99)),
        "mean": float(np.mean(times_arr)),
        "n": iters,
    }


def latency_report(model: nn.Module, batch_size: int = 1, img_size: int = 224,
                   dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward chuẩn: đồng bộ GPU, tính p50/p95/p99 và thông lượng ảnh/giây."""
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    model = deepcopy(model).to(dev)
    model.eval()

    if dtype not in ("fp32", "fp16", "amp"):
        raise ValueError("dtype must be fp32, fp16 or amp")
    model.float()
    if dtype == "fp16":
        model = model.half()
        x = torch.randn(batch_size, 3, img_size, img_size, device=dev, dtype=torch.float16)
    else:
        x = torch.randn(batch_size, 3, img_size, img_size, device=dev, dtype=torch.float32)

    def run_forward():
        with torch.inference_mode():
            if dtype == "amp" and dev.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(x)
            else:
                _ = model(x)

    sync_fn = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else None
    stats = bench(run_forward, warmup=warmup, iters=iters, sync=sync_fn)

    gpu_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
    images_per_s = (batch_size / (stats["p50"] / 1000.0)) if stats["p50"] > 0 else 0.0

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": stats["p50"],
        "p95": stats["p95"],
        "p99": stats["p99"],
        "images_per_s": images_per_s,
        "torch": torch.__version__,
    }


def tta_latency(model: nn.Module, k_views: int, **kw) -> dict:
    """Measure view transforms, forwards and aggregation (excludes image I/O).

    Defaults to identity + horizontal flip for K=2. Other view counts require
    explicit view functions. Use space='logit' to measure logit aggregation.
    """
    from inference import view_identity, view_hflip

    views = kw.pop("views", None)
    space = kw.pop("space", "prob")
    if views is None:
        if k_views == 1:
            views = [view_identity]
        elif k_views == 2:
            views = [view_identity, view_hflip]
        else:
            raise ValueError("Provide actual view functions for k_views > 2")
    if len(views) != k_views or k_views < 1:
        raise ValueError("k_views must match the number of view functions")
    if space not in ("prob", "logit"):
        raise ValueError("space must be prob or logit")

    base = latency_report(model, **kw)
    dev = torch.device(kw.get("device", "cuda"))
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    dtype = kw.get("dtype", "fp32")
    net = deepcopy(model).to(dev).eval()
    net = net.half() if dtype == "fp16" else net.float()
    batch_size = kw.get("batch_size", 1)
    img_size = kw.get("img_size", 224)
    x = torch.randn(batch_size, 3, img_size, img_size,
                    device=dev, dtype=next(net.parameters()).dtype)

    def run_tta():
        context = (torch.autocast(device_type="cuda", dtype=torch.float16)
                   if dtype == "amp" and dev.type == "cuda" else nullcontext())
        with torch.inference_mode(), context:
            logits = [net(view(x)) for view in views]
            if space == "prob":
                return torch.stack([z.softmax(-1) for z in logits]).mean(0)
            return torch.stack(logits).mean(0).softmax(-1)

    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else None
    stats = bench(run_tta, warmup=kw.get("warmup", 10),
                  iters=kw.get("iters", 100), sync=sync)
    return {**base, "k_views": k_views, "space": space,
            "p50": stats["p50"], "p95": stats["p95"], "p99": stats["p99"],
            "base_p50": base["p50"], "relative_cost": stats["p50"] / base["p50"],
            "images_per_s": batch_size * 1000 / stats["p50"],
            "includes_view_transforms_and_aggregation": True}
