"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1)."""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian thực thi một hàm fn() (mili-giây), trả về p50, p95, p99."""
    if sync is None and torch.cuda.is_available():
        sync = torch.cuda.synchronize

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
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = model.to(dev)
    model.eval()

    if dtype == "fp16":
        model = model.half()
        x = torch.randn(batch_size, 3, img_size, img_size, device=dev, dtype=torch.float16)
    else:
        x = torch.randn(batch_size, 3, img_size, img_size, device=dev, dtype=torch.float32)

    def run_forward():
        with torch.inference_mode():
            if dtype == "amp" and dev.type == "cuda":
                with torch.cuda.amp.autocast():
                    _ = model(x)
            else:
                _ = model(x)

    sync_fn = torch.cuda.synchronize if dev.type == "cuda" else None
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
    """Đo độ trễ khi suy luận TTA K view."""
    base = latency_report(model, **kw)
    return {
        "k_views": k_views,
        "p50": base["p50"] * k_views,
        "p95": base["p95"] * k_views,
        "p99": base["p99"] * k_views,
        "base_p50": base["p50"],
    }
