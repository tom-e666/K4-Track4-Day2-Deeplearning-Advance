"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F)."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
try:
    import torch.amp as amp
except ImportError:
    amp = None
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Thêm đường dẫn để import eval.py và các module cùng thư mục
FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR
while REPO_ROOT != REPO_ROOT.parent:
    if (REPO_ROOT / "eval.py").exists():
        break
    REPO_ROOT = REPO_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(FILE_DIR) not in sys.path:
    sys.path.insert(0, str(FILE_DIR))

import dataset
import eval as ev
import losses
import model as md


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(net: nn.Module, cfg: Config) -> AdamW:
    groups = md.param_groups(net, lr_backbone=cfg.lr_backbone, lr_head=cfg.lr_head, weight_decay=cfg.weight_decay)
    return AdamW(groups)


def build_scheduler(optimizer: AdamW, cfg: Config, steps_per_epoch: int):
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class EMA:
    def __init__(self, net: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = deepcopy(net)
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, net: nn.Module) -> None:
        for s_param, param in zip(self.shadow.parameters(), net.parameters()):
            s_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
        for s_buffer, buffer in zip(self.shadow.buffers(), net.buffers()):
            s_buffer.copy_(buffer)


def train_one_epoch(net: nn.Module, loader, criterion, optimizer, scheduler, scaler,
                    cfg: Config, device, ema: EMA | None = None) -> dict:
    net.train()
    total_loss = 0.0
    num_samples = 0

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if cfg.mix and cfg.mix in ("mixup", "cutmix"):
            images, mix_targets = losses.mix_batch(images, targets, alpha=cfg.mix_alpha, mode=cfg.mix)
        else:
            mix_targets = None

        if amp is not None and hasattr(amp, "autocast"):
            autocast_ctx = amp.autocast("cuda", enabled=cfg.amp and torch.cuda.is_available())
        else:
            autocast_ctx = torch.cuda.amp.autocast(enabled=cfg.amp and torch.cuda.is_available())

        with autocast_ctx:
            outputs = net(images)
            if mix_targets is not None:
                loss = losses.mixed_loss(criterion, outputs, mix_targets)
            else:
                loss = criterion(outputs, targets)

        if scaler is not None and cfg.amp and torch.cuda.is_available():
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            if scheduler is not None and scale_before <= scale_after:
                scheduler.step()
        else:
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        if ema is not None:
            ema.update(net)

        total_loss += loss.item() * len(targets)
        num_samples += len(targets)

    current_lr = optimizer.param_groups[0]["lr"]
    return {"train_loss": total_loss / max(1, num_samples), "lr": current_lr}


@torch.no_grad()
def evaluate(net: nn.Module, loader, criterion, device) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    net.eval()
    all_filenames = []
    all_targets = []
    all_logits = []
    total_loss = 0.0
    num_samples = 0

    for images, targets, filenames in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.inference_mode():
            outputs = net(images)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * len(targets)
        num_samples += len(targets)

        all_filenames.extend(filenames)
        all_targets.append(targets.cpu().numpy())
        all_logits.append(outputs.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    avg_loss = total_loss / max(1, num_samples)
    return all_filenames, y_true, logits, avg_loss


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_macro_f1 = [h["val_macro_f1"] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_loss, label="Train Loss", color="red", linestyle="--")
    ax1.plot(epochs, val_loss, label="Val Loss", color="blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_macro_f1, label="Val Macro-F1", color="green", marker="o")
    ax2.set_ylabel("Macro-F1")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.title(title)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def run(cfg: Config) -> dict:
    set_seed(cfg.seed)
    r_dir = run_dir(cfg)
    r_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.curves_dir).mkdir(parents=True, exist_ok=True)

    with open(r_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # 1. Dataset & Split
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, fold=cfg.fold)
    dataset.check_split(train_df, val_df, test_df, cfg.images_dir)

    train_tf = dataset.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = dataset.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = dataset.make_loader(train_df, cfg.images_dir, train_tf, cfg.batch_size, train=True,
                                       sampler=cfg.sampler, num_workers=cfg.num_workers)
    val_loader = dataset.make_loader(val_df, cfg.images_dir, val_tf, cfg.batch_size, train=False,
                                     num_workers=cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Model & Loss
    net = md.build_model(cfg.backbone, pretrained=True, num_classes=ev.NUM_CLASSES,
                         drop_rate=cfg.drop_rate, init=cfg.init).to(device)

    c_weights = None
    if cfg.loss == "ce_weighted":
        counts = train_df["Label"].value_counts().to_dict()
        beta = cfg.class_weight_beta if cfg.class_weight_beta is not None else 0.0
        c_weights = losses.class_weights(counts, beta=beta).to(device)

    criterion = losses.build_criterion(
        cfg.loss,
        smoothing=cfg.label_smoothing,
        gamma=cfg.focal_gamma,
        weight=c_weights,
    )

    optimizer = build_optimizer(net, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    if amp is not None and hasattr(amp, "GradScaler"):
        scaler = amp.GradScaler("cuda", enabled=cfg.amp and torch.cuda.is_available())
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and torch.cuda.is_available())
    ema = EMA(net, decay=cfg.ema_decay) if cfg.ema_decay is not None else None

    # 3. Training Loop
    history = []
    best_f1 = -1.0
    best_epoch = -1
    best_ckpt_path = r_dir / "best_model.pt"

    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        tr_stats = train_one_epoch(net, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)

        eval_net = ema.shadow if ema is not None else net
        v_names, v_true, v_logits, v_loss = evaluate(eval_net, val_loader, criterion, device)
        v_probs = softmax(v_logits)
        v_pred = v_probs.argmax(axis=1)
        v_metrics = ev.compute_metrics(v_true, v_pred, v_probs)

        row = {
            "epoch": ep,
            "train_loss": tr_stats["train_loss"],
            "val_loss": v_loss,
            "val_macro_f1": v_metrics["macro_f1"],
            "val_top1": v_metrics["top1"],
            "lr": tr_stats["lr"],
        }
        history.append(row)

        print(f"Epoch {ep:02d}/{cfg.epochs:02d} | Train Loss: {tr_stats['train_loss']:.4f} | "
              f"Val Loss: {v_loss:.4f} | Val F1: {v_metrics['macro_f1']:.4f} | Val Top1: {v_metrics['top1']:.4f}")

        if v_metrics["macro_f1"] > best_f1:
            best_f1 = v_metrics["macro_f1"]
            best_epoch = ep
            torch.save(eval_net.state_dict(), best_ckpt_path)

    total_train_time = time.time() - t0
    pd.DataFrame(history).to_csv(r_dir / "history.csv", index=False)

    # 4. Lưu biểu đồ
    plot_curves(history, Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.backbone}.png",
                f"{cfg.exp_id}: {cfg.backbone} (Best Val F1={best_f1:.4f} @ Ep {best_epoch})")

    # 5. Đánh giá Checkpoint tốt nhất trên Val và lưu predictions
    eval_net = md.build_model(cfg.backbone, pretrained=False, num_classes=ev.NUM_CLASSES,
                              drop_rate=cfg.drop_rate, init="scratch").to(device)
    eval_net.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    val_names, val_true, val_logits, _ = evaluate(eval_net, val_loader, criterion, device)
    val_probs = softmax(val_logits)
    ev.save_predictions(pred_path(cfg, "val"), val_names, val_true, val_probs)
    np.save(r_dir / "val_logits.npy", val_logits)

    # 6. Test set (chỉ chạy khi cờ save_test_predictions được bật - Bước 4)
    test_metrics = None
    if cfg.save_test_predictions:
        test_tf = dataset.build_transforms(train=False, img_size=cfg.img_size)
        test_loader = dataset.make_loader(test_df, cfg.images_dir, test_tf, cfg.batch_size, train=False,
                                          num_workers=cfg.num_workers)
        test_names, test_true, test_logits, _ = evaluate(eval_net, test_loader, criterion, device)
        test_probs = softmax(test_logits)
        ev.save_predictions(pred_path(cfg, "test"), test_names, test_true, test_probs)
        np.save(r_dir / "test_logits.npy", test_logits)
        test_metrics = ev.compute_metrics(test_true, test_probs.argmax(axis=1), test_probs)

    n_params = md.count_params(eval_net)
    gmacs = md.count_gmacs(eval_net, img_size=cfg.img_size)

    return {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "backbone": cfg.backbone,
        "params_m": n_params,
        "gmacs": gmacs,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "total_train_time_s": total_train_time,
        "time_per_epoch_s": total_train_time / max(1, cfg.epochs),
        "test_metrics": test_metrics,
    }


def parse_overrides(pairs: list[str]) -> dict:
    valid_fields = {f.name: f.type for f in fields(Config)}
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Tham số không hợp lệ (cần KEY=VALUE): '{pair}'")
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k not in valid_fields:
            raise ValueError(f"Field '{k}' không tồn tại trong Config. Các field hợp lệ: {list(valid_fields.keys())}")

        if v.lower() in ("none", "null"):
            overrides[k] = None
        elif v.lower() == "true":
            overrides[k] = True
        elif v.lower() == "false":
            overrides[k] = False
        else:
            try:
                overrides[k] = int(v)
            except ValueError:
                try:
                    overrides[k] = float(v)
                except ValueError:
                    overrides[k] = v
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình DeepWeeds")
    parser.add_argument("--set", nargs="*", default=[], help="Ghi đè tham số: KEY=VALUE ...")
    args = parser.parse_args()

    cfg = Config()
    if args.set:
        overrides = parse_overrides(args.set)
        for k, v in overrides.items():
            setattr(cfg, k, v)

    print(f"=== Bắt đầu chạy exp_id={cfg.exp_id}, backbone={cfg.backbone}, seed={cfg.seed} ===")
    res = run(cfg)
    print("=== Kết quả hoàn thành ===")
    print(f"Best Val Macro-F1: {res['best_val_macro_f1']:.4f} (Epoch {res['best_epoch']})")
    print(f"Params: {res['params_m']:.2f}M | GMACs: {res['gmacs']:.2f} | Time/epoch: {res['time_per_epoch_s']:.1f}s")


if __name__ == "__main__":
    main()
