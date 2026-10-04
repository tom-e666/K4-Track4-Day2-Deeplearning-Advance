"""test_custom.py - Unit test tự viết kiểm tra tính đúng đắn của loss, mixup và model."""
import unittest
import torch
import torch.nn as nn
from losses import FocalLoss, mix_batch
from model import build_model, freeze_backbone, param_groups


class TestCustomComponents(unittest.TestCase):

    def test_focal_loss_gamma_zero_equals_ce(self):
        """Kiểm tra Focal Loss khi gamma = 0 phải trùng 100% với CrossEntropyLoss (sai số < 1e-6)."""
        torch.manual_seed(42)
        logits = torch.randn(16, 9)
        targets = torch.randint(0, 9, (16,))

        ce_loss_fn = nn.CrossEntropyLoss()
        focal_loss_fn = FocalLoss(gamma=0.0)

        ce_val = ce_loss_fn(logits, targets)
        focal_val = focal_loss_fn(logits, targets)

        diff = (ce_val - focal_val).abs().item()
        self.assertLess(diff, 1e-6, f"Focal(gamma=0) lệch với CE: {diff}")

    def test_cutmix_recomputes_real_lambda(self):
        """Kiểm tra CutMix tính lại lambda theo diện tích thực của bbox sau khi clamp."""
        torch.manual_seed(42)
        images = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1).expand(4, 3, 224, 224).clone()
        labels = torch.tensor([0, 1, 2, 3])

        mixed_images, (y_a, y_b, lam) = mix_batch(images, labels, alpha=1.0, mode="cutmix")

        self.assertEqual(mixed_images.shape, (4, 3, 224, 224))
        self.assertGreaterEqual(lam, 0.0)
        self.assertLessEqual(lam, 1.0)
        for i in range(4):
            if y_a[i] != y_b[i]:
                replaced = (mixed_images[i, 0] != images[i, 0]).float().mean().item()
                self.assertAlmostEqual(1.0 - replaced, lam, places=6)

    def test_freeze_backbone_keeps_bn_eval(self):
        """Kiểm tra freeze_backbone giữ BatchNorm ở eval mode ngay cả khi gọi model.train()."""
        model = build_model("resnet50", pretrained=False, num_classes=9, init="frozen")
        model.train()

        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                self.assertFalse(m.training, "Lỗi: BatchNorm bị chuyển sang train mode!")

    def test_param_groups_separate_head_bias(self):
        """Separate head weights and bias while retaining the head learning rate."""
        model = build_model("resnet50", pretrained=False, num_classes=9, init="finetune")
        groups = param_groups(model, lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05)

        self.assertEqual(len(groups), 4)
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        self.assertEqual(groups[2]["lr"], 1e-3)

    def test_each_batch_resets_gradients(self):
        import train
        net = nn.Linear(2, 9)
        x, y = torch.ones(2, 2), torch.tensor([0, 1])
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0)
        gradients = []
        optimizer.register_step_pre_hook(
            lambda opt, args, kwargs: gradients.append(net.weight.grad.detach().clone()))
        loader = [(x, y, ["a", "b"])] * 2
        train.train_one_epoch(net, loader, nn.CrossEntropyLoss(), optimizer,
                              None, None, train.Config(amp=False), torch.device("cpu"))
        torch.testing.assert_close(gradients[0], gradients[1])

    def test_bias_has_no_decay_at_head_learning_rate(self):
        net = build_model("resnet50", pretrained=False)
        groups = param_groups(net, 1e-4, 1e-3, 0.05)
        bias = net.get_classifier().bias
        group = next(g for g in groups if any(p is bias for p in g["params"]))
        self.assertEqual(group["weight_decay"], 0.0)
        self.assertEqual(group["lr"], 1e-3)
        parameters = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(parameters), len(set(parameters)))
        self.assertEqual(set(parameters), {id(p) for p in net.parameters() if p.requires_grad})

    def test_fusion_preserves_dtype_device_and_outputs(self):
        from inference import fuse_conv_bn
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            for affine in (True, False):
                net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4, affine=affine)).to(device).double().eval()
                x = torch.randn(2, 3, 8, 8, device=device, dtype=torch.float64)
                fused = fuse_conv_bn(net)
                self.assertEqual(next(fused.parameters()).device, next(net.parameters()).device)
                self.assertEqual(next(fused.parameters()).dtype, torch.float64)
                torch.testing.assert_close(fused(x), net(x))
                self.assertIsInstance(net[1], nn.BatchNorm2d)

    def test_macs_include_tokens_and_attention(self):
        from model import count_gmacs
        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4, bias=False)
            def forward(self, x):
                q = self.fc(torch.ones(1, 5, 4, device=x.device))
                return torch.nn.functional.scaled_dot_product_attention(q, q, q)
        net = Attention().train()
        # Projection: 5*4*4; QK and AV: 2*5*5*4.
        self.assertAlmostEqual(count_gmacs(net), (80 + 200) / 1e9, places=14)
        self.assertTrue(net.training)

    def test_tta_benchmark_runs_actual_views(self):
        from benchmark import tta_latency
        calls = []
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 9)
            def forward(self, x):
                calls.append(1)
                return self.fc(x.mean((2, 3)))
        result = tta_latency(Tiny(), 2, device="cpu", img_size=8, warmup=1, iters=2)
        self.assertEqual(len(calls), 3 + 2 * 3)
        self.assertTrue(result["includes_view_transforms_and_aggregation"])
        self.assertGreater(result["p95"], 0)

    def test_existing_test_predictions_are_not_overwritten(self):
        import tempfile
        from pathlib import Path
        import train
        with tempfile.TemporaryDirectory() as folder:
            prediction = Path(folder) / "F01_seed0_test.csv"
            prediction.write_text("existing result", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                train.run(train.Config(exp_id="F01", pred_dir=folder, save_test_predictions=True))
            self.assertEqual(prediction.read_text(encoding="utf-8"), "existing result")

    def test_final_runner_exports_each_supported_inference_method(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import pandas as pd
        import numpy as np
        import train
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 9)
            def get_classifier(self):
                return self.fc
            def forward(self, x):
                return self.fc(x.mean((2, 3)))
        x = torch.randn(9, 3, 8, 8)
        y = torch.arange(9)
        df = pd.DataFrame({"Filename": [f"img{i}.jpg" for i in range(9)], "Label": range(9)})
        loader = [(x, y, df["Filename"].tolist())]
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(train.dataset, "load_split", return_value=(df, df, df)), \
             patch.object(train.dataset, "check_split", return_value={}), \
             patch.object(train.dataset, "build_transforms", return_value=None), \
             patch.object(train.dataset, "make_loader", return_value=loader), \
             patch.object(train.md, "build_model", side_effect=lambda *args, **kwargs: Tiny()), \
             patch.object(train, "plot_curves"):
            for method in ("I00", "I01", "I03", "I07", "I08"):
                cfg = train.Config(exp_id=method, epochs=1, amp=False, num_workers=0,
                                   out_dir=folder+"/runs", pred_dir=folder+"/predictions",
                                   curves_dir=folder+"/curves", img_size=8,
                                   inference_method=method, save_test_predictions=True)
                result = train.run(cfg)
                self.assertIsNotNone(result["test_metrics"])
                val = train.ev.read_pred(train.pred_path(cfg, "val"))
                test = train.ev.read_pred(train.pred_path(cfg, "test"))
                self.assertEqual(len(test.y_true), 9)
                np.testing.assert_allclose(test.probs, val.probs)
                uncal = Path(cfg.pred_dir) / f"{method}_uncal_seed0_test.csv"
                self.assertEqual(uncal.exists(), method == "I07")


if __name__ == "__main__":
    unittest.main()
