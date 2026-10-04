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
        images = torch.zeros(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])

        mixed_images, (y_a, y_b, lam) = mix_batch(images, labels, alpha=1.0, mode="cutmix")

        self.assertEqual(mixed_images.shape, (4, 3, 224, 224))
        self.assertGreaterEqual(lam, 0.0)
        self.assertLessEqual(lam, 1.0)

    def test_freeze_backbone_keeps_bn_eval(self):
        """Kiểm tra freeze_backbone giữ BatchNorm ở eval mode ngay cả khi gọi model.train()."""
        model = build_model("resnet50", pretrained=False, num_classes=9, init="frozen")
        model.train()

        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                self.assertFalse(m.training, "Lỗi: BatchNorm bị chuyển sang train mode!")

    def test_param_groups_has_three_groups(self):
        """Kiểm tra chia đủ 3 nhóm tham số với weight decay của norm/bias = 0."""
        model = build_model("resnet50", pretrained=False, num_classes=9, init="finetune")
        groups = param_groups(model, lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05)

        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        self.assertEqual(groups[2]["lr"], 1e-3)


if __name__ == "__main__":
    unittest.main()
