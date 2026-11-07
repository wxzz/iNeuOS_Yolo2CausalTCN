"""因果性测试：扰动未来帧不得影响历史输出。"""
from __future__ import annotations

import pytest
import torch

from config import TCNConfig
from model import CausalConv1d, CausalTCNClassifier, build_model


def test_causal_conv_ignores_future():
    torch.manual_seed(0)
    conv = CausalConv1d(4, 4, kernel_size=3, dilation=2)
    x = torch.randn(2, 4, 20)
    x_future = x.clone()
    x_future[:, :, -1] += 10.0
    with torch.no_grad():
        y1, y2 = conv(x), conv(x_future)
    assert y1.shape == x.shape
    torch.testing.assert_close(y1[:, :, :-1], y2[:, :, :-1])


def test_classifier_output_shape_and_receptive_field():
    cfg = TCNConfig(num_features=8, num_classes=3, channels=(16, 16), dropout=0.0)
    model = build_model(8, 3, channels=(16, 16), dropout=0.0)
    logits = model(torch.randn(5, 8, 30))
    assert logits.shape == (5, 3)
    assert cfg.receptive_field == 1 + 2 * 2 * (2 ** 2 - 1) == model.receptive_field


def test_classifier_is_causal_end_to_end():
    """改动窗口最后一帧，只能影响最后一帧的隐状态，历史位置必须完全一致。"""
    torch.manual_seed(1)
    model = CausalTCNClassifier(
        TCNConfig(num_features=4, num_classes=3, channels=(8, 8), dropout=0.0, pooling="last")
    ).eval()
    x = torch.randn(1, 4, 16)
    x_future = x.clone()
    x_future[:, :, -1] += 5.0
    with torch.no_grad():
        h1 = model.tcn(model.input_proj(x))
        h2 = model.tcn(model.input_proj(x_future))
    torch.testing.assert_close(h1[..., :-1], h2[..., :-1])


def test_predict_proba_sums_to_one():
    model = CausalTCNClassifier(TCNConfig(num_features=6, num_classes=4, channels=(8,)))
    probs = model.predict_proba(torch.randn(3, 6, 12))
    torch.testing.assert_close(probs.sum(dim=1), torch.ones(3))


def test_invalid_input_shape_raises():
    model = CausalTCNClassifier(TCNConfig(num_features=6, num_classes=2, channels=(8,)))
    with pytest.raises(ValueError):
        model(torch.randn(3, 6))
