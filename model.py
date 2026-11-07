"""model.py — 严格因果时序卷积网络（Causal TCN）。

关键点：
1. ``CausalConv1d`` 只在左侧补零，卷积后裁剪回原长度，保证 t 时刻输出只依赖 <= t 的输入；
2. ``TemporalBlock`` 采用指数增长膨胀因子 + 权重归一化 + Dropout + 残差连接；
3. 输入形状 ``(Batch, Features, Seq_Length)``，输出未归一化 logits。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEFAULT_WINDOW_SIZE, TCNConfig

LOGGER = logging.getLogger(__name__)


def maybe_weight_norm(module: nn.Module, enabled: bool = True) -> nn.Module:
    """兼容不同 PyTorch 版本的权重归一化（新版本用 parametrizations）。"""
    if not enabled:
        return module
    try:  # torch >= 2.1
        from torch.nn.utils.parametrizations import weight_norm
    except ImportError:  # 老版本回退
        from torch.nn.utils import weight_norm
    return weight_norm(module)


class CausalConv1d(nn.Module):
    """因果一维卷积：左侧补零 ``(k-1)*d``，卷积后裁剪到输入长度。

    膨胀卷积下输出长度会变成 ``T + (k-1)*d - k + 1``，因此必须显式裁剪回 ``T``，
    否则多层堆叠后长度漂移、且右侧会混入未来信息。

    Note:
        权重归一化必须作用在内层 ``nn.Conv1d``（带 ``weight`` 参数）上，
        不能作用在本封装模块上，否则 parametrization 找不到 ``weight``。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
        use_weight_norm: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size 必须 >= 1")
        self.left_padding = (kernel_size - 1) * dilation
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias,
        )
        self._init_weights(conv)  # 先初始化再包 weight_norm，避免写入只读的 weight
        self.conv = maybe_weight_norm(conv, use_weight_norm)

    @staticmethod
    def _init_weights(conv: nn.Conv1d) -> None:
        """Kaiming 初始化 + 零偏置，缓解深层膨胀卷积的梯度衰减。"""
        nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x 形状 ``(B, C_in, T)``；Returns: ``(B, C_out, T)``。"""
        length = x.size(-1)
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        out = self.conv(x)
        return out[..., :length]


class TemporalBlock(nn.Module):
    """TCN 残差块：两层因果膨胀卷积 + 残差（通道不匹配时 1x1 卷积对齐）。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        use_weight_norm: bool = True,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation,
                use_weight_norm=use_weight_norm,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(
                out_channels,
                out_channels,
                kernel_size,
                dilation,
                use_weight_norm=use_weight_norm,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(out + residual)


class CausalTCNClassifier(nn.Module):
    """因果 TCN 分类器。

    Args:
        cfg: :class:`config.TCNConfig`，含通道数序列、卷积核、Dropout、池化方式。

    Shape:
        - 输入：``(B, num_features, T)``
        - 输出：``(B, num_classes)`` logits
    """

    def __init__(self, cfg: TCNConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg

        self.input_proj = nn.Conv1d(cfg.num_features, cfg.channels[0], kernel_size=1)
        blocks = []
        for level, out_channels in enumerate(cfg.channels):
            blocks.append(
                TemporalBlock(
                    in_channels=(
                        cfg.channels[level - 1] if level > 0 else cfg.channels[0]
                    ),
                    out_channels=out_channels,
                    kernel_size=cfg.kernel_size,
                    dilation=2**level,
                    dropout=cfg.dropout,
                    use_weight_norm=cfg.use_weight_norm,
                )
            )
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(cfg.channels[-1], cfg.num_classes)

        LOGGER.debug(
            "CausalTCN 构建完成: 感受野=%d, 参数量=%d",
            self.receptive_field,
            self.count_parameters(),
        )

    @property
    def receptive_field(self) -> int:
        """理论因果感受野（帧数）。窗口长度应不显著大于该值。"""
        return self.cfg.receptive_field

    def count_parameters(self) -> int:
        """可训练参数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"输入必须是 (B, F, T)，收到 {tuple(x.shape)}")
        h = self.input_proj(x)
        h = self.tcn(h)

        if self.cfg.pooling == "last":
            pooled = h[..., -1]  # 因果：只用最后一帧的表示
        elif self.cfg.pooling == "mean":
            pooled = h.mean(dim=-1)
        elif self.cfg.pooling == "max":
            pooled = h.max(dim=-1).values
        else:  # pragma: no cover - 配置校验已拦截
            raise ValueError(f"未知 pooling: {self.cfg.pooling}")

        return self.head(pooled)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """返回 softmax 概率，便于推理端直接取置信度。"""
        return F.softmax(self.forward(x), dim=-1)


def build_model(
    num_features: int, num_classes: int, **overrides
) -> CausalTCNClassifier:
    """便捷构造器，训练/推理共用，避免两处配置漂移。"""
    cfg = TCNConfig(num_features=num_features, num_classes=num_classes, **overrides)
    return CausalTCNClassifier(cfg)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    """读取 checkpoint（CPU），并确认包含 ``state_dict``。

    训练/推理/Web 服务都必须走本函数，避免各处 ``torch.load`` 的参数与
    错误处理不一致（``weights_only``、默认窗口等）。
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"checkpoint 格式不正确（缺少 state_dict）：{path}")
    return payload


def read_checkpoint_meta(path: Path) -> Dict[str, Any]:
    """只读取 checkpoint 的元信息（不构造模型），字段缺失时用统一默认值。"""
    payload = load_checkpoint(path)
    return {
        "window_size": int(payload.get("window_size") or DEFAULT_WINDOW_SIZE),
        "feature_dim": int(payload.get("feature_dim") or 0),
        "num_classes": int(payload.get("num_classes") or 0),
        "class_names": list(payload.get("class_names") or []),
        "epoch": int(payload.get("epoch") or 0),
        "metrics": payload.get("metrics") or {},
        "path": str(path),
    }
