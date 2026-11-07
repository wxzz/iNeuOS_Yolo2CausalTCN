"""config.py — 全项目集中式配置（数据 / 特征 / 模型 / 训练）。

所有超参数以 dataclass 形式集中定义，脚本通过 argparse 覆盖，
避免超参数散落在各模块内部造成不可追溯。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT: Path = Path(__file__).resolve().parent

# 原始 YOLO 数据集根目录：每个子目录 = 一个数据集
#   权重约定：yolo_datasets/<dataset>/<dataset>.pt
DEFAULT_YOLO_ROOT: Path = PROJECT_ROOT / "yolo_datasets"
DEFAULT_YOLO_DIR: Path = DEFAULT_YOLO_ROOT  # 兼容旧引用
# 转换后的因果 TCN 数据集目录（标签、npy、配置、模型都放这里）
DEFAULT_TCN_DIR: Path = PROJECT_ROOT / "tcn_datasets"
DEFAULT_LABEL_DIR: Path = DEFAULT_TCN_DIR / "labels"
DEFAULT_ARTIFACT_DIR: Path = DEFAULT_TCN_DIR
DEFAULT_CHECKPOINT: Path = DEFAULT_TCN_DIR / "tcn_model.pth"
DEFAULT_LABEL_CSV: Path = DEFAULT_TCN_DIR / "labels.csv"
DEFAULT_DATASET_CONFIG: Path = DEFAULT_TCN_DIR / "tcn_config.json"

# 窗口长度默认值：build_dataset / train / inference / web 校验共用
# （历史上有 8 与 30 两套默认值，checkpoint 缺字段时会得到不同结论）
DEFAULT_WINDOW_SIZE: int = 8

# 关键点可见性阈值：detectors 写入 YOLO 的 keypoint conf，解析器按该阈值筛掉
# 不可信关键点（仅 pose_center_mode="keypoints" 生效）。两处必须共用同一数值。
KPT_VISIBILITY_THRESHOLD: float = 0.5


@dataclass
class DataConfig:
    """数据解析与特征工程配置。

    Attributes:
        window_size: 因果滑动窗口长度（帧）。
        stride: 滑窗步长，1 表示逐帧滑动。
        label_mode: 窗口标签取值方式，``last`` 取窗口末帧标签（因果安全），
            ``majority`` 取窗口内多数标签。
        num_target_classes: 标注中的目标类别数（如 4 类手部状态）。
        target_names: 目标类别名，用于日志与特征可视化。
        kpt_shape: YOLO-pose 关键点形状 ``[关键点数, 维度]``，须与 dataset.yaml 一致。
        pose_center_mode: ``bbox`` 取框中心；``keypoints`` 取可见关键点均值。
        missing_strategy: 某类目标整帧缺失时的降级策略。``interpolate`` 做**因果
            前向填充**（只用最近一次观测，不依赖未来帧），``zero`` 保持 0 且 mask=0。
        max_gap: 允许前向填充的最大缺失帧数（距最近一次观测），超过则回落填零。
        include_size: 是否把框宽高作为特征。
        include_velocity: 是否计算一阶差分速度 (dx, dy)。
        label_source: ``auto`` / ``csv`` / ``filename`` / ``dominant_class``。
        label_csv: 帧名 -> 步骤号的 CSV（由 ``build_dataset.py`` 生成）。
        filename_rules: 文件名关键字 -> 动作标签，例 ``{"pick": 0, "press": 1}``。
        session_pattern: 从文件名提取「视频段」的正则；用于阻止跨视频切窗。
        step_names: 步骤显示名（下标即步骤 id）。为空时回退到内置动作名。
    """

    label_dir: Path = DEFAULT_LABEL_DIR
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR

    window_size: int = DEFAULT_WINDOW_SIZE
    stride: int = 1
    label_mode: str = "last"

    num_target_classes: int = 4
    target_names: Tuple[str, ...] = (
        "right_palm",
        "left_palm",
        "right_fist",
        "left_fist",
    )

    kpt_shape: Tuple[int, int] = (20, 3)
    pose_center_mode: str = "bbox"
    kpt_visibility_threshold: float = KPT_VISIBILITY_THRESHOLD

    missing_strategy: str = "interpolate"
    max_gap: int = 5

    include_size: bool = True
    include_velocity: bool = True

    label_source: str = "auto"
    label_csv: Optional[Path] = DEFAULT_LABEL_CSV
    filename_rules: Dict[str, int] = field(default_factory=dict)
    default_label: Optional[int] = None

    session_pattern: Optional[str] = None
    step_names: Tuple[str, ...] = ()

    def validate(self) -> None:
        """尽早暴露非法配置，避免在深层计算中才报错。"""
        if self.window_size < 2:
            raise ValueError(f"window_size 必须 >= 2，当前 {self.window_size}")
        if self.stride < 1:
            raise ValueError(f"stride 必须 >= 1，当前 {self.stride}")
        if self.num_target_classes < 1:
            raise ValueError("num_target_classes 必须 >= 1")
        if self.label_mode not in ("last", "majority"):
            raise ValueError(f"未知 label_mode: {self.label_mode}")
        if self.missing_strategy not in ("interpolate", "zero"):
            raise ValueError(f"未知 missing_strategy: {self.missing_strategy}")
        if self.label_source not in ("auto", "csv", "filename", "dominant_class"):
            raise ValueError(f"未知 label_source: {self.label_source}")
        if len(self.kpt_shape) != 2 or self.kpt_shape[0] <= 0:
            raise ValueError(f"非法 kpt_shape: {self.kpt_shape}")


@dataclass
class TCNConfig:
    """因果 TCN 结构超参数（输入形状 (Batch, num_features, Seq_Length)）。"""

    num_features: int
    num_classes: int
    channels: Tuple[int, ...] = (64, 64, 64, 64)
    kernel_size: int = 3
    dropout: float = 0.2
    pooling: str = "last"  # last（在线推理一致） | mean | max
    use_weight_norm: bool = True

    def validate(self) -> None:
        if self.num_features < 1 or self.num_classes < 2:
            raise ValueError(
                "num_features 必须 >= 1 且 num_classes >= 2"
                "（至少需要两个步骤/类别，单类别无法训练）"
            )
        if not self.channels:
            raise ValueError("channels 不能为空")
        if self.kernel_size < 2:
            raise ValueError("kernel_size 必须 >= 2")
        if self.pooling not in ("last", "mean", "max"):
            raise ValueError(f"未知 pooling: {self.pooling}")

    @property
    def receptive_field(self) -> int:
        """因果感受野：1 + 2*(k-1)*(2^L - 1)，L 为残差块数量。"""
        return 1 + 2 * (self.kernel_size - 1) * (2 ** len(self.channels) - 1)


@dataclass
class TrainConfig:
    """训练超参数。"""

    epochs: int = 80
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.25
    gap: int = 0  # 训练/验证接缝处丢弃的窗口数（重叠滑窗防泄漏）
    seed: int = 42
    device: str = "auto"  # auto | cpu | cuda | cuda:0
    use_class_weight: bool = True
    scheduler: str = "plateau"  # plateau | cosine | none
    early_stopping_patience: int = 20
    num_workers: int = 0  # Windows 下建议 0，避免 spawn 问题
    checkpoint: Path = DEFAULT_CHECKPOINT

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs 必须 >= 1")
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError("val_ratio 必须位于 [0, 1)")
        if self.scheduler not in ("plateau", "cosine", "none"):
            raise ValueError(f"未知 scheduler: {self.scheduler}")
