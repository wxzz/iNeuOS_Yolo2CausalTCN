"""data_processor.py — YOLOv8 多任务标注解析 + 因果时序特征工程。

分层结构
--------
1. 解析层：``BaseParser`` 抽象出统一策略接口，``Detect/Obb/Seg/Pose`` 四个实现，
   ``YoloParserFactory`` 依据每行 token 数量自动分发（策略 + 工厂）。
2. 聚合层：``aggregate_detections`` 把一帧内的多目标压缩为 ``(C, 5)`` 类别槽位矩阵
   ``[cx, cy, w, h, mask]``，同类多目标取均值（``np.add.at`` 向量化）。
3. 特征层：``FrameFeatureAssembler`` 生成 ``[cx..., cy..., w..., h..., mask..., dx..., dy...]``；
   序列版与流式版共用同一布局，保证训练/推理特征完全一致。
4. 缺失层：``interpolate_missing`` 做**因果前向填充**（只用当前帧与过去观测，
   最近一次观测超过 ``max_gap`` 帧则回落填零）；流式版 ``CausalMissingFiller``
   与批量版逐帧等价，保证训练/推理特征一致。
5. 标签层：``CsvLabelProvider`` / ``FilenameRuleLabelProvider`` / ``DominantClassLabelProvider``。
6. 数据集层：``SequenceDatasetBuilder`` 输出 ``X (N, F, W)``、``y (N,)``。

坐标约定：全部为 YOLO 归一化坐标 ``[0, 1]``，不做像素换算。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import KPT_VISIBILITY_THRESHOLD, DataConfig

LOGGER = logging.getLogger(__name__)

IGNORE_INDEX: int = -1
"""无标签帧/窗口的占位标签。"""


# --------------------------------------------------------------------------- #
# 基础数据结构
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Detection:
    """单个目标在归一化图像坐标系下的几何描述。"""

    class_id: int
    cx: float
    cy: float
    width: float = 0.0
    height: float = 0.0
    source: str = "unknown"
    keypoints: Optional[np.ndarray] = None  # (K, dim)，仅 pose 有效


def polygon_centroid(points: np.ndarray) -> Tuple[float, float]:
    """计算多边形质心（鞋带公式）。

    Args:
        points: 形状 ``(N, 2)`` 的顶点序列，N >= 3。

    Returns:
        ``(cx, cy)``。若多边形面积近似为 0（退化/共线），回退为顶点算术均值。
    """
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError(f"polygon_centroid 需要 (N>=3, 2) 的数组，收到 {points.shape}")

    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    cross = x * y_next - x_next * y
    area2 = float(cross.sum())
    if abs(area2) < 1e-12:
        return float(x.mean()), float(y.mean())

    cx = float(((x + x_next) * cross).sum() / (3.0 * area2))
    cy = float(((y + y_next) * cross).sum() / (3.0 * area2))
    return cx, cy


def _tokenize(line: str) -> Optional[np.ndarray]:
    """把一行文本转成 float 数组；空行/注释/含非法字符行返回 None。"""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    try:
        tokens = np.asarray(stripped.split(), dtype=np.float64)
    except ValueError:
        return None
    if tokens.size == 0 or not np.all(np.isfinite(tokens)):
        return None
    return tokens


# --------------------------------------------------------------------------- #
# 解析策略（策略模式）
# --------------------------------------------------------------------------- #
class BaseParser(ABC):
    """标注行解析策略基类。

    子类只需回答两个问题：token 数量是否属于我（``matches``）以及如何解析（``parse``）。
    上层完全不关心具体格式，新增格式只需新增子类并注册进工厂，符合开闭原则。
    """

    name: str = "base"

    @abstractmethod
    def matches(self, n_fields: int) -> bool:
        """判断 token 数量是否属于本格式。"""

    @abstractmethod
    def parse(self, tokens: np.ndarray) -> Optional[Detection]:
        """解析单行标注；脏数据返回 ``None``，由上层统计并跳过。"""

    @staticmethod
    def _parse_class_id(
        token: float, num_classes: Optional[int] = None
    ) -> Optional[int]:
        """校验并转换类别 ID（整数、非负、可选上界）。"""
        if not np.isfinite(token) or token < 0:
            return None
        class_id = int(round(float(token)))
        if num_classes is not None and class_id >= num_classes:
            return None
        return class_id

    @staticmethod
    def _check_coord(value: float) -> float:
        """归一化坐标合法化：非有限值报错，越界轻微裁剪，明显异常报错。"""
        if not np.isfinite(value):
            raise ValueError("坐标非有限值")
        if abs(value) > 2.0:
            raise ValueError(f"坐标严重越界: {value}")
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _size(value: float) -> float:
        """宽高合法化：取绝对值并裁剪到 [0, 1]。"""
        if not np.isfinite(value):
            raise ValueError("尺寸非有限值")
        return float(np.clip(abs(value), 0.0, 1.0))


class DetectParser(BaseParser):
    """YOLO-detect：``class x_center y_center width height``（5 个数值）。"""

    name = "detect"

    def __init__(self, num_classes: Optional[int] = None) -> None:
        self.num_classes = num_classes

    def matches(self, n_fields: int) -> bool:
        return n_fields == 5

    def parse(self, tokens: np.ndarray) -> Optional[Detection]:
        class_id = self._parse_class_id(tokens[0], self.num_classes)
        if class_id is None:
            return None
        try:
            cx = self._check_coord(float(tokens[1]))
            cy = self._check_coord(float(tokens[2]))
            w = self._size(float(tokens[3]))
            h = self._size(float(tokens[4]))
        except ValueError:
            return None
        return Detection(class_id, cx, cy, w, h, self.name)


class ObbParser(BaseParser):
    """YOLO-obb：``class x1 y1 x2 y2 x3 y3 x4 y4``（9 个数值），取旋转框几何中心。"""

    name = "obb"

    def __init__(self, num_classes: Optional[int] = None) -> None:
        self.num_classes = num_classes

    def matches(self, n_fields: int) -> bool:
        return n_fields == 9

    def parse(self, tokens: np.ndarray) -> Optional[Detection]:
        class_id = self._parse_class_id(tokens[0], self.num_classes)
        if class_id is None:
            return None
        try:
            points = np.clip(tokens[1:].reshape(4, 2), 0.0, 1.0)
        except ValueError:
            return None
        cx, cy = polygon_centroid(points)
        span = points.max(axis=0) - points.min(axis=0)
        return Detection(
            class_id,
            float(np.clip(cx, 0, 1)),
            float(np.clip(cy, 0, 1)),
            float(span[0]),
            float(span[1]),
            self.name,
        )


class SegParser(BaseParser):
    """YOLO-seg：``class x1 y1 x2 y2 ...``（1 + 2N 且 N >= 3），取多边形质心。

    Note:
        65 个数值的 pose 行同样满足「偶数坐标」判据，因此工厂里 ``PoseParser``
        必须先于本策略匹配，否则会被误判为 32 点多边形。
    """

    name = "seg"

    def __init__(self, num_classes: Optional[int] = None) -> None:
        self.num_classes = num_classes

    def matches(self, n_fields: int) -> bool:
        n_points = (n_fields - 1) / 2
        return n_fields >= 7 and (n_fields - 1) % 2 == 0 and n_points >= 3

    def parse(self, tokens: np.ndarray) -> Optional[Detection]:
        class_id = self._parse_class_id(tokens[0], self.num_classes)
        if class_id is None:
            return None
        try:
            points = np.clip(tokens[1:].reshape(-1, 2), 0.0, 1.0)
            cx, cy = polygon_centroid(points)
        except ValueError:
            return None
        span = points.max(axis=0) - points.min(axis=0)
        return Detection(
            class_id,
            float(np.clip(cx, 0, 1)),
            float(np.clip(cy, 0, 1)),
            float(span[0]),
            float(span[1]),
            self.name,
        )


class PoseParser(BaseParser):
    """YOLO-pose：``class x y w h px1 py1 v1 ...``，中心可配置为框中心或可见关键点均值。"""

    name = "pose"

    def __init__(
        self,
        kpt_shape: Tuple[int, int] = (20, 3),
        center_mode: str = "bbox",
        visibility_threshold: float = KPT_VISIBILITY_THRESHOLD,
        num_classes: Optional[int] = None,
    ) -> None:
        self.kpt_count = int(kpt_shape[0])
        self.kpt_dim = int(kpt_shape[1])
        if self.kpt_count <= 0 or self.kpt_dim not in (2, 3):
            raise ValueError(f"非法 kpt_shape: {kpt_shape}")
        if center_mode not in ("bbox", "keypoints"):
            raise ValueError(f"未知 pose center_mode: {center_mode}")
        self.center_mode = center_mode
        self.visibility_threshold = float(visibility_threshold)
        self.num_classes = num_classes
        self.expected_fields = 1 + 4 + self.kpt_count * self.kpt_dim

    def matches(self, n_fields: int) -> bool:
        return n_fields == self.expected_fields

    def parse(self, tokens: np.ndarray) -> Optional[Detection]:
        class_id = self._parse_class_id(tokens[0], self.num_classes)
        if class_id is None:
            return None
        try:
            cx = self._check_coord(float(tokens[1]))
            cy = self._check_coord(float(tokens[2]))
            w = self._size(float(tokens[3]))
            h = self._size(float(tokens[4]))
            keypoints = (
                tokens[5:].reshape(self.kpt_count, self.kpt_dim).astype(np.float32)
            )
        except ValueError:
            return None

        if self.center_mode == "keypoints":
            if self.kpt_dim == 3:
                visible = keypoints[:, 2] >= self.visibility_threshold
            else:
                visible = np.isfinite(keypoints[:, 0]) & np.isfinite(keypoints[:, 1])
            pts = keypoints[visible, :2]
            if pts.size:
                cx = float(np.clip(float(pts[:, 0].mean()), 0.0, 1.0))
                cy = float(np.clip(float(pts[:, 1].mean()), 0.0, 1.0))

        return Detection(class_id, cx, cy, w, h, self.name, keypoints)


class YoloParserFactory:
    """按 token 数量把标注行分发给对应策略，并缓存解析结果统计。

    注册顺序即优先级：pose → obb → detect → seg（pose 必须最先）。
    """

    def __init__(self, parsers: Sequence[BaseParser]) -> None:
        if not parsers:
            raise ValueError("至少需要一个解析策略")
        self._parsers: List[BaseParser] = list(parsers)
        self._resolve_cache: Dict[int, Optional[BaseParser]] = {}
        self._format_counter: Counter = Counter()

    @classmethod
    def default(
        cls,
        kpt_shape: Tuple[int, int] = (20, 3),
        pose_center_mode: str = "bbox",
        kpt_visibility_threshold: float = KPT_VISIBILITY_THRESHOLD,
        num_classes: Optional[int] = None,
    ) -> "YoloParserFactory":
        """构建默认工厂；``num_classes`` 传入后会丢弃类别越界的行。"""
        return cls(
            [
                PoseParser(
                    kpt_shape, pose_center_mode, kpt_visibility_threshold, num_classes
                ),
                ObbParser(num_classes),
                DetectParser(num_classes),
                SegParser(num_classes),
            ]
        )

    def resolve(self, n_fields: int) -> Optional[BaseParser]:
        """根据 token 数量选取策略（带缓存，避免重复线性扫描）。"""
        if n_fields not in self._resolve_cache:
            self._resolve_cache[n_fields] = next(
                (p for p in self._parsers if p.matches(n_fields)), None
            )
        return self._resolve_cache[n_fields]

    def parse_line(self, line: str) -> Optional[Detection]:
        """解析单行文本，非法行返回 ``None``。"""
        tokens = _tokenize(line)
        if tokens is None:
            return None
        parser = self.resolve(tokens.size)
        if parser is None:
            return None
        detection = parser.parse(tokens)
        if detection is not None:
            self._format_counter[detection.source] += 1
        return detection

    def parse_file(self, path: Path) -> Tuple[List[Detection], int]:
        """解析整个标注文件。

        Args:
            path: 标注文件路径。

        Returns:
            ``(检测列表, 被跳过行数)``。文件缺失/编码异常等 IO 问题在此兜底，
            不向上抛出，避免单个坏文件中断整批数据构建。
        """
        detections: List[Detection] = []
        skipped = 0
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            LOGGER.warning("读取标注文件失败 %s: %s", path, exc)
            return detections, 1

        for raw in text.splitlines():
            if not raw.strip():
                continue
            detection = self.parse_line(raw)
            if detection is None:
                skipped += 1
            else:
                detections.append(detection)
        return detections, skipped

    @property
    def format_counter(self) -> Dict[str, int]:
        """各格式成功解析的目标数量，便于确认数据实际是哪种任务类型。"""
        return dict(self._format_counter)


# --------------------------------------------------------------------------- #
# 帧级聚合与特征
# --------------------------------------------------------------------------- #
def aggregate_detections(
    detections: Sequence[Detection], num_classes: int
) -> np.ndarray:
    """把一帧内的目标压缩为 ``(num_classes, 5)`` 槽位矩阵。

    同类多目标取坐标均值（用 ``np.add.at + bincount`` 向量化），无目标的类别槽位
    全 0 且 mask=0，从而保证每帧特征维度恒定，天然抵抗缺帧。

    Args:
        detections: 本帧全部检测。
        num_classes: 目标类别总数。

    Returns:
        形状 ``(C, 5)`` 的 float32 数组，列为 ``[cx, cy, w, h, mask]``。
    """
    out = np.zeros((num_classes, 5), dtype=np.float32)
    if not detections:
        return out

    classes = np.fromiter(
        (d.class_id for d in detections), dtype=np.int64, count=len(detections)
    )
    coords = np.fromiter(
        (v for d in detections for v in (d.cx, d.cy, d.width, d.height)),
        dtype=np.float32,
        count=len(detections) * 4,
    ).reshape(-1, 4)

    valid = (classes >= 0) & (classes < num_classes)
    if not valid.any():
        return out
    classes = classes[valid]
    coords = coords[valid]

    np.add.at(out[:, :4], classes, coords)  # 按类别累加
    counts = np.bincount(classes, minlength=num_classes).astype(np.float32)
    present = counts > 0
    out[present, :4] /= counts[present, None]  # 按类别求均值
    out[:, 4] = counts
    return out


class FrameFeatureAssembler:
    """单帧 -> 特征向量的布局生成器（训练序列版与在线流式版共用同一布局）。

    布局顺序（长度 ``C * (2 + 2 + 1 + 2)``，缺省配置下 C=4 → 28 维）::

        [cx_c0..cx_c3, cy_c0..cy_c3, w_c0..w_c3, h_c0..h_c3, mask_c0..mask_c3,
         dx_c0..dx_c3, dy_c0..dy_c3]
    """

    def __init__(
        self,
        num_classes: int,
        include_size: bool = True,
        include_velocity: bool = True,
    ) -> None:
        if num_classes < 1:
            raise ValueError("num_classes 必须 >= 1")
        self.num_classes = int(num_classes)
        self.include_size = include_size
        self.include_velocity = include_velocity
        self._prev_positions: Optional[np.ndarray] = None

    @property
    def feature_dim(self) -> int:
        """单帧特征维度。"""
        per_class = (
            2
            + (2 if self.include_size else 0)
            + 1
            + (2 if self.include_velocity else 0)
        )
        return self.num_classes * per_class

    @property
    def feature_names(self) -> List[str]:
        """特征列名，便于可解释性排查（例如观察某列长期为 0）。"""
        names: List[str] = []
        for prefix in ("cx", "cy"):
            names += [f"{prefix}_{i}" for i in range(self.num_classes)]
        if self.include_size:
            for prefix in ("w", "h"):
                names += [f"{prefix}_{i}" for i in range(self.num_classes)]
        names += [f"mask_{i}" for i in range(self.num_classes)]
        if self.include_velocity:
            for prefix in ("dx", "dy"):
                names += [f"{prefix}_{i}" for i in range(self.num_classes)]
        return names

    def transform_sequence(self, frames: np.ndarray) -> np.ndarray:
        """把 ``(T, C, 5)`` 帧矩阵一次性转成 ``(T, F)`` 特征序列（向量化）。

        Args:
            frames: 形状 ``(T, C, 5)``，最后一维为 ``[cx, cy, w, h, mask]``。

        Returns:
            ``(T, F)`` float32 特征矩阵。速度为一阶差分且首帧置 0（因果、无未来信息）。
        """
        if frames.ndim != 3 or frames.shape[2] != 5:
            raise ValueError(f"frames 需要 (T, C, 5)，收到 {frames.shape}")
        n_frames = frames.shape[0]

        positions = frames[:, :, :2].reshape(n_frames, -1).astype(np.float32)
        parts = [positions]
        if self.include_size:
            parts.append(frames[:, :, 2:4].reshape(n_frames, -1).astype(np.float32))
        parts.append(frames[:, :, 4].astype(np.float32))
        if self.include_velocity:
            velocity = np.zeros_like(positions)
            if n_frames > 1:
                velocity[1:] = positions[1:] - positions[:-1]
            parts.append(velocity)

        features = np.concatenate(parts, axis=1).astype(np.float32)
        expected = self.feature_dim
        if features.shape[1] != expected:
            raise RuntimeError(f"特征维度异常: {features.shape[1]} != {expected}")
        return features

    def step(self, frame_matrix: np.ndarray) -> np.ndarray:
        """流式处理单帧，内部记忆上一帧位置以计算速度。

        Args:
            frame_matrix: 形状 ``(C, 5)`` 的当前帧槽位矩阵。

        Returns:
            ``(F,)`` 特征向量，与 :meth:`transform_sequence` 的行完全一致。
        """
        if frame_matrix.shape != (self.num_classes, 5):
            raise ValueError(
                f"frame_matrix 需要 ({self.num_classes}, 5)，收到 {frame_matrix.shape}"
            )
        positions = frame_matrix[:, :2].reshape(-1).astype(np.float32)
        parts = [positions]
        if self.include_size:
            parts.append(frame_matrix[:, 2:4].reshape(-1).astype(np.float32))
        parts.append(frame_matrix[:, 4].astype(np.float32))
        if self.include_velocity:
            if self._prev_positions is None:
                velocity = np.zeros_like(positions)
            else:
                velocity = positions - self._prev_positions
            parts.append(velocity)
        self._prev_positions = positions
        return np.concatenate(parts).astype(np.float32)

    def reset(self) -> None:
        """清空速度记忆，用于新视频段开始。"""
        self._prev_positions = None


def interpolate_missing(
    frames: np.ndarray, max_gap: int = 5, strategy: str = "interpolate"
) -> Tuple[np.ndarray, Dict[str, float]]:
    """对缺失目标做**因果前向填充**（函数名保留历史叫法）。

    历史实现用 ``np.interp`` 做双向线性插值，会用到未来帧：既破坏因果性，
    也无法在在线推流时复现同样的数值，造成训练/推理特征分布不一致。
    现在的规则只依赖当前帧与过去观测：

    - 某类本帧缺失、且最近一次观测在 ``max_gap`` 帧以内 → 沿用该次观测的
      ``[cx, cy, w, h]``，并把 ``mask`` 置 1.0；
    - 序列开头（尚无观测）或超过 ``max_gap`` 的缺失保持全 0；
    - ``strategy="zero"`` 时完全不填充。

    流式实现见 :class:`CausalMissingFiller`，两者必须逐帧等价。

    Args:
        frames: ``(T, C, 5)`` 帧矩阵。
        max_gap: 允许前向填充的最大缺失帧数。
        strategy: ``interpolate`` 或 ``zero``。

    Returns:
        ``(填充后的矩阵, 统计信息)``，统计里含被填充的槽位比例。
    """
    if frames.ndim != 3 or frames.shape[2] != 5:
        raise ValueError(f"frames 需要 (T, C, 5)，收到 {frames.shape}")
    if strategy not in ("interpolate", "zero"):
        raise ValueError(f"未知 missing_strategy: {strategy}")
    out = frames.astype(np.float32, copy=True)
    n_frames, n_classes, _ = out.shape
    if strategy == "zero" or max_gap <= 0 or n_frames == 0:
        return out, {"interpolated_ratio": 0.0}

    index = np.arange(n_frames)
    filled_total = 0
    for class_idx in range(n_classes):
        present = out[:, class_idx, 4] > 0
        if not present.any() or present.all():
            continue

        # 最近一次观测的下标（-1 表示此前从未出现）；age = 距该次观测的帧数
        last_seen = np.maximum.accumulate(np.where(present, index, -1))
        age = index - last_seen
        accept = (~present) & (last_seen >= 0) & (age <= max_gap)
        if not accept.any():
            continue

        out[accept, class_idx, :4] = out[last_seen[accept], class_idx, :4]
        out[accept, class_idx, 4] = 1.0
        filled_total += int(accept.sum())

    ratio = filled_total / max(n_frames * n_classes, 1)
    return out, {"interpolated_ratio": float(ratio)}


class CausalMissingFiller:
    """流式缺失填充器：与 :func:`interpolate_missing` 的因果规则逐帧等价。

    推理端必须使用本类（而不是批量实现），否则同一条序列会被填充成不同的
    特征值。内部只保存每个类别的「最近一次观测」与缺失帧数，内存 ``O(C)``。
    """

    def __init__(
        self, num_classes: int, max_gap: int = 5, strategy: str = "interpolate"
    ) -> None:
        if num_classes < 1:
            raise ValueError("num_classes 必须 >= 1")
        if strategy not in ("interpolate", "zero"):
            raise ValueError(f"未知 missing_strategy: {strategy}")
        self.num_classes = int(num_classes)
        self.max_gap = int(max_gap)
        self.strategy = strategy
        self.reset()

    def reset(self) -> None:
        """新视频段开始时清空“最近一次观测”。"""
        self._last = np.zeros((self.num_classes, 4), dtype=np.float32)
        self._age = np.zeros(self.num_classes, dtype=np.int64)
        self._seen = np.zeros(self.num_classes, dtype=bool)

    def step(self, frame_matrix: np.ndarray) -> np.ndarray:
        """填充单帧并推进状态，返回新的 ``(C, 5)``（不修改入参）。"""
        matrix = np.asarray(frame_matrix, dtype=np.float32)
        if matrix.shape != (self.num_classes, 5):
            raise ValueError(
                f"frame_matrix 需要 ({self.num_classes}, 5)，收到 {matrix.shape}"
            )
        out = matrix.copy()
        if self.strategy == "zero" or self.max_gap <= 0:
            return out

        present = matrix[:, 4] > 0
        self._last[present] = matrix[present, :4]
        self._age = np.where(present, 0, self._age + 1)
        self._seen |= present

        fill = (~present) & self._seen & (self._age <= self.max_gap)
        out[fill, :4] = self._last[fill]
        out[fill, 4] = 1.0
        return out


# --------------------------------------------------------------------------- #
# 标签策略
# --------------------------------------------------------------------------- #
class BaseLabelProvider(ABC):
    """帧级标签来源策略接口。

    返回 ``None`` 表示该帧无标签，对应窗口会被丢弃（而不是乱猜一个类别）。
    """

    name: str = "base"
    num_labels: Optional[int] = None

    @abstractmethod
    def label_for(self, frame_id: str, frame_matrix: np.ndarray) -> Optional[int]:
        """返回帧标签；无标签返回 ``None``。"""


class CsvLabelProvider(BaseLabelProvider):
    """从 CSV 读取 ``帧名, 动作标签``（推荐用于真实动作标注）。"""

    name = "csv"

    def __init__(self, csv_path: Path) -> None:
        self._labels: Dict[str, int] = {}
        path = Path(csv_path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) < 2:
                    continue
                key = Path(row[0].strip()).stem.lower()
                if key in ("frame", "frame_id", "filename", "name"):
                    continue
                try:
                    label = int(float(row[1]))
                except ValueError:
                    continue
                if label >= 0:
                    self._labels[key] = label
        if not self._labels:
            raise ValueError(f"标签文件为空或格式不正确: {path}")
        self.num_labels = max(self._labels.values()) + 1

    def label_for(self, frame_id: str, frame_matrix: np.ndarray) -> Optional[int]:
        return self._labels.get(Path(frame_id).stem.lower())


class FilenameRuleLabelProvider(BaseLabelProvider):
    """按文件名关键字/正则生成标签，例：``{"pick": 0, "press": 1, "place": 2}``。

    规则值支持 ``"re:模式"`` 前缀走正则；命中顺序为字典插入顺序。
    """

    name = "filename"

    def __init__(
        self,
        rules: Dict[str, int],
        default_label: Optional[int] = None,
        case_sensitive: bool = False,
    ) -> None:
        if not rules and default_label is None:
            raise ValueError("filename 标签策略需要 rules 或 default_label")
        self._rules = list(rules.items())
        self.default_label = default_label
        self.case_sensitive = case_sensitive
        candidates = list(rules.values()) + (
            [] if default_label is None else [default_label]
        )
        self.num_labels = max(candidates) + 1

    def label_for(self, frame_id: str, frame_matrix: np.ndarray) -> Optional[int]:
        haystack = (
            Path(frame_id).stem if self.case_sensitive else Path(frame_id).stem.lower()
        )
        for keyword, label in self._rules:
            needle = keyword if self.case_sensitive else keyword.lower()
            if needle.startswith("re:"):
                flags = 0 if self.case_sensitive else re.IGNORECASE
                if re.search(keyword[3:], haystack, flags):
                    return label
            elif needle in haystack:
                return label
        return self.default_label


class DominantClassLabelProvider(BaseLabelProvider):
    """以帧内数量最多的目标类别作为伪标签（无动作标注时的降级方案）。

    注意：这只是「手部状态」而非真正的「动作语义」，用于跑通全链路与自测，
    正式训练务必换成 :class:`CsvLabelProvider` 或 :class:`FilenameRuleLabelProvider`。
    """

    name = "dominant_class"

    def __init__(self, num_classes: int) -> None:
        self.num_labels = int(num_classes)

    def label_for(self, frame_id: str, frame_matrix: np.ndarray) -> Optional[int]:
        counts = frame_matrix[:, 4]
        if float(counts.sum()) <= 0.0:
            return None
        return int(np.argmax(counts))


def build_label_provider(cfg: DataConfig) -> BaseLabelProvider:
    """按配置解析标签来源，``auto`` 优先级：CSV > 文件名规则 > 主导类别伪标签。"""
    source = cfg.label_source.lower()

    if source in ("auto", "csv") and cfg.label_csv is not None:
        csv_path = Path(cfg.label_csv)
        if csv_path.exists():
            return CsvLabelProvider(csv_path)
        if source == "csv":
            raise FileNotFoundError(f"指定的标签文件不存在: {csv_path}")

    if source in ("auto", "filename") and cfg.filename_rules:
        return FilenameRuleLabelProvider(cfg.filename_rules, cfg.default_label)
    if source == "filename":
        raise ValueError("label_source=filename 但 filename_rules 为空")

    LOGGER.warning(
        "未找到动作标签（label_csv / filename_rules 均未提供），"
        "回退到 dominant_class 伪标签；正式训练请补充真实动作标签。"
    )
    return DominantClassLabelProvider(cfg.num_target_classes)


# --------------------------------------------------------------------------- #
# 文件名排序与滑窗
# --------------------------------------------------------------------------- #
_DUP_PATTERN = re.compile(r"\(\s*(\d+)\s*\)\s*$")


def natural_frame_key(path: Path) -> Tuple[Tuple[int, ...], int, str]:
    """自然排序键：先按文件名中的数字序列，再按 ``(2)`` 副本编号，最后按名称。

    Windows 导出常出现 ``xxx (2).txt`` 与 ``xxx.txt`` 同秒副本，
    纯字符串排序会把 ``(2)`` 排到前面且破坏时间序，此处显式处理。
    """
    stem = path.stem
    dup_match = _DUP_PATTERN.search(stem)
    dup_index = int(dup_match.group(1)) if dup_match else 0
    cleaned = _DUP_PATTERN.sub("", stem)
    numbers = tuple(int(n) for n in re.findall(r"\d+", cleaned))
    return numbers, dup_index, cleaned.lower()


def session_key(name: str, pattern: Optional[str]) -> str:
    """从帧 ID / 文件名提取会话键（建库与标签回放共用同一规则）。

    未配置 ``pattern`` 时返回 ``"default"``（即不切段）；配置了但未命中时同样
    回落到 ``"default"``，与 :meth:`SequenceDatasetBuilder._session_of` 一致。
    """
    if not pattern:
        return "default"
    match = re.search(pattern, str(name))
    return match.group(0) if match else "default"


def chronological_split_indices(
    n_samples: int, val_ratio: float = 0.2, gap: int = 0
) -> Tuple[slice, slice]:
    """按时间顺序切分训练/验证（禁止随机切分：相邻滑窗高度重叠会造成泄漏）。

    Args:
        n_samples: 样本数。
        val_ratio: 验证集比例。
        gap: 训练集末尾丢弃的窗口数（embargo），用于隔离重叠窗口；真实数据建议设为
            ``window_size``，样本极少时可设 0。

    Returns:
        ``(train_slice, val_slice)``。
    """
    if n_samples <= 0:
        raise ValueError("样本数为 0，无法划分数据集")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio 非法: {val_ratio}")

    n_val = int(round(n_samples * val_ratio))
    if n_samples > 1:
        n_val = min(max(n_val, 1), n_samples - 1)
    else:
        n_val = 0

    train_end = n_samples - n_val
    if gap > 0 and train_end - gap >= 1:
        train_end -= int(gap)
    return slice(0, train_end), slice(n_samples - n_val, n_samples)


def make_causal_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int = 1,
    label_mode: str = "last",
    ignore_index: int = IGNORE_INDEX,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """切分因果滑动窗口。

    使用 ``np.lib.stride_tricks.sliding_window_view`` 零拷贝取窗，再一次性 contiguous
    拷贝，避免 ``for`` 循环拼接造成的内存碎片。

    Args:
        features: ``(T, F)`` 特征序列。
        labels: ``(T,)`` 帧标签，``ignore_index`` 表示无标签。
        window_size: 窗口长度。
        stride: 步长。
        label_mode: ``last`` 取窗口末帧标签；``majority`` 取窗口内多数标签。
        ignore_index: 无标签占位值。

    Returns:
        ``(X, y, end_indices)``，其中 ``X`` 形状 ``(N, F, W)`` 可直接喂给 1D 卷积。
    """
    if features.ndim != 2:
        raise ValueError(f"features 需要 (T, F)，收到 {features.shape}")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels 必须是一维且长度与 features 一致")

    n_frames, n_features = features.shape
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size / stride 必须为正整数")

    if n_frames < window_size:
        LOGGER.warning(
            "帧数 %d 少于窗口长度 %d，无法生成滑动窗口", n_frames, window_size
        )
        return (
            np.empty((0, n_features, window_size), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    end_idx = np.arange(window_size - 1, n_frames, stride, dtype=np.int64)
    end_idx = end_idx[labels[end_idx] != ignore_index]
    if end_idx.size == 0:
        return (
            np.empty((0, n_features, window_size), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    view = np.lib.stride_tricks.sliding_window_view(features, window_size, axis=0)
    start_idx = end_idx - (window_size - 1)
    windows = np.ascontiguousarray(view[start_idx], dtype=np.float32)

    if label_mode == "last":
        target = labels[end_idx].astype(np.int64)
    elif label_mode == "majority":
        n_labels = int(max(int(labels.max()), 0)) + 1
        onehot = np.zeros((n_frames, n_labels), dtype=np.int32)
        known = labels != ignore_index
        onehot[np.flatnonzero(known), labels[known]] = 1
        cumulative = np.cumsum(onehot, axis=0)
        before = np.where(start_idx > 0, cumulative[np.maximum(start_idx - 1, 0)], 0)
        counts = cumulative[end_idx] - before
        target = counts.argmax(axis=1).astype(np.int64)
    else:
        raise ValueError(f"未知 label_mode: {label_mode}")

    return windows, target, end_idx


# --------------------------------------------------------------------------- #
# 数据集构建
# --------------------------------------------------------------------------- #
@dataclass
class BuiltDataset:
    """构建完成的数据集容器。"""

    X: np.ndarray  # (N, F, W) float32
    y: np.ndarray  # (N,) int64
    frame_ids: np.ndarray  # (N,) 每个窗口末帧的文件名
    meta: Dict[str, Any]
    groups: Optional[np.ndarray] = None  # (N,) 每个窗口所属分组（数据集/视频段）


class SequenceDatasetBuilder:
    """把标注目录构建成因果滑窗数据集（解析 → 聚合 → 插值 → 特征 → 滑窗）。"""

    def __init__(self, cfg: DataConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self.factory = YoloParserFactory.default(
            kpt_shape=cfg.kpt_shape,
            pose_center_mode=cfg.pose_center_mode,
            kpt_visibility_threshold=cfg.kpt_visibility_threshold,
            num_classes=cfg.num_target_classes,
        )
        self.assembler = FrameFeatureAssembler(
            num_classes=cfg.num_target_classes,
            include_size=cfg.include_size,
            include_velocity=cfg.include_velocity,
        )
        self.provider = build_label_provider(cfg)

    # -- 内部工具 ---------------------------------------------------------
    def _list_files(self) -> List[Path]:
        label_dir = Path(self.cfg.label_dir)
        if not label_dir.exists():
            raise FileNotFoundError(f"标注目录不存在: {label_dir}")
        files = sorted(label_dir.glob("*.txt"), key=natural_frame_key)
        if not files:
            raise FileNotFoundError(f"标注目录下没有 .txt 文件: {label_dir}")
        return files

    def _session_of(self, path: Path) -> str:
        """提取视频段标识，保证滑窗不跨不同录像（规则与推理回放共用）。"""
        return session_key(path.stem, self.cfg.session_pattern)

    def _load_frames(self, paths: Sequence[Path]) -> Tuple[np.ndarray, List[str], int]:
        """批量读取一个视频段的帧，返回 ``(T, C, 5)`` 与跳过行数。"""
        frames = np.zeros(
            (len(paths), self.cfg.num_target_classes, 5), dtype=np.float32
        )
        frame_ids: List[str] = []
        skipped_total = 0
        for row, path in enumerate(paths):
            detections, skipped = self.factory.parse_file(path)
            skipped_total += skipped
            frames[row] = aggregate_detections(detections, self.cfg.num_target_classes)
            frame_ids.append(path.stem)
        return frames, frame_ids, skipped_total

    def _build_labels(self, frame_ids: Sequence[str], frames: np.ndarray) -> np.ndarray:
        """逐帧取标签，无法确定的帧填 ``IGNORE_INDEX``。"""
        labels = np.full(len(frame_ids), IGNORE_INDEX, dtype=np.int64)
        max_valid = self.provider.num_labels
        for i, frame_id in enumerate(frame_ids):
            label = self.provider.label_for(frame_id, frames[i])
            if label is None or label < 0:
                continue
            if max_valid is not None and label >= max_valid:
                LOGGER.debug("帧 %s 标签 %d 越界，已忽略", frame_id, label)
                continue
            labels[i] = int(label)
        return labels

    # -- 主流程 -----------------------------------------------------------
    def build(self) -> BuiltDataset:
        """构建整个数据集。

        Returns:
            :class:`BuiltDataset`，``X`` 形状 ``(N, F, W)``。

        Raises:
            FileNotFoundError: 目录/文件缺失。
            RuntimeError: 无任何可用滑窗（通常因帧数不足或标签全缺失）。
        """
        files = self._list_files()

        groups: Dict[str, List[Path]] = {}
        for path in files:
            groups.setdefault(self._session_of(path), []).append(path)

        x_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        id_parts: List[np.ndarray] = []
        group_parts: List[np.ndarray] = []
        group_counts: Dict[str, int] = {}
        skipped_lines = 0
        total_frames = 0
        interpolated_ratios: List[float] = []

        for session, paths in groups.items():
            frames, frame_ids, skipped = self._load_frames(paths)
            skipped_lines += skipped
            total_frames += frames.shape[0]

            if frames.shape[0] < self.cfg.window_size:
                LOGGER.warning(
                    "视频段 %s 仅 %d 帧，少于 window_size=%d，跳过",
                    session,
                    frames.shape[0],
                    self.cfg.window_size,
                )
                continue

            filled, stats = interpolate_missing(
                frames, self.cfg.max_gap, self.cfg.missing_strategy
            )
            if stats["interpolated_ratio"] > 0:
                interpolated_ratios.append(stats["interpolated_ratio"])

            features = self.assembler.transform_sequence(filled)
            # 标签必须基于「插值前」的原始帧：插值只是特征层面的补齐，
            # 若用插值后的 mask 统计会凭空造出目标，污染伪标签。
            labels = self._build_labels(frame_ids, frames)
            windows, targets, end_idx = make_causal_windows(
                features,
                labels,
                self.cfg.window_size,
                self.cfg.stride,
                self.cfg.label_mode,
            )
            if windows.shape[0] == 0:
                LOGGER.warning("视频段 %s 未生成有效窗口（标签缺失？）", session)
                continue

            x_parts.append(windows)
            y_parts.append(targets)
            id_parts.append(np.asarray(frame_ids, dtype=object)[end_idx])
            group_parts.append(np.full(windows.shape[0], session, dtype=object))
            group_counts[session] = int(windows.shape[0])

        if not x_parts:
            raise RuntimeError(  # noqa: TRY003
                "没有生成任何滑动窗口：请检查帧数是否 >= window_size，或标签来源是否可用。"
            )

        X = np.concatenate(x_parts, axis=0).astype(np.float32)
        y = np.concatenate(y_parts, axis=0).astype(np.int64)
        frame_ids_arr = np.concatenate(id_parts, axis=0)
        groups_arr = np.concatenate(group_parts, axis=0)

        inferred_labels = int(y.max()) + 1
        label_names = self._resolve_label_names(inferred_labels)
        distribution = {
            int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))
        }

        meta: Dict[str, Any] = {
            "num_windows": int(X.shape[0]),
            "num_frames": int(total_frames),
            "feature_dim": int(self.assembler.feature_dim),
            "window_size": int(self.cfg.window_size),
            "stride": int(self.cfg.stride),
            "num_classes": inferred_labels,
            "label_names": label_names,
            "label_source": self.provider.name,
            "label_distribution": distribution,
            "format_stats": self.factory.format_counter,
            "skipped_lines": int(skipped_lines),
            "mean_interpolated_ratio": (
                float(np.mean(interpolated_ratios)) if interpolated_ratios else 0.0
            ),
            "sessions": len(groups),
            "group_window_counts": group_counts,
            "feature_names": self.assembler.feature_names,
        }
        LOGGER.info(
            "数据集构建完成: 窗口 %d / 帧 %d / 特征 %d / 标签分布 %s / 解析格式 %s",
            meta["num_windows"],
            meta["num_frames"],
            meta["feature_dim"],
            distribution,
            self.factory.format_counter,
        )
        return BuiltDataset(
            X=X, y=y, frame_ids=frame_ids_arr, meta=meta, groups=groups_arr
        )

    def _resolve_label_names(self, num_labels: int) -> List[str]:
        """确定类别显示名。

        优先级：
        1. ``DataConfig.step_names``（由 ``tcn_config.json`` 经 ``build_dataset.py`` 注入）；
        2. ``dominant_class`` 伪标签 → 目标类别名；
        3. 内置动作名，不足时回退 ``class_i``。
        """
        custom = [str(name) for name in (self.cfg.step_names or ())]
        if custom:
            names = custom
        elif self.provider.name == "dominant_class":
            names = list(self.cfg.target_names)
        else:
            names = ["take_part", "press", "place"]
            if num_labels <= len(self.cfg.target_names):
                names = names[:num_labels]
        return [names[i] if i < len(names) else f"class_{i}" for i in range(num_labels)]


def grouped_split_indices(
    groups: np.ndarray, val_ratio: float = 0.2, gap: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """按分组（数据集/视频段）分别做时间顺序切分，再拼接索引。

    多数据集合并时，若对全体样本按时间切分，样本量最大的数据集会独占验证集，
    其余数据集只剩训练样本，验证指标便失去意义。这里让每个分组各自按
    ``val_ratio`` 贡献验证样本，既保留时序性，也保证每个数据集都被验证。
    """
    buckets: Dict[str, List[int]] = {}
    for idx, key in enumerate(groups):
        buckets.setdefault(str(key), []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    for idxs in buckets.values():
        arr = np.asarray(idxs, dtype=np.int64)
        g_train, g_val = chronological_split_indices(arr.size, val_ratio, gap)
        train_idx.extend(arr[g_train].tolist())
        val_idx.extend(arr[g_val].tolist())
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def split_dataset(
    dataset: BuiltDataset, val_ratio: float = 0.2, gap: int = 0
) -> Tuple[np.ndarray, np.ndarray, str]:
    """返回 ``(train_idx, val_idx, split_mode)``，**导出与训练必须共用本函数**。

    历史问题：``train.py`` 曾自己调 ``chronological_split_indices`` 重新切分，
    导致训练用的验证集与写入磁盘的 ``X_val.npy`` 不是同一批样本
    （多数据集时还会使“每个数据集都有验证样本”的保证失效）。

    Returns:
        ``train_idx`` / ``val_idx`` 为窗口下标数组，``split_mode`` 为
        ``grouped``（多分组各自切分）或 ``chronological``（单组）。
    """
    groups = dataset.groups
    if groups is not None and len({str(g) for g in groups}) > 1:
        train_idx, val_idx = grouped_split_indices(groups, val_ratio, gap)
        return train_idx, val_idx, "grouped"

    n_samples = int(dataset.X.shape[0])
    train_slice, val_slice = chronological_split_indices(n_samples, val_ratio, gap)
    return (
        np.arange(0, train_slice.stop, dtype=np.int64),
        np.arange(val_slice.start, n_samples, dtype=np.int64),
        "chronological",
    )


def export_dataset(
    dataset: BuiltDataset,
    cfg: DataConfig,
    val_ratio: float = 0.2,
    gap: int = 0,
) -> Dict[str, Any]:
    """把数据集切分并落盘为 ``.npy`` 与元信息 JSON。

    Args:
        dataset: :meth:`SequenceDatasetBuilder.build` 的产物。
        cfg: 数据配置（用于定位 ``artifact_dir``）。
        val_ratio: 验证集比例。
        gap: 训练/验证之间的隔离窗口数。

    Returns:
        写入磁盘的元信息字典。
    """
    out_dir = Path(cfg.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx, split_mode = split_dataset(dataset, val_ratio, gap)

    X_train, y_train = dataset.X[train_idx], dataset.y[train_idx]
    X_val, y_val = dataset.X[val_idx], dataset.y[val_idx]
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "y_val.npy", y_val)

    meta = dict(dataset.meta)
    meta.update(
        {
            "num_train": int(X_train.shape[0]),
            "num_val": int(X_val.shape[0]),
            "val_ratio": float(val_ratio),
            "gap": int(gap),
            "split_mode": split_mode,
            "train_label_distribution": {
                int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))
            },
            "val_label_distribution": {
                int(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))
            },
        }
    )
    (out_dir / "dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return meta


def load_cached_meta(artifact_dir: Path) -> Dict[str, Any]:
    """读取已导出的 ``dataset_meta.json``（不存在时返回空字典）。"""
    path = Path(artifact_dir) / "dataset_meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    """命令行入口：解析目录并导出训练/验证 npy，便于先验证数据链路。"""
    parser = argparse.ArgumentParser(description="YOLOv8 标注 -> 因果滑窗数据集")
    parser.add_argument("--label-dir", type=Path, default=DataConfig.label_dir)
    parser.add_argument("--artifact-dir", type=Path, default=DataConfig.artifact_dir)
    parser.add_argument("--window", type=int, default=DataConfig.window_size)
    parser.add_argument("--stride", type=int, default=DataConfig.stride)
    parser.add_argument("--pose-center", choices=["bbox", "keypoints"], default="bbox")
    parser.add_argument(
        "--missing", choices=["interpolate", "zero"], default="interpolate"
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--label-source",
        default="auto",
        choices=["auto", "csv", "filename", "dominant_class"],
    )
    parser.add_argument("--label-csv", type=Path, default=None)
    parser.add_argument(
        "--session-pattern", default=None, help="例: \\d{8} 按日期分段，防止滑窗跨视频"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    cfg = DataConfig(
        label_dir=args.label_dir,
        artifact_dir=args.artifact_dir,
        window_size=args.window,
        stride=args.stride,
        pose_center_mode=args.pose_center,
        missing_strategy=args.missing,
        label_source=args.label_source,
        label_csv=args.label_csv,
        session_pattern=args.session_pattern,
    )
    dataset = SequenceDatasetBuilder(cfg).build()
    meta = export_dataset(dataset, cfg, args.val_ratio)
    print(json.dumps(meta, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
