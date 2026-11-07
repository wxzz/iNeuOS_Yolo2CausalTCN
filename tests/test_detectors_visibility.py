"""关键点可见性回归（BUG.MD C11）。

`detectors` 必须把 YOLO 的 ``keypoints.conf`` 写进 Detection 的第 3 列，
而不是恒填 1.0；否则 ``pose_center_mode="keypoints"`` 会把未检出的 (0,0)
点也算进中心。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_processor import PoseParser
from detectors import _result_to_detections


class _Boxes:
    """最小的 ultralytics Boxes 替身（只需要 xywhn / cls / len）。"""

    def __init__(self, xywhn, cls):
        self.xywhn = torch.tensor(xywhn, dtype=torch.float32)
        self.cls = torch.tensor(cls, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.cls.shape[0])


def _make_result(xyn, conf=None):
    # 真实 ultralytics 的 xyn/conf 形状都是 (N, K, 2) / (N, K)
    keypoints = SimpleNamespace(xyn=torch.tensor([xyn], dtype=torch.float32))
    if conf is not None:
        keypoints.conf = torch.tensor([conf], dtype=torch.float32)
    return SimpleNamespace(
        boxes=_Boxes([[0.5, 0.5, 0.2, 0.2]], [0.0]),
        keypoints=keypoints,
    )


def test_visibility_uses_keypoint_conf():
    result = _make_result([[0.2, 0.2], [0.0, 0.0]], conf=[0.9, 0.05])

    detections = _result_to_detections(result)

    assert len(detections) == 1
    kpts = np.asarray(detections[0].keypoints)
    assert kpts.shape == (2, 3)
    assert kpts[0, 2] == pytest.approx(0.9, abs=1e-6)
    assert kpts[1, 2] == pytest.approx(0.05, abs=1e-6)


def test_visibility_falls_back_when_conf_missing():
    result = _make_result([[0.2, 0.2], [0.3, 0.3]])

    detections = _result_to_detections(result)

    assert np.allclose(np.asarray(detections[0].keypoints)[:, 2], 1.0)


def test_visibility_falls_back_on_shape_mismatch():
    # conf 形状与 (N, K) 不一致时不能崩，退回"全部可见"
    result = _make_result([[0.2, 0.2], [0.3, 0.3]], conf=[[0.9, 0.8, 0.7]])

    detections = _result_to_detections(result)

    assert np.allclose(np.asarray(detections[0].keypoints)[:, 2], 1.0)


def test_pose_parser_ignores_low_confidence_keypoints():
    parser = PoseParser(kpt_shape=(2, 3), center_mode="keypoints")
    # 第 1 个点 (0.2,0.2) conf=0.9；第 2 个点 (0,0) conf=0.05 应被排除
    tokens = np.array([0, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.9, 0.0, 0.0, 0.05])

    det = parser.parse(tokens)

    assert det is not None
    assert det.cx == pytest.approx(0.2)
    assert det.cy == pytest.approx(0.2)


def test_pose_parser_falls_back_to_bbox_when_no_visible_point():
    parser = PoseParser(kpt_shape=(2, 3), center_mode="keypoints")
    tokens = np.array([0, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.1, 0.0, 0.0, 0.05])

    det = parser.parse(tokens)

    assert det is not None
    assert det.cx == pytest.approx(0.5)
    assert det.cy == pytest.approx(0.5)
