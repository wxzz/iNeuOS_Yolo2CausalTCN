"""解析器 / 特征 / 滑窗的单元测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_processor import (
    CausalMissingFiller,
    Detection,
    FrameFeatureAssembler,
    ObbParser,
    SegParser,
    YoloParserFactory,
    aggregate_detections,
    interpolate_missing,
    make_causal_windows,
    natural_frame_key,
    polygon_centroid,
)
from detectors import detect_many_with_yolo, list_image_files


def test_detect_line_is_parsed_as_center():
    factory = YoloParserFactory.default()
    det = factory.parse_line("2 0.5 0.25 0.1 0.2")
    assert det is not None and det.source == "detect"
    assert det.class_id == 2
    assert det.cx == pytest.approx(0.5) and det.cy == pytest.approx(0.25)


def test_obb_center_of_square():
    det = ObbParser().parse(np.array([0, 0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8]))
    assert det is not None
    assert det.cx == pytest.approx(0.5, abs=1e-6)
    assert det.cy == pytest.approx(0.5, abs=1e-6)


def test_seg_centroid_of_triangle():
    det = SegParser().parse(np.array([1, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]))
    assert det is not None
    assert det.cx == pytest.approx(1 / 3, abs=1e-6)
    assert det.cy == pytest.approx(1 / 3, abs=1e-6)


def test_pose_is_not_misrouted_to_seg():
    """65 个数值同时满足 seg 判据，工厂必须优先返回 pose。"""
    factory = YoloParserFactory.default(kpt_shape=(20, 3))
    tokens = "3 0.5 0.5 0.2 0.2 " + " ".join(["0.5 0.5 1"] * 20)
    det = factory.parse_line(tokens)
    assert det is not None and det.source == "pose"
    assert det.keypoints.shape == (20, 3)


def test_pose_keypoint_center_mode():
    factory = YoloParserFactory.default(kpt_shape=(2, 3), pose_center_mode="keypoints")
    det = factory.parse_line("0 0.5 0.5 0.2 0.2 0.2 0.2 1 0.8 0.8 1")
    assert det is not None
    assert det.cx == pytest.approx(0.5, abs=1e-6)
    assert det.cy == pytest.approx(0.5, abs=1e-6)


def test_dirty_lines_return_none():
    factory = YoloParserFactory.default()
    for line in ["", "# comment", "a b c", "1 0.5 0.5", "1 0.5 nan 0.1 0.1"]:
        assert factory.parse_line(line) is None


def test_aggregate_averages_same_class_and_flags_missing():
    dets = [Detection(0, 0.2, 0.2), Detection(0, 0.4, 0.4)]
    out = aggregate_detections(dets, num_classes=3)
    assert out[0, 0] == pytest.approx(0.3)
    assert out[0, 1] == pytest.approx(0.3)
    assert out[0, 4] == pytest.approx(2.0)
    assert out[1, 4] == 0.0 and out[2, 4] == 0.0


def test_causal_fill_limited_by_max_gap():
    frames = np.zeros((6, 1, 5), dtype=np.float32)
    frames[0, 0] = [0.0, 0.0, 0.1, 0.1, 1]
    frames[5, 0] = [1.0, 1.0, 0.1, 0.1, 1]

    filled, stats = interpolate_missing(frames, max_gap=2, strategy="interpolate")
    # 只允许向前填充 2 帧，且必须用“最近一次观测”（而不是未来观测）
    assert filled[1, 0, 4] == 1.0 and filled[2, 0, 4] == 1.0
    assert filled[3:5, 0, 4].sum() == 0
    assert filled[2, 0, 0] == pytest.approx(0.0)
    assert stats["interpolated_ratio"] == pytest.approx(2 / 6)

    filled2, _ = interpolate_missing(frames, max_gap=5, strategy="interpolate")
    assert filled2[4, 0, 4] == 1.0
    # 第 5 帧的未来观测不能影响第 1~4 帧的填充值
    assert filled2[1:5, 0, 0].max() == pytest.approx(0.0)

    filled3, _ = interpolate_missing(frames, max_gap=5, strategy="zero")
    assert filled3[2, 0, 4] == 0.0


def test_causal_fill_skips_leading_gap():
    frames = np.zeros((4, 1, 5), dtype=np.float32)
    frames[2, 0] = [0.3, 0.4, 0.1, 0.1, 1]

    filled, _ = interpolate_missing(frames, max_gap=5, strategy="interpolate")
    # 首次观测之前的缺失无法因果填充
    assert filled[0, 0, 4] == 0.0 and filled[1, 0, 4] == 0.0
    assert filled[3, 0, 4] == 1.0
    assert filled[3, 0, 0] == pytest.approx(0.3)


def test_causal_fill_stream_matches_batch():
    rng = np.random.default_rng(7)
    frames = rng.random((12, 3, 5)).astype(np.float32)
    frames[..., 4] = (frames[..., 4] > 0.45).astype(np.float32)
    frames[0, 1, 4] = 0.0  # 类别 1 以缺失开头

    batch, _ = interpolate_missing(frames, max_gap=3, strategy="interpolate")
    filler = CausalMissingFiller(3, max_gap=3, strategy="interpolate")
    stream = np.stack([filler.step(frames[index]) for index in range(len(frames))])
    np.testing.assert_allclose(batch, stream, rtol=1e-6, atol=1e-6)


def test_list_image_files_filters_and_sorts(tmp_path):
    (tmp_path / "b10.jpg").write_bytes(b"x")
    (tmp_path / "b2.png").write_bytes(b"x")
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")

    assert [path.name for path in list_image_files(tmp_path)] == [
        "b2.png",
        "b10.jpg",
    ]
    assert list_image_files(tmp_path / "missing") == []


class _FakeArray:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _FakeBoxes:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    @property
    def xywhn(self):
        return _FakeArray([[0.5, 0.5, 0.2, 0.2] for _ in self.rows])

    @property
    def cls(self):
        return _FakeArray(self.rows)


class _FakeResult:
    keypoints = None

    def __init__(self, rows):
        self.boxes = _FakeBoxes(rows) if rows else None


class _FakeModel:
    names = {0: "a"}

    def __init__(self):
        self.calls = []

    def predict(self, paths, **kwargs):
        if isinstance(paths, str):
            paths = [paths]
        self.calls.append(len(paths))
        return [_FakeResult([0]) for _ in paths]


def test_detect_many_with_yolo_batches_and_preserves_order():
    model = _FakeModel()
    results = detect_many_with_yolo(model, ["a.jpg", "b.jpg", "c.jpg"], batch=2)

    assert model.calls == [2, 1]
    assert len(results) == 3
    assert all(len(items) == 1 and items[0].class_id == 0 for items in results)


def test_stream_and_sequence_features_match():
    rng = np.random.default_rng(0)
    frames = rng.random((10, 4, 5)).astype(np.float32)
    frames[..., 4] = (frames[..., 4] > 0.5).astype(np.float32)

    seq = FrameFeatureAssembler(4).transform_sequence(frames)
    stream_assembler = FrameFeatureAssembler(4)
    stream = np.stack([stream_assembler.step(frames[i]) for i in range(10)])
    np.testing.assert_allclose(seq, stream, rtol=1e-6, atol=1e-6)


def test_windows_are_causal_and_labelled_by_last_frame():
    features = np.arange(12, dtype=np.float32).reshape(12, 1)
    labels = np.array([0] * 6 + [1] * 6, dtype=np.int64)
    X, y, ends = make_causal_windows(features, labels, window_size=4, stride=1)
    assert X.shape == (9, 1, 4)
    np.testing.assert_allclose(X[0].ravel(), [0, 1, 2, 3])
    assert y[0] == labels[3]
    assert y[-1] == labels[ends[-1]] == 1


def test_windows_drop_ignored_labels():
    features = np.zeros((6, 2), dtype=np.float32)
    labels = np.array([-1, -1, -1, 0, 1, 2], dtype=np.int64)
    X, y, _ = make_causal_windows(features, labels, window_size=3, stride=1)
    assert X.shape[0] == 3 and y.tolist() == [0, 1, 2]


def test_natural_key_orders_duplicate_second_files():
    base = Path("WIN_20260903_10_26_29_Pro.txt")
    dup = Path("WIN_20260903_10_26_29_Pro (2).txt")
    later = Path("WIN_20260903_10_26_31_Pro.txt")
    assert natural_frame_key(base) < natural_frame_key(dup) < natural_frame_key(later)


def test_polygon_centroid_degenerate_falls_back_to_mean():
    cx, cy = polygon_centroid(np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]))
    assert cx == pytest.approx(0.5) and cy == pytest.approx(0.5)


def test_parser_rejects_out_of_range_class():
    factory = YoloParserFactory.default(num_classes=4)
    assert factory.parse_line("9 0.5 0.5 0.1 0.1") is None
