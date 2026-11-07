"""训练前置校验的回归测试（T1 / T3 / T4）。"""

from __future__ import annotations

import numpy as np
import pytest

from config import DataConfig
from train import (
    assert_trainable_classes,
    check_class_coverage,
    validate_cached_dataset,
)


def test_assert_trainable_classes_rejects_single_class():
    with pytest.raises(SystemExit) as excinfo:
        assert_trainable_classes(1)
    assert "至少需要 2 个步骤" in str(excinfo.value)
    assert_trainable_classes(2)  # 不抛异常


def test_check_class_coverage_rejects_val_only_class():
    with pytest.raises(SystemExit):
        check_class_coverage(np.array([0, 0, 1]), np.array([1, 2]))
    # 只出现在训练集：仅告警，不阻断
    check_class_coverage(np.array([0, 1]), np.array([1]))


def test_validate_cached_dataset_rejects_stale_window_and_classes():
    cfg = DataConfig(window_size=8, num_target_classes=6)
    X = np.zeros((3, 42, 8), dtype=np.float32)
    meta = {"window_size": 8, "num_classes": 2, "label_names": ["a", "b"]}

    validate_cached_dataset(X, X, meta, cfg, ("a", "b"))

    with pytest.raises(SystemExit):
        validate_cached_dataset(X, X, {**meta, "window_size": 4}, cfg, ("a", "b"))
    with pytest.raises(SystemExit):
        validate_cached_dataset(X, X, {**meta, "num_classes": 3}, cfg, ("a", "b"))
    with pytest.raises(SystemExit):
        validate_cached_dataset(
            np.zeros((3, 35, 8), dtype=np.float32), X, meta, cfg, ("a", "b")
        )
