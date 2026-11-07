"""模型验证来源（Tab4）的路径守卫与数据集判定回归测试。

覆盖：
- ``paths.read_only_path`` 允许工作区外的图片路径（只读）；
- ``validate_service._resolve_dataset_name`` 的判定顺序：
  显式 > 按路径归属 > 按目录名匹配 > 唯一数据集 > 报错。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from web.services import validate_service
from web.services.errors import ApiError
from web.services.paths import read_only_path


def _stub_config(monkeypatch, *datasets) -> None:
    payload = {
        "source": {
            "datasets": [{"name": name, "path": str(path)} for name, path in datasets]
        }
    }
    monkeypatch.setattr(
        validate_service.config_store, "load_config", lambda raw=None: payload
    )


def test_read_only_path_allows_outside_project(tmp_path: Path):
    outside = tmp_path / "copied" / "hand" / "train" / "images"
    outside.mkdir(parents=True)
    image = outside / "frame_0.jpg"
    image.write_bytes(b"")

    assert read_only_path(str(image)) == image.resolve()
    assert read_only_path(str(outside)) == outside.resolve()


def test_read_only_path_rejects_missing(tmp_path: Path):
    with pytest.raises(ApiError) as exc:
        read_only_path(str(tmp_path / "nope"))
    assert exc.value.code == "E_PATH_NOT_FOUND"


def test_resolve_dataset_by_path(tmp_path: Path, monkeypatch):
    hand = tmp_path / "yolo_datasets" / "hand"
    (hand / "train" / "images").mkdir(parents=True)
    _stub_config(monkeypatch, ("hand", hand), ("sitting_pose", tmp_path / "sitting"))

    assert validate_service._resolve_dataset_name({}, hand / "train" / "images") == "hand"


def test_resolve_dataset_by_dir_name_outside_project(tmp_path: Path, monkeypatch):
    """图片被复制到项目外时，按目录名匹配数据集名。"""
    hand = tmp_path / "yolo_datasets" / "hand"
    hand.mkdir(parents=True)
    _stub_config(monkeypatch, ("hand", hand), ("sitting_pose", tmp_path / "sitting"))

    copied = tmp_path / "elsewhere" / "hand" / "train" / "images"
    copied.mkdir(parents=True)
    assert validate_service._resolve_dataset_name({}, copied / "a.jpg") == "hand"


def test_resolve_dataset_explicit_wins(tmp_path: Path, monkeypatch):
    _stub_config(monkeypatch, ("hand", tmp_path / "hand"), ("sitting", tmp_path / "sit"))
    assert validate_service._resolve_dataset_name({"dataset": "sitting"}, tmp_path) == "sitting"


def test_resolve_dataset_single_dataset_fallback(tmp_path: Path, monkeypatch):
    hand = tmp_path / "yolo_datasets" / "hand"
    hand.mkdir(parents=True)
    _stub_config(monkeypatch, ("hand", hand))

    assert validate_service._resolve_dataset_name({}, tmp_path / "unknown") == "hand"


def test_resolve_dataset_unresolved(tmp_path: Path, monkeypatch):
    _stub_config(monkeypatch, ("hand", tmp_path / "hand"), ("sitting", tmp_path / "sit"))

    with pytest.raises(ApiError) as exc:
        validate_service._resolve_dataset_name({}, tmp_path / "unknown")
    assert exc.value.code == "E_DATASET_UNRESOLVED"
