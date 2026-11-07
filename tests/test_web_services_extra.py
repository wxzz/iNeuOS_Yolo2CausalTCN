"""Web 服务层回归测试（第二批/第三批/第四批修复）。

覆盖：日志级别识别、任务锁、kpt_shape 校验、checkpoint 外部路径、
步骤名校验、单图预热估算、失败现场保留、轻量扫描、静态资源白名单、
checkpoint 兼容性、回放关键点形状。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from inference import build_frames_from_labels
from web.services import (
    config_store,
    dataset_service,
    paths,
    task_manager,
    validate_service,
)
from web.services.errors import ApiError
from web.services.task_manager import MANAGER, Task


# --------------------------------------------------------------------------- #
# 日志级别
# --------------------------------------------------------------------------- #
def test_level_of_recognizes_logging_format():
    assert (
        task_manager._level_of("2026-11-06 16:23:00,031 | ERROR   | __main__ | boom")
        == "error"
    )
    assert task_manager._level_of("WARNING | 缺少权重") == "warn"
    assert (
        task_manager._level_of("2026-11-06 16:23:00,031 | INFO    | __main__ | ok")
        == "info"
    )
    assert task_manager._level_of("Traceback (most recent call last):") == "error"


# --------------------------------------------------------------------------- #
# 任务锁
# --------------------------------------------------------------------------- #
def test_ensure_idle_blocks_when_task_running(monkeypatch):
    task = Task(kind="train", argv=[], label="训练模型", state="running")
    monkeypatch.setattr(MANAGER, "_task", task)

    with pytest.raises(ApiError) as excinfo:
        MANAGER.ensure_idle("保存配置")
    assert excinfo.value.code == "E_TASK_BUSY"
    assert "训练模型" in excinfo.value.message

    monkeypatch.setattr(MANAGER, "_task", None)
    MANAGER.ensure_idle("保存配置")  # 空闲时通过


# --------------------------------------------------------------------------- #
# 数据集校验
# --------------------------------------------------------------------------- #
def test_kpt_shape_issues():
    dataset = {"name": "hand", "task": "pose", "kpt_shape": [20, 3]}

    assert dataset_service.kpt_shape_issues(dataset, {"kpt_shape": [20, 3]}) == []
    assert dataset_service.kpt_shape_issues(dataset, {"kpt_shape": [17, 3]})
    assert dataset_service.kpt_shape_issues(dataset, {"kpt_shape": None}) == []
    assert (
        dataset_service.kpt_shape_issues({"task": "detect"}, {"kpt_shape": [1, 3]})
        == []
    )


def test_step_name_map_dedupes_duplicate_ids():
    cfg = {
        "steps": [
            {"id": 0, "name": "抓取"},
            {"id": 0, "name": "抓取"},
            {"id": 1, "name": "放下"},
        ]
    }
    assert config_store.step_name_map(cfg) == {0: "抓取", 1: "放下"}


def test_collect_without_stats_skips_directory_scan(tmp_path, monkeypatch):
    root = tmp_path / "root"
    dataset_dir = root / "hand"
    (dataset_dir / "train" / "images").mkdir(parents=True)
    (dataset_dir / "classes.txt").write_text("a", encoding="utf-8")

    monkeypatch.setattr(
        dataset_service, "user_path", lambda raw, must_exist=False: Path(raw)
    )
    monkeypatch.setattr(
        dataset_service.config_store,
        "load_config",
        lambda raw=None: {"source": {"datasets": []}, "class_aliases": {}},
    )
    monkeypatch.setattr(
        dataset_service.source_store,
        "load",
        lambda: {"extra_datasets": [], "excluded_datasets": [], "root": ""},
    )

    def _boom(_):
        raise AssertionError("with_stats=False 不应扫描划分目录")

    monkeypatch.setattr(dataset_service, "_split_stats", _boom)

    datasets = dataset_service.collect(root, with_stats=False)

    assert [item["name"] for item in datasets] == ["hand"]
    assert "splits" not in datasets[0]


def test_load_meta_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset_service, "data_path", lambda raw, must_exist=False: Path(raw)
    )
    result = dataset_service.load_meta(tmp_path)
    assert result["exists"] is False
    assert result["meta"] == {}


# --------------------------------------------------------------------------- #
# 路径与验证参数
# --------------------------------------------------------------------------- #
def test_checkpoint_path_allows_external_file(tmp_path):
    target = tmp_path / "model.pth"
    target.write_bytes(b"x")

    assert paths.checkpoint_path(target, must_exist=True) == target.resolve()
    with pytest.raises(ApiError):
        paths.checkpoint_path(tmp_path / "model.bin", must_exist=False)
    with pytest.raises(ApiError):
        paths.checkpoint_path(tmp_path / "missing.pth", must_exist=True)


def test_image_context_length(tmp_path):
    for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        (tmp_path / name).write_bytes(b"x")

    target = tmp_path / "c.jpg"
    assert validate_service._image_context_length(target, 8) == 3
    assert validate_service._image_context_length(target, 2) == 2


def test_validate_on_exit_keeps_evidence_on_failure(tmp_path):
    csv_path = tmp_path / "pred.csv"
    csv_path.write_text(
        "frame,step_id,step_name,confidence\nf,0,旧名字,0.9\n",
        encoding="utf-8-sig",
    )
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    task = Task(kind="validate", argv=[])
    task.scratch["_context"] = {
        "csv_path": str(csv_path),
        "work_dir": str(work_dir),
        "frame_count": 2,
        "window": 8,
        "warmup": 1,
        "expected": 1,
        "step_names": {0: "新名字"},
    }

    result = validate_service.on_exit(task, 1)
    assert csv_path.exists() and work_dir.exists()
    assert result["retained"]
    # 名称以当前配置为准
    assert result["predictions"][0]["step_name"] == "新名字"

    result_ok = validate_service.on_exit(task, 0)
    assert not csv_path.exists() and not work_dir.exists()
    assert result_ok["retained"] == []


# --------------------------------------------------------------------------- #
# 静态资源白名单
# --------------------------------------------------------------------------- #
def test_static_only_serves_frontend():
    from web_server import app

    client = app.test_client()
    assert client.get("/services/dataset_service.py").status_code == 404
    assert client.get("/js/main.js").status_code == 200
    assert client.get("/css/base.css").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/api/health").status_code == 200


# --------------------------------------------------------------------------- #
# checkpoint 兼容性
# --------------------------------------------------------------------------- #
def _write_checkpoint(path: Path, num_classes: int) -> None:
    torch.save(
        {
            "state_dict": {},
            "feature_dim": 42,
            "num_classes": num_classes,
            "window_size": 8,
            "class_names": [f"步骤{i}" for i in range(num_classes)],
        },
        path,
    )


def _config() -> dict:
    return {
        "dataset": {},
        "class_aliases": {},
        "source": {
            "datasets": [
                {
                    "name": "hand",
                    "classes": ["a", "b", "c", "d", "e", "f"],
                }
            ]
        },
        "steps": [
            {"id": 0, "name": "步骤0", "match": {"classes": ["a"]}},
            {"id": 1, "name": "步骤1", "match": {"classes": ["b"]}},
        ],
    }


def test_checkpoint_compat_rejects_class_count_mismatch(tmp_path, monkeypatch):
    import web_server

    checkpoint = tmp_path / "model.pth"
    _write_checkpoint(checkpoint, num_classes=3)
    monkeypatch.setattr(
        web_server.config_store, "load_config", lambda raw=None: _config()
    )

    with pytest.raises(ApiError) as excinfo:
        web_server._assert_checkpoint_compatible(
            "hand", {"checkpoint": str(checkpoint)}
        )
    assert excinfo.value.code == "E_CLASS_COUNT_MISMATCH"


def test_checkpoint_compat_accepts_matching_config(tmp_path, monkeypatch):
    import web_server

    checkpoint = tmp_path / "model.pth"
    _write_checkpoint(checkpoint, num_classes=2)
    monkeypatch.setattr(
        web_server.config_store, "load_config", lambda raw=None: _config()
    )

    web_server._assert_checkpoint_compatible(
        "hand", {"checkpoint": str(checkpoint)}
    )  # 不抛异常


# --------------------------------------------------------------------------- #
# 回放关键点形状
# --------------------------------------------------------------------------- #
def test_build_frames_from_labels_uses_recognizer_kpt_shape(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    # pose 单关键点行：class cx cy w h px py v
    (labels / "hand__f1.txt").write_text(
        "0 0.5 0.5 0.2 0.2 0.5 0.5 1\n", encoding="utf-8"
    )
    recognizer = SimpleNamespace(
        assembler=SimpleNamespace(num_classes=2), kpt_shape=(1, 3)
    )

    frames = list(build_frames_from_labels(recognizer, labels, frame_ids=["hand__f1"]))
    assert len(frames[0][1]) == 1

    wrong = SimpleNamespace(assembler=SimpleNamespace(num_classes=2), kpt_shape=(20, 3))
    frames = list(build_frames_from_labels(wrong, labels, frame_ids=["hand__f1"]))
    assert frames[0][1] == []


# --------------------------------------------------------------------------- #
# 训练前一致性闸门（M8）
# --------------------------------------------------------------------------- #
def _config_stub(**payload):
    cfg = {"version": 2, "dataset": {"window_size": 8}, "steps": []}
    cfg.update(payload)
    return cfg


def _write_meta(folder: Path, **payload):
    import json

    meta = {"num_classes": 2, "label_names": ["抓取", "放下"], "window_size": 8}
    meta.update(payload)
    (folder / "dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )


def test_train_rejects_stale_step_count(tmp_path, monkeypatch):
    from web.services import train_service

    tcn_dir = tmp_path / "tcn"
    tcn_dir.mkdir()
    _write_meta(tcn_dir)
    monkeypatch.setattr(
        train_service.config_store,
        "load_config",
        lambda raw=None: _config_stub(steps=[{"id": 0, "name": "抓取"}]),
    )

    with pytest.raises(ApiError) as excinfo:
        train_service.ensure_dataset_fresh(tcn_dir, tmp_path / "tcn_config.json")
    assert excinfo.value.code == "E_DATASET_STALE"
    assert "步骤数" in (excinfo.value.detail or "")


def test_train_rejects_stale_step_names(tmp_path, monkeypatch):
    from web.services import train_service

    tcn_dir = tmp_path / "tcn"
    tcn_dir.mkdir()
    _write_meta(tcn_dir)
    monkeypatch.setattr(
        train_service.config_store,
        "load_config",
        lambda raw=None: _config_stub(
            steps=[{"id": 0, "name": "抓取"}, {"id": 1, "name": "改名了"}]
        ),
    )

    with pytest.raises(ApiError) as excinfo:
        train_service.ensure_dataset_fresh(tcn_dir, tmp_path / "tcn_config.json")
    assert excinfo.value.code == "E_DATASET_STALE"
    assert "步骤名" in (excinfo.value.detail or "")


def test_train_allows_fresh_dataset(tmp_path, monkeypatch):
    from web.services import train_service

    tcn_dir = tmp_path / "tcn"
    tcn_dir.mkdir()
    _write_meta(tcn_dir)
    monkeypatch.setattr(
        train_service.config_store,
        "load_config",
        lambda raw=None: _config_stub(
            steps=[{"id": 0, "name": "抓取"}, {"id": 1, "name": "放下"}]
        ),
    )

    train_service.ensure_dataset_fresh(tcn_dir, tmp_path / "tcn_config.json")


def test_train_skips_check_without_meta(tmp_path, monkeypatch):
    from web.services import train_service

    tcn_dir = tmp_path / "tcn"
    tcn_dir.mkdir()
    monkeypatch.setattr(
        train_service.config_store,
        "load_config",
        lambda raw=None: _config_stub(steps=[{"id": 0, "name": "抓取"}]),
    )

    # 没有 dataset_meta.json（首次训练）不应阻断
    train_service.ensure_dataset_fresh(tcn_dir, tmp_path / "tcn_config.json")


# --------------------------------------------------------------------------- #
# 进程树终止（T10）
# --------------------------------------------------------------------------- #
def test_terminate_process_tree_kills_child():
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert task_manager.terminate_process_tree(proc, force=True) is True
        assert proc.wait(timeout=10) is not None
    finally:
        if proc.poll() is None:  # pragma: no cover - 兜底
            proc.kill()


def test_terminate_process_tree_ignores_finished_process():
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=10)

    assert task_manager.terminate_process_tree(proc, force=True) is True


# --------------------------------------------------------------------------- #
# 图片扫描裁剪（V7）
# --------------------------------------------------------------------------- #
def test_images_scan_truncates_and_reports_total(tmp_path):
    import web_server

    folder = tmp_path / "images"
    folder.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (folder / name).write_bytes(b"\xff\xd8\xff\xe0jpeg-bytes")

    client = web_server.app.test_client()
    response = client.get(
        "/api/images/scan",
        query_string={"path": str(folder), "max_items": "2"},
    )

    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["count"] == 3
    assert data["files"] == ["a.jpg", "b.jpg"]
    assert data["truncated"] is True
    assert data["max_items"] == 2
