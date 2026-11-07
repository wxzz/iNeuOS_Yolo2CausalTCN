"""网页“增加 / 删除数据集”的回归测试。

覆盖：
- ``discover_datasets`` 的 ``extra_dirs`` / ``exclude_dirs`` / ``allow_empty``；
- ``source_store`` 的持久化（增加、删除、撤销删除）；
- ``dataset_service.add_dataset`` / ``remove_dataset`` 只改列表、不动磁盘。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from build_dataset import discover_datasets
from web.services import config_store, dataset_service, source_store
from web.services.errors import ApiError


def _make_dataset(parent: Path, name: str, classes=("a", "b")) -> Path:
    """在 ``parent`` 下造一个最小可用的 YOLO 数据集目录。"""
    folder = parent / name
    (folder / "train" / "images").mkdir(parents=True, exist_ok=True)
    (folder / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (folder / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
    return folder


# --------------------------------------------------------------------------- #
# discover_datasets
# --------------------------------------------------------------------------- #
def test_discover_extra_dir_outside_root(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    _make_dataset(root, "hand")
    outside = _make_dataset(tmp_path / "elsewhere", "sitting")

    assert [item["name"] for item in discover_datasets(root)] == ["hand"]

    datasets = discover_datasets(root, extra_dirs=[outside])
    assert [item["name"] for item in datasets] == ["hand", "sitting"]
    assert datasets[1].get("extra") is True


def test_discover_excluded_dir(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    hand = _make_dataset(root, "hand")
    _make_dataset(root, "sitting")

    datasets = discover_datasets(root, exclude_dirs=[hand])
    assert [item["name"] for item in datasets] == ["sitting"]


def test_discover_allow_empty(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()

    assert discover_datasets(root, allow_empty=True) == []
    with pytest.raises(SystemExit):
        discover_datasets(root)


def test_discover_duplicate_name_skipped(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    _make_dataset(root, "hand")
    other = _make_dataset(tmp_path / "other", "hand")

    datasets = discover_datasets(root, extra_dirs=[other])
    assert [item["name"] for item in datasets] == ["hand"]
    assert datasets[0]["path"] == str(root / "hand")


def test_discover_missing_root_with_extra(tmp_path: Path):
    missing = tmp_path / "no_root"
    extra = _make_dataset(tmp_path, "solo")

    datasets = discover_datasets(missing, extra_dirs=[extra], allow_empty=True)
    assert [item["name"] for item in datasets] == ["solo"]

    # CLI 场景仍要求根目录存在
    with pytest.raises(SystemExit):
        discover_datasets(missing, extra_dirs=[extra])


# --------------------------------------------------------------------------- #
# source_store
# --------------------------------------------------------------------------- #
def test_source_store_roundtrip(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        config_store, "config_path", lambda raw=None: tmp_path / "tcn_config.json"
    )

    extra = tmp_path / "ds"
    extra.mkdir()

    source_store.add(extra)
    assert source_store.load()["extra_datasets"] == [str(extra)]
    assert [path.name for path in source_store.extra_dirs()] == ["ds"]
    assert source_store.excluded_dirs() == []

    source_store.remove(extra)
    state = source_store.load()
    assert state["extra_datasets"] == []
    assert state["excluded_datasets"] == [str(extra)]
    assert source_store.extra_dirs() == []
    assert source_store.excluded_dirs() == [str(extra)]

    # 重新增加等价于撤销删除
    source_store.add(extra)
    assert source_store.load()["excluded_datasets"] == []

    # 目录已复制进根目录时可只移除手工记录
    source_store.drop_extra(extra)
    assert source_store.load()["extra_datasets"] == []


# --------------------------------------------------------------------------- #
# dataset_service
# --------------------------------------------------------------------------- #
@pytest.fixture()
def service_env(tmp_path: Path, monkeypatch):
    """把路径守卫与配置读写都指向临时目录。"""
    monkeypatch.setattr(
        config_store, "config_path", lambda raw=None: tmp_path / "tcn_config.json"
    )
    monkeypatch.setattr(
        config_store,
        "load_config",
        lambda raw=None: {
            "version": 2,
            "source": {"datasets": []},
            "class_aliases": {},
        },
    )
    monkeypatch.setattr(
        dataset_service, "user_path", lambda raw, must_exist=False: Path(raw)
    )

    root = tmp_path / "yolo_datasets"
    root.mkdir()
    _make_dataset(root, "hand")
    outside = _make_dataset(tmp_path / "outside", "sitting")
    return root, outside


def test_add_and_remove_dataset_keeps_files(service_env):
    root, outside = service_env

    assert [item["name"] for item in dataset_service.collect(root)] == ["hand"]

    info = dataset_service.add_dataset(outside, root=str(root))
    assert info["name"] == "sitting"
    assert info["copied"] is True
    assert (root / "sitting" / "classes.txt").exists()
    assert outside.exists()  # 复制而不是移动，原目录保留
    assert [item["name"] for item in dataset_service.collect(root)] == [
        "hand",
        "sitting",
    ]

    dataset_service.remove_dataset(str(root / "hand"))
    datasets = dataset_service.collect(root)
    assert [item["name"] for item in datasets] == ["sitting"]
    assert datasets[0].get("extra") is False
    assert datasets[0]["path"] == str(root / "sitting")
    # 删除只影响列表，磁盘文件必须保留
    assert (root / "hand" / "classes.txt").exists()


def test_add_dataset_rejects_duplicate_name(service_env):
    root, _outside = service_env
    monkeypatch_root = root.parent / "other"
    _make_dataset(monkeypatch_root, "hand")

    with pytest.raises(ApiError):
        dataset_service.add_dataset(monkeypatch_root / "hand", root=str(root))


def test_add_dataset_rejects_non_dataset_dir(service_env, tmp_path: Path):
    root, _outside = service_env
    plain = tmp_path / "plain"
    plain.mkdir()

    with pytest.raises(ApiError):
        dataset_service.add_dataset(plain, root=str(root))


# --------------------------------------------------------------------------- #
# 删除数据集信息（purge）
# --------------------------------------------------------------------------- #
def test_purge_keeps_tcn_artifacts(service_env):
    """purge 只删数据集目录与权重，tcn_datasets 下的产物（labels/config）不动。"""
    root, outside = service_env
    dataset_service.add_dataset(outside, root=str(root))

    tcn_dir = config_store.config_path().parent
    label_dir = tcn_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    label_file = label_dir / "sitting__frame1.txt"
    label_file.write_text("", encoding="utf-8")
    config_file = tcn_dir / "tcn_config.json"
    config_file.write_text('{"version": 2}', encoding="utf-8")

    report = dataset_service.remove_datasets(
        [str(root / "sitting")], purge=True, root=str(root)
    )

    assert report["mode"] == "purge"
    assert not (root / "sitting").exists()
    assert outside.exists()  # 原目录不受影响
    assert label_file.exists()
    assert config_file.read_text(encoding="utf-8") == '{"version": 2}'
    assert [item["name"] for item in dataset_service.collect(root)] == ["hand"]
    assert report["skipped"] == []


def test_purge_deletes_external_weights(service_env, tmp_path: Path, monkeypatch):
    root, outside = service_env
    external = tmp_path / "models" / "sitting.pt"
    external.parent.mkdir()
    external.write_bytes(b"x")
    monkeypatch.setattr(
        config_store,
        "load_config",
        lambda raw=None: {
            "version": 2,
            "source": {"datasets": [{"name": "sitting", "weights": str(external)}]},
            "class_aliases": {},
        },
    )

    dataset_service.add_dataset(outside, root=str(root))
    dataset_service.remove_datasets([str(root / "sitting")], purge=True, root=str(root))

    assert not external.exists()


def test_purge_keeps_shared_weights(service_env, tmp_path: Path, monkeypatch):
    root, outside = service_env
    shared = tmp_path / "models" / "shared.pt"
    shared.parent.mkdir()
    shared.write_bytes(b"x")
    _make_dataset(root, "shared_hand")
    monkeypatch.setattr(
        config_store,
        "load_config",
        lambda raw=None: {
            "version": 2,
            "source": {
                "datasets": [
                    {"name": "sitting", "weights": str(shared)},
                    {"name": "shared_hand", "weights": str(shared)},
                ]
            },
            "class_aliases": {},
        },
    )

    dataset_service.add_dataset(outside, root=str(root))
    report = dataset_service.remove_datasets(
        [str(root / "sitting")], purge=True, root=str(root)
    )

    assert shared.exists()
    assert any("共用" in item["reason"] for item in report["skipped"])


def test_purge_refuses_non_dataset_dir(service_env, tmp_path: Path):
    root, _outside = service_env
    plain = tmp_path / "plain"
    plain.mkdir()

    with pytest.raises(ApiError):
        dataset_service.remove_datasets([str(plain)], purge=True, root=str(root))
    assert plain.exists()


# --------------------------------------------------------------------------- #
# 重新扫描
# --------------------------------------------------------------------------- #
def test_rescan_restores_excluded_dataset(service_env):
    root, outside = service_env
    dataset_service.add_dataset(outside, root=str(root))

    dataset_service.remove_dataset(str(root / "hand"))
    assert [item["name"] for item in dataset_service.collect(root)] == ["sitting"]

    scanned = dataset_service.scan(root, reset=True)
    assert [item["name"] for item in scanned["datasets"]] == ["hand", "sitting"]
    assert source_store.load()["excluded_datasets"] == []


def test_scan_allows_external_root_and_remembers_it(tmp_path: Path, monkeypatch):
    root = tmp_path / "external_root"
    root.mkdir()
    _make_dataset(root, "hand")
    monkeypatch.setattr(
        config_store, "config_path", lambda raw=None: tmp_path / "tcn_config.json"
    )
    monkeypatch.setattr(
        config_store,
        "load_config",
        lambda raw=None: {"source": {}, "class_aliases": {}},
    )

    first = dataset_service.scan(str(root))
    assert [item["name"] for item in first["datasets"]] == ["hand"]
    assert source_store.get_root() == str(root.resolve())

    # 不传 root 时使用上次记住的根目录
    second = dataset_service.scan()
    assert [item["name"] for item in second["datasets"]] == ["hand"]
    assert second["root"].endswith("external_root")


def test_rescan_keeps_external_extra_datasets(service_env):
    root, outside = service_env
    source_store.add(outside)  # 外部手工数据集直接参与，不复制
    assert [item["name"] for item in dataset_service.collect(root)] == [
        "hand",
        "sitting",
    ]

    scanned = dataset_service.scan(root, reset=True)

    assert [item["name"] for item in scanned["datasets"]] == ["hand", "sitting"]
    assert not (root / "sitting").exists()
    assert source_store.load()["extra_datasets"] == [str(outside)]


# --------------------------------------------------------------------------- #
# 模型权重复制
# --------------------------------------------------------------------------- #
def test_import_weights_copies_and_renames(service_env, tmp_path: Path):
    root, _outside = service_env
    dataset_dir = root / "hand"
    (dataset_dir / "old.pt").write_bytes(b"old")
    (dataset_dir / "hand.pt").write_bytes(b"old-canonical")
    picked = tmp_path / "best.pt"
    picked.write_bytes(b"new")

    info = dataset_service.import_weights(dataset_dir, picked)

    target = dataset_dir / "hand.pt"
    assert Path(info["path"]) == target
    assert info["copied"] is True
    assert target.read_bytes() == b"new"
    assert not (dataset_dir / "old.pt").exists()
    assert picked.exists()  # 复制不是移动


def test_import_weights_same_file_keeps_target(service_env):
    root, _outside = service_env
    dataset_dir = root / "hand"
    target = dataset_dir / "hand.pt"
    target.write_bytes(b"same")

    info = dataset_service.import_weights(dataset_dir, target)

    assert info["copied"] is False
    assert target.read_bytes() == b"same"


# --------------------------------------------------------------------------- #
# 步骤编号（同一编号 = 跨数据集同一步骤）
# --------------------------------------------------------------------------- #
def test_validate_config_allows_duplicate_step_ids():
    cfg = {
        "steps": [
            {
                "id": 0,
                "name": "抓取",
                "match": {"dataset": "hand", "classes": ["a"]},
            },
            {
                "id": 0,
                "name": "抓取",
                "match": {"dataset": "sitting", "classes": ["a"]},
            },
            {
                "id": 1,
                "name": "放下",
                "match": {"dataset": "hand", "classes": ["b"]},
            },
        ]
    }
    errors, _warnings = config_store.validate_config(cfg)
    assert errors == []


def test_validate_config_rejects_same_id_with_different_names():
    cfg = {
        "steps": [
            {"id": 0, "name": "抓取", "match": {"classes": ["a"]}},
            {"id": 0, "name": "放下", "match": {"classes": ["a"]}},
        ]
    }
    errors, _warnings = config_store.validate_config(cfg)
    assert [item["code"] for item in errors] == ["E_STEP_NAME_MISMATCH"]


def test_validate_config_rejects_id_gaps():
    cfg = {
        "steps": [
            {"id": 0, "name": "a", "match": {"classes": ["a"]}},
            {"id": 2, "name": "b", "match": {"classes": ["b"]}},
        ]
    }
    errors, _warnings = config_store.validate_config(cfg)
    assert [item["code"] for item in errors] == ["E_STEP_ID_GAP"]


# --------------------------------------------------------------------------- #
# 帧列表（Tab2「帧列表」规则的数据源）
# --------------------------------------------------------------------------- #
def test_list_frames_prefers_existing_labels(service_env, tmp_path: Path, monkeypatch):
    root, _outside = service_env
    (root / "hand" / "train" / "images" / "ignored.jpg").write_bytes(b"x")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "hand__f10.txt").write_text("", encoding="utf-8")
    (labels / "hand__f2.txt").write_text("", encoding="utf-8")
    (labels / "sitting__f1.txt").write_text("", encoding="utf-8")
    monkeypatch.setattr(dataset_service, "DEFAULT_LABEL_DIR", labels)

    result = dataset_service.list_frames("hand", root=root)

    assert result["source"] == "labels"
    # 只含本数据集前缀，且按自然序排列（f2 在 f10 前）
    assert result["frames"] == ["hand__f2", "hand__f10"]


def test_list_frames_falls_back_to_images(service_env, tmp_path: Path, monkeypatch):
    root, _outside = service_env
    images = root / "hand" / "train" / "images"
    (images / "f10.jpg").write_bytes(b"x")
    (images / "f2.png").write_bytes(b"x")
    (images / "notes.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        dataset_service, "DEFAULT_LABEL_DIR", tmp_path / "missing_labels"
    )

    result = dataset_service.list_frames("hand", root=root)

    assert result["source"] == "images"
    assert result["frames"] == ["hand__f2", "hand__f10"]


def test_list_frames_rejects_unknown_dataset(service_env):
    root, _outside = service_env
    with pytest.raises(ApiError):
        dataset_service.list_frames("nope", root=root)


def test_frame_image_path_maps_frame_to_file(service_env):
    root, _outside = service_env
    target = root / "hand" / "train" / "images" / "f2.jpg"
    target.write_bytes(b"x")

    assert (
        dataset_service.frame_image_path("hand", "hand__f2", root=root)
        == target.resolve()
    )
    with pytest.raises(ApiError):
        dataset_service.frame_image_path("hand", "sitting__f2", root=root)
    with pytest.raises(ApiError):
        dataset_service.frame_image_path("hand", "hand__../escape", root=root)


# --------------------------------------------------------------------------- #
# 勾选 = 建库范围（build_service 把未勾选项转成 --exclude-dataset）
# --------------------------------------------------------------------------- #
def test_build_argv_excludes_unselected_datasets(tmp_path: Path, monkeypatch):
    from web.services import build_service

    root = tmp_path / "root"
    root.mkdir()
    hand = _make_dataset(root, "hand")
    sitting = _make_dataset(root, "sitting")

    monkeypatch.setattr(
        build_service, "user_path", lambda raw, must_exist=False: Path(raw)
    )
    monkeypatch.setattr(build_service, "project_root", lambda: tmp_path)
    monkeypatch.setattr(build_service, "weight_path", lambda raw: Path(raw))
    monkeypatch.setattr(
        build_service,
        "config_store",
        SimpleNamespace(config_path=lambda raw=None: tmp_path / "tcn_config.json"),
    )
    monkeypatch.setattr(
        build_service,
        "source_store",
        SimpleNamespace(extra_dirs=lambda: [], excluded_dirs=lambda: []),
    )
    monkeypatch.setattr(
        build_service,
        "dataset_service",
        SimpleNamespace(
            collect=lambda raw=None, with_stats=True: [
                {"name": "hand", "path": str(hand)},
                {"name": "sitting", "path": str(sitting)},
            ]
        ),
    )

    argv = build_service.build_argv(
        {
            "root": str(root),
            "detector": "labels",
            "datasets": [{"name": "hand", "weights": ""}],
        }
    )

    excluded = [
        argv[index + 1]
        for index, item in enumerate(argv)
        if item == "--exclude-dataset"
    ]
    assert excluded == [str(sitting)]

    with pytest.raises(ApiError):
        build_service.build_argv(
            {"root": str(root), "detector": "labels", "datasets": []}
        )


# --------------------------------------------------------------------------- #
# 删除数据集后清理步骤
# --------------------------------------------------------------------------- #
def test_prune_steps_for_removed_dataset_compacts_ids(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        config_store, "config_path", lambda raw=None: tmp_path / "tcn_config.json"
    )
    config = {
        "version": 2,
        "source": {"datasets": []},
        "class_aliases": {},
        "dataset": {},
        "unmatched": None,
        "steps": [
            {"id": 0, "name": "a", "match": {"dataset": "hand", "classes": ["x"]}},
            {"id": 1, "name": "b", "match": {"dataset": "sitting", "classes": ["y"]}},
            {"id": 2, "name": "c", "match": {"dataset": "hand", "classes": ["z"]}},
        ],
    }
    path = tmp_path / "tcn_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    removed = dataset_service._prune_steps_for_removed_datasets({"sitting"})

    assert [item["id"] for item in removed] == [1]
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert [(step["id"], step["name"]) for step in saved["steps"]] == [
        (0, "a"),
        (1, "c"),
    ]


def test_prune_steps_allows_empty_result(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        config_store, "config_path", lambda raw=None: tmp_path / "tcn_config.json"
    )
    config = {
        "version": 2,
        "source": {"datasets": []},
        "class_aliases": {},
        "steps": [{"id": 0, "name": "a", "match": {"dataset": "hand"}}],
    }
    path = tmp_path / "tcn_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    removed = dataset_service._prune_steps_for_removed_datasets({"hand"})

    assert [item["name"] for item in removed] == ["a"]
    assert json.loads(path.read_text(encoding="utf-8"))["steps"] == []
