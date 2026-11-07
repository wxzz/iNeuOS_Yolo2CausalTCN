"""构建数据集：拼参数 + 解析日志（文档 §16.7 / §16.8）。"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from config import DEFAULT_TCN_DIR, DEFAULT_YOLO_ROOT
from build_dataset import normalize_dataset_name
from web.services import config_store, dataset_service, source_store
from web.services.errors import ApiError
from web.services.paths import data_path, project_root, user_path, weight_path
from web.services.task_manager import MANAGER, Task

PREDICT_RE = re.compile(
    r"\[(?P<dataset>.+?)\]\s*预测完成：(?P<frames>\d+)\s*帧\s*/\s*(?P<objects>\d+)\s*个目标\s*/\s*(?P<empty>\d+)\s*帧无检测"
)
STEP_RE = re.compile(
    r"步骤\s+(?P<id>\d+)\s*\|\s*(?P<name>\S+)\s*\|\s*(?P<frames>\d+)\s*帧"
)
CLEAN_RE = re.compile(r"已清理\s+(?P<count>\d+)\s+个陈旧标签文件")
GLOBAL_RE = re.compile(r"全局类别表（(?P<count>\d+)）：(?P<names>.+)")
UNMATCHED_RE = re.compile(r"有\s+(?P<count>\d+)\s*帧未命中任何步骤")


def build_argv(payload: Dict[str, Any]) -> List[str]:
    """把前端请求翻译成 ``build_dataset.py`` 参数数组。"""
    root = user_path(payload.get("root") or DEFAULT_YOLO_ROOT, must_exist=True)
    tcn_dir = data_path(payload.get("tcn_dir") or DEFAULT_TCN_DIR)
    cfg_path = config_store.config_path(payload.get("config"))

    detector = str(payload.get("detector") or "yolo")
    if detector not in ("yolo", "labels"):
        raise ApiError("E_DETECTOR_INVALID", f"未知检测器模式：{detector}")

    splits = payload.get("splits") or ["train"]
    if not isinstance(splits, list) or not splits:
        raise ApiError("E_SPLIT_INVALID", "至少选择一个划分")

    argv: List[str] = [
        sys.executable,
        str(project_root() / "build_dataset.py"),
        "--yolo-root",
        str(root),
        "--tcn-dir",
        str(tcn_dir),
        "--config",
        str(cfg_path),
        "--detector",
        detector,
        "--conf",
        str(float(payload.get("conf", 0.25))),
        "--iou",
        str(float(payload.get("iou", 0.45))),
        "--imgsz",
        str(int(payload.get("imgsz", 640))),
        "--split",
        *[str(item) for item in splits],
    ]

    if payload.get("reset_config"):
        argv.append("--reset-config")
    if payload.get("config_only"):
        argv.append("--config-only")

    # 网页“增加 / 删除数据集”的记录必须同步给建库脚本，否则列表与产物会不一致
    for extra in source_store.extra_dirs():
        argv.extend(["--dataset-dir", str(extra)])
    for excluded in source_store.excluded_dirs():
        argv.extend(["--exclude-dataset", str(excluded)])

    # 勾选 = 建库范围：未勾选的数据集显式排除。否则不仅会“取消勾选仍建库”，
    # 未勾选数据集缺权重时还会让 validate_datasets 整批失败。
    selected = [
        normalize_dataset_name(str(item.get("name") or ""))
        for item in payload.get("datasets") or []
        if str(item.get("name") or "").strip()
    ]
    if not selected:
        raise ApiError("E_NO_DATASET", "请至少勾选一个数据集")
    selected_names = set(selected)
    try:
        known = dataset_service.collect(root, with_stats=False)
    except ApiError:
        known = []
    for item in known:
        if str(item.get("name") or "") not in selected_names:
            argv.extend(["--exclude-dataset", str(item.get("path"))])

    for dataset in payload.get("datasets") or []:
        name = str(dataset.get("name") or "").strip()
        raw_weights = dataset.get("weights")
        if detector == "labels" or not raw_weights:
            continue
        if not name:
            raise ApiError("E_DATASET_UNKNOWN", "数据集名不能为空")
        argv.extend(["--weights", f"{name}={weight_path(str(raw_weights))}"])

    return argv


def on_line(task: Task, text: str) -> None:
    """解析预测进度、步骤统计与清理数量。"""
    match = PREDICT_RE.search(text)
    if match:
        MANAGER.emit(
            task,
            "progress",
            {
                "phase": "predict",
                "dataset": match.group("dataset"),
                "frames": int(match.group("frames")),
                "objects": int(match.group("objects")),
                "empty": int(match.group("empty")),
            },
        )
        return

    match = STEP_RE.search(text)
    if match:
        task.scratch.setdefault("steps", {})[int(match.group("id"))] = {
            "name": match.group("name"),
            "frames": int(match.group("frames")),
        }
        return

    match = CLEAN_RE.search(text)
    if match:
        task.scratch["cleaned"] = int(match.group("count"))
        return

    match = GLOBAL_RE.search(text)
    if match:
        task.scratch["target_names"] = [
            item.strip().strip("'\"")
            for item in match.group("names").strip("[]").split(",")
            if item.strip()
        ]
        return

    match = UNMATCHED_RE.search(text)
    if match:
        task.scratch["unmatched"] = int(match.group("count"))


def on_exit(task: Task, code: int) -> Dict[str, Any]:
    """读取 ``dataset_meta.json`` 与配置，作为 ``done.result``。"""
    if code != 0:
        return {}

    payload = task.scratch.get("_payload") or {}
    tcn_dir = Path(payload.get("tcn_dir") or DEFAULT_TCN_DIR)
    meta_path = tcn_dir / "dataset_meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    cfg_path = Path(payload.get("config") or config_store.config_path())
    return {
        "meta": meta,
        "config_path": str(cfg_path),
        "steps": task.scratch.get("steps", {}),
        "cleaned": task.scratch.get("cleaned", 0),
        "unmatched": task.scratch.get("unmatched", 0),
    }
