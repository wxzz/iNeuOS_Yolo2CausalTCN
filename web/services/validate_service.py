"""模型验证：拼参数 + 读取 CSV 结构化结果（文档 §16.4 / §16.7）。"""

from __future__ import annotations

import csv
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DEFAULT_CHECKPOINT, DEFAULT_WINDOW_SIZE
from detectors import IMAGE_SUFFIXES, list_image_files
from model import read_checkpoint_meta
from web.services import config_store
from web.services.errors import ApiError
from web.services.paths import (
    checkpoint_path,
    cleanup_tmp,
    project_root,
    read_only_path,
    tmp_dir,
)
from web.services.task_manager import Task

IMAGE_LIMIT = 50
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT

DATASET_RE = re.compile(
    r"数据集\s+(?P<dataset>\S+)\s*\|\s*任务\s+(?P<task>\S+)\s*\|\s*权重\s+(?P<weights>.+)"
)
WARMUP_RE = re.compile(r"单图模式：用\s+(?P<warmup>\d+)\s+张前置帧预热")


def _image_context_length(target: Path, window: int) -> int:
    """单图模式可用的前序帧数（与 ``inference.collect_image_context`` 取值一致）。"""
    siblings = list_image_files(target.parent)
    resolved = target.resolve()
    index = next(
        (i for i, item in enumerate(siblings) if item.resolve() == resolved),
        len(siblings) - 1,
    )
    return min(index + 1, max(int(window), 1))


def _resolve_dataset_name(payload: Dict[str, Any], source: Path) -> str:
    """数据集判定：显式 > 按路径归属 > 按目录名匹配 > 唯一数据集（与 inference.resolve_dataset 一致）。

    图片可能被复制到项目外，此时按路径无法归属，退化为「目录名 = 数据集名」匹配
    （如 ``.../hand/train/images`` → ``hand``），取最靠近图片的那一层。
    """
    explicit = str(payload.get("dataset") or "auto")
    if explicit and explicit != "auto":
        return explicit

    cfg = config_store.load_config(payload.get("config"))
    datasets = (cfg.get("source") or {}).get("datasets") or []
    resolved = source.resolve()
    for item in datasets:
        try:
            resolved.relative_to(Path(str(item.get("path", ""))).resolve())
            return str(item["name"])
        except (ValueError, OSError):
            continue

    names = {
        str(item.get("name") or "").lower(): str(item.get("name"))
        for item in datasets
        if item.get("name")
    }
    for part in reversed(resolved.parts):
        matched = names.get(part.lower())
        if matched:
            return matched

    if len(datasets) == 1:
        return str(datasets[0]["name"])

    raise ApiError(
        "E_DATASET_UNRESOLVED",
        "无法判断所属数据集",
        detail="图片不在任何数据集目录下，且目录名未匹配到数据集名；"
        f"可选数据集：{[item.get('name') for item in datasets]}",
    )


def resolve_image(source: str | Path, name: str) -> Path:
    """解析验证来源目录下的某张图片（供 Tab4 列表缩略图，只读）。"""
    if not source or not name:
        raise ApiError("E_SOURCE_REQUIRED", "请提供 path 与 name 参数")

    plain = Path(str(name))
    if plain.name != str(name) or str(name) in (".", ".."):
        raise ApiError("E_SOURCE_INVALID", f"非法图片名：{name}")

    directory = read_only_path(source, must_exist=True)
    folder = (directory if directory.is_dir() else directory.parent).resolve()
    target = (folder / plain.name).resolve()
    try:
        target.relative_to(folder)
    except ValueError as exc:
        raise ApiError("E_SOURCE_INVALID", f"非法图片名：{name}") from exc

    if not target.is_file() or target.suffix.lower() not in IMAGE_SUFFIXES:
        raise ApiError("E_SOURCE_INVALID", f"不是可预览的图片：{name}", status=404)
    return target


def prepare(payload: Dict[str, Any]) -> Dict[str, Any]:
    """校验参数并生成 argv；返回随任务上下文的元信息。"""
    cleanup_tmp()
    checkpoint = checkpoint_path(
        payload.get("checkpoint") or DEFAULT_CHECKPOINT, must_exist=True
    )
    cfg_path = config_store.config_path(payload.get("config"))
    cfg = config_store.load_config(cfg_path)
    step_names = config_store.step_name_map(cfg)
    window = _checkpoint_window(checkpoint)
    mode = str(payload.get("mode") or "dir")
    source_raw = payload.get("source")
    if not source_raw:
        raise ApiError("E_SOURCE_REQUIRED", "请选择图片目录或单张图片")

    source = read_only_path(source_raw, must_exist=True)
    files = [str(item) for item in (payload.get("files") or [])]
    work_dir: Optional[Path] = None
    context_len = 0

    if mode == "file":
        if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ApiError("E_SOURCE_INVALID", "--source 指向文件时必须是图片")
        dataset_name = _resolve_dataset_name(payload, source)
        frame_count = 1
        context_len = _image_context_length(source, window)
    else:
        directory = source if source.is_dir() else source.parent
        # 先按**原始目录**判定数据集：稍后可能把选中图片复制到临时目录，
        # 那时路径已不在数据集中，无法再按路径判断。
        dataset_name = _resolve_dataset_name(payload, directory)
        if files:
            selected = [directory / Path(item).name for item in files]
            missing = [str(item) for item in selected if not item.exists()]
            if missing:
                raise ApiError("E_SOURCE_INVALID", f"文件不存在：{missing[0]}")
            if len(selected) > IMAGE_LIMIT:
                raise ApiError(
                    "E_TOO_MANY_IMAGES",
                    f"最多 {IMAGE_LIMIT} 张，当前 {len(selected)} 张",
                )
            work_dir = tmp_dir() / f"validate_{uuid.uuid4().hex[:8]}"
            work_dir.mkdir(parents=True, exist_ok=True)
            for path in selected:
                shutil.copy2(path, work_dir / path.name)
            source = work_dir
        else:
            images = list_image_files(directory)
            if len(images) > IMAGE_LIMIT:
                raise ApiError(
                    "E_TOO_MANY_IMAGES",
                    f"最多 {IMAGE_LIMIT} 张，当前 {len(images)} 张",
                )
            source = directory
        frame_count = len(list_image_files(source))

    csv_path = tmp_dir() / f"predict_{uuid.uuid4().hex[:8]}.csv"

    argv: List[str] = [
        sys.executable,
        str(project_root() / "inference.py"),
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(cfg_path),
        "--source",
        str(source),
        "--conf",
        str(float(payload.get("conf", 0.25))),
        "--iou",
        str(float(payload.get("iou", 0.45))),
        "--imgsz",
        str(int(payload.get("imgsz", 640))),
        "--smooth",
        str(int(payload.get("smooth", 5))),
        "--save-csv",
        str(csv_path),
    ]
    if dataset_name:
        argv.extend(["--dataset", dataset_name])

    if mode == "file":
        # 单图模式会处理同目录前序帧（预热），只有预热足够时才输出该帧
        expected = 1 if context_len >= max(window, 1) else 0
        warmup = max(0, context_len - 1)
    else:
        expected = max(0, frame_count - max(window - 1, 0))
        warmup = min(max(window - 1, 0), frame_count)

    return {
        "argv": argv,
        "dataset": dataset_name,
        "frame_count": frame_count,
        "window": window,
        "warmup": warmup,
        "expected": expected,
        "work_dir": str(work_dir) if work_dir else "",
        "csv_path": str(csv_path),
        "config_path": str(cfg_path),
        "step_names": step_names,
    }


def _checkpoint_window(path: Path) -> int:
    try:
        return int(read_checkpoint_meta(path)["window_size"])
    except Exception:  # noqa: BLE001 - 读取失败交给 inference 自己报错
        return DEFAULT_WINDOW_SIZE


def on_line(task: Task, text: str) -> None:
    match = DATASET_RE.search(text)
    if match:
        task.scratch["dataset"] = match.group("dataset")
        task.scratch["weights"] = match.group("weights").strip()
        return

    match = WARMUP_RE.search(text)
    if match:
        task.scratch["warmup"] = int(match.group("warmup"))


def on_exit(task: Task, code: int) -> Dict[str, Any]:
    context = task.scratch.get("_context") or {}
    csv_value = str(context.get("csv_path") or "")
    csv_path = Path(csv_value) if csv_value else None
    step_names: Dict[int, str] = context.get("step_names") or {}
    predictions: List[Dict[str, Any]] = []
    ok = code == 0

    if csv_path is not None and csv_path.is_file():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                step_id = int(float(row.get("step_id", -1) or -1))
                predictions.append(
                    {
                        "frame": row.get("frame", ""),
                        "step_id": step_id,
                        # 名称以当前配置为准，避免旧 checkpoint 的 class_names 误导
                        "step_name": step_names.get(step_id)
                        or row.get("step_name", ""),
                        "confidence": float(row.get("confidence", 0) or 0),
                    }
                )
        if ok:
            csv_path.unlink(missing_ok=True)

    work_dir = str(context.get("work_dir") or "")
    retained: List[str] = []
    if work_dir:
        if ok:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            retained.append(work_dir)
    if not ok and csv_path is not None and csv_path.is_file():
        retained.append(str(csv_path))

    result = {
        "predictions": predictions,
        "dataset": task.scratch.get("dataset") or context.get("dataset"),
        "weights": task.scratch.get("weights", ""),
        "processed": int(context.get("frame_count", 0)),
        "window": int(context.get("window", 0)),
        "warmup": int(task.scratch.get("warmup", context.get("warmup", 0))),
        "expected": int(context.get("expected", 0)),
        "outputs": len(predictions),
        "retained": retained,
    }
    if predictions:
        result["mean_confidence"] = round(
            sum(item["confidence"] for item in predictions) / len(predictions), 4
        )
        distribution: Dict[str, int] = {}
        for item in predictions:
            key = str(item["step_id"])
            distribution[key] = distribution.get(key, 0) + 1
        result["distribution"] = dict(
            sorted(distribution.items(), key=lambda kv: int(kv[0]))
        )
    return result


def load_checkpoint_meta(path: Path) -> Dict[str, Any]:
    """读取 checkpoint 关键字段（用于 Tab4 的前置校验）。"""
    try:
        return read_checkpoint_meta(path)
    except Exception as exc:  # noqa: BLE001
        raise ApiError("E_CHECKPOINT_LOAD", f"模型加载失败：{exc}") from exc
