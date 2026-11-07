"""训练：拼参数 + 解析 epoch 指标（文档 §16.7 / §16.8）。"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from config import DEFAULT_CHECKPOINT, DEFAULT_TCN_DIR
from model import read_checkpoint_meta
from web.services import config_store
from web.services.errors import ApiError
from web.services.paths import checkpoint_path, data_path, project_root
from web.services.task_manager import MANAGER, Task

METRIC_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+)\s*\|\s*train_loss=(?P<train_loss>[-\d.]+)\s+acc=(?P<train_acc>[-\d.]+)"
    r"\s*\|\s*val_loss=(?P<val_loss>[-\d.]+)\s+acc=(?P<val_acc>[-\d.]+)"
    r"\s*\|\s*lr=(?P<lr>[-\d.eE+]+)"
)
SPLIT_RE = re.compile(
    r"切分模式:\s*(?P<mode>\w+)（train\s+(?P<train>\d+)\s*/\s*val\s+(?P<val>\d+)）"
)


def ensure_dataset_fresh(tcn_dir: Path, cfg_path: Path) -> None:
    """训练前校验「数据集产物」与「当前配置」是否同源。

    保存过步骤/窗口却没重新构建时，``labels/`` 仍是旧配置的产物；此时训练脚本
    会按新配置重新解析，结果与界面/元信息不一致（静默训到旧数据）。这里比对
    ``dataset_meta.json`` 里的**步骤数 / 步骤名 / 窗口**，不一致直接拒绝。

    meta 不存在（首次训练）时跳过，由训练脚本自行生成。
    """
    meta_path = tcn_dir / "dataset_meta.json"
    if not meta_path.is_file():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApiError(
            "E_META_INVALID",
            f"dataset_meta.json 读取失败：{meta_path}",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    cfg = config_store.load_config(cfg_path)
    names = config_store.step_name_map(cfg)
    expect_names = [names[key] for key in sorted(names)]
    expect_window = (cfg.get("dataset") or {}).get("window_size")

    problems: List[str] = []
    meta_classes = meta.get("num_classes")
    if meta_classes is not None and int(meta_classes) != len(expect_names):
        problems.append(f"步骤数 {meta_classes} → {len(expect_names)}")
    meta_names = meta.get("label_names")
    if meta_names is not None and [str(item) for item in meta_names] != expect_names:
        problems.append(f"步骤名 {list(meta_names)} → {expect_names}")
    meta_window = meta.get("window_size")
    if meta_window and expect_window and int(meta_window) != int(expect_window):
        problems.append(f"窗口 {meta_window} → {expect_window}")

    if problems:
        raise ApiError(
            "E_DATASET_STALE",
            "数据集产物与当前配置不一致（步骤或窗口已变），请先在【数据映射】重新构建数据集",
            status=409,
            detail="；".join(problems),
        )


def build_argv(payload: Dict[str, Any]) -> List[str]:
    """把前端请求翻译成 ``train.py`` 参数数组。"""
    tcn_dir = data_path(payload.get("tcn_dir") or DEFAULT_TCN_DIR)
    cfg_path = config_store.config_path(payload.get("config"))
    checkpoint = checkpoint_path(
        payload.get("checkpoint") or DEFAULT_CHECKPOINT, must_exist=False
    )
    ensure_dataset_fresh(tcn_dir, cfg_path)

    epochs = int(payload.get("epochs", 80))
    batch_size = int(payload.get("batch_size", 16))
    lr = float(payload.get("lr", 1e-3))
    if epochs < 1:
        raise ApiError("E_TRAIN_PARAM", "epochs 必须 >= 1")
    if batch_size < 1:
        raise ApiError("E_TRAIN_PARAM", "batch_size 必须 >= 1")

    channels = payload.get("channels") or [64, 64, 64, 64]
    if isinstance(channels, str):
        channels = [int(item) for item in channels.split() if item.strip()]
    if not channels:
        raise ApiError("E_TRAIN_PARAM", "channels 不能为空")

    argv: List[str] = [
        sys.executable,
        str(project_root() / "train.py"),
        "--dataset-config",
        str(cfg_path),
        "--artifact-dir",
        str(tcn_dir),
        "--label-dir",
        str(tcn_dir / "labels"),
        "--label-csv",
        str(tcn_dir / "labels.csv"),
        "--checkpoint",
        str(checkpoint),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--channels",
        *[str(int(item)) for item in channels],
        "--dropout",
        str(float(payload.get("dropout", 0.2))),
        "--device",
        str(payload.get("device") or "auto"),
    ]
    if payload.get("from_npy"):
        argv.append("--from-npy")
    return argv


def on_line(task: Task, text: str) -> None:
    match = METRIC_RE.search(text)
    if match:
        MANAGER.emit(
            task,
            "metric",
            {
                "epoch": int(match.group("epoch")),
                "train_loss": float(match.group("train_loss")),
                "train_acc": float(match.group("train_acc")),
                "val_loss": float(match.group("val_loss")),
                "val_acc": float(match.group("val_acc")),
                "lr": float(match.group("lr")),
            },
        )
        return

    match = SPLIT_RE.search(text)
    if match:
        task.scratch["split"] = {
            "mode": match.group("mode"),
            "train": int(match.group("train")),
            "val": int(match.group("val")),
        }


def on_exit(task: Task, code: int) -> Dict[str, Any]:
    if code != 0:
        return {}

    payload = task.scratch.get("_payload") or {}
    checkpoint = checkpoint_path(
        payload.get("checkpoint") or DEFAULT_CHECKPOINT, must_exist=False
    )
    tcn_dir = Path(payload.get("tcn_dir") or DEFAULT_TCN_DIR)

    result: Dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "exists": checkpoint.exists(),
        "split": task.scratch.get("split", {}),
    }

    if checkpoint.exists():
        try:
            info = read_checkpoint_meta(checkpoint)
            result.update(
                {
                    "best_epoch": info["epoch"],
                    "val_acc": float(info["metrics"].get("val_acc", 0.0)),
                    "feature_dim": info["feature_dim"],
                    "num_classes": info["num_classes"],
                    "class_names": info["class_names"],
                    "window_size": info["window_size"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            result["checkpoint_error"] = f"{type(exc).__name__}: {exc}"

    meta_path = tcn_dir / "dataset_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        result["num_train"] = meta.get("num_train")
        result["num_val"] = meta.get("num_val")
        result["group_window_counts"] = meta.get("group_window_counts")
        result["split_mode"] = meta.get("split_mode")

    return result
