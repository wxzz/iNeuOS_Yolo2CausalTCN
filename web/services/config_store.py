"""`tcn_datasets/tcn_config.json` 的读写与校验（文档 §15.1 / §16.3）。

写入策略：
- 先写 ``*.tmp`` 再 ``os.replace`` 原子替换；
- 保留一份 ``*.bak``，避免误覆盖；
- 只接受白名单字段（``class_aliases`` / ``steps`` / ``dataset`` / ``unmatched``），
  ``source`` 与 ``object_classes`` 由后端重算后写回。
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config import DEFAULT_DATASET_CONFIG
from build_dataset import validate_match_rule

from .errors import ApiError, err, warn
from .paths import data_path

WRITABLE_KEYS = ("class_aliases", "steps", "dataset", "unmatched")


def config_path(raw: str | Path | None = None) -> Path:
    if raw:
        return data_path(raw)
    return DEFAULT_DATASET_CONFIG


def load_config(raw: str | Path | None = None) -> Dict[str, Any]:
    """读取配置；不存在时返回可用的最小骨架。"""
    path = config_path(raw)
    if not path.exists():
        return {
            "version": 2,
            "exists": False,
            "source": {
                "yolo_root": "",
                "detector_defaults": {"type": "yolo"},
                "datasets": [],
            },
            "class_aliases": {},
            "object_classes": {
                "num_target_classes": 0,
                "target_names": [],
                "kpt_shape": None,
                "tasks": {},
            },
            "dataset": {},
            "steps": [],
            "unmatched": None,
            "path": str(path),
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _bad_config(path, exc) from exc

    payload["exists"] = True
    payload["path"] = str(path)
    return payload


def _bad_config(path: Path, exc: Exception):
    return ApiError("E_CONFIG_INVALID", f"配置不是合法 JSON：{path}", detail=str(exc))


def save_config(
    payload: Dict[str, Any], raw: str | Path | None = None
) -> Dict[str, Any]:
    """保存配置（白名单合并 + 原子写 + 备份），返回保存后的配置。"""
    path = config_path(raw)
    current = load_config(path)
    if not current.get("exists"):
        raise _missing_config(path)

    merged: Dict[str, Any] = dict(current)
    for key in WRITABLE_KEYS:
        if key in payload:
            merged[key] = payload[key]

    merged.pop("exists", None)
    merged.pop("path", None)

    errors, warnings = validate_config(merged)
    if errors:
        return {
            "saved": False,
            "errors": errors,
            "warnings": warnings,
            "config": merged,
        }

    _write_config(path, merged)

    merged["exists"] = True
    merged["path"] = str(path)
    return {"saved": True, "errors": [], "warnings": warnings, "config": merged}


def write_steps(
    steps: List[Dict[str, Any]], raw: str | Path | None = None
) -> Dict[str, Any]:
    """仅改写 ``steps``（删除数据集后清理步骤用），允许写成空列表。

    与 :func:`save_config` 共用原子写 + 备份；其余字段（source / object_classes /
    dataset 等）保持原样。除“至少一个步骤”外的校验错误仍会拒绝写入。
    """
    path = config_path(raw)
    current = load_config(path)
    if not current.get("exists"):
        raise _missing_config(path)

    merged: Dict[str, Any] = dict(current)
    merged["steps"] = steps
    merged.pop("exists", None)
    merged.pop("path", None)

    errors, warnings = validate_config(merged)
    errors = [item for item in errors if item.get("code") != "E_STEPS_EMPTY"]
    if errors:
        raise ApiError(
            "E_CONFIG_INVALID",
            "清理步骤后的配置校验失败",
            detail=json.dumps(errors, ensure_ascii=False),
        )

    _write_config(path, merged)
    merged["exists"] = True
    merged["path"] = str(path)
    return {"saved": True, "errors": [], "warnings": warnings, "config": merged}


def _write_config(path: Path, payload: Dict[str, Any]) -> None:
    """原子写入 + 保留 ``*.bak``（调用方负责已去掉 exists/path 临时字段）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copyfile(path, path.with_suffix(".json.bak"))

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _missing_config(path: Path):
    from .errors import ApiError

    return ApiError(
        "E_CONFIG_NOT_FOUND",
        f"未找到数据集配置：{path}",
        detail="请先在【数据映射】完成一次构建，生成 tcn_config.json",
    )


def validate_config(
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """校验步骤与别名（不校验数据集，那部分由 dataset_service 负责）。"""
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    steps = cfg.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append(err("E_STEPS_EMPTY", "至少需要一个步骤"))
        return errors, warnings

    # 编号 = 训练类别下标：同一编号可以出现在多个步骤（= 跨数据集同一步骤），
    # 但「唯一编号」必须从 0 连续，否则训练集里会出现永远无样本的幽灵类别。
    ids: List[int] = []
    names_by_id: Dict[int, str] = {}
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or "id" not in step:
            errors.append(err("E_STEP_INVALID", f"第 {index + 1} 个步骤缺少 id"))
            continue
        try:
            step_id = int(step["id"])
        except (TypeError, ValueError):
            errors.append(err("E_STEP_INVALID", f"第 {index + 1} 个步骤的 id 不是整数"))
            continue
        name = str(step.get("name", "")).strip()
        if step_id < 0:
            errors.append(
                err("E_STEP_INVALID", f"第 {index + 1} 个步骤的 id 不能为负数")
            )
        ids.append(step_id)
        if step_id in names_by_id and names_by_id[step_id] != name:
            errors.append(
                err(
                    "E_STEP_NAME_MISMATCH",
                    f"编号 #{step_id} 的步骤名称不一致："
                    f"{names_by_id[step_id]!r} / {name!r}（相同编号必须是同一步骤）",
                )
            )
        names_by_id.setdefault(step_id, name)
        if not name:
            warnings.append(warn("W_STEP_NO_NAME", f"步骤 {step_id} 没有名称"))
        for message in validate_match_rule(step, label=f"步骤 {step_id}"):
            errors.append(err("E_STEP_RULE_INVALID", message))

    unique_ids = sorted(set(ids))
    if unique_ids and unique_ids != list(range(len(unique_ids))):
        errors.append(
            err(
                "E_STEP_ID_GAP",
                "步骤编号必须从 0 连续（同一编号可重复，表示跨数据集的同一步骤），"
                f"当前唯一编号为 {unique_ids}",
            )
        )

    aliases = cfg.get("class_aliases")
    if aliases is not None and not isinstance(aliases, dict):
        errors.append(err("E_ALIAS_INVALID", "class_aliases 必须是对象"))

    return errors, warnings


def step_name_map(cfg: Dict[str, Any]) -> Dict[int, str]:
    """步骤 id -> 显示名（验证结果按**当前配置**显示，避免旧 checkpoint 的名称误导）。"""
    names: Dict[int, str] = {}
    for step in cfg.get("steps") or []:
        if isinstance(step, dict) and "id" in step:
            names[int(step["id"])] = str(step.get("name", f"步骤{step['id']}"))
    return names
