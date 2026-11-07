"""网页“增加 / 删除数据集”的持久化记录。

只记录**网页层面**的数据集增减，不写 ``tcn_config.json``：
那份配置由 ``build_dataset.py`` 维护，每次建库都会用扫描结果覆盖。

文件位置：与 ``tcn_config.json`` 同目录的 ``web_datasets.json``。

```json
{
  "extra_datasets": ["D:/models/my_dataset"],
  "excluded_datasets": ["C:/.../yolo_datasets/hand"]
}
```

- ``extra_datasets``：手工增加的目录（可位于数据集根目录之外），建库时以
  ``build_dataset.py --dataset-dir`` 传入；
- ``excluded_datasets``：从列表删除、但磁盘上仍存在的目录，建库时以
  ``--exclude-dataset`` 传入，避免“列表里删了、建库又跑起来”。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from . import config_store
from .paths import path_key as _norm

STATE_FILENAME = "web_datasets.json"


def state_path() -> Path:
    """状态文件路径（与 ``tcn_config.json`` 同目录）。"""
    return Path(config_store.config_path()).parent / STATE_FILENAME


def _dedupe(items: List[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        key = _norm(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(str(item))
    return result


def load() -> Dict[str, List[str]]:
    """读取状态；文件缺失或损坏时返回空记录，不阻塞页面。"""
    path = state_path()
    payload: Dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return {
        "extra_datasets": _dedupe(
            [
                str(item)
                for item in payload.get("extra_datasets") or []
                if str(item).strip()
            ]
        ),
        "excluded_datasets": _dedupe(
            [
                str(item)
                for item in payload.get("excluded_datasets") or []
                if str(item).strip()
            ]
        ),
        "root": str(payload.get("root") or "").strip(),
    }


def save(state: Dict[str, List[str]]) -> None:
    """原子写入（``*.tmp`` + ``os.replace``），避免半截文件。"""
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "extra_datasets": _dedupe(state.get("extra_datasets") or []),
        "excluded_datasets": _dedupe(state.get("excluded_datasets") or []),
        "root": str(state.get("root") or "").strip(),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def set_root(path: str | Path) -> None:
    """记住用户上次使用的数据集根目录（留空表示恢复默认）。"""
    state = load()
    root = str(path).strip()
    if state.get("root") == root:
        return
    state["root"] = root
    save(state)


def get_root() -> str:
    """用户上次使用的数据集根目录；未设置时返回空字符串。"""
    return str(load().get("root") or "")


def add(path: str | Path) -> None:
    """把目录加入“手工数据集”，并从排除列表中移除（相当于撤销删除）。"""
    key = _norm(path)
    state = load()
    state["extra_datasets"] = _dedupe([*state["extra_datasets"], str(path)])
    state["excluded_datasets"] = [
        item for item in state["excluded_datasets"] if _norm(item) != key
    ]
    save(state)


def remove(path: str | Path) -> None:
    """从列表移除目录：删掉手工记录并写入排除列表（不删除磁盘文件）。"""
    key = _norm(path)
    state = load()
    state["extra_datasets"] = [
        item for item in state["extra_datasets"] if _norm(item) != key
    ]
    state["excluded_datasets"] = _dedupe([*state["excluded_datasets"], str(path)])
    save(state)


def drop_extra(path: str | Path) -> None:
    """只移除手工数据集记录，不写入排除列表（目录已复制进根目录时使用）。"""
    key = _norm(path)
    state = load()
    items = [item for item in state["extra_datasets"] if _norm(item) != key]
    if len(items) != len(state["extra_datasets"]):
        state["extra_datasets"] = items
        save(state)


def clear_exclusion(path: str | Path) -> None:
    """取消对某个目录的排除（目录重新可用时使用）。"""
    key = _norm(path)
    state = load()
    items = [item for item in state["excluded_datasets"] if _norm(item) != key]
    if len(items) != len(state["excluded_datasets"]):
        state["excluded_datasets"] = items
        save(state)


def extra_dirs() -> List[Path]:
    """存在的手工数据集目录（建库时传给 ``--dataset-dir``）。"""
    return [Path(item) for item in load()["extra_datasets"] if Path(item).is_dir()]


def excluded_dirs() -> List[str]:
    """排除列表（建库时传给 ``--exclude-dataset``）。"""
    return list(load()["excluded_datasets"])


def clear_exclusions() -> List[str]:
    """清空排除列表并返回被清除的路径。

    “重新扫描”表示按根目录重新发现全部数据集，因此会调用这里：
    仅删除记录（未删磁盘）的数据集会重新出现；已删除目录或已移出
    ``extra_datasets`` 的手工数据集不会凭空恢复。
    """
    state = load()
    cleared = list(state["excluded_datasets"])
    if cleared:
        state["excluded_datasets"] = []
        save(state)
    return cleared
