"""模型权重元信息与类别比对（文档 §3.4 / §16.4）。

加载复用 ``detectors.load_yolo``，不在这里重新实现 YOLO 调用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from detectors import load_yolo

_CACHE: Dict[str, Dict[str, Any]] = {}


def _kpt_shape(model: Any) -> Any:
    """尽力从 ultralytics 模型里取 ``kpt_shape``（不同版本位置不同）。"""
    candidates = (
        getattr(getattr(model, "model", None), "kpt_shape", None),
        getattr(model, "kpt_shape", None),
        (getattr(model, "overrides", None) or {}).get("kpt_shape"),
    )
    for value in candidates:
        if value:
            return list(value)
    return None


def _compare(model_names: Sequence[str], classes: Sequence[str]) -> Dict[str, Any]:
    have = [str(name) for name in model_names]
    want = [str(name) for name in classes]
    return {
        "missing": [name for name in want if name not in have],
        "extra": [name for name in have if name not in want],
    }


def inspect(weights: Path, classes: Sequence[str]) -> Dict[str, Any]:
    """返回权重元信息；加载失败时返回 ``{"error": "..."}`` 而不是抛异常。"""
    path = Path(weights)
    key = f"{path}|{path.stat().st_mtime if path.exists() else 0}"
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    try:
        model = load_yolo(path)
        names_map = getattr(model, "names", {}) or {}
        names: List[str] = [str(names_map[i]) for i in sorted(names_map)]
        stat = path.stat()
        info: Dict[str, Any] = {
            "path": str(path),
            "name": path.name,
            "size": int(stat.st_size),
            "mtime": _fmt_time(stat.st_mtime),
            "task": str(getattr(model, "task", "") or ""),
            "names": names,
            "kpt_shape": _kpt_shape(model),
            "compare": _compare(names, classes),
        }
    except SystemExit as exc:  # load_yolo 用 SystemExit 报错
        info = {"path": str(path), "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - 原样回传给前端
        info = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}

    _CACHE[key] = info
    return info


def _fmt_time(timestamp: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
