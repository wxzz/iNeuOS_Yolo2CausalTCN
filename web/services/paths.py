"""路径守卫（文档 §16.9）。

规则：
- 数据路径（数据集根 / 产出目录 / checkpoint / 配置）必须在项目根内，防目录穿越；
- **权重 `.pt` 例外**：允许工作区外的任意绝对路径，但只读；
- **验证来源（图片目录 / 单张图片）例外**：待验证图片常被复制到项目外，
  允许工作区外任意绝对路径，但只读（不写入、不删除）。
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from config import PROJECT_ROOT

from .errors import ApiError

TMP_DIR = PROJECT_ROOT / ".web_tmp"


def project_root() -> Path:
    return PROJECT_ROOT


def path_key(value: str | Path) -> str:
    """路径比较键：绝对化 + 大小写归一（Windows 不区分大小写）。

    仅供“是不是同一个路径”的判断使用，不负责存在性与越界校验。
    """
    return os.path.normcase(str(Path(str(value)).expanduser().resolve()))


def data_path(raw: str | Path, must_exist: bool = False) -> Path:
    """校验并返回工作区内的绝对路径。"""
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()

    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ApiError(
            "E_PATH_OUTSIDE",
            f"路径不允许：{path}",
            detail=f"数据路径必须位于 {PROJECT_ROOT} 内",
        ) from exc

    if must_exist and not path.exists():
        raise ApiError("E_PATH_NOT_FOUND", f"路径不存在：{path}")
    return path


def weight_path(raw: str | Path) -> Path:
    """校验权重路径（允许工作区外，仅要求存在且是 .pt 文件）。"""
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()

    if not path.exists():
        raise ApiError("E_WEIGHTS_NOT_FOUND", f"文件不存在：{path}")
    if path.suffix.lower() != ".pt":
        raise ApiError("E_WEIGHTS_NOT_FOUND", f"权重必须是 .pt 文件：{path}")
    return path


def read_only_path(raw: str | Path, must_exist: bool = True) -> Path:
    """校验只读资源路径（验证来源：图片目录 / 单张图片）。

    允许位于工作区之外（待验证图片常与项目不在一处），只做读取；
    调用方不得写入或删除该路径下的内容。
    """
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()

    if must_exist and not path.exists():
        raise ApiError("E_PATH_NOT_FOUND", f"路径不存在：{path}")
    return path


def user_path(raw: str | Path, must_exist: bool = True) -> Path:
    """用户显式选择的路径（数据集根目录等）：允许位于工作区之外。

    与 :func:`data_path` 的区别：``tcn_dir`` / 配置 / 默认 checkpoint 等
    **产物**必须留在工作区内，而用户选择的数据集根目录可能在任意盘符。
    读操作直接使用；写操作（如“增加数据集”复制进根目录）必须是用户明确触发。
    """
    return read_only_path(raw, must_exist=must_exist)


def checkpoint_path(raw: str | Path, must_exist: bool = False) -> Path:
    """checkpoint 路径：允许工作区外（用户通过弹窗明确选择）。

    训练需要写入新文件（``must_exist=False``），验证/推理需要读取已有文件
    （``must_exist=True``）。只校验后缀，不限制盘符——与 YOLO 权重策略一致。
    """
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()

    if path.suffix.lower() not in (".pth", ".pt"):
        raise ApiError(
            "E_CHECKPOINT_INVALID", f"checkpoint 必须是 .pth / .pt 文件：{path}"
        )
    if path.exists() and path.is_dir():
        raise ApiError("E_CHECKPOINT_INVALID", f"checkpoint 不能是目录：{path}")
    if must_exist and not path.is_file():
        raise ApiError("E_CHECKPOINT_NOT_FOUND", f"权重不存在：{path}")
    return path


def tmp_dir() -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    return TMP_DIR


def cleanup_tmp(max_age_hours: float = 24.0) -> None:
    """清理 ``.web_tmp`` 下超过保留期的临时目录/文件。

    验证失败时会保留现场（便于复现），因此需要定期回收，避免无限增长。
    """
    if not TMP_DIR.is_dir():
        return
    cutoff = time.time() - max(0.0, float(max_age_hours)) * 3600.0
    for item in TMP_DIR.iterdir():
        try:
            if item.stat().st_mtime >= cutoff:
                continue
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
        except OSError:
            continue


def display(path: Path | str | None) -> str:
    """统一用正斜杠展示路径，便于跨平台复制。"""
    if not path:
        return ""
    return str(path).replace("\\", "/")
