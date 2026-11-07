"""数据集扫描、全局类别表预览与跨数据集校验（文档 §3 / §14.1 / §14.2）。"""

from __future__ import annotations

import copy
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

from build_dataset import (
    build_global_classes,
    discover_datasets,
    normalize_dataset_name,
    read_yolo_meta,
)
from config import (
    DEFAULT_LABEL_DIR,
    DEFAULT_TCN_DIR,
    DEFAULT_YOLO_ROOT,
    PROJECT_ROOT,
)
from data_processor import load_cached_meta, natural_frame_key
from detectors import IMAGE_SUFFIXES, list_image_files

from . import config_store, model_service, source_store
from .errors import ApiError, err, warn
from .paths import (
    data_path,
    display,
    project_root,
    user_path,
    weight_path,
)
from .paths import path_key as _norm

SPLITS = ("train", "val", "test")
LOGGER = logging.getLogger(__name__)


def _split_stats(dataset_dir: Path) -> Dict[str, Dict[str, int]]:
    """统计每个划分的图片/标签数量。"""
    stats: Dict[str, Dict[str, int]] = {}
    for split in SPLITS:
        image_dir = dataset_dir / split / "images"
        label_dir = dataset_dir / split / "labels"
        images = (
            sum(
                1
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if image_dir.is_dir()
            else 0
        )
        labels = len(list(label_dir.glob("*.txt"))) if label_dir.is_dir() else 0
        stats[split] = {"images": images, "labels": labels}
    return stats


def per_class_components(dataset_params: Dict[str, Any]) -> int:
    """每类特征分量数：坐标 2 + 宽高 2（可选）+ mask 1 + 速度 2（可选）。"""
    include_size = bool(dataset_params.get("include_size", True))
    include_velocity = bool(dataset_params.get("include_velocity", True))
    return 2 + (2 if include_size else 0) + 1 + (2 if include_velocity else 0)


def collect(
    root_raw: str | Path | None = None, with_stats: bool = True
) -> List[Dict[str, Any]]:
    """扫描数据集目录，返回内嵌 ``class_map`` 的描述列表（权重按优先级解析）。

    结果 = 根目录下的子目录 + 网页“增加数据集”加入的目录 - 网页“删除”的目录。
    根目录可以不存在（只用手工数据集时），此时按空目录处理。

    ``with_stats=False`` 跳过逐划分的图片/标签计数：帧列表、缩略图、
    建库参数等只需要名称/路径的入口必须用它，否则每个缩略图都会全量扫目录。
    """
    root = user_path(root_raw or DEFAULT_YOLO_ROOT, must_exist=False)
    cfg = config_store.load_config()
    state = source_store.load()

    saved_weights: Dict[str, str] = {}
    for item in (cfg.get("source") or {}).get("datasets") or []:
        if item.get("name") and item.get("weights"):
            saved_weights[str(item["name"])] = str(item["weights"])

    try:
        datasets = discover_datasets(
            root,
            {},
            saved_weights,
            extra_dirs=state["extra_datasets"],
            exclude_dirs=state["excluded_datasets"],
            allow_empty=True,
        )
    except SystemExit as exc:  # discover_datasets 用 SystemExit 报错
        raise ApiError("E_NO_DATASET", str(exc)) from exc

    for dataset in datasets:
        if with_stats:
            dataset["splits"] = _split_stats(Path(dataset["dir"]))
        dataset["saved_weights"] = saved_weights.get(str(dataset["name"]))
        dataset["extra"] = bool(dataset.get("extra"))
        dataset.pop("dir", None)
    return datasets


def _find_dataset(dataset_name: str, root: str | Path | None = None) -> Dict[str, Any]:
    """按名称查找数据集快照；未知名称直接报错。"""
    name = str(dataset_name or "").strip()
    if not name:
        raise ApiError("E_DATASET_UNKNOWN", "请提供数据集名 dataset")
    for item in collect(root, with_stats=False):
        if str(item["name"]) == name:
            return item
    raise ApiError("E_DATASET_UNKNOWN", f"未知数据集：{name}")


def list_frames(
    dataset_name: str,
    root: str | Path | None = None,
    labels_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """列出数据集可选的帧 ID（``<数据集>__<文件名>``）。

    优先用 ``tcn_datasets/labels`` 里已生成的标签（与建库完全一致，且带
    ``<数据集>__`` 前缀）；尚未建库时回退到各划分的 ``images`` 目录。
    """
    entry = _find_dataset(dataset_name, root)
    name = str(entry["name"])
    dataset_dir = Path(str(entry["path"]))

    label_root = Path(labels_dir) if labels_dir else DEFAULT_LABEL_DIR
    frames: List[str] = []
    source = "labels"
    if label_root.is_dir():
        frames = sorted(
            (path.stem for path in label_root.glob(f"{name}__*.txt")),
            key=lambda stem: natural_frame_key(Path(stem)),
        )
    if not frames:
        source = "images"
        seen: set[str] = set()
        for split in SPLITS:
            image_dir = dataset_dir / split / "images"
            if not image_dir.is_dir():
                continue
            for path in list_image_files(image_dir):
                frame = f"{name}__{path.stem}"
                if frame not in seen:
                    seen.add(frame)
                    frames.append(frame)
    return {"dataset": name, "source": source, "count": len(frames), "frames": frames}


def frame_image_path(
    dataset_name: str,
    frame_id: str,
    root: str | Path | None = None,
) -> Path:
    """把帧 ID 映射回数据集里的图片文件（供弹窗缩略图使用）。"""
    entry = _find_dataset(dataset_name, root)
    name = str(entry["name"])
    prefix = f"{name}__"
    frame = str(frame_id or "").strip()
    if not frame.startswith(prefix):
        raise ApiError("E_FRAME_UNKNOWN", f"帧 {frame!r} 不属于数据集 {name!r}")
    stem = frame[len(prefix) :]
    if not stem or "/" in stem or "\\" in stem or stem in (".", ".."):
        raise ApiError("E_FRAME_UNKNOWN", f"非法帧 ID：{frame!r}")

    dataset_dir = Path(str(entry["path"]))
    for split in SPLITS:
        image_dir = dataset_dir / split / "images"
        if not image_dir.is_dir():
            continue
        for candidate in image_dir.glob(f"{stem}.*"):
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                return candidate.resolve()
    raise ApiError(
        "E_FRAME_NOT_FOUND",
        f"找不到帧对应的图片：{frame}",
        detail=f"已查找 {dataset_dir} 下 train/val/test 的 images 目录",
        status=404,
    )


def _resolve_dataset_dir(raw: str | Path) -> Path:
    """校验“增加数据集”选中的目录。

    与 ``paths.data_path`` 不同：允许位于工作区之外（数据集常与项目不在一处），
    但必须存在且看起来像 YOLO 数据集（有 ``classes.txt`` 或 ``dataset.yaml``）。
    """
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    path = path.resolve()

    if not path.is_dir():
        raise ApiError("E_DATASET_NOT_FOUND", f"数据集目录不存在：{path}")
    if not (path / "classes.txt").exists() and not (path / "dataset.yaml").exists():
        raise ApiError(
            "E_DATASET_INVALID",
            f"不是 YOLO 数据集目录（缺少 classes.txt / dataset.yaml）：{path}",
            detail="请选择包含 classes.txt 或 dataset.yaml 的数据集目录",
        )
    return path


def _is_within(path: Path, root: Path) -> bool:
    """``path`` 是否位于 ``root`` 内（含 root 本身）。"""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _copy_dataset_into_root(source: Path, root: Path) -> Path:
    """把根目录之外的数据集整体复制到 ``root/<目录名>``，返回目标目录。"""
    target = root / source.name
    if _norm(target) == _norm(source):
        return target
    if target.exists():
        raise ApiError(
            "E_DATASET_EXISTS",
            f"根目录下已存在同名目录：{target}",
            detail="请先删除或重命名根目录下的同名数据集，再重新加入",
        )
    try:
        root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
    except OSError as exc:
        raise ApiError(
            "E_DATASET_COPY_FAILED",
            f"复制数据集到根目录失败：{source} -> {target}",
            detail=f"{type(exc).__name__}: {exc}",
            status=500,
        ) from exc
    return target


def add_dataset(raw: str | Path, root: str | Path | None = None) -> Dict[str, Any]:
    """``POST /api/datasets/add``：把数据集加入列表。

    选中的目录不在数据集根目录下时，会先**整体复制**到 ``root/<目录名>``，
    之后按根目录内的普通数据集处理；原目录保持不变。
    """
    path = _resolve_dataset_dir(raw)
    root_dir = user_path(root or DEFAULT_YOLO_ROOT, must_exist=False)
    try:
        meta = read_yolo_meta(path)
    except SystemExit as exc:
        raise ApiError("E_DATASET_INVALID", str(exc)) from exc

    name = normalize_dataset_name(path.name)
    for existing in collect(root, with_stats=False):
        if str(existing["name"]) != name:
            continue
        if Path(str(existing["path"])).resolve() == path:
            break
        raise ApiError(
            "E_DATASET_NAME_CONFLICT",
            f"数据集名 {name} 已被占用：{existing['path']}",
            detail="数据集名用于帧名前缀，必须唯一；请先删除旧数据集或重命名目录",
        )

    copied = False
    source = path
    if not _is_within(path, root_dir):
        path = _copy_dataset_into_root(path, root_dir)
        source_store.drop_extra(source)
        source_store.clear_exclusion(path)
        copied = True
        try:
            meta = read_yolo_meta(path)
        except SystemExit as exc:
            raise ApiError("E_DATASET_INVALID", str(exc)) from exc
    else:
        source_store.add(path)

    return {
        "name": name,
        "path": display(path),
        "source": display(source) if copied else "",
        "copied": copied,
        "task": meta["task"],
        "classes": meta["classes"],
    }


def import_weights(dataset_raw: str | Path, source_raw: str | Path) -> Dict[str, Any]:
    """把选中的模型复制到数据集目录，重命名为 ``<数据集目录名>.pt``。

    同时删除该数据集目录下其它 ``.pt``，保证目录里只保留这一份模型；
    原文件若在目录之外则保留（复制，不移动）。
    """
    if not str(source_raw).strip():
        raise ApiError("E_WEIGHTS_MISSING", "请提供模型文件路径")
    dataset_dir = _resolve_dataset_dir(dataset_raw)
    source = Path(str(source_raw)).expanduser()
    if not source.is_absolute():
        source = project_root() / source
    source = source.resolve()

    if not source.is_file():
        raise ApiError("E_WEIGHTS_NOT_FOUND", f"模型文件不存在：{source}")
    if source.suffix.lower() != ".pt":
        raise ApiError("E_WEIGHTS_NOT_FOUND", f"模型必须是 .pt 文件：{source}")

    target = dataset_dir / f"{dataset_dir.name}.pt"
    copied = False
    if _norm(source) != _norm(target):
        try:
            shutil.copy2(source, target)
        except OSError as exc:
            raise ApiError(
                "E_WEIGHTS_COPY_FAILED",
                f"复制模型失败：{source} -> {target}",
                detail=f"{type(exc).__name__}: {exc}",
                status=500,
            ) from exc
        copied = True

    removed: List[str] = []
    skipped: List[Dict[str, str]] = []
    for file in sorted(dataset_dir.glob("*.pt")):
        if _norm(file) == _norm(target):
            continue
        try:
            file.unlink()
            removed.append(display(file))
        except OSError as exc:
            skipped.append(
                {"path": display(file), "reason": f"{type(exc).__name__}: {exc}"}
            )

    return {
        "dataset": display(dataset_dir),
        "path": display(target),
        "copied": copied,
        "removed": removed,
        "skipped": skipped,
    }


def remove_dataset(
    raw: str | Path,
    purge: bool = False,
    root: str | Path | None = None,
) -> Dict[str, Any]:
    """``POST /api/datasets/remove``（单条）：从列表移除，``purge`` 时同时删磁盘。"""
    return remove_datasets([raw], purge=purge, root=root)


def remove_datasets(
    paths: Sequence[str | Path],
    purge: bool = False,
    root: str | Path | None = None,
) -> Dict[str, Any]:
    """批量移除数据集记录；``purge=True`` 时连磁盘上的数据一起删除。

    删除范围（**只动数据集目录与权重，``tcn_datasets`` 下的产物一律保留**）：
    - 数据集目录（含 images/labels/classes.txt 等）；
    - 该数据集的 模型权重 ``.pt``（位于目录内随目录删除；目录外且未被其它数据集共用才删）。

    ``tcn_datasets`` 里的 ``labels/``、``labels.csv``、npy 与 ``tcn_model.pth``
    都不在这里处理：重新构建时会自动清理陈旧的逐帧标签与配置快照。
    但 ``tcn_config.json`` 的 ``steps`` 中引用被删数据集的规则会同步移除，
    避免下一次构建因“引用了不存在的数据集”而失败。

    **记录（``web_datasets.json``）在磁盘删除成功后才会移除**，失败时保持现场。
    """
    report: Dict[str, Any] = {
        "mode": "purge" if purge else "record",
        "removed": [],
        "deleted": [],
        "skipped": [],
    }
    if not paths:
        return report

    targets = [_resolve_remove_path(item) for item in paths]
    if purge:
        target_keys = {_norm(item) for item in targets}
        records = {
            _norm(item["path"]): item for item in collect(root, with_stats=False)
        }
        remaining_weights = {
            _norm(item["weights"])
            for key, item in records.items()
            if key not in target_keys and item.get("weights")
        }
        removed_names: set[str] = set()

        for path in targets:
            info = records.get(_norm(path))
            _purge_dataset_dir(path, root)
            report["deleted"].append(display(path))
            _purge_weights(info, path, remaining_weights, report)
            source_store.remove(path)
            report["removed"].append(display(path))
            name = str(info["name"]) if info else normalize_dataset_name(path.name)
            removed_names.add(name)

        pruned = _prune_steps_for_removed_datasets(removed_names)
        if pruned:
            report["steps_removed"] = pruned
        return report

    for path in targets:
        source_store.remove(path)
        report["removed"].append(display(path))
    return report


def _prune_steps_for_removed_datasets(names: set[str]) -> List[Dict[str, Any]]:
    """删除数据集后清理引用它的步骤，并把剩余编号压缩为 ``0..K-1``。

    - 只处理带 ``match.dataset`` 且命中被删数据集的步骤；跨数据集共用的
      步骤（同一编号、不同 scope）不受影响；
    - 编号即训练类别下标，删除后必须压缩，否则 ``validate_steps`` 会因
      “编号不连续”再次阻断构建；
    - 清理失败不影响已完成的磁盘删除，只记 warning。
    """
    if not names:
        return []

    cfg = config_store.load_config()
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    for step in cfg.get("steps") or []:
        if not isinstance(step, dict):
            continue
        rule = step.get("match")
        scope = rule.get("dataset") if isinstance(rule, dict) else None
        if scope and str(scope) in names:
            removed.append({"id": step.get("id"), "name": step.get("name", "")})
        else:
            kept.append(step)
    if not removed:
        return []

    try:
        unique = sorted({int(item["id"]) for item in kept if "id" in item})
        remap = {step_id: index for index, step_id in enumerate(unique)}
        for item in kept:
            if "id" in item:
                item["id"] = remap[int(item["id"])]
    except (TypeError, ValueError):
        LOGGER.warning("配置里的步骤 id 不是整数，跳过步骤清理")
        return []

    try:
        config_store.write_steps(kept)
    except ApiError as exc:
        LOGGER.warning("清理步骤失败（数据集已删除）：%s", exc.message)
        return []
    return removed


def _resolve_remove_path(raw: str | Path) -> Path:
    if not str(raw).strip():
        raise ApiError("E_DATASET_UNKNOWN", "数据集路径不能为空")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def _protected_paths(root_raw: str | Path | None) -> set[str]:
    """绝不允许删除的目录：项目根、数据集根及其祖先，以及磁盘根目录。"""
    protected = {_norm(PROJECT_ROOT), _norm(DEFAULT_YOLO_ROOT)}
    try:
        protected.add(_norm(user_path(root_raw or DEFAULT_YOLO_ROOT, must_exist=False)))
    except ApiError:
        pass
    return protected


def _purge_dataset_dir(path: Path, root_raw: str | Path | None) -> None:
    """删除数据集目录（带类型与保护目录校验）。"""
    if not path.is_dir():
        raise ApiError("E_DATASET_NOT_FOUND", f"数据集目录不存在：{path}")
    if not (path / "classes.txt").exists() and not (path / "dataset.yaml").exists():
        raise ApiError(
            "E_DATASET_INVALID",
            f"拒绝删除非数据集目录（缺少 classes.txt / dataset.yaml）：{path}",
        )
    if (
        path.parent == path
        or _norm(path) in _protected_paths(root_raw)
        or path in Path(PROJECT_ROOT).resolve().parents
    ):
        raise ApiError("E_PURGE_REFUSED", f"拒绝删除受保护的目录：{path}")

    try:
        shutil.rmtree(path)
    except OSError as exc:
        raise ApiError(
            "E_PURGE_FAILED",
            f"删除数据集目录失败：{path}",
            detail=f"{type(exc).__name__}: {exc}",
            status=500,
        ) from exc


def _purge_weights(
    info: Dict[str, Any] | None,
    dataset_dir: Path,
    remaining_weights: set[str],
    report: Dict[str, Any],
) -> None:
    """删除该数据集的 YOLO 权重；目录内的已随 rmtree 删除，共用权重跳过。"""
    if not info:
        return
    seen: set[str] = set()
    for key in ("weights", "default_weights"):
        raw = info.get(key)
        if not raw:
            continue
        candidate = Path(str(raw)).expanduser()
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        if _norm(candidate) in seen:
            continue
        seen.add(_norm(candidate))
        if not candidate.is_file() or candidate.suffix.lower() != ".pt":
            continue
        try:
            candidate.relative_to(dataset_dir)
            continue  # 位于数据集目录内，已随目录删除
        except ValueError:
            pass
        if _norm(candidate) in remaining_weights:
            report["skipped"].append(
                {"path": display(candidate), "reason": "该权重仍被其它数据集共用"}
            )
            continue
        try:
            candidate.unlink()
            report["deleted"].append(display(candidate))
        except OSError as exc:
            report["skipped"].append(
                {"path": display(candidate), "reason": f"{type(exc).__name__}: {exc}"}
            )


def global_preview(
    datasets: Sequence[Dict[str, Any]], aliases: Dict[str, str], per_class: int
) -> Dict[str, Any]:
    """按名称并集推导全局类别表（不修改入参）。"""
    copies = copy.deepcopy([dict(item) for item in datasets])
    for item in copies:
        item.setdefault("classes", [])
    names = build_global_classes(copies, dict(aliases or {}))
    slot_of: Dict[str, Dict[str, int]] = {}
    for item, copy_item in zip(datasets, copies):
        slot_of[str(item.get("name"))] = {
            str(name): int(slot)
            for name, slot in zip(copy_item["classes"], copy_item["class_map"])
        }
    return {
        "target_names": list(names),
        "num_target_classes": len(names),
        "per_class_components": per_class,
        "feature_dim": len(names) * per_class,
        "slot_of": slot_of,
    }


def kpt_shape_issues(dataset: Dict[str, Any], info: Dict[str, Any]) -> List[str]:
    """模型 ``kpt_shape`` 与数据集声明不一致时返回错误（仅 pose）。

    不一致会导致建库时关键点行无法解析、``skipped_lines`` 大量增加，
    因此作为校验错误而不是提示。
    """
    if str(dataset.get("task")) != "pose":
        return []
    want = dataset.get("kpt_shape")
    have = info.get("kpt_shape")
    if not want or not have:
        return []
    try:
        want_pair = [int(want[0]), int(want[1])]
        have_pair = [int(have[0]), int(have[1])]
    except (TypeError, ValueError, IndexError):
        return []
    if want_pair == have_pair:
        return []
    return [
        f"{dataset.get('name')}：模型 kpt_shape={have_pair} 与数据集 {want_pair} 不一致"
        "（建库会丢弃关键点行/产生 skipped_lines）"
    ]


def load_meta(tcn_dir: str | Path | None = None) -> Dict[str, Any]:
    """读取 ``dataset_meta.json``（页面刷新后恢复训练/验证页的数据集摘要）。"""
    folder = data_path(tcn_dir or DEFAULT_TCN_DIR, must_exist=False)
    path = folder / "dataset_meta.json"
    try:
        meta = load_cached_meta(folder)
    except (OSError, json.JSONDecodeError) as exc:
        raise ApiError(
            "E_META_INVALID",
            f"dataset_meta.json 读取失败：{path}",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    return {"exists": path.is_file(), "path": display(path), "meta": meta}


def scan(root_raw: str | Path | None = None, reset: bool = False) -> Dict[str, Any]:
    """``GET /api/dataset/scan`` 的实现。

    - 根目录优先级：显式参数 > 上次使用的根目录 > 配置里的 ``source.yolo_root`` > 默认；
    - 显式传入根目录时持久化到 ``web_datasets.json``，刷新页面后仍使用该目录；
    - ``reset=True``（“重新扫描”）只清空排除列表并重新发现，**不再复制任何文件**。
    """
    cfg = config_store.load_config()
    explicit = bool(str(root_raw or "").strip())
    raw = (
        root_raw
        or source_store.get_root()
        or (cfg.get("source") or {}).get("yolo_root")
        or DEFAULT_YOLO_ROOT
    )
    root = user_path(raw, must_exist=False)
    if explicit:
        source_store.set_root(root)
    if reset:
        source_store.clear_exclusions()
    datasets = collect(root)
    per_class = per_class_components(cfg.get("dataset") or {})
    global_info = global_preview(datasets, cfg.get("class_aliases") or {}, per_class)

    warnings: List[Dict[str, Any]] = []
    if not root.is_dir():
        warnings.append(warn("W_ROOT_MISSING", f"数据集根目录不存在：{display(root)}"))
    if not datasets:
        warnings.append(
            warn(
                "W_DATASET_EMPTY",
                "还没有任何数据集：可选择根目录后点“重新扫描”，或点“增加数据集”加入单个目录",
            )
        )
    for dataset in datasets:
        if dataset["task"] == "pose" and not dataset.get("kpt_shape"):
            warnings.append(
                warn("W_KPT_DEFAULT", f"{dataset['name']} 未声明 kpt_shape")
            )
        if not dataset.get("weights"):
            warnings.append(
                warn(
                    "W_WEIGHTS_MISSING",
                    f"{dataset['name']} 还没有可用权重",
                    dataset=dataset["name"],
                )
            )
        total = sum(s["images"] for s in dataset["splits"].values())
        if total == 0:
            warnings.append(
                err(
                    "E_DATASET_NO_IMAGES",
                    f"{dataset['name']} 没有任何图片",
                    dataset=dataset["name"],
                )
            )

    return {
        "root": display(root),
        "config_path": display(cfg.get("path", "")),
        "config_version": cfg.get("version"),
        "detector_defaults": (cfg.get("source") or {}).get("detector_defaults") or {},
        "datasets": datasets,
        "global": global_info,
        "steps": cfg.get("steps") or [],
        "aliases": cfg.get("class_aliases") or {},
        "warnings": warnings,
    }


def validate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """``POST /api/datasets/validate``：校验权重 + 跨数据集一致性。"""
    detector = str(payload.get("detector") or "yolo")
    root = payload.get("root")
    selections = payload.get("datasets") or []

    base = collect(root)
    by_name = {str(item["name"]): item for item in base}

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    models: Dict[str, Any] = {}

    chosen: List[Dict[str, Any]] = []
    for selection in selections:
        name = normalize_dataset_name(str(selection.get("name") or ""))
        dataset = by_name.get(name)
        if dataset is None:
            errors.append(err("E_DATASET_UNKNOWN", f"未知数据集：{name}", dataset=name))
            continue
        chosen.append(dataset)

        raw_weights = selection.get("weights")
        if detector == "labels":
            dataset["weights"] = str(raw_weights) if raw_weights else None
            continue
        if not raw_weights:
            errors.append(
                err(
                    "E_WEIGHTS_MISSING",
                    f"缺少 YOLO 权重：{name}（期望 {dataset.get('default_weights')}）",
                    dataset=name,
                )
            )
            continue
        try:
            path = weight_path(str(raw_weights))
        except ApiError as exc:
            errors.append(err(exc.code, f"{name}：{exc.message}", dataset=name))
            continue

        info = model_service.inspect(path, dataset.get("classes") or [])
        dataset["weights"] = str(path)
        models[name] = info
        if info.get("error"):
            errors.append(
                err("E_WEIGHTS_LOAD", f"{name}：{info['error']}", dataset=name)
            )
            continue
        if str(info.get("task")) != str(dataset.get("task")):
            errors.append(
                err(
                    "E_TASK_MISMATCH",
                    f"{name}：模型任务 {info.get('task')} 与数据集任务 {dataset.get('task')} 不一致",
                    dataset=name,
                )
            )
        for message in kpt_shape_issues(dataset, info):
            errors.append(err("E_KPT_SHAPE_MISMATCH", message, dataset=name))
        compare = info.get("compare") or {}
        if compare.get("missing"):
            warnings.append(
                warn(
                    "W_MODEL_MISSING_CLASSES",
                    f"{name}：模型缺少类别 {compare['missing']}，这些类别永远检测不到",
                    dataset=name,
                )
            )
        if compare.get("extra"):
            warnings.append(
                warn(
                    "W_MODEL_EXTRA_CLASSES",
                    f"{name}：模型多出类别 {compare['extra']}，这些检测会被丢弃",
                    dataset=name,
                )
            )

    # kpt_shape 必须统一（仅 pose 参与）
    shapes = {
        tuple(d["kpt_shape"])
        for d in chosen
        if d.get("task") == "pose" and d.get("kpt_shape")
    }
    if len(shapes) > 1:
        errors.append(
            err("E_KPT_SHAPE_CONFLICT", f"关键点形状不一致：{sorted(shapes)}")
        )

    if not chosen:
        errors.append(err("E_NO_DATASET", "至少选择一个数据集"))

    for dataset in chosen:
        total = sum(s["images"] for s in dataset.get("splits", {}).values())
        if total == 0:
            errors.append(
                err(
                    "E_DATASET_NO_IMAGES",
                    f"{dataset['name']} 没有任何图片",
                    dataset=dataset["name"],
                )
            )

    cfg = config_store.load_config()
    per_class = per_class_components(cfg.get("dataset") or {})
    aliases = payload.get("class_aliases")
    if aliases is None:
        aliases = cfg.get("class_aliases") or {}
    global_info = global_preview(chosen, aliases, per_class)

    return {
        "ok": not errors,
        "global": global_info,
        "models": models,
        "datasets": chosen,
        "errors": errors,
        "warnings": warnings,
    }
