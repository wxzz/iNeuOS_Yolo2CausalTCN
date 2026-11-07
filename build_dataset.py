"""build_dataset.py — 把 ``yolo_datasets/<dataset>/`` 转换成因果 TCN 训练集。

目录约定（不再支持"单数据集平铺"的旧布局）
---------------------------------------
```
yolo_datasets/
├─ hand/
│  ├─ classes.txt          # 一行一个类别名
│  ├─ dataset.yaml         # task / nc / names / kpt_shape
│  ├─ hand.pt              # 权重名 = 目录名 + .pt
│  └─ train|val|test/{images,labels}/
└─ sitting_pose/
   ├─ ...
   └─ sitting_pose.pt
```

处理流程
--------
1. 扫描 ``yolo_datasets/*/``，每个子目录视为一个数据集；
2. 按**类别名**建立全局类别表：同名合并到同一个全局槽位，异名各占一个槽位；
   可用 ``class_aliases`` 手工把同义异名合并（例：``{"left_hand": "hand_left"}``）；
3. 每个数据集用 ``<dataset>/<dataset>.pt`` 推理（或复制人工标注），
   写出带前缀的逐帧标签：``tcn_datasets/labels/<dataset>__<stem>.txt``；
4. 规则匹配时支持 ``"dataset"`` 字段，避免不同数据集的时间戳互相串味；
5. 生成 ``labels.csv`` 并调用 ``SequenceDatasetBuilder`` 导出 npy + meta。

说明
----
- 逐帧标签里的 ``class_id`` 一律是**全局 id**，因此不同数据集可以共用一套特征槽位；
- 数据集前缀用 ``__`` 分隔，默认 ``session_pattern`` 为 ``^.+?__``，
  保证滑窗不会跨数据集。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import Counter, OrderedDict
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from config import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_TCN_DIR,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_YOLO_ROOT,
    DataConfig,
)
from data_processor import (
    SequenceDatasetBuilder,
    YoloParserFactory,
    export_dataset,
    natural_frame_key,
)

LOGGER = logging.getLogger("build_dataset")

_TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{2})_(\d{2})_(\d{2})")
_SPLITS: Tuple[str, ...] = ("train",)
_CONFIG_VERSION = 2
_DEFAULT_WINDOW = DEFAULT_WINDOW_SIZE
_DEFAULT_SESSION_PATTERN = r"^.+?__"
"""默认按数据集前缀分会话，确保滑窗不跨数据集。"""


# --------------------------------------------------------------------------- #
# 数据集发现与元信息
# --------------------------------------------------------------------------- #
def read_yolo_meta(dataset_dir: Path) -> Dict[str, Any]:
    """读取单个数据集的 ``classes.txt`` / ``dataset.yaml``。"""
    names: List[str] = []
    classes_file = dataset_dir / "classes.txt"
    if classes_file.exists():
        names = [
            line.strip()
            for line in classes_file.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines()
            if line.strip()
        ]

    task, kpt_shape = "detect", None
    yaml_file = dataset_dir / "dataset.yaml"
    if yaml_file.exists():
        text = yaml_file.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"^\s*task:\s*(\w+)", text, re.MULTILINE)
        if match:
            task = match.group(1).strip().lower()
        match = re.search(
            r"^\s*kpt_shape:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]", text, re.MULTILINE
        )
        if match:
            kpt_shape = [int(match.group(1)), int(match.group(2))]

        inline = re.search(r"^\s*names:\s*\[(.*?)\]", text, re.MULTILINE | re.DOTALL)
        if inline and not names:
            names = [
                item.strip().strip("'\"")
                for item in inline.group(1).split(",")
                if item.strip()
            ]
        elif not names:
            # 支持 YAML 块状列表：
            # names:
            # - a
            # - b
            block = re.search(r"^\s*names:\s*\n((?:\s*-\s*.+\n?)+)", text, re.MULTILINE)
            if block:
                names = [
                    line.split("-", 1)[1].strip().strip("'\"")
                    for line in block.group(1).splitlines()
                    if "-" in line
                ]

    if not names:
        raise SystemExit(f"{dataset_dir} 缺少 classes.txt / names，无法确定类别")
    if kpt_shape is None and task == "pose":
        kpt_shape = [20, 3]
        LOGGER.warning("%s 未声明 kpt_shape，按默认 [20, 3] 处理", dataset_dir.name)

    return {
        "task": task,
        "classes": names,
        "kpt_shape": kpt_shape,
        "num_classes": len(names),
    }


def normalize_dataset_name(name: str) -> str:
    """数据集名用作帧名前缀，清洗掉空白与 ``__``。"""
    cleaned = re.sub(r"\s+", "_", name.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "dataset"


def _resolve_dataset_weights(
    name: str,
    default_path: Path,
    overrides: Dict[str, str],
    config_weights: Dict[str, str],
) -> "tuple[Optional[Path], Optional[str]]":
    """按优先级确定权重路径，返回 ``(可用路径, 已失效的旧路径)``。

    优先级：CLI ``--weights`` > 配置里的既有路径 > 约定路径 ``<目录>/<目录名>.pt``。
    保留配置里的失效路径到 ``weights_previous``，便于界面提示“原路径已失效”。
    """
    override = overrides.get(name)
    if override:
        path = Path(override).expanduser()
        if not path.exists():
            raise SystemExit(f"--weights 指定的权重不存在: {name} -> {path}")
        return path.resolve(), None

    previous: Optional[str] = None
    saved = config_weights.get(name)
    if saved:
        saved_path = Path(saved).expanduser()
        if saved_path.exists():
            return saved_path.resolve(), None
        previous = saved
        LOGGER.warning("[%s] 配置中的权重已失效: %s", name, saved)

    if default_path.exists():
        if previous is None:
            LOGGER.info("[%s] 使用约定权重路径: %s", name, default_path)
        else:
            LOGGER.warning("[%s] 回退到约定权重: %s", name, default_path)
        return default_path.resolve(), previous

    return None, previous


def _norm_path(value: "str | Path") -> str:
    """路径比较键：绝对化 + 大小写归一（Windows 不区分大小写）。"""
    return os.path.normcase(str(Path(str(value)).expanduser().resolve()))


def _describe_dataset(
    entry: Path,
    overrides: Dict[str, str],
    saved: Dict[str, str],
    extra: bool = False,
) -> Dict[str, Any]:
    """读取单个数据集目录的元信息并解析权重路径（``discover_datasets`` 的唯一实现）。"""
    meta = read_yolo_meta(entry)
    name = normalize_dataset_name(entry.name)
    default_weights = entry / f"{entry.name}.pt"
    resolved, previous = _resolve_dataset_weights(
        name, default_weights, overrides, saved
    )
    dataset: Dict[str, Any] = {
        "name": name,
        "dir": entry,
        "path": str(entry),
        "weights": str(resolved) if resolved else None,
        "weights_previous": previous,
        "weights_expected": str(default_weights),
        "default_weights": str(default_weights),
        **meta,
    }
    if extra:
        dataset["extra"] = True
    return dataset


def discover_datasets(
    root: Path,
    weights_override: Optional[Dict[str, str]] = None,
    config_weights: Optional[Dict[str, str]] = None,
    extra_dirs: Optional[Sequence["str | Path"]] = None,
    exclude_dirs: Optional[Sequence["str | Path"]] = None,
    allow_empty: bool = False,
) -> List[Dict[str, Any]]:
    """扫描 ``root`` 下的每个子目录，返回数据集描述列表。

    - ``extra_dirs``：网页“增加数据集”加入的目录，可以位于 ``root`` 之外；
    - ``exclude_dirs``：网页“删除数据集”后仍留在磁盘、但不参与建库的目录；
    - ``allow_empty``：没有数据集时返回空列表而不是 ``SystemExit``（网页端需要）。

    目录按路径去重，名称重复时只保留先出现的那个（帧名前缀必须唯一）。
    """
    if not root.is_dir():
        if not allow_empty:
            raise SystemExit(f"YOLO 数据集根目录不存在: {root}")
        LOGGER.warning("YOLO 数据集根目录不存在，仅使用手工加入的目录: %s", root)

    overrides = weights_override or {}
    saved = config_weights or {}
    excluded = {_norm_path(item) for item in (exclude_dirs or [])}

    candidates: List["tuple[Path, bool]"] = (
        [
            (entry, False)
            for entry in sorted(root.iterdir(), key=lambda p: p.name.lower())
        ]
        if root.is_dir()
        else []
    )
    for raw in extra_dirs or []:
        entry = Path(str(raw)).expanduser()
        if not entry.is_absolute():
            entry = Path.cwd() / entry
        candidates.append((entry.resolve(), True))

    datasets: List[Dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_names: set[str] = set()
    for entry, is_extra in candidates:
        if not entry.is_dir():
            if is_extra:
                LOGGER.warning("跳过不存在的手工数据集目录: %s", entry)
            continue
        if (
            not (entry / "classes.txt").exists()
            and not (entry / "dataset.yaml").exists()
        ):
            LOGGER.warning("跳过非数据集目录: %s", entry)
            continue
        path_key = _norm_path(entry)
        if path_key in excluded:
            LOGGER.info("跳过已删除的数据集目录: %s", entry)
            continue
        if path_key in seen_paths:
            continue
        name = normalize_dataset_name(entry.name)
        if name in seen_names:
            LOGGER.warning("数据集名重复，已跳过: %s（%s）", entry, name)
            continue

        seen_paths.add(path_key)
        seen_names.add(name)
        datasets.append(_describe_dataset(entry, overrides, saved, extra=is_extra))

    if not datasets and not allow_empty:
        raise SystemExit(f"{root} 下没有发现任何 YOLO 数据集目录")
    return datasets


def validate_datasets(datasets: Sequence[Dict[str, Any]], detector: str) -> None:
    """跨数据集一致性校验：pose 关键点形状必须统一，权重必须存在。"""
    pose_shapes = {
        tuple(ds["kpt_shape"])
        for ds in datasets
        if ds["task"] == "pose" and ds.get("kpt_shape")
    }
    if len(pose_shapes) > 1:
        raise SystemExit(
            "多个 pose 数据集的 kpt_shape 不一致，无法共用同一个特征解析器："
            f"{sorted(pose_shapes)}"
        )
    if detector == "yolo":
        missing = [ds["name"] for ds in datasets if not ds["weights"]]
        if missing:
            detail = ", ".join(
                f"{ds['name']} -> {ds['weights_expected']}"
                for ds in datasets
                if not ds["weights"]
            )
            raise SystemExit(
                f"以下数据集缺少 YOLO 权重（约定为 <目录名>.pt）：{detail}"
            )


def build_global_classes(
    datasets: Sequence[Dict[str, Any]],
    aliases: Dict[str, str],
) -> List[str]:
    """按名称建立全局类别表，并把每个数据集的本地 id 映射成全局 id。"""
    global_names: List[str] = []
    for ds in datasets:
        mapping: List[int] = []
        for local_name in ds["classes"]:
            canonical = str(aliases.get(local_name, local_name))
            if canonical not in global_names:
                global_names.append(canonical)
            mapping.append(global_names.index(canonical))
        ds["class_map"] = mapping
    return global_names


# --------------------------------------------------------------------------- #
# 步骤定义
# --------------------------------------------------------------------------- #
def parse_timestamp(stem: str) -> Optional[datetime]:
    match = _TIMESTAMP_RE.search(stem)
    if not match:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def merge_small_groups(groups: List[List[str]], min_size: int) -> List[List[str]]:
    groups = [list(group) for group in groups]
    while len(groups) > 1:
        merged = False
        for index, group in enumerate(groups):
            if len(group) >= min_size:
                continue
            if index == 0:
                groups[1] = group + groups[1]
                groups.pop(0)
            elif index == len(groups) - 1:
                groups[index - 1] = groups[index - 1] + group
                groups.pop(index)
            elif len(groups[index - 1]) <= len(groups[index + 1]):
                groups[index - 1] = groups[index - 1] + group
                groups.pop(index)
            else:
                groups[index + 1] = group + groups[index + 1]
                groups.pop(index)
            merged = True
            break
        if not merged:
            break
    return groups


def generate_default_steps(
    frames_by_dataset: "OrderedDict[str, List[Tuple[str, Optional[datetime]]]]",
    gap_seconds: float = 3.0,
    min_frames: int = _DEFAULT_WINDOW,
) -> List[Dict[str, Any]]:
    """按数据集 + 时间空隙生成默认步骤，规则带 ``dataset`` 作用域。"""
    steps: List[Dict[str, Any]] = []
    step_id = 0
    for dataset, frames in frames_by_dataset.items():
        groups: List[List[str]] = []
        current: List[str] = []
        prev_dt: Optional[datetime] = None
        for stem, dt in frames:
            if dt is not None and prev_dt is not None:
                if (dt - prev_dt).total_seconds() > gap_seconds and current:
                    groups.append(current)
                    current = []
            current.append(stem)
            if dt is not None:
                prev_dt = dt
        if current:
            groups.append(current)

        groups = merge_small_groups(groups, min_frames)
        dt_map = dict(frames)
        for index, group in enumerate(groups):
            times = [dt_map[s] for s in group if dt_map.get(s) is not None]
            match: Dict[str, Any] = {
                "dataset": dataset,
                "frames": [f"{dataset}__{s}" for s in group],
            }
            description = f"{len(group)} 帧"
            if times:
                start, end = min(times), max(times)
                match = {
                    "dataset": dataset,
                    "datetime_from": start.strftime("%Y-%m-%d %H:%M:%S"),
                    "datetime_to": end.strftime("%Y-%m-%d %H:%M:%S"),
                }
                description = (
                    f"{start.strftime('%m-%d %H:%M:%S')} ~ {end.strftime('%H:%M:%S')}，"
                    f"{len(group)} 帧"
                )
            steps.append(
                {
                    "id": step_id,
                    "name": f"{dataset}-阶段{index + 1}",
                    "description": description,
                    "match": match,
                }
            )
            step_id += 1
    return steps


def summarize_match(match: Dict[str, Any], limit: int = 3) -> str:
    """把 match 规则压成一行日志，避免打印上百个帧 id。"""
    parts: List[str] = []
    for key, value in match.items():
        if isinstance(value, (list, tuple)):
            head = ", ".join(str(v) for v in list(value)[:limit])
            more = f", …共{len(value)}" if len(value) > limit else ""
            parts.append(f"{key}=[{head}{more}]")
        else:
            parts.append(f"{key}={value}")
    return "{" + ", ".join(parts) + "}"


def _match_of(step: Dict[str, Any]) -> Dict[str, Any]:
    """安全取出 ``step["match"]``（非 dict 一律视为空规则）。"""
    match = step.get("match")
    return match if isinstance(match, dict) else {}


def _step_scope(step: Dict[str, Any]) -> Optional[str]:
    """步骤限定的数据集名（未限定返回 None）。"""
    scope = _match_of(step).get("dataset")
    return str(scope) if scope else None


def validate_match_rule(step: Dict[str, Any], label: str = "") -> List[str]:
    """校验 ``steps[].match`` 的参数格式，返回可读错误列表（空 = 通过）。

    只拦截会让 :func:`match_step` 抛异常的输入（非法正则 / 非法时间 /
    非数字帧号），不改动任何规则语义。保存配置与命令行构建共用本函数。
    """
    errors: List[str] = []
    match = step.get("match")
    where = label or f"步骤 {step.get('id')}"
    if match is None:
        return [f"{where} 没有任何匹配条件（match 不能为空）"]
    if not isinstance(match, dict):
        return [f"{where} 的 match 必须是对象"]

    dataset = match.get("dataset")
    if dataset is not None and not isinstance(dataset, str):
        errors.append(f"{where} 的 match.dataset 必须是字符串")

    frames = match.get("frames")
    if frames is not None:
        if not isinstance(frames, (list, tuple)):
            errors.append(f"{where} 的 match.frames 必须是数组")
        elif any(not isinstance(item, str) or not item.strip() for item in frames):
            errors.append(f"{where} 的 match.frames 只能包含非空字符串")

    classes = match.get("classes")
    if classes is not None:
        if not isinstance(classes, (list, tuple)):
            errors.append(f"{where} 的 match.classes 必须是数组")
        elif any(
            isinstance(item, bool) or not isinstance(item, (str, int))
            for item in classes
        ):
            errors.append(f"{where} 的 match.classes 只能包含类别名或整数 id")

    pattern = match.get("pattern")
    if pattern is not None:
        if not isinstance(pattern, str):
            errors.append(f"{where} 的 match.pattern 必须是字符串")
        else:
            try:
                re.compile(pattern)
            except re.error as exc:
                errors.append(f"{where} 的正则无效：{exc.msg}（pattern={pattern!r}）")

    for key in ("datetime_from", "datetime_to"):
        value = match.get(key)
        if value in (None, ""):
            continue
        if not isinstance(value, str):
            errors.append(f"{where} 的 {key} 必须是字符串")
            continue
        try:
            datetime.fromisoformat(value)
        except ValueError:
            errors.append(
                f"{where} 的 {key} 不是合法时间（YYYY-MM-DD HH:MM:SS）：{value!r}"
            )

    for key in ("time_from", "time_to"):
        value = match.get(key)
        if value in (None, ""):
            continue
        if not isinstance(value, str):
            errors.append(f"{where} 的 {key} 必须是字符串")
            continue
        try:
            time.fromisoformat(value)
        except ValueError:
            errors.append(f"{where} 的 {key} 不是合法时间（HH:MM[:SS]）：{value!r}")

    indexes: Dict[str, int] = {}
    for key in ("index_from", "index_to"):
        value = match.get(key)
        if value in (None, ""):
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            errors.append(f"{where} 的 {key} 必须是整数：{value!r}")
            continue
        if number < 0:
            errors.append(f"{where} 的 {key} 不能为负数：{number}")
        indexes[key] = number
    if (
        "index_from" in indexes
        and "index_to" in indexes
        and indexes["index_from"] > indexes["index_to"]
    ):
        errors.append(f"{where} 的 index_from 不能大于 index_to")

    if not has_match_condition(match):
        errors.append(
            f"{where} 没有任何匹配条件（至少选择类别/帧列表/正则/时间/帧号之一）"
        )

    return errors


def has_match_condition(match: Dict[str, Any]) -> bool:
    """规则是否包含至少一个有效条件（空数组/空字符串不算）。"""
    frames = match.get("frames")
    if isinstance(frames, (list, tuple)) and len(frames):
        return True
    classes = match.get("classes")
    if isinstance(classes, (list, tuple)) and len(classes):
        return True
    pattern = match.get("pattern")
    if isinstance(pattern, str) and pattern.strip():
        return True
    for key in ("datetime_from", "datetime_to", "time_from", "time_to"):
        value = match.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return match.get("index_from") is not None or match.get("index_to") is not None


def add_missing_dataset_steps(
    cfg: Dict[str, Any],
    frames_by_dataset: "OrderedDict[str, List[Tuple[str, Optional[datetime]]]]",
) -> List[str]:
    """为**没有任何步骤覆盖**的数据集追加默认步骤，保留已有手工步骤。

    历史行为：配置已存在时不再生成步骤，新加入的数据集全部帧 unmatched，
    被静默排除在训练集之外。现在只为缺失的数据集补步骤，并把编号接在
    现有最大编号之后；存在全局规则（未限定 dataset）时不自动补，避免重复。
    """
    steps = list(cfg.get("steps") or [])
    valid = [step for step in steps if isinstance(step, dict)]
    if any(_step_scope(step) is None for step in valid):
        return []
    covered = {_step_scope(step) for step in valid if _step_scope(step)}
    missing = [
        str(name)
        for name, frames in frames_by_dataset.items()
        if frames and str(name) not in covered
    ]
    if not missing:
        return []

    next_id = max((int(step["id"]) for step in valid if "id" in step), default=-1) + 1
    for name in missing:
        generated = generate_default_steps(
            OrderedDict([(name, frames_by_dataset[name])])
        )
        for step in generated:
            step["id"] = next_id
            next_id += 1
        steps.extend(generated)
        LOGGER.warning(
            "数据集 %s 没有步骤，已自动生成 %d 个默认步骤（请在网页中确认/调整规则）",
            name,
            len(generated),
        )
    cfg["steps"] = steps
    return missing


def validate_steps(steps: List[Dict[str, Any]], dataset_names: Sequence[str]) -> None:
    """校验步骤编号、名称与规则参数。

    编号即训练类别下标：**同一编号可以出现在多个步骤上**（跨数据集的同一步骤
    共用一个类别），但唯一编号必须从 0 连续，且同一编号的名称必须完全一致。

    引用了已删除数据集的步骤只告警：这类步骤永远不会命中，降级为告警后
    删除数据集不会让整批构建失败。
    """
    if not steps:
        raise SystemExit("tcn_config.json 的 steps 为空，至少定义一个步骤")

    ids: List[int] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or "id" not in step:
            raise SystemExit(f"第 {index + 1} 个步骤缺少 id")
        try:
            ids.append(int(step["id"]))
        except (TypeError, ValueError):
            raise SystemExit(f"第 {index + 1} 个步骤的 id 不是整数：{step['id']!r}")

    unique_ids = sorted(set(ids))
    if unique_ids != list(range(len(unique_ids))):
        raise SystemExit(
            "步骤编号必须从 0 连续（同一编号可重复，表示跨数据集的同一步骤），"
            f"当前唯一编号为 {unique_ids}"
        )

    names_by_id: Dict[int, str] = {}
    for step in steps:
        step_id = int(step["id"])
        name = str(step.get("name", ""))
        if step_id in names_by_id and names_by_id[step_id] != name:
            raise SystemExit(
                f"步骤编号 {step_id} 的名称不一致：{names_by_id[step_id]!r} / {name!r}；"
                "相同编号必须是同一步骤（名称必须相同）"
            )
        names_by_id[step_id] = name

    rule_errors: List[str] = []
    for step in steps:
        rule_errors.extend(validate_match_rule(step, label=f"步骤 {step['id']}"))
    if rule_errors:
        raise SystemExit("步骤规则参数不合法：\n- " + "\n- ".join(rule_errors))

    ids_by_name: Dict[str, Set[int]] = {}
    for step in steps:
        ids_by_name.setdefault(str(step.get("name", "")), set()).add(int(step["id"]))
    ambiguous = [name for name, id_set in ids_by_name.items() if len(id_set) > 1]
    if ambiguous:
        LOGGER.warning(
            "以下步骤名对应多个编号（不影响训练，但显示会混淆）：%s", ambiguous
        )

    known = set(dataset_names)
    for step in steps:
        scope = _match_of(step).get("dataset")
        if scope and scope not in known:
            LOGGER.warning(
                "步骤 %s 引用了不存在的数据集 %r（该步骤不会命中任何帧，已跳过）；"
                "可选：%s",
                step["id"],
                scope,
                sorted(known),
            )


def assert_steps_have_frames(
    steps: Sequence[Dict[str, Any]],
    counts: Dict[int, int],
    dataset_names: Sequence[str],
) -> None:
    """确保每个步骤都有帧命中；否则训练会出现永远无样本的“幽灵类别”。

    属于**未参与本次构建**的数据集的步骤只告警（相当于临时排除数据集），
    其余步骤命中 0 帧直接报错。
    """
    active = {str(name) for name in dataset_names}
    dead: List[str] = []
    inactive: List[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = int(step["id"])
        if counts.get(step_id, 0) > 0:
            continue
        label = f"{step_id}-{step.get('name', '')}"
        scope = _step_scope(step)
        if scope and scope not in active:
            inactive.append(f"{label}（数据集 {scope}）")
        else:
            dead.append(label)
    if dead:
        raise SystemExit(
            "以下步骤没有命中任何帧，训练会出现永远无样本的类别：\n- "
            + "\n- ".join(dead)
            + "\n请检查规则参数、类别名/帧 ID 是否与数据集一致，或删除这些步骤。"
        )
    if inactive:
        LOGGER.warning(
            "以下步骤属于未参与本次构建的数据集，会产生无样本类别（建议删除或勾选该数据集）：%s",
            inactive,
        )


def assert_steps_have_windows(
    steps: Sequence[Dict[str, Any]],
    present_ids: Set[int],
    dataset_names: Sequence[str],
) -> None:
    """确保每个步骤至少切出一个窗口（构建后、导出前调用）。"""
    active = {str(name) for name in dataset_names}
    dead: List[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = int(step["id"])
        if step_id in present_ids:
            continue
        scope = _step_scope(step)
        if scope and scope not in active:
            continue
        dead.append(f"{step_id}-{step.get('name', '')}")
    if dead:
        raise SystemExit(
            "以下步骤没有生成任何滑动窗口（命中帧太分散或帧数不足 window_size）：\n- "
            + "\n- ".join(dead)
            + "\n请补充样本/增大窗口，或删除这些步骤。"
        )


def warn_single_step_datasets(
    frames_by_dataset: Dict[str, List[Tuple[str, Optional[datetime]]]],
    steps: Sequence[Dict[str, Any]],
    counts: Dict[int, int],
) -> None:
    """提示“整个数据集只落到一个步骤”的退化情况。

    这类数据集只有单一标签，模型学到的其实是“这是哪个数据集”，
    与真实步骤时序无关，开训后验证指标会失去意义。
    """
    scoped: Dict[str, List[int]] = {}
    for step in steps:
        scope = _match_of(step).get("dataset")
        if scope:
            scoped.setdefault(str(scope), []).append(int(step["id"]))

    for dataset, frames in frames_by_dataset.items():
        step_ids = scoped.get(dataset) or []
        covered = sum(counts.get(sid, 0) for sid in step_ids)
        if len(step_ids) <= 1 and covered >= len(frames) * 0.9 and len(frames) > 1:
            LOGGER.warning(
                "数据集 %s 的 %d 帧全部落在单个步骤上，训练只能记住“数据集身份”；"
                "请在 %s 里用 pattern/时间窗拆出真实阶段",
                dataset,
                len(frames),
                "tcn_config.json",
            )


def _frame_sets(steps: Sequence[Dict[str, Any]]) -> List[Optional[Set[str]]]:
    """把每步的 ``match.frames`` 预编译成集合，顺序与 ``steps`` 对齐。

    自动生成的「帧列表」规则会把整组帧写进列表；逐帧 ``in list`` 是 O(n)，
    万级帧时会退化成 O(帧数 × 组内帧数)。这里只做一次转换，语义完全不变。
    """
    prepared: List[Optional[Set[str]]] = []
    for step in steps:
        frames = _match_of(step).get("frames")
        if isinstance(frames, (list, tuple)) and frames:
            prepared.append({str(item) for item in frames})
        else:
            prepared.append(None)
    return prepared


def _rule_hits(
    rule: Dict[str, Any],
    frame_set: Optional[Set[str]],
    frame_id: str,
    dataset: str,
    index: int,
    dt: Optional[datetime],
    present_classes: Optional[Set[int]],
    class_map: Optional[Dict[str, int]],
) -> bool:
    """判断**单个步骤**的规则是否命中；规则内部顺序固定为

    ``frames → classes → pattern → datetime → time → index``。
    """
    scope = rule.get("dataset")
    if scope and scope != dataset:
        return False

    frames = rule.get("frames")
    if frames:
        if frame_set is not None:
            if frame_id in frame_set:
                return True
        elif frame_id in frames:
            return True

    wanted = rule.get("classes")
    if wanted:
        if present_classes is None:
            # 与历史行为一致：没有该帧的类别信息时，**整条步骤规则**判定为未命中
            # （不再尝试后续的 pattern / 时间 / 序号规则）
            return False
        mapping = class_map or {}
        wanted_ids: Set[int] = set()
        items = wanted if isinstance(wanted, (list, tuple, set)) else [wanted]
        for item in items:
            if isinstance(item, int):
                wanted_ids.add(item)
                continue
            slot = mapping.get(str(item))
            if slot is not None:
                wanted_ids.add(int(slot))
        if wanted_ids & present_classes:
            return True

    pattern = rule.get("pattern")
    if pattern and re.search(pattern, frame_id):
        return True

    if rule.get("datetime_from") or rule.get("datetime_to"):
        if dt is None:
            return False
        lower = (
            datetime.fromisoformat(rule["datetime_from"])
            if rule.get("datetime_from")
            else datetime.min
        )
        upper = (
            datetime.fromisoformat(rule["datetime_to"])
            if rule.get("datetime_to")
            else datetime.max
        )
        if lower <= dt <= upper:
            return True

    if rule.get("time_from") or rule.get("time_to"):
        if dt is None:
            return False
        lower_t = (
            time.fromisoformat(rule["time_from"]) if rule.get("time_from") else time.min
        )
        upper_t = (
            time.fromisoformat(rule["time_to"]) if rule.get("time_to") else time.max
        )
        if lower_t <= dt.time() <= upper_t:
            return True

    if rule.get("index_from") is not None or rule.get("index_to") is not None:
        lower_i = int(rule.get("index_from", 1))
        upper_i = int(rule.get("index_to", 10**9))
        if lower_i <= index <= upper_i:
            return True

    return False


def match_step(
    frame_id: str,
    dataset: str,
    index: int,
    dt: Optional[datetime],
    steps: Sequence[Dict[str, Any]],
    *,
    present_classes: Optional[Set[int]] = None,
    class_map: Optional[Dict[str, int]] = None,
    frame_sets: Optional[Sequence[Optional[Set[str]]]] = None,
) -> Optional[int]:
    """返回该帧命中的**第一个**步骤 id；规则可限定 ``dataset``。

    ``present_classes`` 是该帧标签里出现的**全局类别 id**，``class_map`` 是
    当前数据集的「类别名 → 全局 id」映射（``steps[].match.classes`` 存类别名）。
    未提供 ``present_classes`` 时类别规则不会命中（向后兼容）。

    ``frame_sets`` 是 :func:`_frame_sets` 的预编译结果；批量逐帧匹配时传入可
    避免重复建集合，不传则内部自行编译（单次调用/测试用）。
    """
    prepared = frame_sets if frame_sets is not None else _frame_sets(steps)
    for position, step in enumerate(steps):
        if _rule_hits(
            _match_of(step),
            prepared[position],
            frame_id,
            dataset,
            index,
            dt,
            present_classes,
            class_map,
        ):
            return int(step["id"])
    return None


def match_all_steps(
    frame_id: str,
    dataset: str,
    index: int,
    dt: Optional[datetime],
    steps: Sequence[Dict[str, Any]],
    *,
    present_classes: Optional[Set[int]] = None,
    class_map: Optional[Dict[str, int]] = None,
    frame_sets: Optional[Sequence[Optional[Set[str]]]] = None,
) -> List[int]:
    """返回该帧命中的**全部**步骤 id（按步骤顺序、按 id 去重）。

    只用于「命中/被抢占」统计：判定结果仍然以 :func:`match_step` 的第一个为准。
    """
    prepared = frame_sets if frame_sets is not None else _frame_sets(steps)
    hits: List[int] = []
    seen: Set[int] = set()
    for position, step in enumerate(steps):
        if not _rule_hits(
            _match_of(step),
            prepared[position],
            frame_id,
            dataset,
            index,
            dt,
            present_classes,
            class_map,
        ):
            continue
        step_id = int(step["id"])
        if step_id in seen:
            # 同编号步骤（跨数据集同一步骤）只算一次
            continue
        seen.add(step_id)
        hits.append(step_id)
    return hits


# --------------------------------------------------------------------------- #
# 逐帧标签生成（预测 / 人工标注）
# --------------------------------------------------------------------------- #
def _write_dataset_label_file(
    out_dir: Path,
    dataset: str,
    stem: str,
    lines: Iterable[str],
) -> Path:
    target = out_dir / f"{dataset}__{stem}.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def write_predictions_for_dataset(
    ds: Dict[str, Any],
    splits: Sequence[str],
    out_dir: Path,
    target_names: Sequence[str],
    conf: float,
    iou: float,
    imgsz: int,
) -> "OrderedDict[str, Path]":
    """用该数据集的 YOLO 权重推理并写出全局 id 的逐帧标签。"""
    from detectors import (
        detect_many_with_yolo,
        format_detection_line,
        list_image_files,
        load_yolo,
        remap_class_ids,
    )

    images: "OrderedDict[str, Path]" = OrderedDict()
    for split in splits:
        image_dir = ds["dir"] / split / "images"
        if not image_dir.is_dir():
            LOGGER.warning("[%s] 图像目录不存在，已跳过: %s", ds["name"], image_dir)
            continue
        for path in list_image_files(image_dir):
            images.setdefault(path.stem, path)
    if not images:
        raise SystemExit(f"[{ds['name']}] 所选划分下没有图片: {splits}")

    model = load_yolo(Path(ds["weights"]))
    labels: "OrderedDict[str, Path]" = OrderedDict()
    total, empty = 0, 0
    dropped: Counter = Counter()

    stems = list(images.keys())
    results = detect_many_with_yolo(
        model,
        [images[stem] for stem in stems],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
    )
    for stem, detections in zip(stems, results):
        detections, bad = remap_class_ids(detections, model.names, target_names)
        if bad:
            dropped.update(bad)
        lines = [format_detection_line(det, ds.get("kpt_shape")) for det in detections]
        labels[stem] = _write_dataset_label_file(out_dir, ds["name"], stem, lines)
        total += len(lines)
        if not lines:
            empty += 1

    LOGGER.info(
        "[%s] 预测完成：%d 帧 / %d 个目标 / %d 帧无检测",
        ds["name"],
        len(labels),
        total,
        empty,
    )
    if dropped:
        LOGGER.warning("[%s] 类别不在全局表中，已丢弃: %s", ds["name"], dict(dropped))
    return labels


def write_labels_for_dataset(
    ds: Dict[str, Any],
    splits: Sequence[str],
    out_dir: Path,
    target_names: Sequence[str],
) -> "OrderedDict[str, Path]":
    """复制人工标注并重映射成全局类别 id。"""
    from detectors import format_detection_line

    classes = list(ds["classes"])
    factory = YoloParserFactory.default(
        kpt_shape=tuple(ds.get("kpt_shape") or (20, 3)),
        num_classes=len(classes),
    )
    name_to_global = {name: index for index, name in enumerate(target_names)}

    files: "OrderedDict[str, Path]" = OrderedDict()
    for split in splits:
        label_dir = ds["dir"] / split / "labels"
        if not label_dir.is_dir():
            LOGGER.warning("[%s] 标签目录不存在，已跳过: %s", ds["name"], label_dir)
            continue
        for path in sorted(label_dir.glob("*.txt"), key=natural_frame_key):
            detections, _ = factory.parse_file(path)
            lines: List[str] = []
            for det in detections:
                local_name = (
                    classes[det.class_id] if det.class_id < len(classes) else None
                )
                global_id = name_to_global.get(str(local_name))
                if global_id is None:
                    continue
                remapped = type(det)(
                    global_id,
                    det.cx,
                    det.cy,
                    det.width,
                    det.height,
                    det.source,
                    det.keypoints,
                )
                lines.append(format_detection_line(remapped, ds.get("kpt_shape")))
            files[path.stem] = _write_dataset_label_file(
                out_dir, ds["name"], path.stem, lines
            )

    if not files:
        raise SystemExit(f"[{ds['name']}] 所选划分下没有标签文件: {splits}")
    LOGGER.info("[%s] 人工标注已复制：%d 帧", ds["name"], len(files))
    return files


def sync_dataset_labels(
    dataset_labels: Dict[str, "OrderedDict[str, Path]"],
    out_dir: Path,
) -> int:
    """删除不属于当前数据集/帧集合的陈旧标签。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        f"{dataset}__{stem}"
        for dataset, labels in dataset_labels.items()
        for stem in labels
    }
    removed = 0
    for stale in out_dir.glob("*.txt"):
        if stale.stem not in expected:
            stale.unlink()
            removed += 1
    return removed


# --------------------------------------------------------------------------- #
# labels.csv 与 DataConfig
# --------------------------------------------------------------------------- #
def collect_frames(
    dataset_labels: Dict[str, "OrderedDict[str, Path]"],
) -> "OrderedDict[str, List[Tuple[str, Optional[datetime]]]]":
    frames: "OrderedDict[str, List[Tuple[str, Optional[datetime]]]]" = OrderedDict()
    for dataset, labels in dataset_labels.items():
        frames[dataset] = [(stem, parse_timestamp(stem)) for stem in labels]
    return frames


def _read_frame_class_ids(path: Path) -> Set[int]:
    """读取一帧标签文件里出现的全局类别 id（每行首列）。"""
    ids: Set[int] = set()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ids
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            ids.add(int(parts[0]))
        except ValueError:
            continue
    return ids


def write_labels_csv(
    frames_by_dataset: "OrderedDict[str, List[Tuple[str, Optional[datetime]]]]",
    steps: Sequence[Dict[str, Any]],
    unmatched: Any,
    out_csv: Path,
    labels_dir: Optional[Path] = None,
    class_maps: Optional[Dict[str, Dict[str, int]]] = None,
) -> Tuple[Dict[str, int], List[str]]:
    name_by_id = {
        int(step["id"]): str(step.get("name", f"步骤{step['id']}")) for step in steps
    }
    counts: Counter = Counter()
    unmatched_frames: List[str] = []
    lines = ["frame,label,step_name"]
    # 只有存在类别规则时才需要逐帧读标签，避免无谓 IO
    needs_classes = any((step.get("match") or {}).get("classes") for step in steps)
    class_maps = class_maps or {}
    frame_sets = _frame_sets(steps)

    for dataset, frames in frames_by_dataset.items():
        dataset_map = class_maps.get(str(dataset)) or {}
        for index, (stem, dt) in enumerate(frames, start=1):
            frame_id = f"{dataset}__{stem}"
            present: Optional[Set[int]] = None
            if needs_classes and labels_dir is not None:
                present = _read_frame_class_ids(Path(labels_dir) / f"{frame_id}.txt")
            step_id = match_step(
                frame_id,
                dataset,
                index,
                dt,
                steps,
                present_classes=present,
                class_map=dataset_map,
                frame_sets=frame_sets,
            )
            if step_id is None:
                if isinstance(unmatched, int):
                    step_id = int(unmatched)
                else:
                    unmatched_frames.append(frame_id)
                    continue
            counts[step_id] += 1
            lines.append(
                f"{frame_id},{step_id},{name_by_id.get(step_id, f'class_{step_id}')}"
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return dict(sorted(counts.items())), unmatched_frames


def match_shadow_stats(
    frames_by_dataset: "OrderedDict[str, List[Tuple[str, Optional[datetime]]]]",
    steps: Sequence[Dict[str, Any]],
    labels_dir: Optional[Path] = None,
    class_maps: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[int, Dict[int, int]]:
    """统计「本应命中、却被更早的步骤抢占」的帧数（只读，仅用于日志提示）。

    匹配语义不变：一帧只属于**第一个**命中的步骤。返回
    ``{被抢占的步骤 id: {抢占它的步骤 id: 帧数}}``；只有一个步骤时返回空字典。
    """
    if len({int(step["id"]) for step in steps}) < 2:
        return {}
    frame_sets = _frame_sets(steps)
    needs_classes = any((step.get("match") or {}).get("classes") for step in steps)
    class_maps = class_maps or {}
    shadowed: Dict[int, Dict[int, int]] = {}

    for dataset, frames in frames_by_dataset.items():
        dataset_map = class_maps.get(str(dataset)) or {}
        for index, (stem, dt) in enumerate(frames, start=1):
            frame_id = f"{dataset}__{stem}"
            present: Optional[Set[int]] = None
            if needs_classes and labels_dir is not None:
                present = _read_frame_class_ids(Path(labels_dir) / f"{frame_id}.txt")
            hits = match_all_steps(
                frame_id,
                dataset,
                index,
                dt,
                steps,
                present_classes=present,
                class_map=dataset_map,
                frame_sets=frame_sets,
            )
            if len(hits) < 2:
                continue
            winner = hits[0]
            for loser in hits[1:]:
                by_winner = shadowed.setdefault(loser, {})
                by_winner[winner] = by_winner.get(winner, 0) + 1
    return shadowed


def build_data_config(
    tcn_dir: Path,
    cfg: Dict[str, Any],
) -> DataConfig:
    dataset = cfg.get("dataset") or {}
    objects = cfg.get("object_classes") or {}
    steps = cfg.get("steps") or []
    names_by_id = {
        int(step["id"]): str(step.get("name", f"步骤{step['id']}")) for step in steps
    }
    step_names = tuple(names_by_id[i] for i in sorted(names_by_id))

    return DataConfig(
        label_dir=tcn_dir / "labels",
        artifact_dir=tcn_dir,
        window_size=int(dataset.get("window_size", DataConfig.window_size)),
        stride=int(dataset.get("stride", DataConfig.stride)),
        label_mode=str(dataset.get("label_mode", DataConfig.label_mode)),
        num_target_classes=int(objects["num_target_classes"]),
        target_names=tuple(objects["target_names"]),
        kpt_shape=tuple(objects.get("kpt_shape") or DataConfig.kpt_shape),
        pose_center_mode=str(
            dataset.get("pose_center_mode", DataConfig.pose_center_mode)
        ),
        missing_strategy=str(
            dataset.get("missing_strategy", DataConfig.missing_strategy)
        ),
        max_gap=int(dataset.get("max_gap", DataConfig.max_gap)),
        include_size=bool(dataset.get("include_size", DataConfig.include_size)),
        include_velocity=bool(
            dataset.get("include_velocity", DataConfig.include_velocity)
        ),
        label_source="csv",
        label_csv=tcn_dir / "labels.csv",
        session_pattern=dataset.get("session_pattern") or _DEFAULT_SESSION_PATTERN,
        step_names=step_names,
    )


# --------------------------------------------------------------------------- #
# 配置读写
# --------------------------------------------------------------------------- #
def default_dataset_params() -> Dict[str, Any]:
    return {
        "window_size": _DEFAULT_WINDOW,
        "stride": 1,
        "label_mode": "last",
        "pose_center_mode": "bbox",
        "missing_strategy": "interpolate",
        "max_gap": 5,
        "include_size": True,
        "include_velocity": True,
        "session_pattern": _DEFAULT_SESSION_PATTERN,
        "val_ratio": 0.25,
        "gap": 0,
    }


def dataset_snapshots(datasets: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """生成写进 ``tcn_config.json`` 的数据集快照。

    ``weights`` 是用户实际使用的权威路径；``default_weights`` 仅作为下次的推荐值；
    若上次路径已失效则保留 ``weights_previous`` 供界面提示。
    """
    items: List[Dict[str, Any]] = []
    for ds in datasets:
        item: Dict[str, Any] = {
            "name": ds["name"],
            "path": ds["path"],
            "weights": ds["weights"],
            "default_weights": ds.get("default_weights") or ds.get("weights_expected"),
            "task": ds["task"],
            "kpt_shape": ds["kpt_shape"],
            "classes": list(ds["classes"]),
            "class_map": list(ds["class_map"]),
        }
        if ds.get("weights_previous"):
            item["weights_previous"] = ds["weights_previous"]
        items.append(item)
    return items


def load_or_create_config(
    config_path: Path,
    datasets: Sequence[Dict[str, Any]],
    frames_by_dataset: "OrderedDict[str, List[Tuple[str, Optional[datetime]]]]",
    global_names: Sequence[str],
    aliases: Dict[str, str],
    detector_defaults: Dict[str, Any],
    yolo_root: Path,
    force: bool = False,
) -> Dict[str, Any]:
    if config_path.exists() and not force:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if cfg.get("version") != _CONFIG_VERSION:
            LOGGER.warning(
                "检测到旧版配置（version=%s），已按多数据集结构重建", cfg.get("version")
            )
            force = True
        else:
            cfg.setdefault("dataset", default_dataset_params())
            cfg.setdefault("steps", [])
            cfg.setdefault("unmatched", None)
            cfg.setdefault("class_aliases", {})
            add_missing_dataset_steps(cfg, frames_by_dataset)
            return cfg

    pose_shapes = [
        ds["kpt_shape"] for ds in datasets if ds["task"] == "pose" and ds["kpt_shape"]
    ]
    steps = generate_default_steps(frames_by_dataset)
    cfg = {
        "version": _CONFIG_VERSION,
        "source": {
            "yolo_root": str(yolo_root),
            "detector_defaults": dict(detector_defaults),
            "datasets": dataset_snapshots(datasets),
        },
        "class_aliases": dict(aliases),
        "object_classes": {
            "num_target_classes": len(global_names),
            "target_names": list(global_names),
            "kpt_shape": pose_shapes[0] if pose_shapes else None,
            "tasks": {ds["name"]: ds["task"] for ds in datasets},
        },
        "dataset": default_dataset_params(),
        "steps": steps,
        "unmatched": None,
        "_说明": (
            "多数据集：source.datasets[].class_map 把本地类别 id 映射到 object_classes.target_names；"
            "steps[].match 可用 dataset 字段限定作用域；改完重新运行 build_dataset.py 生效。"
        ),
    }
    return cfg


def save_config(config_path: Path, cfg: Dict[str, Any]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def refresh_config_dataset_state(
    cfg: Dict[str, Any],
    datasets: Sequence[Dict[str, Any]],
    global_names: Sequence[str],
    detector_defaults: Dict[str, Any],
    yolo_root: Path,
    aliases: Dict[str, str],
) -> None:
    """用最新扫描结果更新配置里的数据集快照与全局类别表。

    注意：``weights`` 取自解析后的权威值（CLI > 配置 > 约定），
    **不得**在这里回退成约定路径，否则用户指定的外部权重会被静默覆盖。
    """
    pose_shapes = [
        ds["kpt_shape"] for ds in datasets if ds["task"] == "pose" and ds["kpt_shape"]
    ]
    cfg["version"] = _CONFIG_VERSION
    cfg["source"] = {
        "yolo_root": str(yolo_root),
        "detector_defaults": dict(detector_defaults),
        "datasets": dataset_snapshots(datasets),
    }
    cfg["class_aliases"] = dict(aliases)
    cfg["object_classes"] = {
        "num_target_classes": len(global_names),
        "target_names": list(global_names),
        "kpt_shape": pose_shapes[0] if pose_shapes else None,
        "tasks": {ds["name"]: ds["task"] for ds in datasets},
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_weight_overrides(items: Sequence[str]) -> Dict[str, str]:
    """把 ``name=path`` 列表解析成字典（路径可含 ``=``，只切第一个）。"""
    result: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--weights 需要 NAME=PATH 形式，收到: {item}")
        name, path = item.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise SystemExit(f"--weights 需要 NAME=PATH 形式，收到: {item}")
        result[name] = path
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="yolo_datasets/<dataset>/ -> tcn_datasets（多数据集）"
    )
    parser.add_argument("--yolo-root", type=Path, default=DEFAULT_YOLO_ROOT)
    parser.add_argument("--tcn-dir", type=Path, default=DEFAULT_TCN_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument(
        "--detector",
        choices=["yolo", "labels"],
        default="yolo",
        help="特征来源：yolo=用各数据集自己的权重预测；labels=复制人工标注",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--split",
        nargs="+",
        default=list(_SPLITS),
        help="每个数据集使用哪些划分，默认只用 train",
    )
    parser.add_argument(
        "--reset-config", action="store_true", help="重建步骤（会覆盖手工规则）"
    )
    parser.add_argument(
        "--config-only", action="store_true", help="只生成标签与 labels.csv"
    )
    parser.add_argument(
        "--weights",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="覆盖某个数据集的 YOLO 权重，可重复，例：--weights hand=D:/models/hand.pt",
    )
    parser.add_argument(
        "--dataset-dir",
        action="append",
        default=[],
        metavar="PATH",
        help="额外加入的数据集目录（可不在 --yolo-root 下），可重复",
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=[],
        metavar="PATH",
        help="排除某个数据集目录（仍保留磁盘文件），可重复",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s | %(message)s",
    )

    tcn_dir: Path = args.tcn_dir

    # 先读旧配置（别名 + 已保存的权重路径），再扫描数据集：权重的优先级依赖旧配置
    aliases: Dict[str, str] = {}
    saved_weights: Dict[str, str] = {}
    if args.config.exists():
        try:
            raw_cfg = json.loads(args.config.read_text(encoding="utf-8"))
            aliases = raw_cfg.get("class_aliases") or {}
            for ds_cfg in (raw_cfg.get("source") or {}).get("datasets") or []:
                if ds_cfg.get("name") and ds_cfg.get("weights"):
                    saved_weights[str(ds_cfg["name"])] = str(ds_cfg["weights"])
        except json.JSONDecodeError:
            LOGGER.warning("旧配置不是合法 JSON，忽略其中内容")

    datasets = discover_datasets(
        args.yolo_root,
        weights_override=parse_weight_overrides(args.weights),
        config_weights=saved_weights,
        extra_dirs=args.dataset_dir,
        exclude_dirs=args.exclude_dataset,
    )
    validate_datasets(datasets, args.detector)
    splits = tuple(str(item) for item in args.split)
    detector_defaults = {
        "type": args.detector,
        "conf": float(args.conf),
        "iou": float(args.iou),
        "imgsz": int(args.imgsz),
    }

    LOGGER.info(
        "发现 %d 个数据集：%s",
        len(datasets),
        ", ".join(
            f"{ds['name']}({ds['task']},{ds['num_classes']}类)" for ds in datasets
        ),
    )

    # 别名已在上面读入
    global_names = build_global_classes(datasets, aliases)
    LOGGER.info("全局类别表（%d）：%s", len(global_names), global_names)

    # 逐数据集生成逐帧标签
    out_labels = tcn_dir / "labels"
    dataset_labels: Dict[str, "OrderedDict[str, Path]"] = {}
    for ds in datasets:
        if args.detector == "yolo":
            dataset_labels[ds["name"]] = write_predictions_for_dataset(
                ds, splits, out_labels, global_names, args.conf, args.iou, args.imgsz
            )
        else:
            dataset_labels[ds["name"]] = write_labels_for_dataset(
                ds, splits, out_labels, global_names
            )

    removed = sync_dataset_labels(dataset_labels, out_labels)
    if removed:
        LOGGER.info("已清理 %d 个陈旧标签文件", removed)

    frames_by_dataset = collect_frames(dataset_labels)
    cfg = load_or_create_config(
        args.config,
        datasets,
        frames_by_dataset,
        global_names,
        aliases,
        detector_defaults,
        args.yolo_root,
        force=args.reset_config,
    )
    refresh_config_dataset_state(
        cfg, datasets, global_names, detector_defaults, args.yolo_root, aliases
    )
    validate_steps(cfg["steps"], [ds["name"] for ds in datasets])
    cfg.setdefault("dataset", default_dataset_params())
    cfg["dataset"].setdefault("session_pattern", _DEFAULT_SESSION_PATTERN)
    save_config(args.config, cfg)

    class_maps = {
        str(ds["name"]): {
            str(name): int(slot)
            for name, slot in zip(ds.get("classes") or [], ds.get("class_map") or [])
        }
        for ds in datasets
    }
    counts, unmatched = write_labels_csv(
        frames_by_dataset,
        cfg["steps"],
        cfg.get("unmatched"),
        tcn_dir / "labels.csv",
        labels_dir=out_labels,
        class_maps=class_maps,
    )
    if unmatched:
        LOGGER.warning(
            "有 %d 帧未命中任何步骤，将被排除：%s ...",
            len(unmatched),
            ", ".join(unmatched[:5]),
        )

    dataset_names = [ds["name"] for ds in datasets]
    assert_steps_have_frames(cfg["steps"], counts, dataset_names)
    shadowed = match_shadow_stats(
        frames_by_dataset, cfg["steps"], labels_dir=out_labels, class_maps=class_maps
    )

    LOGGER.info("步骤配置: %s", args.config)
    seen_step_ids: Set[int] = set()
    for step in cfg["steps"]:
        step_id = int(step["id"])
        if step_id in seen_step_ids:
            LOGGER.info(
                "  步骤 %d | %-16s | 与前面的同编号步骤合并为同一类别 | %s",
                step_id,
                step.get("name", ""),
                summarize_match(step.get("match") or {}),
            )
            continue
        seen_step_ids.add(step_id)
        LOGGER.info(
            "  步骤 %d | %-16s | %3d 帧 | %s",
            step_id,
            step.get("name", ""),
            counts.get(step_id, 0),
            summarize_match(step.get("match") or {}),
        )
        if shadowed.get(step_id):
            detail = "，".join(
                f"被步骤 {winner} 抢占 {num} 帧"
                for winner, num in sorted(shadowed[step_id].items())
            )
            LOGGER.warning(
                "  步骤 %d（%s）：另有帧本应命中但%s；同一帧只归第一个命中的步骤，"
                "如需归到该步骤请把它的编号调小或收窄前面的规则。",
                step_id,
                step.get("name", ""),
                detail,
            )
    warn_single_step_datasets(frames_by_dataset, cfg["steps"], counts)

    data_cfg = build_data_config(tcn_dir, cfg)
    data_cfg.validate()

    if args.config_only:
        print(
            json.dumps(
                {
                    "steps": counts,
                    "unmatched": len(unmatched),
                    "target_names": global_names,
                },
                ensure_ascii=False,
            )
        )
        return

    dataset = SequenceDatasetBuilder(data_cfg).build()
    present_ids = {
        int(step_id) for step_id in (dataset.meta.get("label_distribution") or {})
    }
    assert_steps_have_windows(cfg["steps"], present_ids, dataset_names)
    meta = export_dataset(
        dataset,
        data_cfg,
        val_ratio=float(cfg["dataset"].get("val_ratio", 0.25)),
        gap=int(cfg["dataset"].get("gap", 0)),
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    print(f"\n数据集已写入: {tcn_dir}")


if __name__ == "__main__":
    main()
