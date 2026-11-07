"""inference.py — 因果 TCN 步骤识别推理（多数据集）。

支持四种输入源：

1. ``--source <labels目录>``：回放合并后的逐帧标签（全局类别 id）；
   存在 ``labels.csv`` 时只回放其中参与训练的帧，并按数据集/视频段重置窗口；
2. ``--source <images目录>``：目录内逐帧检测后送入 TCN（需确定所属数据集）；
3. ``--source <单张图片>``：用同目录前面的图片预热窗口，只输出这张图的判定；
4. ``--npy X_val.npy``：对已导出的窗口数组做批量推理。

数据集的权重与类别映射来自 ``tcn_datasets/tcn_config.json``：
``source.datasets[].weights`` / ``class_map``。图像模式可用 ``--dataset`` 指定，
不指定时会根据路径自动识别（``--source`` 位于 ``yolo_datasets/<name>/`` 下）。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from config import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_LABEL_DIR,
    DEFAULT_WINDOW_SIZE,
    TCNConfig,
)
from data_processor import (
    Detection,
    CausalMissingFiller,
    FrameFeatureAssembler,
    YoloParserFactory,
    aggregate_detections,
    natural_frame_key,
    session_key,
)
from detectors import (
    IMAGE_SUFFIXES,
    detect_many_with_yolo,
    list_image_files,
    load_yolo,
    remap_class_ids,
)
from model import CausalTCNClassifier, load_checkpoint

LOGGER = logging.getLogger(__name__)


@dataclass
class Prediction:
    """单次推理结果。"""

    label: int
    name: str
    confidence: float
    frame_id: str = ""


class RealtimeActionRecognizer:
    """流式步骤识别器：逐帧喂入检测结果，窗口满后输出步骤与置信度。

    Args:
        checkpoint: ``train.py`` 保存的权重文件。
        device: ``auto`` / ``cpu`` / ``cuda``。
        window_size: 覆盖权重里记录的时间窗长度（一般不用改）。
        smooth_window: 概率滑动平均的历史预测数，抑制单帧抖动。
    """

    def __init__(
        self,
        checkpoint: Path,
        device: str = "auto",
        window_size: Optional[int] = None,
        smooth_window: int = 5,
    ) -> None:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(
                f"权重不存在: {path}，请先运行 build_dataset.py + train.py"
            )

        payload = load_checkpoint(path)
        model_config = dict(payload["model_config"])
        model_config["channels"] = tuple(model_config.get("channels", (64, 64, 64, 64)))
        self.model_cfg = TCNConfig(**model_config)
        self.model = CausalTCNClassifier(self.model_cfg)
        self.model.load_state_dict(payload["state_dict"])

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device).eval()

        data_cfg: Dict = payload.get("data_config", {})
        self.window_size = int(
            window_size or payload.get("window_size") or DEFAULT_WINDOW_SIZE
        )
        self.kpt_shape = tuple(data_cfg.get("kpt_shape") or (20, 3))
        self.assembler = FrameFeatureAssembler(
            num_classes=int(data_cfg.get("num_target_classes", 4)),
            include_size=bool(data_cfg.get("include_size", True)),
            include_velocity=bool(data_cfg.get("include_velocity", True)),
        )
        if self.assembler.feature_dim != self.model_cfg.num_features:
            raise ValueError(
                f"特征维度不匹配: 权重期望 {self.model_cfg.num_features}，"
                f"当前布局给出 {self.assembler.feature_dim}"
            )

        # 缺失填充必须与建库时一致，否则同一条序列会得到不同的特征值。
        # 旧权重没有这两个字段时回落到默认策略（与 DataConfig 默认值一致）。
        self.missing_strategy = str(data_cfg.get("missing_strategy", "interpolate"))
        self.missing_max_gap = int(data_cfg.get("max_gap", 5))
        self.filler = CausalMissingFiller(
            self.assembler.num_classes, self.missing_max_gap, self.missing_strategy
        )

        normalization = payload.get("normalization", {})
        self.mean = np.asarray(
            normalization.get("mean", np.zeros(self.model_cfg.num_features)),
            dtype=np.float32,
        ).reshape(1, -1, 1)
        self.std = np.asarray(
            normalization.get("std", np.ones(self.model_cfg.num_features)),
            dtype=np.float32,
        ).reshape(1, -1, 1)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)

        self.class_names: List[str] = list(payload.get("class_names", [])) or [
            f"class_{i}" for i in range(self.model_cfg.num_classes)
        ]
        self.metrics = payload.get("metrics", {})
        self.epoch = payload.get("epoch")

        # buffer 中每个元素是一帧的 (F,) 特征，stack(axis=1) 后得到 (F, W) 的输入矩阵
        self.buffer: Deque[np.ndarray] = deque(maxlen=self.window_size)
        self.prob_history: Deque[np.ndarray] = deque(maxlen=max(int(smooth_window), 1))
        self.assembler.reset()

    # -- 公共接口 ---------------------------------------------------------
    def reset(self) -> None:
        """清空缓冲，用于换线/换视频段。"""
        self.buffer.clear()
        self.prob_history.clear()
        self.assembler.reset()
        self.filler.reset()

    def push_detections(
        self, detections: Sequence[Detection], frame_id: str = ""
    ) -> Optional[Prediction]:
        """推入一帧的原始检测结果（``Detection`` 列表）。"""
        frame_matrix = aggregate_detections(detections, self.assembler.num_classes)
        return self.push_frame(frame_matrix, frame_id)

    def push_frame(
        self, frame_matrix: np.ndarray, frame_id: str = ""
    ) -> Optional[Prediction]:
        """推入一帧已聚合的 ``(C, 5)`` 槽位矩阵（先做因果缺失填充）。"""
        filled = self.filler.step(frame_matrix)
        feature = self.assembler.step(filled)
        self.buffer.append(feature)
        if len(self.buffer) < self.window_size:
            return None

        batch = np.ascontiguousarray(np.stack(self.buffer, axis=1), dtype=np.float32)
        batch = (batch[None, ...] - self.mean) / self.std
        tensor = torch.from_numpy(batch).to(self.device)

        with torch.no_grad():
            probs = self.model.predict_proba(tensor)[0].cpu().numpy()
        self.prob_history.append(probs)

        smoothed = np.mean(np.stack(self.prob_history, axis=0), axis=0)
        label = int(np.argmax(smoothed))
        name = (
            self.class_names[label]
            if label < len(self.class_names)
            else f"class_{label}"
        )
        return Prediction(label, name, float(smoothed[label]), frame_id)


# --------------------------------------------------------------------------- #
# 检测结果来源
# --------------------------------------------------------------------------- #
def list_frames(directory: Path, suffixes: Optional[set] = None) -> List[Path]:
    """按自然时间序列出目录下的文件（支持 txt 或图片）。"""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"目录不存在: {directory}")
    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and (suffixes is None or path.suffix.lower() in suffixes)
    ]
    if not files:
        raise FileNotFoundError(f"目录下没有匹配文件: {directory}")
    return sorted(files, key=natural_frame_key)


def detect_with_labels(
    factory: YoloParserFactory,
    labels_dir: Path,
    frame_id: str,
    class_map: Optional[Dict[int, int]] = None,
) -> List[Detection]:
    """用同名标注文件当作检测结果（无 YOLO 权重时的链路自测）。

    ``class_map`` 把数据集的本地类别 id 映射成全局 id。
    """
    label_path = Path(labels_dir) / f"{frame_id}.txt"
    if not label_path.exists():
        return []
    detections, _ = factory.parse_file(label_path)
    if not class_map:
        return detections

    mapped: List[Detection] = []
    for det in detections:
        global_id = class_map.get(int(det.class_id))
        if global_id is None:
            continue
        mapped.append(
            Detection(
                global_id,
                det.cx,
                det.cy,
                det.width,
                det.height,
                det.source,
                det.keypoints,
            )
        )
    return mapped


def find_sibling_labels(image_dir: Path) -> Optional[Path]:
    """在 images 同级或上一级寻找 labels 目录。"""
    for candidate in (image_dir.parent / "labels", image_dir.parent.parent / "labels"):
        if candidate.is_dir():
            return candidate
    return None


# --------------------------------------------------------------------------- #
# 运行模式
# --------------------------------------------------------------------------- #
def run_stream(
    recognizer: RealtimeActionRecognizer,
    frames: Iterable[Tuple[str, Sequence[Detection]]],
    fps: float = 0.0,
    save_csv: Optional[Path] = None,
    verbose: bool = True,
    session_of: Optional[Callable[[str], str]] = None,
) -> List[Prediction]:
    """逐帧推流并打印结果。

    ``verbose=False`` 时只在最后打印一条（单张图片预热模式使用）。
    ``session_of`` 用于标签/多段回放：帧 ID 的会话键变化时先 ``reset``，
    避免时间窗跨数据集或视频段（与建库的 ``session_pattern`` 同一规则）。
    """
    predictions: List[Prediction] = []
    delay = 1.0 / fps if fps and fps > 0 else 0.0
    current_session: Optional[str] = None

    for frame_id, detections in frames:
        if session_of is not None:
            session = session_of(frame_id)
            if current_session is None:
                current_session = session
            elif session != current_session:
                recognizer.reset()
                current_session = session
                if verbose:
                    print(f"-- 会话切换：{session}，已重置时间窗 --")

        prediction = recognizer.push_detections(detections, frame_id=frame_id)
        if prediction is not None:
            predictions.append(prediction)
            if verbose:
                print(
                    f"[{len(recognizer.buffer):>2}/{recognizer.window_size}] {frame_id}"
                    f" -> 步骤 {prediction.label} ({prediction.name})"
                    f" 置信度 {prediction.confidence:.3f}"
                )
        if delay:
            time.sleep(delay)

    if not verbose and predictions:
        last = predictions[-1]
        print(
            f"{last.frame_id} -> 步骤 {last.label} ({last.name})"
            f" 置信度 {last.confidence:.3f}"
        )

    if save_csv is not None and predictions:
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        with Path(save_csv).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "step_id", "step_name", "confidence"])
            for item in predictions:
                writer.writerow(
                    [item.frame_id, item.label, item.name, f"{item.confidence:.6f}"]
                )
        print(f"预测已保存: {Path(save_csv).resolve()}")

    return predictions


def load_replay_frames(
    labels_dir: Path, csv_path: Optional[Path] = None
) -> Tuple[List[str], str]:
    """确定标签回放使用的帧 ID 列表与来源（``csv`` / ``labels``）。

    - 有 ``labels.csv``：只回放其中列出的帧（也就是真正参与训练的样本），
      保持 CSV 行顺序；文件存在但为空时返回空列表，由调用方给出可读错误；
    - 否则回退到目录内全部 ``*.txt``（按自然时间序）。
    """
    if csv_path is not None and Path(csv_path).is_file():
        frame_ids: List[str] = []
        seen: set[str] = set()
        with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                frame = str(row.get("frame") or "").strip()
                if frame and frame not in seen:
                    seen.add(frame)
                    frame_ids.append(frame)
        return frame_ids, "csv"

    files = list_frames(labels_dir, {".txt"})
    return [path.stem for path in files], "labels"


def load_session_pattern(config_path: Path) -> Optional[str]:
    """读取配置里的 ``dataset.session_pattern``（回放分段必须与建库一致）。"""
    path = Path(config_path)
    if not path.exists():
        return None
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return str((cfg.get("dataset") or {}).get("session_pattern") or "") or None


def build_frames_from_labels(
    recognizer: RealtimeActionRecognizer,
    label_dir: Path,
    factory: Optional[YoloParserFactory] = None,
    frame_ids: Optional[Sequence[str]] = None,
) -> Iterable[Tuple[str, List[Detection]]]:
    """标注目录回放（可按 ``frame_ids`` 限定与排序，保证与训练样本一致）。"""
    factory = factory or YoloParserFactory.default(
        kpt_shape=recognizer.kpt_shape,
        num_classes=recognizer.assembler.num_classes,
    )
    if frame_ids is None:
        paths = list_frames(label_dir, {".txt"})
    else:
        paths = []
        missing = 0
        for frame_id in frame_ids:
            path = Path(label_dir) / f"{frame_id}.txt"
            if path.is_file():
                paths.append(path)
            else:
                missing += 1
        if missing:
            LOGGER.warning("有 %d 帧在 %s 下找不到标签文件，已跳过", missing, label_dir)
        if not paths:
            raise FileNotFoundError(f"要回放的帧在标注目录中都不存在: {label_dir}")

    for path in paths:
        detections, skipped = factory.parse_file(path)
        if skipped:
            LOGGER.debug("%s 跳过 %d 行非法标注", path.name, skipped)
        yield path.stem, detections


def collect_image_context(target: Path, window_size: int) -> List[Path]:
    """取同目录下该图及其前面的 ``window_size-1`` 张，用于填满时间窗。"""
    target = Path(target)
    siblings = list_image_files(target.parent)
    target_resolved = target.resolve()
    index = next(
        (i for i, path in enumerate(siblings) if path.resolve() == target_resolved),
        len(siblings) - 1,
    )
    start = max(0, index - (window_size - 1))
    return siblings[start : index + 1]


def load_dataset_registry(config_path: Path) -> Dict[str, Any]:
    """读取 tcn_config.json 中的多数据集信息与全局类别表。"""
    path = Path(config_path)
    if not path.exists():
        raise SystemExit(f"数据集配置不存在: {path}，请先运行 build_dataset.py")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    datasets = (cfg.get("source") or {}).get("datasets") or []
    if not datasets:
        raise SystemExit(f"{path} 中缺少 source.datasets（是否由旧版脚本生成？）")
    return {
        "datasets": datasets,
        "target_names": (cfg.get("object_classes") or {}).get("target_names") or [],
    }


def resolve_dataset(
    registry: Dict[str, Any], source: Path, explicit: Optional[str]
) -> Dict[str, Any]:
    """确定当前图像属于哪个 YOLO 数据集。"""
    datasets = registry["datasets"]
    if explicit:
        for ds in datasets:
            if ds.get("name") == explicit:
                return ds
        raise SystemExit(
            f"未知数据集 {explicit!r}；可选：{[ds.get('name') for ds in datasets]}"
        )

    resolved = Path(source).resolve()
    for ds in datasets:
        try:
            resolved.relative_to(Path(ds["path"]).resolve())
            return ds
        except (ValueError, OSError):
            continue

    # 图片可能被复制到项目外：退化为「目录名 = 数据集名」匹配，取最接近图片的一层
    by_name = {
        str(ds.get("name") or "").lower(): ds for ds in datasets if ds.get("name")
    }
    for part in reversed(resolved.parts):
        matched = by_name.get(part.lower())
        if matched:
            return matched

    if len(datasets) == 1:
        LOGGER.warning(
            "无法从路径判断数据集，按唯一数据集 %s 处理", datasets[0]["name"]
        )
        return datasets[0]

    raise SystemExit(
        "无法从 --source 路径判断所属数据集，请显式指定 --dataset；"
        f"可选：{[ds.get('name') for ds in datasets]}"
    )


def build_frames_from_images(
    recognizer: RealtimeActionRecognizer,
    image_dir: Path,
    dataset_cfg: Dict[str, Any],
    target_names: Sequence[str],
    yolo_weights: Optional[Path],
    conf: float,
    iou: float,
    imgsz: int,
    paths: Optional[Sequence[Path]] = None,
    verbose: bool = True,
) -> Iterable[Tuple[str, List[Detection]]]:
    """图像逐帧检测，并把类别映射成全局 id。

    ``verbose=True`` 时逐张打印进度，供 Web 端实时展示验证过程
    （预热帧同样会打印，便于看清窗口填充进度）。
    """
    image_paths = list(paths) if paths is not None else list_image_files(image_dir)
    total = len(image_paths)

    weights = Path(yolo_weights) if yolo_weights else dataset_cfg.get("weights")
    if weights:
        model = load_yolo(Path(weights))
        batch_size = 16
        for start in range(0, total, batch_size):
            chunk = image_paths[start : start + batch_size]
            batch_detections = detect_many_with_yolo(
                model, chunk, conf=conf, iou=iou, imgsz=imgsz, batch=batch_size
            )
            for offset, (path, detections) in enumerate(
                zip(chunk, batch_detections), start=1
            ):
                index = start + offset
                detections, dropped = remap_class_ids(
                    detections, model.names, target_names
                )
                if dropped:
                    LOGGER.debug("[%s] 丢弃类别: %s", dataset_cfg.get("name"), dropped)
                if verbose:
                    print(
                        f"[{index}/{total}] {path.name} → 检测 {len(detections)} 个目标",
                        flush=True,
                    )
                yield path.stem, detections
        return

    labels_dir = find_sibling_labels(Path(image_dir))
    if labels_dir is None:
        raise SystemExit(
            f"数据集 {dataset_cfg.get('name')} 没有可用权重，且找不到同名 labels 目录"
        )
    LOGGER.warning(
        "[%s] 未找到权重，改用同名标注作为检测结果（仅用于链路自测）: %s",
        dataset_cfg.get("name"),
        labels_dir,
    )
    classes = dataset_cfg.get("classes") or []
    class_map = {
        int(local): int(global_id)
        for local, global_id in enumerate(dataset_cfg.get("class_map") or [])
    }
    factory = YoloParserFactory.default(
        kpt_shape=tuple(dataset_cfg.get("kpt_shape") or (20, 3)),
        num_classes=len(classes) or recognizer.assembler.num_classes,
    )
    for index, path in enumerate(image_paths, 1):
        detections = detect_with_labels(factory, labels_dir, path.stem, class_map)
        if verbose:
            print(
                f"[{index}/{total}] {path.name} → 标注 {len(detections)} 个目标",
                flush=True,
            )
        yield path.stem, detections


def predict_npy(
    recognizer: RealtimeActionRecognizer, npy_path: Path, limit: int = 20
) -> None:
    """对已导出的窗口数组做批量推理（``(N, F, W)``）。"""
    X = np.load(npy_path)
    if X.ndim != 3:
        raise ValueError(f"期望 (N, F, W)，收到 {X.shape}")
    X = ((X - recognizer.mean) / recognizer.std).astype(np.float32)
    with torch.no_grad():
        probs = (
            recognizer.model.predict_proba(torch.from_numpy(X).to(recognizer.device))
            .cpu()
            .numpy()
        )
    for i, prob in enumerate(probs[:limit]):
        label = int(np.argmax(prob))
        name = (
            recognizer.class_names[label]
            if label < len(recognizer.class_names)
            else "?"
        )
        print(f"window {i:03d} -> 步骤 {label} ({name}) 置信度 {prob[label]:.3f}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="因果 TCN 步骤识别推理")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_LABEL_DIR,
        help="标注目录(txt) / 图像目录 / 单张图片路径，自动识别",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="图像模式使用的数据集名；source 位于 yolo_datasets/<name>/ 下时可自动识别",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=None,
        help="覆盖数据集约定的 <dataset>/<dataset>.pt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DATASET_CONFIG,
        help="tcn_config.json（读取各数据集权重与全局类别表）",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--npy", type=Path, default=None, help="改为对 X_val.npy 批量推理"
    )
    parser.add_argument("--fps", type=float, default=0.0, help="回放节奏，0 表示不等待")
    parser.add_argument("--window", type=int, default=None, help="覆盖窗口长度")
    parser.add_argument("--smooth", type=int, default=5, help="概率平滑窗口")
    parser.add_argument(
        "--save-csv", type=Path, default=None, help="把逐帧预测写到 CSV"
    )
    parser.add_argument(
        "--label-csv",
        type=Path,
        default=None,
        help="标签回放使用的 labels.csv（默认取 --source 同级的 labels.csv）",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")

    recognizer = RealtimeActionRecognizer(
        args.checkpoint,
        device=args.device,
        window_size=args.window,
        smooth_window=args.smooth,
    )
    print(
        f"模型已加载 | 窗口 {recognizer.window_size} 帧 | "
        f"特征 {recognizer.model_cfg.num_features} 维 | "
        f"缺失填充 {recognizer.missing_strategy}(max_gap={recognizer.missing_max_gap}) | "
        f"步骤 {recognizer.class_names} | epoch {recognizer.epoch} | "
        f"离线指标 {recognizer.metrics}"
    )

    if args.npy is not None:
        predict_npy(recognizer, args.npy)
        return

    source = Path(args.source)

    registry = None
    dataset_cfg = None
    target_names: Sequence[str] = []
    if source.is_file() or (
        source.is_dir()
        and {p.suffix.lower() for p in source.iterdir() if p.is_file()} & IMAGE_SUFFIXES
    ):
        registry = load_dataset_registry(args.config)
        dataset_cfg = resolve_dataset(registry, source, args.dataset)
        target_names = registry["target_names"]
        print(
            f"数据集 {dataset_cfg.get('name')} | 任务 {dataset_cfg.get('task')} | "
            f"权重 {dataset_cfg.get('weights')}"
        )

    # ---- 单张图片：用同目录前置帧预热窗口 ----
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise SystemExit(f"--source 指向文件时必须是图片: {source}")
        context = collect_image_context(source, recognizer.window_size)
        if len(context) < recognizer.window_size:
            LOGGER.warning(
                "窗口需要 %d 帧，但 %s 及之前只有 %d 张图；按现有帧数执行，预计输出 0 条。",
                recognizer.window_size,
                source.name,
                len(context),
            )
        else:
            print(
                f"单图模式：用 {len(context) - 1} 张前置帧预热窗口，"
                f"输出 {context[-1].name} 的判定"
            )
        frames = build_frames_from_images(
            recognizer,
            source.parent,
            dataset_cfg,
            target_names,
            args.yolo_weights,
            args.conf,
            args.iou,
            args.imgsz,
            paths=context,
        )
        predictions = run_stream(
            recognizer, frames, fps=args.fps, save_csv=args.save_csv, verbose=False
        )
        print(f"\n共输出 {len(predictions)} 次预测")
        return

    if not source.is_dir():
        raise SystemExit(f"输入路径不存在: {source}")

    suffixes = {path.suffix.lower() for path in source.iterdir() if path.is_file()}
    session_of: Optional[Callable[[str], str]] = None
    if suffixes & IMAGE_SUFFIXES:
        frames = build_frames_from_images(
            recognizer,
            source,
            dataset_cfg,
            target_names,
            args.yolo_weights,
            args.conf,
            args.iou,
            args.imgsz,
        )
    else:
        # 优先按 labels.csv 回放：它才是真正参与训练的帧集合，且顺序与建库一致；
        # 没有 CSV（尚未建库）时回退到目录内全部标签。两种情况都按会话分隔，
        # 避免时间窗跨数据集/视频段。
        csv_path = (
            Path(args.label_csv) if args.label_csv else source.parent / "labels.csv"
        )
        frame_ids, replay_source = load_replay_frames(source, csv_path)
        if replay_source == "csv":
            if not frame_ids:
                raise SystemExit(
                    f"labels.csv 没有任何帧，无法回放：{csv_path}（请先构建数据集）"
                )
            print(
                f"标签回放：使用 {csv_path} 的 {len(frame_ids)} 帧" "（与训练样本一致）"
            )
        else:
            print(
                f"标签回放：未找到 {csv_path}，回放 {source} 内全部 "
                f"{len(frame_ids)} 帧（可能包含未参与训练的帧）"
            )
        pattern = load_session_pattern(args.config) or r"^.+?__"
        session_of = partial(session_key, pattern=pattern)
        frames = build_frames_from_labels(recognizer, source, frame_ids=frame_ids)

    predictions = run_stream(
        recognizer,
        frames,
        fps=args.fps,
        save_csv=args.save_csv,
        session_of=session_of,
    )
    print(f"\n共输出 {len(predictions)} 次预测")


if __name__ == "__main__":
    main()
