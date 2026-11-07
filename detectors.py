"""detectors.py — YOLO 检测结果到 ``Detection`` 的共用转换层。

``build_dataset.py``（离线生成 TCN 训练特征）和 ``inference.py``（在线推理）
**必须共用这里的实现**，否则两端的坐标约定、关键点补齐、类别映射会各自演化，
最终导致"训练/推理特征不同分布"。

约定：
- 坐标全部为 YOLO 归一化坐标 ``[0, 1]``；
- ``class_id`` 以 ``classes.txt`` 的下标为准（按**类别名**映射，不按模型内 id）；
- 写了 ``kpt_shape`` 但模型没有关键点时，补零占位，保证 token 数一致。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from data_processor import Detection, natural_frame_key

LOGGER = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_image_files(directory: Path) -> list[Path]:
    """目录内的图片文件，按自然时间序（所有图片入口共用，避免多份实现漂移）。"""
    folder = Path(directory)
    if not folder.is_dir():
        return []
    files = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(files, key=natural_frame_key)


def load_yolo(model_path: Path):
    """延迟导入 ultralytics，避免无图像需求时也要装它。"""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("图像模式需要 ultralytics：pip install ultralytics") from exc

    path = Path(model_path)
    if not path.exists():
        raise SystemExit(f"YOLO 权重不存在: {path}")
    model = YOLO(str(path))
    LOGGER.info(
        "YOLO 已加载 | %s | task=%s | 类别 %s", path.name, model.task, model.names
    )
    return model


def _result_to_detections(result) -> List[Detection]:
    """把单张图的 ultralytics Result 转成 ``Detection`` 列表（归一化坐标）。"""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xywhn = boxes.xywhn.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    keypoints_all = None
    keypoints = getattr(result, "keypoints", None)
    if keypoints is not None:
        try:
            xy = keypoints.xyn.cpu().numpy()  # (N, K, 2) 归一化
            # 第 3 列必须是**真实关键点置信度**：PoseParser 用
            # ``vis >= kpt_visibility_threshold`` 判断该点是否参与中心计算。
            # 旧实现恒填 1.0，导致未检出的 (0,0) 点也被算进中心。
            conf = getattr(keypoints, "conf", None)
            vis = None
            if conf is not None:
                candidate = np.asarray(conf.cpu().numpy(), dtype=np.float32)
                if candidate.shape == xy.shape[:2]:
                    vis = candidate[..., None]
            if vis is None:
                # 少数旧版本/非 pose 结果没有 conf：退回"全部可见"并保持形状一致
                vis = np.ones((*xy.shape[:2], 1), dtype=np.float32)
            keypoints_all = np.concatenate([xy, vis], axis=2)  # (N, K, 3)
        except Exception:  # pragma: no cover - 非 pose 模型
            keypoints_all = None

    detections: List[Detection] = []
    for index in range(len(classes)):
        cx, cy, width, height = (float(v) for v in xywhn[index][:4])
        kpts = keypoints_all[index] if keypoints_all is not None else None
        detections.append(
            Detection(int(classes[index]), cx, cy, width, height, "yolo", kpts)
        )
    return detections


def detect_with_yolo(
    model,
    image_path: Path,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
) -> List[Detection]:
    """把一张图的 YOLO 结果转成 ``Detection`` 列表（归一化坐标）。"""
    result = model.predict(
        str(image_path), conf=conf, iou=iou, imgsz=imgsz, verbose=False
    )[0]
    return _result_to_detections(result)


def detect_many_with_yolo(
    model,
    image_paths: Sequence[Path],
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    batch: int = 16,
) -> List[List[Detection]]:
    """批量检测：按 ``batch`` 切块调用一次 ``predict``，返回与输入同序的结果。

    建库与验证都走这里：逐张 ``predict`` 在几千张图时会慢数倍；
    分块可避免一次性把所有图片塞进显存。
    """
    paths = [str(Path(item)) for item in image_paths]
    results: List[List[Detection]] = []
    size = max(int(batch), 1)
    for start in range(0, len(paths), size):
        chunk = paths[start : start + size]
        outputs = model.predict(chunk, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        results.extend(_result_to_detections(item) for item in outputs)
    return results


def remap_class_ids(
    detections: Sequence[Detection],
    model_names: Dict[int, str],
    target_names: Sequence[str],
) -> Tuple[List[Detection], Dict[str, int]]:
    """按**类别名**把模型内 id 映射到 ``classes.txt`` 的下标。

    Returns:
        ``(重映射后的检测列表, 被丢弃的类别计数)``。
    """
    name_to_index = {str(name): index for index, name in enumerate(target_names)}
    dropped: Dict[str, int] = {}
    remapped: List[Detection] = []

    for det in detections:
        name = model_names.get(int(det.class_id))
        if name is None:
            dropped[f"id:{det.class_id}"] = dropped.get(f"id:{det.class_id}", 0) + 1
            continue
        index = name_to_index.get(str(name))
        if index is None:
            dropped[str(name)] = dropped.get(str(name), 0) + 1
            continue
        remapped.append(
            Detection(
                index, det.cx, det.cy, det.width, det.height, det.source, det.keypoints
            )
        )
    return remapped, dropped


def format_detection_line(
    det: Detection, kpt_shape: Optional[Sequence[int]] = None
) -> str:
    """按 YOLO 标注格式输出一行（detect 5 个值 / pose ``1+4+K*D`` 个值）。"""
    parts = [
        str(int(det.class_id)),
        f"{det.cx:.6f}",
        f"{det.cy:.6f}",
        f"{det.width:.6f}",
        f"{det.height:.6f}",
    ]

    keypoints = det.keypoints
    if keypoints is not None:
        flat = np.asarray(keypoints).reshape(-1)
        parts.extend(f"{float(value):.6f}" for value in flat)
    elif kpt_shape:
        # pose 数据集但模型没给关键点：补零占位，保证 token 数与 PoseParser 一致
        parts.extend(["0.000000"] * (int(kpt_shape[0]) * int(kpt_shape[1])))

    return " ".join(parts)
