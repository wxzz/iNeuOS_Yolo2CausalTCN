"""train.py — 因果 TCN 训练与验证。

要点：
1. 时间顺序切分（+ 可选 embargo）避免重叠滑窗泄漏；
2. 归一化统计量只在训练集上拟合，并随权重一起保存，推理端复用；
3. 自动选择 cuda / mps / cpu，tqdm 展示进度，按验证准确率保存最优权重。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_LABEL_CSV,
    DEFAULT_LABEL_DIR,
    DataConfig,
    TCNConfig,
    TrainConfig,
)
from data_processor import (
    BuiltDataset,
    FrameFeatureAssembler,
    SequenceDatasetBuilder,
    export_dataset,
    split_dataset,
)
from model import CausalTCNClassifier

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(preferred: str = "auto") -> torch.device:
    """自动选择训练设备。"""
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


class ArrayDataset(Dataset):
    """把 ``(N, F, W)`` / ``(N,)`` 的 NumPy 数组包装成 PyTorch Dataset。"""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X/y 样本数不一致: {X.shape[0]} vs {y.shape[0]}")
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


def fit_normalization(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """在训练集上拟合逐特征通道的均值/标准差，形状 ``(1, F, 1)``。"""
    mean = X_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = X_train.std(axis=(0, 2), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)  # 全零特征通道不放大噪声
    return mean, std


def apply_normalization(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """按训练集统计量标准化。"""
    return ((X - mean) / std).astype(np.float32)


def compute_class_weights(y: np.ndarray, num_classes: int) -> Optional[torch.Tensor]:
    """按类别频次的倒数计算权重，缓解动作类别不平衡。"""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    if (counts == 0).any():
        LOGGER.warning("存在训练集中未出现的类别，已关闭类别加权: %s", counts)
        return None
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    desc: str = "train",
) -> Tuple[float, float]:
    """跑一个 epoch，返回 ``(平均 loss, 准确率)``。``optimizer`` 为 None 时只做评估。"""
    training = optimizer is not None
    model.train(training)

    total_loss, correct, total = 0.0, 0, 0
    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch_x, batch_y in progress:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            if training and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = int(batch_y.size(0))
        total_loss += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == batch_y).sum().item())
        total += batch_size
        progress.set_postfix(
            loss=f"{total_loss / max(total, 1):.4f}",
            acc=f"{correct / max(total, 1):.3f}",
        )

    return total_loss / max(total, 1), correct / max(total, 1)


def save_checkpoint(
    path: Path,
    model: CausalTCNClassifier,
    model_cfg: TCNConfig,
    data_cfg: DataConfig,
    meta: Dict[str, object],
    mean: np.ndarray,
    std: np.ndarray,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    """保存结构配置 + 权重 + 归一化统计，实现推理端零配置复现。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": asdict(model_cfg),
            "data_config": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in asdict(data_cfg).items()
            },
            "window_size": int(meta.get("window_size", data_cfg.window_size)),
            "num_classes": int(meta.get("num_classes", model_cfg.num_classes)),
            "class_names": list(meta.get("label_names", [])),
            "feature_dim": int(model_cfg.num_features),
            "normalization": {
                "mean": mean.reshape(-1).tolist(),
                "std": std.reshape(-1).tolist(),
            },
            "epoch": int(epoch),
            "metrics": metrics,
        },
        path,
    )
    LOGGER.info(
        "权重已保存: %s (epoch=%d, val_acc=%.4f)", path, epoch, metrics["val_acc"]
    )


def load_npy_arrays(
    artifact_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """直接读取四个 npy（已按 ``split_mode`` 切完），**不再二次切分**。

    历史问题：旧版把 ``X_train``/``X_val`` 拼回整体再切一次，导致训练集
    与磁盘上的 npy 不一致；缓存训练必须原样使用已导出结果。
    """
    out_dir = Path(artifact_dir)
    meta = json.loads((out_dir / "dataset_meta.json").read_text("utf-8"))
    return (
        np.load(out_dir / "X_train.npy"),
        np.load(out_dir / "y_train.npy"),
        np.load(out_dir / "X_val.npy"),
        np.load(out_dir / "y_val.npy"),
        meta,
    )


def load_npy_dataset(artifact_dir: Path) -> BuiltDataset:
    """读取 ``export_dataset`` 落盘的 npy，跳过解析阶段（适合反复调参）。"""
    out_dir = Path(artifact_dir)
    meta = json.loads((out_dir / "dataset_meta.json").read_text("utf-8"))
    X_train = np.load(out_dir / "X_train.npy")
    y_train = np.load(out_dir / "y_train.npy")
    X_val = np.load(out_dir / "X_val.npy")
    y_val = np.load(out_dir / "y_val.npy")
    X = np.concatenate([X_train, X_val], axis=0) if X_val.size else X_train
    y = np.concatenate([y_train, y_val], axis=0) if y_val.size else y_train
    return BuiltDataset(X=X, y=y, frame_ids=np.array([], dtype=object), meta=meta)


def load_dataset_config(path: Path) -> Dict[str, object]:
    """读取 ``build_dataset.py`` 生成的 ``tcn_config.json``；缺失时返回空字典。"""
    config_path = Path(path)
    if not config_path.exists():
        LOGGER.warning(
            "未找到数据集配置 %s，回退到内置默认值；建议先运行 build_dataset.py",
            config_path,
        )
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_step_names(dataset_cfg: Dict[str, object]) -> Tuple[str, ...]:
    """从配置里按步骤 id 还原显示名（用于 checkpoint 的 class_names）。

    同一编号允许对应多个步骤（跨数据集同一步骤），但名称必须一致；
    不一致时直接报错，避免 checkpoint 里静默留下任意一个名字。
    """
    steps = dataset_cfg.get("steps") or []
    names_by_id: Dict[int, str] = {}
    for step in steps:
        if not isinstance(step, dict) or "id" not in step:
            continue
        step_id = int(step["id"])
        name = str(step.get("name", f"步骤{step_id}"))
        previous = names_by_id.get(step_id)
        if previous is not None and previous != name:
            raise SystemExit(
                f"配置里步骤编号 {step_id} 的名称不一致：{previous!r} / {name!r}；"
                "相同编号必须是同一步骤（名称必须相同）"
            )
        names_by_id[step_id] = name
    return tuple(names_by_id[i] for i in sorted(names_by_id))


def assert_trainable_classes(num_classes: int) -> None:
    """提前拦住“只有一个步骤”的退化情况，给出可读原因。"""
    if num_classes < 2:
        raise SystemExit(
            f"训练至少需要 2 个步骤（类别），当前只有 {num_classes} 个。\n"
            "请在【数据映射】增加步骤，或换用包含多个步骤的数据集。"
        )


def check_class_coverage(y_train: np.ndarray, y_val: np.ndarray) -> None:
    """验证集不能出现训练集没有的类别，否则该类别永远学不到、指标失真。"""
    train_classes = {int(value) for value in np.unique(y_train)}
    val_classes = {int(value) for value in np.unique(y_val)}
    only_val = sorted(val_classes - train_classes)
    if only_val:
        raise SystemExit(
            f"验证集出现训练集没有的类别 id：{only_val}；模型无法学到这些类别。\n"
            "请调整 val_ratio / gap、补充样本，或检查数据集切分口径。"
        )
    only_train = sorted(train_classes - val_classes)
    if only_train:
        LOGGER.warning("以下类别只出现在训练集（验证指标无法反映）：%s", only_train)


def validate_cached_dataset(
    X_train: np.ndarray,
    X_val: np.ndarray,
    meta: Dict[str, object],
    data_cfg: DataConfig,
    step_names: Sequence[str],
) -> None:
    """校验缓存 npy 与当前配置一致（``--from-npy`` 专用）。

    缓存与配置不一致时静默训练会得到“看起来正常、实际用旧数据”的模型，
    因此这里直接终止并给出重建指引。
    """
    assembler = FrameFeatureAssembler(
        data_cfg.num_target_classes,
        data_cfg.include_size,
        data_cfg.include_velocity,
    )
    expected_features = assembler.feature_dim
    problems: List[str] = []
    if X_train.ndim != 3 or X_train.shape[1] != expected_features:
        problems.append(
            f"缓存特征维度 {X_train.shape[1] if X_train.ndim == 3 else '?'} "
            f"!= 当前配置 {expected_features}"
        )
    if X_train.ndim == 3 and X_train.shape[2] != data_cfg.window_size:
        problems.append(
            f"缓存窗口 {X_train.shape[2]} != 当前配置 {data_cfg.window_size}"
        )
    cached_window = int(meta.get("window_size") or 0)
    if cached_window and cached_window != data_cfg.window_size:
        problems.append(
            f"dataset_meta.window_size={cached_window} != 当前配置 {data_cfg.window_size}"
        )
    cached_classes = int(meta.get("num_classes") or 0)
    if cached_classes and cached_classes != len(step_names):
        problems.append(f"缓存类别数 {cached_classes} != 当前步骤数 {len(step_names)}")
    if X_val.ndim == 3 and X_train.ndim == 3 and X_val.shape[1] != X_train.shape[1]:
        problems.append("训练/验证缓存的特征维度不一致")
    if problems:
        raise SystemExit(
            "缓存 npy 与当前配置不一致：\n- "
            + "\n- ".join(problems)
            + "\n请先重新构建数据集（或在网页取消勾选“使用缓存 npy”）。"
        )

    cached_names = [str(item) for item in (meta.get("label_names") or [])]
    if cached_names and cached_names != list(step_names):
        LOGGER.warning(
            "缓存里的步骤名与当前配置不同（只影响显示，不影响训练）：%s -> %s",
            cached_names,
            list(step_names),
        )
    cached_stride = int(meta.get("stride") or 0)
    if cached_stride and cached_stride != data_cfg.stride:
        LOGGER.warning(
            "缓存 stride=%s 与当前配置 %s 不同（样本数会不同）",
            cached_stride,
            data_cfg.stride,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="训练因果 TCN 动作识别模型")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DEFAULT_DATASET_CONFIG,
        help="build_dataset.py 生成的配置（步骤定义 / 窗口参数）",
    )
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DataConfig.artifact_dir)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--window", type=int, default=None, help="覆盖配置里的窗口长度")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--pose-center", choices=["bbox", "keypoints"], default=None)
    parser.add_argument(
        "--label-source",
        default="csv",
        choices=["auto", "csv", "filename", "dominant_class"],
    )
    parser.add_argument("--label-csv", type=Path, default=DEFAULT_LABEL_CSV)
    parser.add_argument("--session-pattern", default=None)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument(
        "--gap",
        type=int,
        default=None,
        help="训练/验证接缝处丢弃的窗口数（防重叠泄漏，建议设为 window）",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=None,
        help="TCN 各层通道数，例: 64 64 64",
    )
    parser.add_argument("--dropout", type=float, default=TCNConfig.dropout)
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument(
        "--from-npy",
        action="store_true",
        help="直接读取 tcn_datasets 下已导出的 npy，跳过解析",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    set_seed(args.seed)

    dataset_cfg = load_dataset_config(args.dataset_config)
    ds_params = dataset_cfg.get("dataset") or {}
    objects = dataset_cfg.get("object_classes") or {}
    step_names = resolve_step_names(dataset_cfg)

    window = (
        args.window
        if args.window is not None
        else int(ds_params.get("window_size", DataConfig.window_size))
    )
    stride = (
        args.stride
        if args.stride is not None
        else int(ds_params.get("stride", DataConfig.stride))
    )
    pose_center = args.pose_center or str(
        ds_params.get("pose_center_mode", DataConfig.pose_center_mode)
    )
    session_pattern = (
        args.session_pattern
        if args.session_pattern is not None
        else (ds_params.get("session_pattern") or None)
    )
    val_ratio = (
        args.val_ratio
        if args.val_ratio is not None
        else float(ds_params.get("val_ratio", TrainConfig.val_ratio))
    )
    gap = (
        args.gap if args.gap is not None else int(ds_params.get("gap", TrainConfig.gap))
    )

    data_cfg = DataConfig(
        label_dir=args.label_dir,
        artifact_dir=args.artifact_dir,
        window_size=window,
        stride=stride,
        label_mode=str(ds_params.get("label_mode", DataConfig.label_mode)),
        num_target_classes=int(
            objects.get("num_target_classes", DataConfig.num_target_classes)
        ),
        target_names=tuple(objects.get("target_names", DataConfig.target_names)),
        kpt_shape=tuple(objects.get("kpt_shape") or DataConfig.kpt_shape),
        pose_center_mode=pose_center,
        missing_strategy=str(
            ds_params.get("missing_strategy", DataConfig.missing_strategy)
        ),
        max_gap=int(ds_params.get("max_gap", DataConfig.max_gap)),
        include_size=bool(ds_params.get("include_size", DataConfig.include_size)),
        include_velocity=bool(
            ds_params.get("include_velocity", DataConfig.include_velocity)
        ),
        label_source=args.label_source,
        label_csv=args.label_csv,
        session_pattern=session_pattern,
        step_names=step_names,
    )
    data_cfg.validate()

    device = select_device(args.device)
    LOGGER.info("训练设备: %s", device)

    # ---------------- 数据准备 ----------------
    if args.from_npy:
        X_train, y_train, X_val, y_val, meta = load_npy_arrays(args.artifact_dir)
        validate_cached_dataset(X_train, X_val, meta, data_cfg, step_names)
        LOGGER.info(
            "已加载缓存数据: train=%s val=%s（沿用已导出的切分）",
            X_train.shape,
            X_val.shape,
        )
    else:
        dataset = SequenceDatasetBuilder(data_cfg).build()
        meta = export_dataset(dataset, data_cfg, val_ratio, gap)
        train_idx, val_idx, split_mode = split_dataset(dataset, val_ratio, gap)
        X_train, y_train = dataset.X[train_idx], dataset.y[train_idx]
        X_val, y_val = dataset.X[val_idx], dataset.y[val_idx]
        LOGGER.info(
            "切分模式: %s（train %d / val %d）",
            split_mode,
            int(train_idx.size),
            int(val_idx.size),
        )
        if dataset.X.shape[0] < 4:
            LOGGER.warning(
                "样本数仅 %d，模型极可能过拟合，仅适合验证代码链路。",
                dataset.X.shape[0],
            )

    if X_val.shape[0] == 0:
        raise SystemExit("验证集为空，请调小 --gap 或增大数据量。")

    mean, std = fit_normalization(X_train)
    X_train = apply_normalization(X_train, mean, std)
    X_val = apply_normalization(X_val, mean, std)

    num_classes = int(max(int(y_train.max()), int(y_val.max()))) + 1
    assert_trainable_classes(num_classes)
    check_class_coverage(y_train, y_val)
    num_features = int(X_train.shape[1])
    LOGGER.info(
        "训练集 %d / 验证集 %d，特征 %d 维，窗口 %d 帧，类别数 %d",
        X_train.shape[0],
        X_val.shape[0],
        num_features,
        data_cfg.window_size,
        num_classes,
    )
    if step_names:
        LOGGER.info("步骤名: %s", list(step_names[:num_classes]))

    # 验证集必须分批：旧实现用 batch_size=len(X_val) 一次性进显存，
    # 验证集大时会出现显存尖峰。run_epoch 按样本数加权累计指标，
    # 因此分批不改变 loss/acc。
    pin_memory = device.type == "cuda"
    num_workers = TrainConfig.num_workers
    val_batch = max(1, min(int(args.batch_size), len(X_val)))
    loader_common = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        ArrayDataset(X_train, y_train),
        batch_size=min(args.batch_size, len(X_train)),
        shuffle=True,
        drop_last=False,
        **loader_common,
    )
    val_loader = DataLoader(
        ArrayDataset(X_val, y_val),
        batch_size=val_batch,
        shuffle=False,
        **loader_common,
    )
    LOGGER.info(
        "DataLoader：train_batch=%d val_batch=%d num_workers=%d pin_memory=%s",
        min(args.batch_size, len(X_train)),
        val_batch,
        num_workers,
        pin_memory,
    )

    # ---------------- 模型与优化器 ----------------
    model_cfg = TCNConfig(
        num_features=num_features,
        num_classes=num_classes,
        channels=tuple(args.channels) if args.channels else TCNConfig.channels,
        dropout=args.dropout,
    )
    model = CausalTCNClassifier(model_cfg).to(device)
    LOGGER.info(
        "模型感受野 %d 帧，参数量 %d", model.receptive_field, model.count_parameters()
    )
    if model.receptive_field < data_cfg.window_size:
        LOGGER.warning(
            "感受野(%d) 小于窗口(%d)，窗口前段信息不会被利用，可增加 --channels 层数。",
            model.receptive_field,
            data_cfg.window_size,
        )

    class_weights = compute_class_weights(y_train, num_classes)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=TrainConfig.weight_decay
    )
    scheduler = None
    if TrainConfig.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=8
        )
    elif TrainConfig.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1)
        )

    # ---------------- 训练循环 ----------------
    best_acc, best_epoch, patience = -1.0, -1, 0
    epochs = tqdm(range(1, args.epochs + 1), desc="epoch", dynamic_ncols=True)
    for epoch in epochs:
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, desc=f"train@{epoch}"
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device, optimizer=None, desc=f"val@{epoch}"
        )
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        epochs.set_postfix(train_loss=f"{train_loss:.4f}", val_acc=f"{val_acc:.3f}")
        LOGGER.info(
            "epoch %03d | train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f | lr=%.2e",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            optimizer.param_groups[0]["lr"],
        )

        if val_acc > best_acc:
            best_acc, best_epoch, patience = val_acc, epoch, 0
            save_checkpoint(
                args.checkpoint,
                model,
                model_cfg,
                data_cfg,
                meta,
                mean,
                std,
                epoch,
                {
                    "val_acc": float(val_acc),
                    "val_loss": float(val_loss),
                    "train_acc": float(train_acc),
                    "train_loss": float(train_loss),
                },
            )
        else:
            patience += 1
            if patience >= TrainConfig.early_stopping_patience:
                LOGGER.info("验证指标 %d 轮未提升，提前停止。", patience)
                break

    print(f"\n训练完成：最优验证准确率 {best_acc:.4f}（epoch {best_epoch}）")
    print(f"权重路径：{args.checkpoint}")


if __name__ == "__main__":
    main()
