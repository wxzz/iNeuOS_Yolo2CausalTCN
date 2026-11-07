"""标签回放 / 会话切分的回归测试（V1 修复）。

覆盖：
- ``load_replay_frames`` 优先使用 labels.csv（只回放训练样本），缺失时回退目录；
- ``build_frames_from_labels`` 按 ``frame_ids`` 限定与排序；
- ``run_stream`` 在会话键变化时重置窗口，避免跨数据集拼接。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from data_processor import session_key
from inference import build_frames_from_labels, load_replay_frames, run_stream


def _write_labels(label_dir: Path, names: list[str]) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (label_dir / f"{name}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")


def test_load_replay_frames_prefers_csv(tmp_path: Path):
    labels = tmp_path / "labels"
    _write_labels(labels, ["hand__2", "hand__1", "sitting__1"])
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "frame,label,step_name\nhand__2,0,a\nhand__1,0,a\nhand__1,0,a\n",
        encoding="utf-8-sig",
    )

    frames, source = load_replay_frames(labels, csv_path)

    assert source == "csv"
    assert frames == ["hand__2", "hand__1"]  # CSV 顺序，且去重


def test_load_replay_frames_falls_back_to_directory(tmp_path: Path):
    labels = tmp_path / "labels"
    _write_labels(labels, ["hand__1", "hand__2"])

    frames, source = load_replay_frames(labels, tmp_path / "missing.csv")

    assert source == "labels"
    assert sorted(frames) == ["hand__1", "hand__2"]


def test_build_frames_from_labels_respects_frame_ids(tmp_path: Path):
    labels = tmp_path / "labels"
    _write_labels(labels, ["hand__1", "hand__2"])
    recognizer = SimpleNamespace(
        assembler=SimpleNamespace(num_classes=2), kpt_shape=(20, 3)
    )

    frames = list(
        build_frames_from_labels(recognizer, labels, frame_ids=["hand__2", "nope"])
    )

    assert [frame_id for frame_id, _ in frames] == ["hand__2"]
    assert len(frames[0][1]) == 1


class _FakeRecognizer:
    """只记录 push/reset 的假识别器，用于验证会话切分。"""

    window_size = 2

    def __init__(self) -> None:
        self.buffer: list[str] = []
        self.resets = 0
        self.pushed: list[str] = []

    def reset(self) -> None:
        self.resets += 1
        self.buffer.clear()

    def push_detections(self, detections, frame_id: str = ""):
        self.pushed.append(frame_id)
        self.buffer.append(frame_id)
        return None


def test_run_stream_resets_on_session_change():
    recognizer = _FakeRecognizer()
    frames = [("hand__1", []), ("hand__2", []), ("sit__1", []), ("sit__2", [])]

    run_stream(
        recognizer,
        frames,
        verbose=False,
        session_of=lambda frame_id: session_key(frame_id, r"^.+?__"),
    )

    assert recognizer.resets == 1
    assert recognizer.pushed == ["hand__1", "hand__2", "sit__1", "sit__2"]
