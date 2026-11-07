"""多数据集合并 / 切分 / 权重解析的回归测试。

覆盖文档 §14 的关键不变式：
- 全局类别表按名称并集，别名先于编号；
- 分组切分保证每个分组都进验证集；
- 导出与训练共用 ``split_dataset``；
- 流式特征与序列特征等价；
- 步骤匹配的「顺序 + 数据集 + 数据集内序号」语义；
- 权重解析优先级 CLI > 配置 > 约定路径。
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from build_dataset import (
    _frame_sets,
    _resolve_dataset_weights,
    add_missing_dataset_steps,
    assert_steps_have_frames,
    build_global_classes,
    match_all_steps,
    match_shadow_stats,
    match_step,
    parse_weight_overrides,
    validate_match_rule,
    validate_steps,
    write_labels_csv,
)
from data_processor import (
    BuiltDataset,
    FrameFeatureAssembler,
    grouped_split_indices,
    split_dataset,
)


def _ds(name: str, classes: list[str]) -> dict:
    return {"name": name, "classes": list(classes)}


# --------------------------------------------------------------------------- #
# 全局类别表
# --------------------------------------------------------------------------- #
def test_global_classes_union_by_name():
    datasets = [
        _ds("hand", ["right_palm", "left_palm"]),
        _ds("sitting", ["left_palm", "bad"]),
    ]
    names = build_global_classes(datasets, {})
    assert names == ["right_palm", "left_palm", "bad"]
    assert datasets[0]["class_map"] == [0, 1]
    assert datasets[1]["class_map"] == [1, 2]


def test_alias_merges_slots():
    datasets = [_ds("a", ["right_palm"]), _ds("b", ["palm"])]
    names = build_global_classes(datasets, {"palm": "right_palm"})
    assert names == ["right_palm"]
    assert datasets[0]["class_map"] == [0]
    assert datasets[1]["class_map"] == [0]


def test_class_map_invariants():
    datasets = [
        _ds("hand", ["a", "b", "c"]),
        _ds("pose", ["a", "d"]),
    ]
    names = build_global_classes(datasets, {})
    for ds in datasets:
        assert len(ds["class_map"]) == len(ds["classes"])
        assert all(0 <= i < len(names) for i in ds["class_map"])
    # 同名类别必须共用槽位
    assert ds_slot(datasets[0], "a") == ds_slot(datasets[1], "a")


def ds_slot(dataset: dict, name: str) -> int:
    """取某类别在某数据集内映射到的全局槽位。"""
    return dataset["class_map"][dataset["classes"].index(name)]


# --------------------------------------------------------------------------- #
# 切分
# --------------------------------------------------------------------------- #
def test_grouped_split_keeps_every_group_in_val():
    groups = np.array(["a__"] * 10 + ["b__"] * 4)
    train_idx, val_idx = grouped_split_indices(groups, val_ratio=0.25, gap=0)

    assert train_idx.size + val_idx.size == groups.size
    assert set(groups[val_idx].tolist()) == {"a__", "b__"}
    for key in ("a__", "b__"):
        group_idx = set(np.flatnonzero(groups == key).tolist())
        covered = set(train_idx.tolist()) | set(val_idx.tolist())
        assert group_idx <= covered


def test_grouped_split_preserves_group_time_order():
    groups = np.array(["a__"] * 8)
    train_idx, val_idx = grouped_split_indices(groups, val_ratio=0.25, gap=0)
    assert train_idx.max() < val_idx.min()


def test_split_dataset_reports_grouped_mode():
    size = 10
    dataset = BuiltDataset(
        X=np.zeros((size, 3, 4), dtype=np.float32),
        y=np.zeros((size,), dtype=np.int64),
        frame_ids=np.array(["f"] * size, dtype=object),
        meta={},
        groups=np.array(["a__"] * 6 + ["b__"] * 4, dtype=object),
    )
    train_idx, val_idx, mode = split_dataset(dataset, 0.25, 0)
    assert mode == "grouped"
    assert train_idx.size + val_idx.size == size
    assert val_idx.size >= 2  # 每个分组至少一个验证窗口


def test_split_dataset_single_group_is_chronological():
    size = 8
    dataset = BuiltDataset(
        X=np.zeros((size, 2, 3), dtype=np.float32),
        y=np.zeros((size,), dtype=np.int64),
        frame_ids=np.array(["f"] * size, dtype=object),
        meta={},
        groups=None,
    )
    train_idx, val_idx, mode = split_dataset(dataset, 0.25, 0)
    assert mode == "chronological"
    assert train_idx.max() < val_idx.min()


# --------------------------------------------------------------------------- #
# 特征一致性
# --------------------------------------------------------------------------- #
def test_assembler_sequence_matches_stream():
    rng = np.random.RandomState(0)
    frames = rng.rand(6, 3, 5).astype(np.float32)
    frames[:, :, 4] = 1.0

    asm = FrameFeatureAssembler(3)
    seq = asm.transform_sequence(frames)

    asm.reset()
    stream = np.stack([asm.step(frames[i]) for i in range(frames.shape[0])], axis=0)

    np.testing.assert_allclose(seq, stream, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- #
# 步骤匹配
# --------------------------------------------------------------------------- #
def test_match_step_scope_and_priority():
    steps = [
        {
            "id": 0,
            "name": "a",
            "match": {"dataset": "hand", "pattern": "^hand__.*10_26"},
        },
        {
            "id": 1,
            "name": "b",
            "match": {"frames": ["hand__WIN_20260903_10_26_09_Pro"]},
        },
    ]
    frame = "hand__WIN_20260903_10_26_09_Pro"
    # 先匹配先得：步骤 0 的 pattern 命中即返回
    assert match_step(frame, "hand", 1, None, steps) == 0
    # 数据集：其他数据集不应命中 dataset=hand 的规则
    assert (
        match_step(frame.replace("hand", "sitting"), "sitting", 1, None, steps) is None
    )


def test_match_step_index_is_per_dataset():
    steps = [{"id": 3, "name": "c", "match": {"index_from": 1, "index_to": 2}}]
    assert match_step("hand__x", "hand", 2, None, steps) == 3
    assert match_step("hand__x", "hand", 3, None, steps) is None


def test_match_step_time_rule_skips_frames_without_timestamp():
    steps = [
        {
            "id": 5,
            "name": "t",
            "match": {
                "datetime_from": "2026-09-03 10:00:00",
                "datetime_to": "2026-09-03 11:00:00",
            },
        }
    ]
    assert match_step("hand__x", "hand", 1, None, steps) is None


def test_match_step_class_rule_uses_present_classes():
    steps = [
        {
            "id": 6,
            "name": "cls",
            "match": {"dataset": "hand", "classes": ["right_palm"]},
        }
    ]
    mapping = {"right_palm": 0, "left_palm": 1}
    assert (
        match_step(
            "hand__x", "hand", 1, None, steps, present_classes={0}, class_map=mapping
        )
        == 6
    )
    assert (
        match_step(
            "hand__x", "hand", 1, None, steps, present_classes={1}, class_map=mapping
        )
        is None
    )
    # 未提供帧类别时不命中（向后兼容）
    assert match_step("hand__x", "hand", 1, None, steps) is None


def test_match_step_class_rule_any_of_multiple():
    steps = [
        {"id": 1, "name": "cls", "match": {"classes": ["right_palm", "left_palm"]}}
    ]
    mapping = {"right_palm": 0, "left_palm": 5}
    assert (
        match_step(
            "sitting__x",
            "sitting",
            1,
            None,
            steps,
            present_classes={5},
            class_map=mapping,
        )
        == 1
    )


def test_match_all_steps_reports_every_hit_in_step_order():
    steps = [
        {"id": 0, "name": "a", "match": {"pattern": ".*"}},
        {"id": 1, "name": "b", "match": {"pattern": "hand__"}},
        {"id": 2, "name": "c", "match": {"pattern": "nope"}},
    ]
    assert match_all_steps("hand__x", "hand", 1, None, steps) == [0, 1]
    # 判定仍然取第一个命中
    assert match_step("hand__x", "hand", 1, None, steps) == 0


def test_match_all_steps_dedupes_same_id_across_scopes():
    steps = [
        {"id": 0, "name": "a", "match": {"dataset": "hand", "pattern": ".*"}},
        {"id": 0, "name": "a", "match": {"dataset": "sitting", "pattern": ".*"}},
    ]
    assert match_all_steps("hand__x", "hand", 1, None, steps) == [0]


def test_match_shadow_stats_counts_shadowed_frames():
    steps = [
        {"id": 0, "name": "宽", "match": {"pattern": ".*"}},
        {"id": 1, "name": "窄", "match": {"pattern": "hand__a"}},
    ]
    frames = OrderedDict([("hand", [("a", None), ("b", None)])])

    shadowed = match_shadow_stats(frames, steps)

    assert shadowed == {1: {0: 1}}


def test_match_shadow_stats_empty_for_single_step():
    steps = [{"id": 0, "name": "only", "match": {"pattern": ".*"}}]
    frames = OrderedDict([("hand", [("a", None)])])

    assert match_shadow_stats(frames, steps) == {}


def test_frame_sets_precompile_keeps_semantics():
    steps = [
        {"id": 0, "name": "a", "match": {"frames": ["hand__a"]}},
        {"id": 1, "name": "b", "match": {"pattern": ".*"}},
    ]
    prepared = _frame_sets(steps)

    assert prepared[0] == {"hand__a"}
    assert prepared[1] is None
    assert match_step("hand__a", "hand", 1, None, steps, frame_sets=prepared) == 0
    assert match_step("hand__b", "hand", 2, None, steps, frame_sets=prepared) == 1
    # 与不预编译时的结果一致
    assert match_step("hand__a", "hand", 1, None, steps) == 0


def test_validate_steps_allows_duplicate_ids_for_same_step():
    steps = [
        {
            "id": 0,
            "name": "抓取",
            "match": {"dataset": "hand", "classes": ["a"]},
        },
        {
            "id": 0,
            "name": "抓取",
            "match": {"dataset": "sitting", "classes": ["a"]},
        },
        {
            "id": 1,
            "name": "放下",
            "match": {"dataset": "hand", "classes": ["b"]},
        },
    ]
    validate_steps(steps, ["hand", "sitting"])  # 同一编号可对应多个数据集的同一步骤


def test_validate_steps_rejects_name_mismatch_on_same_id():
    steps = [{"id": 0, "name": "a"}, {"id": 0, "name": "b"}]
    with pytest.raises(SystemExit):
        validate_steps(steps, ["hand"])


def test_validate_steps_rejects_id_gaps():
    steps = [{"id": 0, "name": "a"}, {"id": 2, "name": "b"}]
    with pytest.raises(SystemExit):
        validate_steps(steps, ["hand"])


def test_validate_match_rule_accepts_valid_rule():
    step = {
        "id": 0,
        "match": {
            "dataset": "hand",
            "classes": ["right_palm", 1],
            "pattern": r"^hand__WIN_\d+",
            "datetime_from": "2026-09-03 10:00:00",
            "time_to": "18:30",
            "index_from": 2,
            "index_to": 10,
        },
    }
    assert validate_match_rule(step) == []


def test_validate_match_rule_rejects_bad_regex():
    errors = validate_match_rule({"id": 3, "match": {"pattern": "["}})
    assert len(errors) == 1 and "正则" in errors[0]


def test_validate_match_rule_rejects_empty_or_missing_match():
    assert validate_match_rule({"id": 0, "match": {}})
    assert validate_match_rule({"id": 0})
    assert validate_match_rule({"id": 0, "match": {"classes": []}})


def test_add_missing_dataset_steps_appends_only_for_new_dataset():
    cfg = {
        "steps": [
            {
                "id": 0,
                "name": "手工",
                "match": {"dataset": "hand", "classes": ["a"]},
            }
        ]
    }
    frames = OrderedDict(
        [("hand", [("f1", None)]), ("sitting", [("f1", None), ("f2", None)])]
    )

    added = add_missing_dataset_steps(cfg, frames)

    assert added == ["sitting"]
    assert [step["id"] for step in cfg["steps"]] == [0, 1]
    assert cfg["steps"][1]["match"]["dataset"] == "sitting"


def test_add_missing_dataset_steps_skips_when_global_rule_exists():
    cfg = {"steps": [{"id": 0, "name": "全局", "match": {"classes": ["a"]}}]}
    frames = OrderedDict([("hand", [("f1", None)])])
    assert add_missing_dataset_steps(cfg, frames) == []


def test_assert_steps_have_frames_rejects_active_dead_step():
    steps = [{"id": 0, "name": "a", "match": {"dataset": "hand", "classes": ["x"]}}]
    with pytest.raises(SystemExit):
        assert_steps_have_frames(steps, {}, ["hand"])


def test_assert_steps_have_frames_warns_inactive_scope(caplog):
    steps = [{"id": 0, "name": "a", "match": {"dataset": "gone", "classes": ["x"]}}]
    with caplog.at_level("WARNING", logger="build_dataset"):
        assert_steps_have_frames(steps, {}, ["hand"])
    assert "未参与本次构建" in caplog.text


def test_validate_match_rule_rejects_bad_time_and_index():
    errors = validate_match_rule(
        {
            "id": 4,
            "match": {
                "datetime_from": "2026-13-40 99:00:00",
                "time_from": "25:99",
                "index_from": "abc",
                "index_to": -3,
            },
        }
    )
    text = "\n".join(errors)
    assert "datetime_from" in text
    assert "time_from" in text
    assert "index_from" in text
    assert "index_to" in text


def test_validate_steps_rejects_bad_rule():
    steps = [{"id": 0, "name": "a", "match": {"pattern": "("}}]
    with pytest.raises(SystemExit) as excinfo:
        validate_steps(steps, ["hand"])
    assert "规则参数" in str(excinfo.value)


def test_validate_steps_warns_deleted_dataset_scope(caplog):
    steps = [
        {
            "id": 0,
            "name": "a",
            "match": {"dataset": "gone", "frames": ["gone__f1"]},
        }
    ]
    with caplog.at_level("WARNING", logger="build_dataset"):
        validate_steps(steps, ["hand"])  # 不再整批失败
    assert "不存在的数据集" in caplog.text


def test_write_labels_csv_merges_duplicate_step_ids(tmp_path: Path):
    frames = {"hand": [("f1", None)], "sitting": [("f1", None)]}
    steps = [
        {"id": 0, "name": "抓取", "match": {"dataset": "hand", "frames": ["hand__f1"]}},
        {
            "id": 0,
            "name": "抓取",
            "match": {"dataset": "sitting", "frames": ["sitting__f1"]},
        },
    ]

    counts, unmatched = write_labels_csv(frames, steps, None, tmp_path / "labels.csv")

    assert counts == {0: 2}  # 两个数据集的同一步骤合并为同一类别
    assert unmatched == []
    rows = (tmp_path / "labels.csv").read_text(encoding="utf-8-sig").splitlines()[1:]
    assert all(row.split(",")[1] == "0" for row in rows)
    assert all(row.split(",")[2] == "抓取" for row in rows)


def test_write_labels_csv_class_rule_reads_label_files(tmp_path: Path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "hand__f1.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (labels / "hand__f2.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (labels / "hand__f3.txt").write_text("", encoding="utf-8")
    frames = {"hand": [("f1", None), ("f2", None), ("f3", None)]}
    steps = [
        {
            "id": 0,
            "name": "抓握",
            "match": {"dataset": "hand", "classes": ["right_palm"]},
        }
    ]

    counts, unmatched = write_labels_csv(
        frames,
        steps,
        None,
        tmp_path / "labels.csv",
        labels_dir=labels,
        class_maps={"hand": {"right_palm": 0, "left_palm": 1}},
    )

    assert counts == {0: 1}
    assert unmatched == ["hand__f2", "hand__f3"]
    rows = (tmp_path / "labels.csv").read_text(encoding="utf-8-sig").splitlines()
    assert rows[1].startswith("hand__f1,0,")


# --------------------------------------------------------------------------- #
# 权重解析
# --------------------------------------------------------------------------- #
def test_weight_override_parsing():
    assert parse_weight_overrides(["hand=D:/m/h.pt", "pose=C:/a=b/c.pt"]) == {
        "hand": "D:/m/h.pt",
        "pose": "C:/a=b/c.pt",
    }
    with pytest.raises(SystemExit):
        parse_weight_overrides(["bad"])


def test_resolve_weights_priority(tmp_path: Path):
    default = tmp_path / "hand.pt"
    default.write_bytes(b"x")
    external = tmp_path / "ext.pt"
    external.write_bytes(b"x")

    # CLI 覆盖优先
    path, previous = _resolve_dataset_weights(
        "hand", default, {"hand": str(external)}, {}
    )
    assert path == external.resolve()
    assert previous is None

    # 配置路径优先于约定路径
    path, previous = _resolve_dataset_weights(
        "hand", default, {}, {"hand": str(external)}
    )
    assert path == external.resolve()
    assert previous is None

    # 配置失效 → 回退约定路径，并保留旧路径
    missing = tmp_path / "gone.pt"
    path, previous = _resolve_dataset_weights(
        "hand", default, {}, {"hand": str(missing)}
    )
    assert path == default.resolve()
    assert previous == str(missing)

    # 都不存在 → None
    path, previous = _resolve_dataset_weights("x", tmp_path / "nope.pt", {}, {})
    assert path is None
    assert previous is None


def test_resolve_weights_override_missing_raises(tmp_path: Path):
    with pytest.raises(SystemExit):
        _resolve_dataset_weights(
            "hand", tmp_path / "hand.pt", {"hand": str(tmp_path / "nope.pt")}, {}
        )
