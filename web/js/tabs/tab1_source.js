// Tab1 源数据集：扫描 → 勾选 → 逐个指定权重 → 校验 → 全局类别表
import * as api from "../api.js";
import { state, update, selectedNames } from "../state.js";
import { h, clear, toast, badge, dot, section, choose } from "../ui.js";

export function render(container) {
  clear(container);
  container.append(
    section("数据集根目录", rootRow()),
    // 检测器设置影响“校验权重与一致性”（labels 模式不加载 .pt），放在列表与校验之前
    detectorCard(),
    datasetsCard(),
    globalCard(),
    messageCard(),
    actionRow()
  );
  if (!state.datasets.length && !state.scanning && !state.scanAttempted) {
    scan();
  }
}

function rootRow() {
  const input = h("input", {
    type: "text",
    class: "path",
    value: state.root,
    placeholder: "数据集根目录（留空使用上次设置）",
    onchange: (event) => {
      state.root = event.target.value.trim();
    },
  });
  const checked = selectedNames();
  return h(
    "div",
    { class: "toolbar" },
    input,
    h("button", {
      text: "选择数据目录",
      onclick: pickRoot,
      disabled: state.scanning,
    }),
    h("button", {
      text: "重新扫描",
      onclick: () => scan(true),
      disabled: state.scanning,
    }),
    h("button", {
      text: "增加数据集",
      onclick: addDataset,
      disabled: state.scanning || state.running,
    }),
    h("button", {
      text: `删除选择数据集${checked.length ? `（${checked.length}）` : ""}`,
      class: "danger",
      onclick: removeChecked,
      disabled: state.scanning || state.running || !checked.length,
    }),
    h("span", {
      class: "hint",
      text: "重新扫描只重新发现数据集，不会复制文件；根目录之外的目录用“增加数据集”加入（会复制进根目录）；权重“浏览…”会把模型复制到数据集目录并重命名为 <数据集>.pt",
    })
  );
}

// 只打开目录选择框并回填根目录，不触发扫描（扫描由“重新扫描”按钮负责）
async function pickRoot() {
  try {
    const picked = await api.post("/api/fs/pick", {
      mode: "dir",
      start: state.root || "",
    });
    if (picked.cancelled || !picked.path) return;
    state.root = picked.path;
    update({});
    toast("已选择数据集目录，点“重新扫描”加载数据集");
  } catch (err) {
    toast(err.message, "err");
  }
}

async function addDataset() {
  if (state.running) {
    toast("任务运行中，暂不能修改数据集", "err");
    return;
  }
  try {
    const picked = await api.post("/api/fs/pick", {
      mode: "dir",
      start: state.root || "",
    });
    if (picked.cancelled || !picked.path) return;
    update({ scanning: true });
    toast("正在加入数据集（根目录外的目录会先复制，可能较慢）…");
    const data = await api.post("/api/datasets/add", {
      path: picked.path,
      root: state.root,
    });
    const added = data.added || {};
    applyScan(data, { dirty: true });
    toast(
      added.copied
        ? `已复制到根目录并加入：${added.path}`
        : "已增加数据集，请确认权重后重新构建",
      "ok"
    );
  } catch (err) {
    update({ scanning: false });
    toast(err.message, "err");
  }
}

function removeChecked() {
  if (state.running) {
    toast("任务运行中，暂不能修改数据集", "err");
    return;
  }
  const names = selectedNames();
  if (!names.length) {
    toast("请先勾选要删除的数据集", "err");
    return;
  }
  const datasets = state.datasets.filter((dataset) =>
    names.includes(dataset.name)
  );
  askRemoveDatasets(datasets);
}

// 删除前的两个选项：仅删除记录 / 连磁盘上的数据集与模型一起删除
async function askRemoveDatasets(datasets) {
  if (!datasets.length) return;
  const mode = await choose({
    title:
      datasets.length > 1
        ? `删除 ${datasets.length} 个数据集`
        : `删除数据集：${datasets[0].name}`,
    message: datasets
      .map((dataset) => `${dataset.name}：${dataset.path}`)
      .join("\n"),
    options: [
      { value: "record", label: "① 保留现有数据集信息（只删除记录）" },
      {
        value: "purge",
        label: "② 删除记录和数据集信息（同时删除数据集与模型）",
        kind: "danger",
      },
    ],
    note: "选 ① 只删记录，之后点“重新扫描”可按根目录恢复；选 ② 会永久删除磁盘上的数据集目录与该数据集的模型权重（未被其它数据集共用时）。",
  });
  if (!mode) return;
  await removeDatasets(
    datasets.map((dataset) => dataset.path),
    mode === "purge"
  );
}

async function removeDatasets(paths, purge = false) {
  if (!paths.length) return;
  if (state.running) {
    toast("任务运行中，暂不能修改数据集", "err");
    return;
  }
  update({ scanning: true });
  try {
    const data = await api.post("/api/datasets/remove", {
      root: state.root,
      paths,
      purge,
    });
    applyScan(data, { dirty: true });
    if (purge) {
      const report = data.purge || {};
      const pruned = (report.steps_removed || []).length;
      toast(
        `已删除 ${paths.length} 个数据集记录，磁盘删除 ${
          (report.deleted || []).length
        } 项${pruned ? `，清理步骤 ${pruned} 个` : ""}`,
        "ok"
      );
      (report.skipped || []).forEach((item) =>
        toast(`未删除 ${item.path}：${item.reason}`, "err")
      );
    } else {
      toast(`已从列表删除 ${paths.length} 个数据集（磁盘文件保留）`, "ok");
    }
  } catch (err) {
    update({ scanning: false });
    toast(err.message, "err");
  }
}

// 把 /api/dataset/scan、/api/datasets/add|remove 的返回合并进状态
function applyScan(data, { dirty = false } = {}) {
  const selected = {};
  (data.datasets || []).forEach((dataset) => {
    const previous = state.selected[dataset.name];
    // 原路径已失效时权重输入保持为空（“原路径已失效”提示仍照常显示）
    const weights = dataset.weights_previous
      ? ""
      : previous?.weights || dataset.weights || dataset.default_weights || "";
    selected[dataset.name] = {
      checked: previous ? previous.checked : true,
      weights,
    };
  });
  const models = {};
  Object.entries(state.models || {}).forEach(([name, info]) => {
    if (selected[name]) models[name] = info;
  });
  update({
    root: data.root || state.root,
    datasets: data.datasets || [],
    global: data.global,
    selected,
    models,
    steps: data.steps || [],
    warnings: data.warnings || [],
    errors: [],
    scanning: false,
    scanAttempted: true,
    configLoaded: false,
    // dirty 只增不减：普通扫描不能把“需重新构建”误清掉
    dirtyDataset: dirty || state.dirtyDataset,
    // 数据集变了，之前的校验结论自动失效
    validation: null,
  });
}

async function scan(reset = false) {
  update({ scanning: true });
  try {
    const data = await api.get("/api/dataset/scan", {
      root: state.root,
      reset: reset ? 1 : "",
    });
    applyScan(data);
  } catch (err) {
    update({ scanning: false, scanAttempted: true });
    toast(err.message, "err");
  }
}

function datasetsCard() {
  const body = h("tbody");
  state.datasets.forEach((dataset) => {
    const item = state.selected[dataset.name] || { checked: false, weights: "" };
    const status = weightsStatus(item.weights, state.models[dataset.name]);

    const checkbox = h("input", {
      type: "checkbox",
      checked: item.checked,
      onchange: (event) => {
        item.checked = event.target.checked;
        state.dirtyDataset = true;
        update({});
      },
    });

    const weightsInput = h("input", {
      type: "text",
      class: "path",
      value: item.weights,
      placeholder: "选择 .pt 文件（可在任意目录）",
      onchange: (event) => {
        item.weights = event.target.value.trim();
        state.dirtyDataset = true;
        // 路径变了，旧的校验结果不再适用
        if (state.models) delete state.models[dataset.name];
        update({});
      },
    });

    const browse = h("button", {
      text: "浏览…",
      onclick: async () => {
        if (state.running) {
          toast("任务运行中，暂不能导入权重", "err");
          return;
        }
        try {
          // 原路径已失效时不要用它做浏览起点（输入框里可能只是推荐路径）
          const start = dataset.weights_previous ? "" : item.weights || "";
          const picked = await api.post("/api/fs/pick", {
            mode: "file",
            filters: ["pt"],
            start,
          });
          if (picked.cancelled || !picked.path) return;
          // 复制到数据集目录并重命名为 <数据集目录名>.pt，同时清理旧 .pt
          update({ scanning: true });
          const imported = await api.post("/api/weights/import", {
            dataset: dataset.path,
            source: picked.path,
          });
          item.weights = imported.path;
          state.dirtyDataset = true;
          // 导入的是新文件，旧的校验结果不再适用
          if (state.models) delete state.models[dataset.name];
          toast(`模型已复制为 ${imported.path}`, "ok");
        } catch (err) {
          toast(err.message, "err");
        } finally {
          update({ scanning: false });
        }
      },
    });

    const splits = Object.entries(dataset.splits || {})
      .filter(([name]) => name === "train" || name === "val")
      .map(([name, stat]) => `${name} ${stat.images}`)
      .join(" / ");

    body.append(
      h(
        "tr",
        {},
        h("td", {}, checkbox),
        h(
          "td",
          {},
          h(
            "div",
            { class: "dataset-name" },
            dataset.name,
            dataset.extra ? badge(" 手动添加", "warn") : null
          ),
          h("div", { class: "dataset-sub mono", text: dataset.path })
        ),
        h(
          "td",
          {},
          badge(dataset.task || "-", dataset.task === "pose" ? "ok" : ""),
          h("div", { class: "dataset-sub", text: `${dataset.num_classes} 类` }),
          h("div", {
            class: "dataset-sub",
            text: dataset.kpt_shape ? `kpt ${dataset.kpt_shape.join("×")}` : "无关键点",
          })
        ),
        h("td", { class: "split-list", text: splits || "-" }),
        h(
          "td",
          {},
          h(
            "div",
            { class: "weights-cell" },
            dot(status.kind),
            weightsInput,
            browse,
            h("span", { class: "hint", text: status.text })
          ),
          dataset.weights_previous
            ? h("div", { class: "warn-text", text: `原路径已失效：${dataset.weights_previous}` })
            : null
        ),
        h(
          "td",
          { class: "actions" },
          h("button", {
            class: "danger",
            text: "删除",
            onclick: () => askRemoveDatasets([dataset]),
          })
        )
      )
    );
  });

  if (!state.datasets.length) {
    body.append(h("tr", {}, h("td", { colspan: 6, class: "hint", text: "尚未扫描到数据集" })));
  }

  return section(
    "数据集列表",
    h(
      "div",
      { class: "table-wrap" },
      h(
        "table",
        {},
        h(
          "thead",
          {},
          h(
            "tr",
            {},
            h("th", { text: "选择" }),
            h("th", { text: "数据集 / 路径" }),
            h("th", { text: "任务 / 类别" }),
            h("th", { text: "图片数" }),
            h("th", { text: "模型权重（可单独指定）" }),
            h("th", { text: "操作" })
          )
        ),
        body
      )
    ),
    h(
      "div",
      { class: "toolbar" },
      h("button", { text: "校验权重与一致性", class: "primary", onclick: validate }),
      h("span", {
        class: "hint",
        text: "校验会逐个加载 .pt，确认 task / 类别 / kpt_shape 与数据集一致",
      })
    )
  );
}

function detectorCard() {
  const detector = state.detector;
  const modeSelect = h(
    "select",
    {
      onchange: (event) => {
        detector.mode = event.target.value;
        update({});
      },
    },
    [
      ["yolo", "yolo（用各自 .pt 预测）"],
      ["labels", "labels（直接使用人工标注）"],
    ].map(([value, label]) =>
      h("option", { value, text: label, selected: detector.mode === value })
    )
  );

  const PARAM_TIPS = {
    conf: "置信度阈值（0~1）：只保留置信度不低于该值的检测框。调高更严格（漏检增多），调低更宽松（误检增多）。",
    iou: "NMS 的 IoU 阈值（0~1）：交并比超过该值的重叠框会被抑制。调高保留更多重叠框，调低抑制更激进。",
    imgsz: "YOLO 推理输入尺寸（像素，如 640）：越大越精确但越慢，需为 32 的倍数。",
  };

  const numberInput = (key, step) =>
    h("input", {
      type: "number",
      step,
      value: detector[key],
      title: PARAM_TIPS[key] || "",
      onchange: (event) => {
        detector[key] = Number(event.target.value);
      },
    });

  const splitBoxes = ["train", "val", "test"].map((name) =>
    h(
      "label",
      {},
      h("input", {
        type: "checkbox",
        checked: detector.splits.includes(name),
        onchange: (event) => {
          const next = new Set(detector.splits);
          if (event.target.checked) next.add(name);
          else next.delete(name);
          detector.splits = [...next];
          update({});
        },
      }),
      ` ${name}`
    )
  );

  return section(
    "检测器设置（所有数据集共用）",
    h(
      "div",
      { class: "row" },
      h("label", {}, "模式 ", modeSelect),
      h("label", { title: PARAM_TIPS.conf }, "conf ", numberInput("conf", "0.05")),
      h("label", { title: PARAM_TIPS.iou }, "iou ", numberInput("iou", "0.05")),
      h("label", { title: PARAM_TIPS.imgsz }, "imgsz ", numberInput("imgsz", "32")),
      h("span", { class: "hint", text: "使用划分：" }),
      ...splitBoxes
    ),
    h("div", { class: "hint", text: "建库与验证必须使用同一组 conf / iou / imgsz，否则特征分布会有偏差" })
  );
}

function weightsStatus(weights, info) {
  if (!weights) return { kind: "err", text: "未指定" };
  if (info && info.error) return { kind: "err", text: "加载失败" };
  // 必须“校验过的路径”等于“输入框里的路径”：旧实现只看 info.path 是否存在，
  // 改完权重路径后仍会显示“已加载”，让人误以为已校验。
  if (info && info.path && samePath(info.path, weights)) {
    return { kind: "ok", text: "已加载" };
  }
  return { kind: "warn", text: "未校验" };
}

// 仅用于提示的路径比较：统一斜杠 + 忽略大小写（Windows 路径不区分大小写）
function samePath(a, b) {
  const norm = (value) =>
    String(value || "")
      .replace(/\\/g, "/")
      .replace(/\/+$/, "")
      .toLowerCase();
  if (!a || !b) return false;
  return norm(a) === norm(b);
}

async function validate() {
  const datasets = selectedNames().map((name) => ({
    name,
    weights: state.selected[name].weights,
  }));
  if (!datasets.length) {
    toast("请至少勾选一个数据集", "err");
    return;
  }
  try {
    const data = await api.post("/api/datasets/validate", {
      root: state.root,
      detector: state.detector.mode,
      datasets,
    });
    update({
      models: data.models || {},
      global: data.global,
      errors: data.errors || [],
      warnings: data.warnings || [],
      validation: {
        ok: Boolean(data.ok),
        errors: data.errors || [],
        warnings: data.warnings || [],
        datasetCount: datasets.length,
      },
    });
    if (data.ok) toast("校验通过", "ok");
    else toast(data.errors[0]?.message || "校验未通过", "err");
  } catch (err) {
    toast(err.message, "err");
  }
}

function globalCard() {
  const names = state.global?.target_names || [];
  const slotOf = state.global?.slot_of || {};
  const rows = names.map((name, index) => {
    const sources = Object.entries(slotOf)
      .filter(([, map]) => Object.prototype.hasOwnProperty.call(map, name))
      .map(([dataset, map]) => `${dataset}#${map[name]}`);
    return h(
      "tr",
      {},
      h("td", { text: String(index) }),
      h("td", { class: "names", text: name }),
      h("td", { class: "hint", text: sources.join("、") || "-" }),
      h("td", {}, sources.length > 1 ? badge("共享", "warn") : badge("独立"))
    );
  });

  return section(
    "全局类别表",
    h(
      "div",
      { class: "row" },
      badge(`类别 ${state.global?.num_target_classes || 0}`, "ok"),
      badge(`每类 ${state.global?.per_class_components || 7} 维`),
      badge(`feature_dim ${state.global?.feature_dim || 0}`, "ok")
    ),
    h(
      "table",
      { class: "global-table" },
      h(
        "thead",
        {},
        h("tr", {}, h("th", { text: "全局 id" }), h("th", { text: "名称" }), h("th", { text: "来源" }), h("th", { text: "状态" }))
      ),
      h("tbody", {}, rows.length ? rows : h("tr", {}, h("td", { colspan: 4, class: "hint", text: "勾选并校验后可预览" })))
    )
  );
}

function messageCard() {
  const result = state.validation;
  if (!result) {
    // 还没点过“校验权重与一致性”：只展示扫描提示，避免与校验结果混淆
    if (!state.warnings.length) return h("div");
    return section(
      "校验结果",
      h("div", { class: "hint", text: "尚未校验；以下为扫描提示" }),
      ...(state.warnings || []).map((item) =>
        h("div", { class: "warn-text", text: `⚠ ${item.message}` })
      )
    );
  }

  const summary = result.ok
    ? `✔ 校验通过${result.warnings.length ? `（${result.warnings.length} 条提示）` : ""}`
    : `✖ 校验未通过（${result.errors.length} 项错误）`;
  return section(
    "校验结果",
    h("div", { class: result.ok ? "ok-text" : "err-text", text: summary }),
    ...(result.errors || []).map((item) =>
      h("div", {
        class: "err-text",
        text: `✖ ${item.dataset ? `[${item.dataset}] ` : ""}${item.message}`,
      })
    ),
    ...(result.warnings || []).map((item) =>
      h("div", { class: "warn-text", text: `⚠ ${item.message}` })
    )
  );
}

function actionRow() {
  return h(
    "div",
    { class: "toolbar" },
    h("button", {
      class: "primary",
      text: "下一步：数据映射 →",
      disabled: !(selectedNames().length > 0 && state.errors.length === 0),
      onclick: () => document.dispatchEvent(new CustomEvent("goto-tab", { detail: 2 })),
    }),
    state.dirtyDataset ? h("span", { class: "warn-text", text: "配置已变更，需重新构建与训练" }) : null
  );
}
