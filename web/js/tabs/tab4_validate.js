// Tab4 模型验证：来源选择 + 50 张上限 + SSE 结果
import * as api from "../api.js";
import { openTask } from "../sse.js";
import { state, update } from "../state.js";
import { h, clear, toast, badge, section, download } from "../ui.js";

// 与后端 validate_service.IMAGE_LIMIT 一致；扫描结果会带回同样的 limit
const DEFAULT_IMAGE_LIMIT = 50;

// 验证参数的中文悬停说明（与训练页一致：鼠标移到标签或输入框上即可看到）
const TIPS = {
  smooth: "smooth：概率平滑窗口，对最近 N 次预测的概率取平均以抑制抖动。越大越平稳但响应越慢；默认 5。",
  conf: "conf：YOLO 目标置信度阈值，低于该值的检测会被丢弃。调低保留更多目标但易误检，调高更严格；默认 0.25。",
  iou: "iou：YOLO NMS 的 IoU 阈值，控制重叠检测框的抑制程度。调低合并更积极，调高保留更多重叠框；默认 0.45。",
  imgsz: "imgsz：YOLO 推理输入尺寸（像素），需为 32 的倍数。越大越精确但越慢；默认 640，且应与建库时一致。",
};

export function render(container) {
  clear(container);
  container.append(sourceCard());
  const logs = logCard();
  if (logs) container.append(logs);
  container.append(resultCard());
}

// 验证日志：开始验证后才出现，实时追加 inference.py 的输出
function logCard() {
  if (!state.running && !(state.validateLog || []).length) return null;
  const log = h("div", { class: "log" });
  (state.validateLog || []).forEach((item) => appendLog(log, item));
  return section("验证日志", log);
}

function appendLog(node, item) {
  node.append(
    h("div", { class: item.level === "info" ? "" : item.level, text: item.text })
  );
  node.scrollTop = node.scrollHeight;
}

function sourceCard() {
  const payloadMeta = state.meta;

  const modeSelect = h(
    "select",
    {
      onchange: (event) => {
        // 只切模式：扫描结果与勾选保留，切回目录模式时无需重新扫描
        state.validate.mode = event.target.value;
        update({});
      },
    },
    [
      ["dir", "目录"],
      ["file", "单张图片"],
    ].map(([value, label]) =>
      h("option", { value, text: label, selected: state.validate.mode === value })
    )
  );

  const sourceInput = h("input", {
    type: "text",
    class: "path",
    value: state.validate.source,
    placeholder: state.validate.mode === "dir" ? "图片目录" : "单张图片路径",
    onchange: (event) => {
      state.validate.source = event.target.value.trim();
      state.validate.files = [];
      state.imageScan = null;
      update({});
    },
  });

  const browse = h("button", {
    text: "浏览…",
    onclick: async () => {
      try {
        const picked = await api.post("/api/fs/pick", {
          mode: state.validate.mode === "dir" ? "dir" : "file",
          filters: ["jpg", "jpeg", "png", "bmp", "webp"],
          start: state.validate.source,
        });
        if (picked.path) {
          state.validate.source = picked.path;
          state.validate.files = [];
          state.imageScan = null;
          update({});
        }
      } catch (err) {
        toast(err.message, "err");
      }
    },
  });

  return section(
    "验证来源",
    h(
      "div",
      { class: "row" },
      h("label", {}, "来源 ", modeSelect),
      sourceInput,
      browse,
      state.validate.mode === "dir"
        ? h("button", {
            text: "扫描图片",
            onclick: scanImages,
            disabled: !state.validate.source,
          })
        : null
    ),
    h(
      "div",
      { class: "row" },
      h("label", { "data-tip": TIPS.smooth }, "smooth ", h("input", {
        type: "number",
        value: state.validate.smooth,
        onchange: (event) => {
          state.validate.smooth = Number(event.target.value);
        },
      })),
      h("label", { "data-tip": TIPS.conf }, "conf ", h("input", {
        type: "number",
        step: "0.05",
        value: state.validate.conf,
        onchange: (event) => {
          state.validate.conf = Number(event.target.value);
        },
      })),
      h("label", { "data-tip": TIPS.iou }, "iou ", h("input", {
        type: "number",
        step: "0.05",
        value: state.validate.iou,
        onchange: (event) => {
          state.validate.iou = Number(event.target.value);
        },
      })),
      h("label", { "data-tip": TIPS.imgsz }, "imgsz ", h("input", {
        type: "number",
        value: state.validate.imgsz,
        onchange: (event) => {
          state.validate.imgsz = Number(event.target.value);
        },
      }))
    ),
    workloadHint(payloadMeta),
    state.validate.mode === "dir" ? fileList() : null,
    h(
      "div",
      { class: "toolbar verify-actions" },
      h("button", { class: "primary", text: "开始验证", onclick: runValidate, disabled: state.running }),
      h("span", { class: "hint", text: `目录模式最多 ${imageLimit()} 张；预热帧不占用额度` })
    )
  );
}

// 缩略图走后端只读接口：按来源目录 + 文件名取图，避免跨目录读取
function thumbUrl(name) {
  const query = new URLSearchParams({ path: state.validate.source, name });
  return `/api/images/thumb?${query}`;
}

function imageLimit() {
  return state.imageScan?.limit || DEFAULT_IMAGE_LIMIT;
}

function workloadHint(meta) {
  const window =
    meta?.window_size || state.config?.dataset?.window_size || 8;
  if (state.validate.mode !== "dir") {
    return h("div", {
      class: "inline-note",
      text: `单张图片：用同目录前置 ${window - 1} 帧预热窗口，只输出该帧的判定`,
    });
  }
  const scan = state.imageScan;
  if (!scan) {
    return h("div", { class: "inline-note", text: `窗口 ${window} 帧：目录模式请先“扫描图片”确认数量` });
  }
  const count = state.validate.files.length;
  if (!count) {
    return h("div", {
      class: "inline-note",
      text: `窗口 ${window} 帧：已选 0 张（目录共 ${scan.count} 张），请点“全选”或手动勾选`,
    });
  }
  const expected = Math.max(0, count - (window - 1));
  return h("div", {
    class: "inline-note",
    text: `已选 ${count} 张 → 预热 ${Math.min(window - 1, count)} 帧 → 预计输出 ${expected} 条`,
  });
}

// 目录上万张时不可能全部渲染：只画前 N 行，其余用文件名过滤定位
const RENDER_LIMIT = 200;

function fileList() {
  const scan = state.imageScan;
  if (!scan) return h("div");
  const selected = new Set(state.validate.files);
  const limit = scan.limit || DEFAULT_IMAGE_LIMIT;
  const list = h("div", { class: "file-list" });
  const summary = h("span", { class: "hint" });

  const matchedFiles = () => {
    const keyword = String(state.imageFilter || "").trim().toLowerCase();
    if (!keyword) return scan.files;
    return scan.files.filter((name) => name.toLowerCase().includes(keyword));
  };

  const renderList = () => {
    clear(list);
    const files = matchedFiles();
    const shown = files.slice(0, RENDER_LIMIT);
    shown.forEach((name) => {
      list.append(
        h(
          "label",
          {},
          h("input", {
            type: "checkbox",
            checked: selected.has(name),
            onchange: (event) => {
              const next = new Set(state.validate.files);
              if (event.target.checked) next.add(name);
              else next.delete(name);
              // 按扫描顺序保存，提交时预热帧顺序才正确
              state.validate.files = scan.files.filter((item) => next.has(item));
              update({});
            },
          }),
          h("img", {
            class: "thumb",
            src: thumbUrl(name),
            alt: "",
            loading: "lazy",
            decoding: "async",
            title: name,
          }),
          h("span", { class: "mono", text: name })
        )
      );
    });
    if (!shown.length) {
      list.append(h("div", { class: "hint", text: "没有匹配的文件名" }));
    }
    const parts = [`共 ${scan.count} 张`];
    if (scan.truncated) parts.push(`仅加载前 ${scan.files.length} 张`);
    if (files.length > shown.length) {
      parts.push(`匹配 ${files.length} 张，仅显示前 ${shown.length} 张`);
    }
    parts.push(`已选 ${selected.size} 张`);
    summary.textContent = parts.join(" · ");
    summary.classList.toggle("warn-text", scan.count > limit);
  };

  const filterInput = h("input", {
    type: "search",
    placeholder: "按文件名过滤…",
    value: state.imageFilter || "",
    oninput: (event) => {
      // 只重画列表，不整页重渲染，避免输入框失焦
      state.imageFilter = event.target.value;
      renderList();
    },
  });

  const selectAll = () => {
    // 只选当前过滤结果，且不超过提交上限
    const files = matchedFiles();
    state.validate.files = files.slice(0, limit);
    update({});
    if (files.length > limit) {
      toast(`最多 ${limit} 张，已选择前 ${limit} 张`, "err");
    }
  };

  const clearAll = () => {
    state.validate.files = [];
    update({});
  };

  renderList();

  return h(
    "div",
    {},
    h(
      "div",
      { class: "row" },
      h("button", { text: "全选", onclick: selectAll }),
      h("button", { text: "取消全选", onclick: clearAll }),
      filterInput,
      summary
    ),
    list
  );
}

async function scanImages() {
  try {
    const scan = await api.get("/api/images/scan", { path: state.validate.source });
    // 换了目录，旧的过滤词不再适用
    state.imageFilter = "";
    // 上限内默认全选；超出上限默认不选，由“全选”或手动勾选决定
    state.validate.files = scan.files.length <= scan.limit ? [...scan.files] : [];
    update({ imageScan: scan });
  } catch (err) {
    toast(err.message, "err");
  }
}

async function runValidate() {
  if (!state.validate.source) {
    toast("请选择图片目录或单张图片", "err");
    return;
  }
  if (state.validate.mode === "dir" && state.imageScan) {
    const limit = imageLimit();
    const count = state.validate.files.length;
    if (!count) {
      toast("请至少选择一张图片", "err");
      return;
    }
    if (count > limit) {
      toast(`最多 ${limit} 张`, "err");
      return;
    }
  }

  update({ running: true, predictions: [], lastValidate: null, validateLog: [] });
  render(document.getElementById("panel-4"));
  const logNode = document.querySelector("#panel-4 .log");

  try {
    await openTask(
      "/api/validate",
      {
        checkpoint: state.checkpoint,
        mode: state.validate.mode,
        source: state.validate.source,
        files: state.validate.mode === "dir" ? state.validate.files : [],
        conf: state.validate.conf,
        iou: state.validate.iou,
        imgsz: state.validate.imgsz,
        smooth: state.validate.smooth,
      },
      {
        onEvent(name, data) {
          if (name === "log") {
            const item = { level: data.level || "info", text: data.text || "" };
            (state.validateLog ||= []).push(item);
            if (logNode) appendLog(logNode, item);
          } else if (name === "done") {
            update({
              running: false,
              predictions: data.result?.predictions || [],
              lastValidate: data.result || null,
            });
            toast(`验证完成，输出 ${data.result?.outputs ?? 0} 条`, "ok");
            render(document.getElementById("panel-4"));
          } else if (name === "error") {
            update({
              running: false,
              predictions: data.result?.predictions || [],
              lastValidate: data.result || null,
            });
            toast(data.message || "验证失败", "err");
            render(document.getElementById("panel-4"));
          }
        },
        onError(err) {
          update({ running: false });
          toast(err.message, "err");
          render(document.getElementById("panel-4"));
        },
      }
    );
  } catch (err) {
    update({ running: false });
    toast(err.message, "err");
    render(document.getElementById("panel-4"));
  }
}

function resultCard() {
  const result = state.lastValidate;
  const predictions = state.predictions || [];
  if (!result) {
    return section("验证结果", h("div", { class: "hint", text: "尚未运行验证" }));
  }

  const rows = predictions.map((item) =>
    h(
      "tr",
      {},
      h("td", { class: "mono", text: item.frame }),
      h("td", { text: String(item.step_id) }),
      h("td", { text: item.step_name }),
      h("td", { text: Number(item.confidence).toFixed(3) })
    )
  );

  // CSV 单元格转义：含逗号/引号/换行的字段必须用双引号包裹，内部引号双写
  const csvCell = (value) => {
    const text = value === undefined || value === null ? "" : String(value);
    return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };

  const exportCsv = () => {
    const header = ["frame", "step_id", "step_name", "confidence"].map(csvCell).join(",");
    const body = predictions
      .map((item) =>
        [item.frame, item.step_id, item.step_name, item.confidence].map(csvCell).join(",")
      )
      .join("\n");
    // 带 BOM 便于 Excel 正确识别中文
    download(`predictions_${Date.now()}.csv`, `\ufeff${header}\n${body}\n`);
  };

  return section(
    "验证结果",
    h(
      "div",
      { class: "row" },
      badge(`数据集 ${result.dataset || "-"}`, "ok"),
      badge(`输出 ${result.outputs} 条`),
      badge(`预计 ${result.expected ?? "-"} 条`),
      badge(`处理 ${result.processed} 帧`),
      badge(`预热 ${result.warmup} 帧`),
      result.mean_confidence !== undefined ? badge(`平均置信度 ${Number(result.mean_confidence).toFixed(3)}`) : null,
      h("button", { text: "导出 CSV", onclick: exportCsv, disabled: !predictions.length })
    ),
    result.weights ? h("div", { class: "hint mono", text: `权重：${result.weights}` }) : null,
    result.expected !== undefined && result.outputs !== result.expected
      ? h("div", {
          class: "warn-text",
          text: `输出 ${result.outputs} 条与预计 ${result.expected} 条不一致，请检查帧数/窗口/预热设置`,
        })
      : null,
    (result.retained || []).length
      ? h("div", {
          class: "warn-text",
          text: `失败现场已保留：${result.retained.join("；")}`,
        })
      : null,
    result.outputs === 0
      ? h("div", { class: "inline-note", text: `帧数不足（需要 ${result.window} 帧，实际 ${result.processed} 帧），输出 0 条` })
      : null,
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
            h("th", { text: "frame" }),
            h("th", { text: "步骤" }),
            h("th", { text: "步骤名" }),
            h("th", { text: "置信度" })
          )
        ),
        h("tbody", {}, rows.length ? rows : h("tr", {}, h("td", { colspan: 4, class: "hint", text: "无输出" })))
      )
    )
  );
}
