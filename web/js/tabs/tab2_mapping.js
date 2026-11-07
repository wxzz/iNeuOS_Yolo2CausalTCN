// Tab2 数据映射：步骤/规则编辑 + 预览 + 构建（SSE）
import * as api from "../api.js";
import { openTask } from "../sse.js";
import { state, update, selectedNames } from "../state.js";
import { h, clear, toast, section, pickMany } from "../ui.js";

// 规则类型只在这里定义一次：新增类型时同步维护 has/create，
// 避免 ruleTypeOf / rebuildMatch / 下发选项三处各自判断而漏改。
const RULE_TYPES = [
  {
    value: "class",
    label: "类别名称",
    has: (match) => Array.isArray(match.classes),
    create: () => ({ classes: [] }),
  },
  {
    value: "time",
    label: "时间区间",
    // 必须按“键是否存在”判断：空字符串也是时间规则（只是未填写）
    has: (match) => "datetime_from" in match || "datetime_to" in match,
    create: () => ({ datetime_from: "", datetime_to: "" }),
  },
  {
    value: "pattern",
    label: "帧名正则",
    has: (match) => typeof match.pattern === "string",
    create: () => ({ pattern: "" }),
  },
  {
    value: "index",
    label: "数据集内帧号",
    has: (match) =>
      match.index_from !== undefined || match.index_to !== undefined,
    create: () => ({ index_from: 1, index_to: 99999 }),
  },
  {
    value: "frames",
    label: "帧列表",
    has: (match) => Array.isArray(match.frames),
    create: () => ({ frames: [] }),
  },
];

export function render(container) {
  clear(container);
  container.append(stepsCard(), buildCard());
  // 首次进入/构建完成后自动加载，不弹“已重新加载”提示
  if (!state.configLoaded) loadConfig({ notify: false });
}

let configLoading = false;

async function loadConfig({ notify = true } = {}) {
  if (configLoading) return;
  configLoading = true;
  const reloadButtons = [...document.querySelectorAll("#panel-2 button")].filter(
    (button) => button.textContent.includes("重新加载配置")
  );
  reloadButtons.forEach((button) => {
    button.disabled = true;
    button.textContent = "加载中…";
  });
  try {
    const config = await api.get("/api/config");
    const steps = (config.steps || []).map((step) => ({
      id: Number(step.id),
      name: String(step.name || ""),
      match: step.match || {},
    }));
    update({ config, steps, configLoaded: true });
    if (notify) {
      if (config.exists === false) {
        toast("配置文件不存在，已加载空配置", "err");
      } else {
        toast(`已重新加载配置：${steps.length} 个步骤`, "ok");
      }
    }
  } catch (err) {
    toast(describeApiError(err), "err");
  } finally {
    configLoading = false;
    // 出错时不会重渲染，这里恢复按钮；成功时按钮已被新 DOM 替换
    document.querySelectorAll("#panel-2 button").forEach((button) => {
      if (button.textContent === "加载中…") {
        button.disabled = false;
        button.textContent = "重新加载配置";
      }
    });
  }
}

// 类型必须按「键是否存在」判断：空字符串/空数组也代表该类型，
// 用真值判断会把刚切换的步骤判回其它类型，下拉框选了也会立刻跳回去。
function ruleTypeOf(match = {}) {
  const found = RULE_TYPES.find((item) => item.has(match));
  return found ? found.value : "class";
}

// 至少包含一个有效条件（空数组/空字符串不算）
function hasCondition(match = {}) {
  if (Array.isArray(match.classes) && match.classes.length) return true;
  if (Array.isArray(match.frames) && match.frames.length) return true;
  if (typeof match.pattern === "string" && match.pattern.trim()) return true;
  if ((match.datetime_from || "").trim() || (match.datetime_to || "").trim()) {
    return true;
  }
  if ((match.time_from || "").trim() || (match.time_to || "").trim()) {
    return true;
  }
  return match.index_from !== undefined || match.index_to !== undefined;
}

function isRegexValid(pattern) {
  try {
    new RegExp(pattern);
    return true;
  } catch {
    return false;
  }
}

// 配置接口的 error.detail 是后端校验错误数组的 JSON，优先展示里面的具体原因
function describeApiError(err) {
  const detail = err && err.detail;
  if (detail) {
    try {
      const parsed = JSON.parse(detail);
      if (Array.isArray(parsed) && parsed.length) {
        return parsed
          .map((item) => item.message || item.code || "")
          .filter(Boolean)
          .join("；");
      }
    } catch {
      return detail;
    }
  }
  return (err && err.message) || "请求失败";
}

// 同一编号 = 跨数据集同一步骤：允许重复，但名称必须一致
function idConflict(stepId) {
  const same = state.steps.filter((item) => item.id === stepId);
  if (same.length < 2) return "";
  const names = [...new Set(same.map((item) => (item.name || "").trim()))];
  if (names.length < 2) return "";
  return `编号 #${stepId} 的步骤名称不一致：${names
    .map((name) => name || "(空)")
    .join(" / ")}（相同编号必须是同一步骤）`;
}

function idGroupNote(stepId) {
  const same = state.steps.filter((item) => item.id === stepId);
  if (same.length < 2 || idConflict(stepId)) return "";
  return `编号 #${stepId} 共 ${same.length} 个步骤，合并为同一类别`;
}

// 名称改动后刷新所有卡片的编号提示（同编号跨卡片联动）
function refreshAllIdNotes() {
  document.querySelectorAll("#panel-2 .step-card").forEach((card) => {
    const stepId = Number(card.dataset.id);
    const conflict = card.querySelector(".id-conflict");
    if (conflict) {
      const message = idConflict(stepId);
      conflict.textContent = message;
      conflict.hidden = !message;
    }
    const group = card.querySelector(".id-group");
    if (group) {
      const message = idGroupNote(stepId);
      group.textContent = message;
      group.hidden = !message;
    }
  });
}

// 删除后把唯一编号压缩为 0..K-1（同一编号的多个步骤作为一组移动）
function compactStepIds() {
  const unique = [...new Set(state.steps.map((item) => item.id))].sort((a, b) => a - b);
  const remap = new Map(unique.map((id, index) => [id, index]));
  state.steps.forEach((item) => {
    item.id = remap.get(item.id);
  });
}

function stepErrors() {
  const errors = [];
  const unique = [...new Set(state.steps.map((step) => step.id))].sort((a, b) => a - b);
  if (unique.join(",") !== unique.map((_, index) => index).join(",")) {
    errors.push(`步骤编号必须从 0 连续（相同编号可重复），当前唯一编号为 [${unique.join(", ")}]`);
  }
  const namesById = new Map();
  for (const step of state.steps) {
    if (!namesById.has(step.id)) namesById.set(step.id, new Set());
    namesById.get(step.id).add((step.name || "").trim());
  }
  for (const [stepId, names] of namesById) {
    if (names.size > 1) {
      errors.push(
        `编号 #${stepId} 的步骤名称不一致：${[...names].map((name) => name || "(空)").join(" / ")}`
      );
    }
  }
  state.steps.forEach((step) => {
    if (!hasCondition(step.match || {})) {
      errors.push(
        `步骤 #${step.id}（${step.name || "未命名"}）没有任何匹配条件，请选择类别/帧列表/正则/时间/帧号`
      );
    }
  });
  return errors;
}

function stepsCard() {
  const list = h("div");
  const steps = [...state.steps].sort((a, b) => a.id - b.id);

  steps.forEach((step, index) => {
    const type = ruleTypeOf(step.match);
    const datasetOptions = ["", ...selectedNames()];

    const datasetSelect = h(
      "select",
      {
        onchange: (event) => {
          const previous = step.match.dataset;
          const value = event.target.value || undefined;
          step.match.dataset = value;
          if (!value) delete step.match.dataset;
          // 帧 ID 带数据集前缀、类别名也是数据集本地的：换数据集后旧选择会失效，直接清空
          if (value && value !== previous) {
            let cleared = false;
            if ((step.match.frames || []).length) {
              step.match.frames = [];
              cleared = true;
            }
            if ((step.match.classes || []).length) {
              step.match.classes = [];
              cleared = true;
            }
            if (cleared) update({});
          }
        },
      },
      h("option", { value: "", text: "（不限数据集）" }),
      datasetOptions
        .filter(Boolean)
        .map((name) => h("option", { value: name, text: name, selected: step.match.dataset === name }))
    );

    const typeSelect = h(
      "select",
      {
        onchange: (event) => {
          rebuildMatch(step, event.target.value);
          update({});
        },
      },
      RULE_TYPES.map(({ value, label }) =>
        h("option", { value, text: label, selected: value === type })
      )
    );

    const hint = h("div", { class: "hint" });
    const refreshHint = () => {
      const text = describe(step) || "命中 0 帧（预览可确认）";
      hint.textContent = text;
      hint.classList.toggle("err-text", text.startsWith("正则无效"));
    };
    refreshHint();

    // 参数紧跟在「规则」下拉后面、删除按钮之前，平铺在头部以保证换行时删除按钮仍在最后
    const params = paramControls(step, type, refreshHint);

    // 编号可自定义：相同编号 = 同一步骤（可跨数据集），但名称必须一致
    const idInput = h("input", {
      type: "number",
      class: "step-id",
      min: 0,
      step: 1,
      value: step.id,
      title: "步骤编号：相同编号表示同一步骤（可跨数据集），名称必须一致",
      onchange: (event) => {
        const next = Number(event.target.value);
        if (!Number.isInteger(next) || next < 0) {
          toast("步骤编号必须是非负整数", "err");
          event.target.value = String(step.id);
          return;
        }
        step.id = next;
        update({});
      },
    });

    const conflictNode = h("div", { class: "hint err-text id-conflict" });
    const groupNode = h("div", { class: "hint id-group" });
    const conflict = idConflict(step.id);
    conflictNode.textContent = conflict;
    conflictNode.hidden = !conflict;
    const groupNote = idGroupNote(step.id);
    groupNode.textContent = groupNote;
    groupNode.hidden = !groupNote;

    list.append(
      h(
        "div",
        { class: "step-card", "data-id": step.id },
        h(
          "div",
          { class: "step-head" },
          h("span", { class: "id", text: "#" }),
          idInput,
          h("input", {
            type: "text",
            value: step.name,
            placeholder: "步骤名",
            onchange: (event) => {
              step.name = event.target.value.trim();
              refreshAllIdNotes();
            },
          }),
          h("span", { class: "hint", text: "数据集" }),
          datasetSelect,
          h("span", { class: "hint", text: "规则" }),
          typeSelect,
          params,
          h("button", {
            class: "danger",
            text: "删除",
            onclick: () => {
              state.steps = state.steps.filter((item) => item.id !== step.id);
              compactStepIds();
              update({});
            },
          })
        ),
        hint,
        conflictNode,
        groupNode
      )
    );
  });

  if (!steps.length) {
    list.append(h("div", { class: "inline-note", text: "还没有步骤。可点“添加步骤”或“重新生成步骤”（会覆盖手工规则）。" }));
  }

  return section(
    "步骤与映射规则",
    h(
      "div",
      { class: "toolbar" },
      h("button", {
        text: "＋ 添加步骤",
        onclick: () => {
          const nextId = state.steps.length
            ? Math.max(...state.steps.map((item) => item.id)) + 1
            : 0;
          state.steps.push({
            id: nextId,
            name: `步骤${nextId + 1}`,
            match: { dataset: selectedNames()[0] || undefined, classes: [] },
          });
          update({});
        },
      }),
      h("button", {
        text: "重新加载配置",
        title: "从 tcn_config.json 重新读取步骤（会丢弃未保存的修改）",
        onclick: () => loadConfig(),
      }),
      h("span", {
        class: "hint",
        text: "编号可重复（相同编号=同一类别，名称必须一致）；唯一编号需从 0 连续",
      })
    ),
    h("div", {
      class: "hint",
      text:
        "匹配顺序固定：按上方步骤顺序逐个试，同一步骤内 frames → 类别 → 帧名正则 → 日期 → 时间 → 帧号；" +
        "同一帧命中多个步骤时，只归第一个命中的步骤（构建日志会提示“被抢占了多少帧”）。",
    }),
    list,
    // 保存/构建放在最后一步下面（先保存配置，再构建数据集）
    h(
      "div",
      { class: "toolbar step-actions" },
      h("button", { class: "primary", text: "保存配置", onclick: saveConfig, disabled: state.running }),
      h("button", {
        class: "primary",
        text: "构建数据集",
        onclick: () => startBuild(),
        disabled: state.running,
      }),
      h("span", { class: "hint", text: "构建前会自动保存当前步骤；尚无配置时由构建生成默认步骤；构建会覆盖 tcn_datasets 下的 labels / labels.csv / *.npy" })
    )
  );
}

function rebuildMatch(step, type) {
  const dataset = step.match.dataset;
  const spec =
    RULE_TYPES.find((item) => item.value === type) || RULE_TYPES[0];
  step.match = { ...(dataset ? { dataset } : {}), ...spec.create() };
}

// 弹窗多选类别：确定写回 match.classes，取消保持原样
async function editClasses(step) {
  const dataset = step.match.dataset;
  if (!dataset) {
    toast("请先选择该步骤的数据集，再选择类别", "err");
    return;
  }
  const entry = (state.datasets || []).find((item) => item.name === dataset);
  const classes = (entry && entry.classes) || [];
  if (!classes.length) {
    toast(`数据集 ${dataset} 没有类别信息（请先在【源数据集】扫描）`, "err");
    return;
  }
  const picked = await pickMany({
    title: `选择类别（${dataset}）`,
    message: "帧内出现任一选中类别即判定命中该步骤。",
    options: classes.map((name) => ({ value: name, label: name })),
    selected: step.match.classes || [],
    searchPlaceholder: "搜索类别名…",
  });
  if (picked === null) return; // 取消：不保存
  step.match.classes = picked;
  update({});
  toast(`已选择 ${picked.length} 个类别，记得点“保存配置”`, picked.length ? "ok" : "");
}

// 弹窗多选帧：确定写回 match.frames，取消保持原样
async function editFrames(step) {
  const dataset = step.match.dataset;
  if (!dataset) {
    toast("请先选择该步骤的数据集，再编辑帧列表", "err");
    return;
  }
  let data;
  try {
    data = await api.get("/api/frames/list", { dataset, root: state.root });
  } catch (err) {
    toast(err.message, "err");
    return;
  }
  const options = (data.frames || []).map((frame) => ({ value: frame, label: frame }));
  if (!options.length) {
    toast(`数据集 ${dataset} 下没有可用帧（先确认图片或标签已就绪）`, "err");
    return;
  }
  const picked = await pickMany({
    title: `选择帧（${dataset}）`,
    message:
      data.source === "labels"
        ? "列表来自已生成的 labels，与建库使用的帧 ID 完全一致。"
        : "列表来自数据集图片目录（尚未建库时按图片展开）。",
    options,
    selected: step.match.frames || [],
    searchPlaceholder: "搜索帧名…",
    thumbUrl: (frame) =>
      `/api/frames/image?dataset=${encodeURIComponent(dataset)}&frame=${encodeURIComponent(frame)}`,
  });
  if (picked === null) return; // 取消：不保存
  step.match.frames = picked;
  update({});
  toast(`已选择 ${picked.length} 帧，记得点“保存配置”`, picked.length ? "ok" : "");
}

// 每个参数用 label.param 成组，保证「起/止 + 输入框」不会被换行拆开
function paramControls(step, type, refresh = () => {}) {
  if (type === "class") {
    const count = (step.match.classes || []).length;
    return [
      h("button", {
        type: "button",
        text: count ? `选择类别（${count}）` : "选择类别",
        onclick: () => editClasses(step),
      }),
      h(
        "span",
        { class: "hint", text: step.match.dataset ? "帧中出现任一选中类别即命中" : "请先选择数据集" }
      ),
    ];
  }
  if (type === "pattern") {
    return [
      h(
        "label",
        { class: "param grow" },
        h("span", { class: "hint", text: "正则" }),
        h("input", {
          type: "text",
          class: "path",
          value: step.match.pattern || "",
          placeholder: "例如 ^hand__WIN_20260903_10_26",
          oninput: (event) => {
            step.match.pattern = event.target.value;
            refresh();
          },
        })
      ),
      h("span", { class: "hint", text: "作用于帧 ID（<数据集>__<文件名>）" }),
    ];
  }
  if (type === "index") {
    return [
      h(
        "label",
        { class: "param" },
        h("span", { class: "hint", text: "起" }),
        h("input", {
          type: "number",
          value: step.match.index_from ?? 1,
          oninput: (event) => {
            step.match.index_from = Number(event.target.value);
            refresh();
          },
        })
      ),
      h(
        "label",
        { class: "param" },
        h("span", { class: "hint", text: "止" }),
        h("input", {
          type: "number",
          value: step.match.index_to ?? 99999,
          oninput: (event) => {
            step.match.index_to = Number(event.target.value);
            refresh();
          },
        })
      ),
    ];
  }
  if (type === "frames") {
    const count = (step.match.frames || []).length;
    return [
      h("button", {
        type: "button",
        text: count ? `编辑帧列表（${count}）` : "编辑帧列表",
        onclick: () => editFrames(step),
      }),
      h(
        "span",
        { class: "hint", text: step.match.dataset ? "点击后多选样本图片" : "请先选择数据集" }
      ),
    ];
  }
  return [
    h(
      "label",
      { class: "param" },
      h("span", { class: "hint", text: "起" }),
      h("input", {
        type: "text",
        value: step.match.datetime_from || "",
        placeholder: "YYYY-MM-DD HH:MM:SS",
        oninput: (event) => {
          step.match.datetime_from = event.target.value;
          refresh();
        },
      })
    ),
    h(
      "label",
      { class: "param" },
      h("span", { class: "hint", text: "止" }),
      h("input", {
        type: "text",
        value: step.match.datetime_to || "",
        placeholder: "YYYY-MM-DD HH:MM:SS",
        oninput: (event) => {
          step.match.datetime_to = event.target.value;
          refresh();
        },
      })
    ),
  ];
}

function describe(step) {
  const match = step.match || {};
  if (Array.isArray(match.classes)) {
    return match.classes.length
      ? `类别=${match.classes.join(", ")}`
      : "类别未选择（该步骤暂不命中任何帧）";
  }
  if (Array.isArray(match.frames)) {
    return match.frames.length ? `${match.frames.length} 帧` : "帧列表为空（该步骤暂不命中任何帧）";
  }
  if (typeof match.pattern === "string") {
    if (!match.pattern.trim()) return "正则未填写（该步骤暂不命中任何帧）";
    if (!isRegexValid(match.pattern)) return `正则无效：${match.pattern}`;
    return `pattern=${match.pattern}`;
  }
  if (match.index_from !== undefined) return `帧号 ${match.index_from}~${match.index_to}`;
  if ("datetime_from" in match || "datetime_to" in match) {
    const from = (match.datetime_from || "").trim();
    const to = (match.datetime_to || "").trim();
    if (!from && !to) return "时间区间未填写（该步骤暂不命中任何帧）";
    return `${from || "…"} ~ ${to || "…"}`;
  }
  return "";
}

let savingConfig = false;

// 保存步骤配置，返回是否成功；“构建数据集”会先复用它，
// 保证建库脚本读到的磁盘配置包含最新编辑。
async function persistSteps({ button = null, notify = true } = {}) {
  const errors = stepErrors();
  if (errors.length) {
    toast(errors[0], "err");
    return false;
  }

  const originalText = button ? button.textContent : "保存配置";
  if (button) {
    button.disabled = true;
    button.textContent = "正在保存";
    button.setAttribute("aria-busy", "true");
  }

  try {
    const data = await api.put("/api/config", { steps: state.steps });
    // 先弹提示再重渲染，避免重渲染耗时让用户以为按钮没响应
    if (notify) toast("配置已保存", "ok");
    update({ config: data.config, dirtyDataset: true });
    return true;
  } catch (err) {
    toast(describeApiError(err), "err");
    return false;
  } finally {
    // 成功时 update 已整页重渲染，旧按钮不在文档里了
    if (button && button.isConnected) {
      button.disabled = false;
      button.textContent = originalText;
      button.removeAttribute("aria-busy");
    }
  }
}

async function saveConfig(event) {
  if (savingConfig) return;
  if (state.running) {
    toast("任务运行中，暂不能保存配置", "err");
    return;
  }
  savingConfig = true;
  try {
    await persistSteps({
      button: event && event.currentTarget ? event.currentTarget : null,
    });
  } finally {
    savingConfig = false;
  }
}

// 日志统一进 state.log，并追加到当前 DOM 里的日志节点（避免重渲染后旧节点失效）
function pushLog(item) {
  state.log.push(item);
  const node = document.querySelector("#panel-2 .log");
  if (node) appendLog(node, item);
}

async function startBuild() {
  if (state.running) {
    toast("已有任务在运行", "err");
    return;
  }
  if (savingConfig) {
    toast("正在保存配置，请稍候", "err");
    return;
  }
  const datasets = selectedNames().map((name) => ({
    name,
    weights: state.selected[name].weights,
  }));
  if (!datasets.length) {
    toast("请先在【源数据集】勾选数据集", "err");
    return;
  }

  // 建库脚本读磁盘上的 tcn_config.json：必须先保存，否则未保存的步骤编辑会被静默丢弃。
  // 首次使用（配置尚不存在）时不能先保存，否则会死锁：交给建库脚本生成默认配置。
  if (state.config ? state.config.exists !== false : true) {
    savingConfig = true;
    let saved = false;
    try {
      saved = await persistSteps({ notify: false });
    } finally {
      savingConfig = false;
    }
    if (!saved) {
      toast("构建已取消：配置未保存", "err");
      return;
    }
  }

  update({ running: true, log: [] });
  const log = document.querySelector("#panel-2 .log");
  if (log) clear(log);

  const payload = {
    root: state.root,
    detector: state.detector.mode,
    splits: state.detector.splits,
    conf: state.detector.conf,
    iou: state.detector.iou,
    imgsz: state.detector.imgsz,
    datasets,
  };

  try {
    await openTask("/api/dataset/build", payload, {
      onEvent(name, data) {
        if (name === "log") {
          pushLog({ level: data.level || "info", text: data.text || "" });
        } else if (name === "progress") {
          pushLog({
            level: "info",
            text: `[${data.dataset}] ${data.frames} 帧 / ${data.objects} 目标 / ${data.empty} 帧无检测`,
          });
        } else if (name === "done") {
          update({
            running: false,
            meta: data.result?.meta || null,
            lastBuild: data.result || null,
            dirtyDataset: false,
            configLoaded: false,
          });
          toast("构建完成", "ok");
          render(document.getElementById("panel-2"));
        } else if (name === "error") {
          pushLog({ level: "error", text: data.message || "构建失败" });
          update({ running: false });
          toast(data.message || "构建失败", "err");
        }
      },
      onError(err) {
        pushLog({ level: "error", text: err.message });
        update({ running: false });
        toast(err.message, "err");
      },
    });
  } catch (err) {
    pushLog({ level: "error", text: err.message });
    update({ running: false });
    toast(err.message, "err");
  }
}

function buildCard() {
  const log = h("div", { class: "log" });
  state.log.forEach((item) => appendLog(log, item));
  return section("构建数据集日志", log, metaSummary());
}

function metaSummary() {
  const meta = state.meta;
  if (!meta) return h("div");
  const items = [
    ["窗口数", meta.num_windows],
    ["帧数", meta.num_frames],
    ["特征维度", meta.feature_dim],
    ["步骤数", meta.num_classes],
    ["训练/验证", `${meta.num_train} / ${meta.num_val}`],
    ["切分模式", meta.split_mode],
    ["sessions", meta.sessions],
    ["skipped_lines", meta.skipped_lines],
  ];
  return h(
    "div",
    { class: "card" },
    h("h2", { text: "dataset_meta.json 摘要" }),
    h(
      "div",
      { class: "metric-grid" },
      items.map(([key, value]) =>
        h(
          "div",
          { class: "metric" },
          h("div", { class: "k", text: key }),
          h("div", { class: "v", text: value === undefined ? "-" : String(value) })
        )
      )
    ),
    meta.label_distribution
      ? h("div", {
          class: "hint",
          text: `标签分布：${Object.entries(meta.label_distribution)
            .map(([key, value]) => `${key}:${value}`)
            .join("  ")}`,
        })
      : null
  );
}

function appendLog(node, item) {
  const line = h("div", { class: item.level === "info" ? "" : item.level, text: item.text });
  node.append(line);
  node.scrollTop = node.scrollHeight;
}
