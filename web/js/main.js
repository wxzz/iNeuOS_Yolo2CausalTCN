// 入口：Tab 切换 + 全局渲染调度 + 状态条
import * as api from "./api.js";
import { state, update, subscribe, selectedNames } from "./state.js";
import { h, clear } from "./ui.js";
import { render as renderTab1 } from "./tabs/tab1_source.js";
import { render as renderTab2 } from "./tabs/tab2_mapping.js";
import { render as renderTab3 } from "./tabs/tab3_train.js";
import { render as renderTab4 } from "./tabs/tab4_validate.js";

const RENDERERS = { 1: renderTab1, 2: renderTab2, 3: renderTab3, 4: renderTab4 };

function renderStatus() {
  const bar = document.getElementById("statusbar");
  clear(bar);

  const names = selectedNames();
  const frames = (state.datasets || [])
    .filter((item) => names.includes(item.name))
    .reduce((total, item) => {
      const splits = item.splits || {};
      const train = splits.train?.images || 0;
      return total + train;
    }, 0);

  const stepCount = (state.steps || []).length;
  const classCount = new Set((state.steps || []).map((step) => step.id)).size;
  const parts = [
    `数据集 ${names.length} 个（${names.join(" + ") || "未选择"}${frames ? ` = ${frames} 帧` : ""}）`,
    `全局类别 ${state.global?.num_target_classes || 0}`,
    `特征维度 ${state.global?.feature_dim || 0}`,
    `步骤 ${stepCount}${classCount < stepCount ? `（类别 ${classCount}）` : ""}`,
    state.running
      ? `任务运行中${state.taskLabel ? `：${state.taskLabel}` : ""}…`
      : "空闲",
    state.dirtyDataset ? "⚠ 需重新构建" : "",
  ].filter(Boolean);

  bar.append(h("div", { text: parts.join("  |  ") }));
  if (state.warnings?.length) {
    bar.append(h("div", { text: `⚠ ${state.warnings.length} 条提示：${state.warnings[0].message}` }));
  }
  if (state.errors?.length) {
    bar.append(h("div", { class: "err-text", text: `✖ ${state.errors[0].message}` }));
  }
}

function renderAll() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.tab) === state.tab);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `panel-${state.tab}`);
  });

  renderStatus();

  const panel = document.getElementById(`panel-${state.tab}`);
  const renderer = RENDERERS[state.tab];
  if (panel && renderer) renderer(panel);
}

function bindTabs() {
  document.getElementById("tabs").addEventListener("click", (event) => {
    const button = event.target.closest(".tab");
    if (!button) return;
    const tab = Number(button.dataset.tab);
    if (!Number.isFinite(tab) || tab === state.tab) return;
    if (tab > 1 && selectedNames().length === 0) return;
    state.tab = tab;
    renderAll();
  });

  document.addEventListener("goto-tab", (event) => {
    const tab = Number(event.detail);
    if (RENDERERS[tab]) {
      state.tab = tab;
      renderAll();
    }
  });
}

// 悬停提示（[data-tip]）：靠近视口右边界时自动向左对齐，避免提示被裁掉
function bindTooltips() {
  document.addEventListener("mouseover", (event) => {
    const target = event.target.closest?.("[data-tip]");
    if (!target) return;
    const width = Math.min(280, window.innerWidth - 24);
    const rect = target.getBoundingClientRect();
    target.classList.toggle("tip-end", rect.left + width > window.innerWidth - 8);
  });
}

subscribe(renderAll);
bindTabs();
bindTooltips();
renderAll();

// --------------------------------------------------------------------------- //
// 运行时状态恢复：页面刷新后恢复“任务运行中”与 dataset_meta（训练/验证页需要）
// --------------------------------------------------------------------------- //
let metaChecked = false;

async function syncRuntimeState() {
  try {
    const status = await api.get("/api/task/status");
    const running = Boolean(status.running);
    const label = status.task?.label || "";
    const patch = {};
    if (running !== state.running) patch.running = running;
    if (label !== state.taskLabel) patch.taskLabel = label;

    if (!metaChecked) {
      const meta = await api.get("/api/dataset/meta");
      metaChecked = true;
      if (meta.exists && meta.meta && !state.meta) patch.meta = meta.meta;
    }
    if (Object.keys(patch).length) update(patch);
  } catch (err) {
    // 服务尚未就绪或短暂异常：静默等待下一次轮询
  }
}

syncRuntimeState();
window.setInterval(syncRuntimeState, 4000);
