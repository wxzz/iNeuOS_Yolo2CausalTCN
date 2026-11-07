// Tab3 模型训练：超参 + SSE 训练日志（epoch 指标直接看日志）
import { openTask } from "../sse.js";
import { state, update, selectedNames } from "../state.js";
import { h, clear, toast, badge, section } from "../ui.js";

// 训练参数的中文悬停说明（鼠标移到标签或输入框上即可看到）
const TIPS = {
  epochs: "epochs：训练轮数，整个训练集被完整遍历的次数。轮数越多拟合越充分，过大易过拟合且耗时更长；默认 80。",
  batch_size: "batch：批大小，每次参数更新使用的窗口样本数量。显存/内存不足时调小；默认 16。",
  lr: "lr：学习率，控制每次参数更新的步长。太大易震荡不收敛，太小收敛很慢；默认 0.001。",
  dropout: "dropout：随机丢弃比例，训练时按该概率丢弃神经元以抑制过拟合，取值 0~1；默认 0.2。",
  channels: "channels：各层通道数，用空格分隔，数量即层数。例如 64 64 64 64 表示 4 层、每层 64 通道。",
  device: "device：计算设备。auto 自动选择（有 CUDA 用 GPU，否则 CPU）；cpu 强制用 CPU；cuda 强制用 GPU。",
  from_npy: "使用缓存 npy：勾选后直接读取已生成的 X_train/X_val 等缓存文件，跳过重新构建，启动训练更快；数据映射变更后需重新构建，否则该选项会被禁用。",
  checkpoint: "checkpoint：模型权重保存路径（.pth）。训练完成后写入该文件，已存在则覆盖；用于后续推理与验证。",
};

export function render(container) {
  clear(container);
  container.append(formCard(), runCard());
  // 挂到 DOM 后再滚到底部，保证重渲染后直接看到最新的 epoch 行
  const log = container.querySelector(".log");
  if (log) log.scrollTop = log.scrollHeight;
}

function numeric(key, value, step = "1") {
  return h("input", {
    type: "number",
    step,
    value,
    onchange: (event) => {
      state.train[key] = Number(event.target.value);
    },
  });
}

function formCard() {
  const dataReady = Boolean(state.meta) && !state.dirtyDataset;
  const cacheBlocked = Boolean(state.dirtyDataset);
  return section(
    "训练参数",
    h(
      "div",
      { class: "inline-note" },
      state.meta
        ? `数据集：${state.meta.num_windows} 窗口 / ${state.meta.num_frames} 帧 / 特征 ${state.meta.feature_dim} 维 / 步骤 ${state.meta.num_classes} / ${state.meta.split_mode}`
        : "尚未构建数据集（请先在【数据映射】构建）"
    ),
    h(
      "div",
      { class: "row" },
      h(
        "label",
        { "data-tip": TIPS.epochs },
        "epochs ",
        numeric("epochs", state.train.epochs)
      ),
      h(
        "label",
        { "data-tip": TIPS.batch_size },
        "batch ",
        numeric("batch_size", state.train.batch_size)
      ),
      h(
        "label",
        { "data-tip": TIPS.lr },
        "lr ",
        numeric("lr", state.train.lr, "0.0001")
      ),
      h(
        "label",
        { "data-tip": TIPS.dropout },
        "dropout ",
        numeric("dropout", state.train.dropout, "0.05")
      ),
      h(
        "label",
        { "data-tip": TIPS.channels },
        "channels ",
        h("input", {
          type: "text",
          value: state.train.channels,
          onchange: (event) => {
            state.train.channels = event.target.value;
          },
        })
      ),
      h(
        "label",
        { "data-tip": TIPS.device },
        "device ",
        h(
          "select",
          {
            onchange: (event) => {
              state.train.device = event.target.value;
            },
          },
          ["auto", "cpu", "cuda"].map((value) =>
            h("option", { value, text: value, selected: state.train.device === value })
          )
        )
      ),
      h(
        "label",
        { "data-tip": TIPS.from_npy },
        h("input", {
          type: "checkbox",
          checked: state.train.from_npy,
          onchange: (event) => {
            state.train.from_npy = event.target.checked;
          },
        }),
        " 使用缓存 npy",
        cacheBlocked
          ? h("span", {
              class: "warn-text",
              text: "（配置已变更：需重新构建后才能使用缓存，否则训练会被拒绝）",
            })
          : null
      )
    ),
    h(
      "div",
      { class: "row" },
      h(
        "span",
        { class: "tip-group", "data-tip": TIPS.checkpoint },
        h("label", { class: "hint" }, "checkpoint "),
        h("input", {
          type: "text",
          class: "path",
          value: state.checkpoint,
          onchange: (event) => {
            state.checkpoint = event.target.value;
          },
        }),
        h("button", {
          text: "浏览…",
          onclick: async () => {
            const { post } = await import("../api.js");
            try {
              const picked = await post("/api/fs/pick", { mode: "file", filters: ["pth"] });
              if (picked.path) {
                state.checkpoint = picked.path;
                update({});
              }
            } catch (err) {
              toast(err.message, "err");
            }
          },
        })
      )
    ),
    dataReady ? null : h("div", { class: "warn-text", text: "数据集未构建或已变更，请先在【数据映射】构建" })
  );
}

function runCard() {
  const log = h("div", { class: "log" });
  (state.trainLog || []).forEach((item) => appendLog(log, item));

  const start = async () => {
    if (state.running) {
      toast("已有任务在运行", "err");
      return;
    }
    // 配置/数据集改过就必须先重建：否则 labels/ 还是旧步骤的产物，
    // 训练出来的标签与界面不一致（以前只拦 from_npy，漏掉了普通训练）
    if (state.dirtyDataset) {
      toast("配置或数据集已变更，请先在【数据映射】重新构建数据集", "err");
      return;
    }
    update({ running: true, trainLog: [] });
    clear(log);

    const payload = {
      checkpoint: state.checkpoint,
      epochs: state.train.epochs,
      batch_size: state.train.batch_size,
      lr: state.train.lr,
      dropout: state.train.dropout,
      device: state.train.device,
      from_npy: state.train.from_npy,
      channels: String(state.train.channels)
        .split(/\s+/)
        .filter(Boolean)
        .map(Number),
    };

    try {
      await openTask("/api/train", payload, {
        onEvent(name, data) {
          if (name === "log") {
            // epoch 指标行由 train.py 直接打印，日志里已经包含每轮的
            // epoch / train_loss / train_acc / val_loss / val_acc / lr，
            // 这里不再单独渲染指标卡片。
            const item = { level: data.level || "info", text: data.text || "" };
            (state.trainLog ||= []).push(item);
            appendLog(log, item);
          } else if (name === "done") {
            update({ running: false, lastTrain: data.result || null });
            appendLog(log, { level: "info", text: `训练结束：${JSON.stringify(data.result || {})}` });
            toast(`训练完成，val_acc=${data.result?.val_acc ?? "-"}`, "ok");
            render(document.getElementById("panel-3"));
          } else if (name === "error") {
            update({ running: false });
            appendLog(log, { level: "error", text: `${data.message}\n${data.detail || ""}` });
            toast(data.message || "训练失败", "err");
          }
        },
        onError(err) {
          update({ running: false });
          toast(err.message, "err");
        },
      });
    } catch (err) {
      update({ running: false });
      toast(err.message, "err");
    }
  };

  return section(
    "训练",
    h(
      "div",
      { class: "toolbar" },
      h("button", { class: "primary", text: "开始训练", onclick: start, disabled: state.running }),
      h("button", {
        text: "停止训练",
        class: "danger",
        disabled: !state.running,
        onclick: async () => {
          const { post } = await import("../api.js");
          try {
            await post("/api/train/stop");
            toast("已请求停止", "ok");
          } catch (err) {
            toast(err.message, "err");
          }
        },
      }),
      selectedNames().length ? badge(`数据集 ${selectedNames().join(" + ")}`) : null
    ),
    log,
    resultCard()
  );
}

// 训练结果卡各字段的中文悬停说明
const RESULT_TIPS = {
  checkpoint: "checkpoint：本次训练保存的模型权重文件路径（保存的是验证准确率最高的那一轮），供推理与验证使用。",
  best_epoch: "best_epoch：验证准确率最高的轮次编号，该轮的权重被写入 checkpoint。",
  val_acc: "val_acc：最优权重在验证集上的准确率。验证集窗口很少（如 3 个）时，该值参考意义有限。",
  feature_dim: "feature_dim：每个窗口每帧的特征维度（全局类别数 × 7）。",
  num_classes: "num_classes：训练使用的步骤（类别）数量，对应配置里 0..K-1 连续的步骤编号。",
  "train/val": "train/val：训练集与验证集的窗口数量（按时间顺序或多分组切分后的结果）。",
  split_mode: "split_mode：数据集切分方式。chronological = 单个分组按时间顺序切分；grouped = 多个数据集/会话各自切分。",
};

function resultCard() {
  const result = state.lastTrain;
  if (!result) return h("div");
  const items = [
    ["checkpoint", result.checkpoint, "full"],
    ["best_epoch", result.best_epoch],
    ["val_acc", result.val_acc],
    ["feature_dim", result.feature_dim],
    ["num_classes", result.num_classes],
    ["train/val", `${result.num_train ?? "-"} / ${result.num_val ?? "-"}`],
    ["split_mode", result.split_mode],
  ];
  return h(
    "div",
    { class: "card" },
    h("h2", { text: "训练结果" }),
    h(
      "div",
      { class: "metric-grid" },
      items.map(([key, value, extra]) =>
        h(
          "div",
          { class: `metric ${extra || ""}`.trim(), "data-tip": RESULT_TIPS[key] },
          h("div", { class: "k", text: key }),
          h("div", { class: "v mono", text: value === undefined || value === null ? "-" : String(value) })
        )
      )
    ),
    result.class_names?.length
      ? h("div", { class: "hint", text: `步骤：${result.class_names.join(" / ")}` })
      : null
  );
}

// train.py 每轮的指标行，例如：
// 2026-11-06 15:29:39 | INFO | __main__ | epoch 001 | train_loss=0.7227 acc=0.300 | ...
const EPOCH_LINE = /epoch\s+\d+\s*\|\s*train_loss=/i;
// tqdm 进度条行（\r 刷新会产生大量重复行，且带 ANSI 控制符）
const ANSI_RE = /\x1b\[[0-9;?]*[A-Za-z]/g;
const TQDM_LINE = /^\s*[\w@.\- ]*:?\s*\d+%\s*\|/;

// 把一行原始输出拆成要显示的日志条目：
// - 带 epoch 指标的行只保留指标段并高亮（可能被 tqdm 前缀拼在同一行）
// - 纯 tqdm 进度行丢弃，避免刷屏
function logEntries(text, level) {
  const clean = String(text ?? "").replace(ANSI_RE, "").trim();
  if (!clean) return [];
  const metric = EPOCH_LINE.exec(clean);
  if (metric) return [{ level, text: clean.slice(metric.index), epoch: true }];
  if (TQDM_LINE.test(clean)) return [];
  return [{ level, text: clean, epoch: false }];
}

function appendLog(node, item) {
  logEntries(item.text, item.level).forEach((entry) => {
    const cls = entry.level === "info" ? (entry.epoch ? "epoch-line" : "") : entry.level;
    node.append(h("div", { class: cls, text: entry.text }));
  });
  node.scrollTop = node.scrollHeight;
}
