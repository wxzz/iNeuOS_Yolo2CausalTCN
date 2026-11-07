// 全局状态与最小订阅机制
export const state = {
  tab: 1,
  // 空字符串 = 使用后端记住的根目录（刷新后不丢失）
  root: "",
  datasets: [],
  selected: {},        // name -> { checked, weights }
  models: {},          // name -> inspect 结果
  global: { target_names: [], num_target_classes: 0, feature_dim: 0 },
  errors: [],
  warnings: [],
  validation: null,   // { ok, errors, warnings, datasetCount }
  config: null,
  steps: [],
  meta: null,
  lastBuild: null,
  lastTrain: null,
  checkpoint: "tcn_datasets/tcn_model.pth",
  detector: { mode: "yolo", conf: 0.25, iou: 0.45, imgsz: 640, splits: ["train"] },
  imageScan: null,
  imageFilter: "",
  scanning: false,
  scanAttempted: false,
  configLoaded: false,
  train: { epochs: 80, batch_size: 16, lr: 0.001, channels: "64 64 64 64", dropout: 0.2, device: "auto", from_npy: false },
  validate: { mode: "dir", source: "", files: [], smooth: 5, conf: 0.25, iou: 0.45, imgsz: 640 },
  predictions: [],
  validateLog: [],
  lastValidate: null,
  running: false,
  taskLabel: "",
  dirtyDataset: false,
  log: [],
};

const listeners = new Set();

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function update(patch) {
  Object.assign(state, patch);
  listeners.forEach((fn) => fn());
}

export function selectedNames() {
  return Object.entries(state.selected)
    .filter(([, item]) => item.checked)
    .map(([name]) => name);
}
