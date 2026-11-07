---
name: tcn-dev
description: '**WORKFLOW SKILL** — 在本仓库开发/改造 YOLO→TCN 时序模型流水线与配套前端。USE FOR: 修改 build_dataset.py / train.py / inference.py / data_processor.py / detectors.py / config.py / model.py；新增或调整 YOLO 类别映射、步骤匹配规则、窗口与特征参数、训练超参、推理模式；开发“时序模型训练”页面（HTML/CSS/JS 分离）；排查 dataset_meta、特征维度、类别槽位、标签对齐、训练/推理不一致问题。DO NOT USE FOR: 纯概念问答（直接回答即可）；训练 YOLO 模型本身；与 TCN 无关的通用编程问题。'
argument-hint: '说明要开发或修改的功能，例如：给 build_dataset 增加 class 匹配规则 / 实现 Tab2 页面'
---

# TCN 开发（tcn_dev）

面向本仓库（YOLO 数据集 → 因果 TCN）的开发规范。目标是：**逻辑清晰、复用现有实现、安全稳定、性能可控、不遗留问题、不打补丁式修改、歧义交给人决定**。

## 何时使用

- 修改 `build_dataset.py` / `train.py` / `inference.py` / `data_processor.py` / `detectors.py` / `config.py` / `model.py`
- 新增类别映射、步骤匹配规则、窗口/特征参数、训练超参、推理来源
- 开发“时序模型训练”Web 页面（4 个 Tab）
- 排查：`skipped_lines > 0`、特征维度不匹配、某步骤消失、训练/推理结果不一致

## 第一原则：先读代码，再写代码

动手前必须完成：

1. 用搜索/阅读确认**是否已有实现**（下表）；
2. 若已有，优先**扩展参数或新增函数**，不复制粘贴；
3. 若必须改接口，先找出所有调用点；
4. 不确定数据语义时，停下来问（见「歧义处理」）。

## 仓库地图（复用清单）

| 模块 | 可直接复用的能力 | 禁止重写 |
|---|---|---|
| `config.py` | `DEFAULT_YOLO_ROOT`（多数据集根目录，每个子目录 = 一个数据集）/ `DEFAULT_TCN_DIR` / `DEFAULT_CHECKPOINT` / `DEFAULT_DATASET_CONFIG`；`DataConfig` / `TCNConfig` / `TrainConfig` | 路径常量、超参默认值 |
| `data_processor.py` | `YoloParserFactory`、`Detection`、`aggregate_detections`、`FrameFeatureAssembler`、`interpolate_missing`、`make_causal_windows`、`chronological_split_indices`、`grouped_split_indices`、`SequenceDatasetBuilder`、`export_dataset` | 解析各格式、聚合、插值、滑窗 |
| `detectors.py` | `load_yolo`、`detect_with_yolo`、`remap_class_ids`、`format_detection_line`、`IMAGE_SUFFIXES` | **YOLO → Detection 的唯一实现**，任何地方都不得再写一份 |
| `build_dataset.py` | `read_yolo_meta`、`discover_datasets`、`validate_datasets`、`build_global_classes`、`sync_dataset_labels`、`write_predictions_for_dataset`、`write_labels_for_dataset`、`generate_default_steps`、`merge_small_groups`、`match_step`、`summarize_match`、`warn_single_step_datasets`、`write_labels_csv`、`build_data_config` | 建库流程、全局类别表与步骤匹配 |
| `train.py` | `load_dataset_config`、`resolve_step_names`、`ArrayDataset`、`fit_normalization`、`save_checkpoint` | 切分、归一化、保存权重 |
| `inference.py` | `RealtimeActionRecognizer`、`list_frames`、`load_dataset_registry`、`resolve_dataset`、`build_frames_from_images`、`build_frames_from_labels`、`run_stream`、`collect_image_context`、`predict_npy` | 流式推理、单图预热、数据集判定 |
| `model.py` | `CausalTCNClassifier` | 因果卷积结构 |

## 开发流程

1. **定位**：确认改动落在哪个模块，列出受影响的调用点；
2. **设计**：写清输入/输出/失败路径，明确是否改变 `tcn_config.json` 或 checkpoint 字段；
3. **实现**：函数职责单一、参数显式、类型标注、docstring 说明「为什么」；
4. **自测**：跑下面的冒烟命令；
5. **回归**：`python -m pytest tests -q` 必须全绿；
6. **同步文档**：改了 CLI/配置结构就更新 `README.md` 与 `docs/`；
7. **总结**：说明改了什么、影响哪些产物、需要重跑哪几步。

## 安全与稳定

- **不删除/覆盖用户数据**：`yolo_datasets/` 只读；重建 `tcn_datasets` 会覆盖 `labels/`、`labels.csv`、`*.npy`、`tcn_model.pth`，执行前必须确认，或在临时目录先验证。
- **不改 checkpoint 字段名**（`window_size` / `model_config` / `data_config` / `normalization` / `class_names`），否则旧模型无法加载。
- 改特征布局（`num_target_classes` / `include_size` / `include_velocity`）会让旧 `.pth` 失效，必须显式提示“需重训”。
- **不硬编码绝对路径**，统一从 `config.py` 取。
- Windows + 中文路径：文件读写显式 `encoding`，子进程设 `PYTHONUTF8=1`。
- 启动子进程用**参数数组**，禁止拼接 shell 字符串。
- 异常必须带原始错误摘要，不能只显示“失败”。
- 失败时保持现场：不要先清空输出目录再执行可能失败的步骤。

## 性能

- 逐帧处理必须向量化（`sliding_window_view`、`np.add.at`、`bincount`），不允许 Python 双重循环。
- YOLO 推理复用已加载的 `model` 实例；批量时优先一次 `predict(paths)`，注意显存/内存上限。
- 流式推理只保留 `deque(maxlen=window_size)`，不要每帧重算整个窗口。
- 模型/权重**懒加载**，不要放在模块 import 时。
- 日志按行推送，不要每个 token 发一次 HTTP 请求。

## 歧义处理（人工决定）

**遇到以下问题必须停下来问，不得自行假设**：

| 歧义点 | 为什么不能猜 |
|---|---|
| 步骤（工序）的语义与划分 | 直接决定监督信号，猜错等于模型学错 |
| 类别 → 步骤的映射方式 | 同义/异义/粒度不同，无法从数据推断 |
| 是否覆盖已有 `tcn_datasets` / `.pth` | 可能造成不可恢复的数据丢失 |
| 验证集划分口径 | 影响指标是否可信 |
| 多数据集合并策略 | 涉及全局类别表与伪相关风险 |
| 前端交互形态（目录上限、预热口径等） | 影响验收标准 |

提问要求：给出 **2–3 个具体选项 + 推荐项 + 后果**，不要开放式提问。未确认前用最小默认值，并在代码/回复中标注 `TODO(待确认)`。

## 已确认的技术决策

| 项 | 决策 |
|---|---|
| 技能名称 | `tcn-dev`（平台要求小写字母/数字/连字符） |
| 后端 | **Flask**（单进程本地服务） |
| 前端 | **原生 HTML / CSS / JS，分文件，无框架、无构建步骤** |
| 前端页面 | 4 个 Tab：源数据集 / 数据映射 / 模型训练 / 模型验证 |

## 多数据集约定（强制）

| 规则 | 说明 |
|---|---|
| 数据集 = 目录 | `yolo_datasets/<name>/` 即一个数据集，`name` 与目录名一致 |
| 权重约定 | `yolo_datasets/<name>/<name>.pt`；`--yolo-weights` 仅作临时覆盖 |
| 全局类别表 | 按**类别名并集**排序，`class_map[i]` = 该数据集第 i 类在全局表中的槽位 |
| 特征维度 | `num_target_classes × (include_size?2:0 + include_velocity?2:0 + 1)` = 7/类（默认） |
| 帧名前缀 | 建库写盘时统一 `<name>__<stem>`，`session_pattern` 默认 `^.+?__` |
| 步骤作用域 | `steps[].match.dataset` 限定数据集，自动生成的步骤一定带此字段 |
| 切分 | 多分组时走 `grouped_split_indices`（每组各自时序切分），不得退回全局切分 |
| 兼容性 | **不需要兼容旧版单数据集目录结构**；`tcn_config.json` 版本号非 2 时自动重建 |


## 后端规则（Flask）

- 入口建议 `web_server.py`，只监听 `127.0.0.1`，不对外网开放；关闭 `debug` reloader（否则会启动两次、端口冲突）。
- 长任务用 SSE：`Response(stream_with_context(gen()), mimetype="text/event-stream")`，事件类型 `log / metric / done / error`。
- 执行脚本必须用 `subprocess.Popen([sys.executable, "train.py", "--epochs", "60", ...], cwd=项目根, env=含 PYTHONUTF8=1)`，**禁止 shell=True**。
- 读取输出时同时处理 `\r`（tqdm）与 `\n`，逐行解析；stderr 也要转发。
- **全局任务锁**：同一时刻只允许一个构建/训练/验证任务；提供 `/api/train/stop` 终止子进程。
- 路径参数必须 `resolve()` 后区分：产物类路径（`tcn_dir` / `tcn_config.json`）限制在工作区内；用户指定的数据集根、YOLO 权重、checkpoint 允许工作区外（`web/services/paths.py` 的 `user_path` / `checkpoint_path`），但仍需校验后缀与存在性。
- 接口建议：`/api/dataset/scan`、`/api/dataset/meta`、`/api/datasets/validate`、`/api/config`、`/api/dataset/build`、`/api/train`、`/api/train/stop`、`/api/validate`。

## 前端规则（强制）

**HTML / CSS / JavaScript 必须分文件，禁止内联。**

```
web/
├─ index.html
├─ css/style.css
└─ js/
   ├─ app.js        # 入口与状态
   ├─ api.js        # 后端请求封装
   └─ tabs/*.js     # 各 Tab 逻辑（可选）
```

- 不引入框架（除非用户明确要求）；使用原生 ES Module。
- 状态集中管理：一个 `state` 对象 + `render()`，避免分散的 DOM 读写。- 用户数据一律用 `textContent` 渲染，**禁止用 `innerHTML` 拼接**（防 XSS）。
- 长任务用 SSE/轮询，必须提供“停止”与失败提示。
- 前端只传路径字符串，路径解析与校验由后端负责。
- 界面文案中文，与 `docs/yolo数据集转换成tcn.md` 保持一致。
- 页面包含 4 个 Tab：源数据集 / 数据映射 / 模型训练 / 模型验证。

## 验证清单（每次改完都要跑）

```powershell
$env:PYTHONUTF8='1'

# 1) 单测
python -m pytest tests -q

# 2) 建库冒烟（输出到临时目录，不动正式数据）
python build_dataset.py --tcn-dir tcn_datasets_smoke --config-only

# 3) 训练冒烟
python train.py --epochs 2 --batch-size 4 `
  --artifact-dir tcn_datasets_smoke `
  --checkpoint tcn_datasets_smoke/tcn_model.pth

# 4) 推理冒烟
python inference.py --checkpoint tcn_datasets_smoke/tcn_model.pth `
  --source tcn_datasets_smoke/labels
```

必须确认：

| 检查项 | 期望 |
|---|---|
| `pytest` | 全部通过 |
| `dataset_meta.json` 的 `skipped_lines` | 0 |
| `feature_dim` | `num_target_classes × 7` |
| `label_distribution` | 每个步骤都有窗口 |
| 推理输出条数 | `窗口数`（目录回放）或 `图片数 − (window_size−1)`（图像目录） |
| 预测标签 token 数 | detect=5 / pose=`1+4+K×D` |

## 完成定义

- 逻辑：函数职责单一、无隐式全局状态、失败路径明确
- 复用：没有复制已有实现（尤其 `detectors.py`）
- 安全：未覆盖用户数据；异常信息可定位
- 性能：无逐帧 Python 循环；无重复模型加载
- 验证：单测 + 冒烟全过
- 文档：CLI/配置变更同步到 `README.md` / `docs/`
- 歧义：所有未决项已列出并交给用户确认，而非默默假设
