"""时序模型训练 —— 本地 Web 服务入口（文档 §16）。

启动：
    python web_server.py --port 5000
只监听 127.0.0.1，不对外网开放；``debug`` reloader 必须关闭，否则会重复启动子进程。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from flask import (
    Flask,
    Response,
    jsonify,
    request,
    send_file,
    send_from_directory,
    stream_with_context,
)

from config import PROJECT_ROOT
from detectors import list_image_files
from web.services import (
    build_service,
    config_store,
    dataset_service,
    paths,
    train_service,
    validate_service,
)
from web.services.errors import ApiError
from web.services.task_manager import MANAGER

WEB_DIR = PROJECT_ROOT / "web"
WEB_CSS_DIR = WEB_DIR / "css"
WEB_JS_DIR = WEB_DIR / "js"

# 图片扫描单次最多返回多少文件名：目录上万张时不必把整份列表传给页面
# （count 仍是真实总数，前端另有渲染上限）
IMAGE_SCAN_MAX_ITEMS = 1000

# 只暴露 index.html + css/js：Flask 默认静态目录会把 web/services/*.py 也
# 当作静态文件供下载（本机服务仍不应暴露后端源码）。
app = Flask(__name__, static_folder=None)
app.config["JSON_AS_ASCII"] = False


# --------------------------------------------------------------------------- #
# 通用响应
# --------------------------------------------------------------------------- #
def ok(data: Any) -> Response:
    return jsonify({"ok": True, "data": data})


@app.errorhandler(ApiError)
def handle_api_error(exc: ApiError) -> Response:
    response = jsonify({"ok": False, "error": exc.to_dict()})
    response.status_code = exc.status
    return response


@app.errorhandler(Exception)
def handle_unexpected(exc: Exception) -> Response:
    # 静态资源 404 等 HTTP 异常按原样返回，不要包装成 500
    from werkzeug.exceptions import HTTPException

    if isinstance(exc, HTTPException):
        return exc

    response = jsonify(
        {
            "ok": False,
            "error": {
                "code": "E_INTERNAL",
                "message": "服务内部错误",
                "detail": f"{type(exc).__name__}: {exc}",
            },
        }
    )
    response.status_code = 500
    return response


# --------------------------------------------------------------------------- #
# 静态页面
# --------------------------------------------------------------------------- #
@app.get("/")
def index() -> Response:
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/css/<path:filename>")
def web_css(filename: str) -> Response:
    return send_from_directory(WEB_CSS_DIR, filename)


@app.get("/js/<path:filename>")
def web_js(filename: str) -> Response:
    return send_from_directory(WEB_JS_DIR, filename)


@app.get("/api/health")
def health() -> Response:
    return ok({"python": sys.version.split()[0], "root": str(PROJECT_ROOT)})


# --------------------------------------------------------------------------- #
# 数据集与模型
# --------------------------------------------------------------------------- #
@app.get("/api/dataset/meta")
def dataset_meta() -> Response:
    """读取 dataset_meta.json（页面刷新后恢复训练/验证页的数据集摘要）。"""
    return ok(dataset_service.load_meta(request.args.get("tcn_dir")))


@app.get("/api/dataset/scan")
def dataset_scan() -> Response:
    reset = str(request.args.get("reset") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    return ok(dataset_service.scan(request.args.get("root"), reset=reset))


@app.post("/api/datasets/validate")
def datasets_validate() -> Response:
    payload = _json_body()
    return ok(dataset_service.validate(payload))


@app.post("/api/datasets/add")
def datasets_add() -> Response:
    """加入单个数据集目录；目录不在根目录下时先复制进根目录，再返回最新扫描。"""
    MANAGER.ensure_idle("增加数据集")
    payload = _json_body()
    raw = str(payload.get("path") or "").strip()
    if not raw:
        raise ApiError("E_DATASET_UNKNOWN", "请提供数据集目录 path")
    added = dataset_service.add_dataset(raw, root=payload.get("root"))
    data = dataset_service.scan(payload.get("root"))
    data["added"] = added
    return ok(data)


@app.post("/api/weights/import")
def weights_import() -> Response:
    """把选中的模型复制到数据集目录，并重命名为 <数据集目录名>.pt。"""
    MANAGER.ensure_idle("导入权重")
    payload = _json_body()
    dataset = str(payload.get("dataset") or "").strip()
    source = str(payload.get("source") or "").strip()
    if not dataset:
        raise ApiError("E_DATASET_UNKNOWN", "请提供数据集目录 dataset")
    if not source:
        raise ApiError("E_WEIGHTS_MISSING", "请提供模型文件 source")
    return ok(dataset_service.import_weights(dataset, source))


@app.post("/api/datasets/remove")
def datasets_remove() -> Response:
    """移除数据集记录；``purge=true`` 时连同数据集目录与权重一起删除。"""
    MANAGER.ensure_idle("删除数据集")
    payload = _json_body()
    paths = payload.get("paths") or []
    if not isinstance(paths, list) or not paths:
        raise ApiError("E_DATASET_UNKNOWN", "请提供要删除的数据集 paths")
    report = dataset_service.remove_datasets(
        paths,
        purge=bool(payload.get("purge")),
        root=payload.get("root"),
    )
    data = dataset_service.scan(payload.get("root"))
    data["purge"] = report
    return ok(data)


@app.get("/api/images/scan")
def images_scan() -> Response:
    raw = request.args.get("path")
    if not raw:
        raise ApiError("E_SOURCE_REQUIRED", "请提供 path 参数")
    directory = paths.read_only_path(raw, must_exist=True)
    folder = directory if directory.is_dir() else directory.parent
    all_files = [path.name for path in list_image_files(folder)]
    try:
        max_items = int(request.args.get("max_items") or IMAGE_SCAN_MAX_ITEMS)
    except ValueError:
        max_items = IMAGE_SCAN_MAX_ITEMS
    max_items = max(1, min(max_items, 5000))
    files = all_files[:max_items]
    limit = validate_service.IMAGE_LIMIT
    return ok(
        {
            "count": len(all_files),
            "files": files,
            "limit": limit,
            "exceeds": len(all_files) > limit,
            "truncated": len(files) < len(all_files),
            "max_items": max_items,
        }
    )


@app.get("/api/images/thumb")
def images_thumb() -> Response:
    """返回验证来源目录中的单张图片（Tab4 图片列表缩略图，只读）。"""
    path = validate_service.resolve_image(
        request.args.get("path") or "", request.args.get("name") or ""
    )
    return send_file(path, conditional=True, max_age=3600)


@app.get("/api/frames/list")
def frames_list() -> Response:
    """列出某数据集的帧 ID，供「帧列表」规则多选。"""
    return ok(
        dataset_service.list_frames(
            request.args.get("dataset") or "", request.args.get("root")
        )
    )


@app.get("/api/frames/image")
def frames_image() -> Response:
    """返回帧对应图片（弹窗缩略图）；只允许已知数据集内的图片。"""
    dataset = request.args.get("dataset") or ""
    frame = request.args.get("frame") or ""
    path = dataset_service.frame_image_path(dataset, frame, request.args.get("root"))
    return send_file(path, conditional=True, max_age=3600)


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@app.get("/api/config")
def config_get() -> Response:
    return ok(config_store.load_config(request.args.get("path")))


@app.put("/api/config")
def config_put() -> Response:
    MANAGER.ensure_idle("保存配置")
    payload = _json_body()
    raw = request.args.get("path")
    result = config_store.save_config(payload, raw)
    if not result["saved"]:
        response = jsonify(
            {
                "ok": False,
                "error": {
                    "code": "E_CONFIG_INVALID",
                    "message": "配置校验失败",
                    "detail": json.dumps(result["errors"], ensure_ascii=False),
                },
                "data": result,
            }
        )
        response.status_code = 400
        return response
    return ok(result)


# --------------------------------------------------------------------------- #
# 长任务（SSE）
# --------------------------------------------------------------------------- #
@app.post("/api/dataset/build")
def dataset_build() -> Response:
    payload = _json_body()
    argv = build_service.build_argv(payload)
    task = MANAGER.start(
        "build",
        argv,
        label="构建数据集",
        on_line=build_service.on_line,
        on_exit=build_service.on_exit,
        context={"_payload": payload},
    )
    return _sse(task)


@app.post("/api/train")
def train() -> Response:
    payload = _json_body()
    argv = train_service.build_argv(payload)
    task = MANAGER.start(
        "train",
        argv,
        label="训练模型",
        on_line=train_service.on_line,
        on_exit=train_service.on_exit,
        context={"_payload": payload},
    )
    return _sse(task)


@app.post("/api/validate")
def validate() -> Response:
    payload = _json_body()
    context = validate_service.prepare(payload)
    _assert_checkpoint_compatible(context.get("dataset"), payload)
    task = MANAGER.start(
        "validate",
        context["argv"],
        label="模型验证",
        on_line=validate_service.on_line,
        on_exit=validate_service.on_exit,
        context={"_context": context},
    )
    return _sse(task)


@app.post("/api/train/stop")
def task_stop() -> Response:
    return ok({"cancelled": MANAGER.stop()})


@app.get("/api/task/status")
def task_status() -> Response:
    return ok(MANAGER.status())


# --------------------------------------------------------------------------- #
# 文件选择
# --------------------------------------------------------------------------- #
@app.post("/api/fs/pick")
def fs_pick() -> Response:
    payload = _json_body()
    mode = str(payload.get("mode") or "dir")
    start = payload.get("start") or ""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if mode == "file":
            filters = [
                (item, f"*.{item}") for item in (payload.get("filters") or ["pt"])
            ]
            picked = filedialog.askopenfilename(
                initialdir=start or None, filetypes=filters
            )
        else:
            picked = filedialog.askdirectory(initialdir=start or None)
        root.destroy()
    except Exception as exc:  # noqa: BLE001
        raise ApiError(
            "E_PICKER_UNAVAILABLE",
            "当前环境不支持弹窗选择，请手动输入路径",
            status=501,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    if not picked:
        return ok({"path": "", "cancelled": True})
    return ok({"path": picked})


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _json_body() -> Dict[str, Any]:
    payload = request.get_json(silent=True)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ApiError("E_BODY_INVALID", "请求体必须是 JSON 对象")
    return payload


def _sse(task) -> Response:
    return Response(
        stream_with_context(MANAGER.stream(task)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _assert_checkpoint_compatible(dataset: str | None, payload: Dict[str, Any]) -> None:
    """Tab4 前置校验：特征维度与类别数都必须与当前配置一致。"""
    if not dataset:
        return
    checkpoint = paths.checkpoint_path(
        payload.get("checkpoint") or validate_service.DEFAULT_CHECKPOINT_PATH,
        must_exist=False,
    )
    if not checkpoint.exists():
        return
    info = validate_service.load_checkpoint_meta(checkpoint)

    cfg = config_store.load_config(payload.get("config"))
    per_class = dataset_service.per_class_components(cfg.get("dataset") or {})
    datasets = (cfg.get("source") or {}).get("datasets") or []
    if not datasets:
        return
    preview = dataset_service.global_preview(
        datasets, cfg.get("class_aliases") or {}, per_class
    )
    expected = int(preview["feature_dim"])
    if int(info["feature_dim"] or 0) not in (0, expected):
        raise ApiError(
            "E_FEATURE_DIM_MISMATCH",
            f"特征维度不匹配：权重 {info['feature_dim']} / 当前 {expected}",
            detail="数据集已变更，请重新构建并重训",
        )
    expected_classes = len(config_store.step_name_map(cfg))
    actual_classes = int(info["num_classes"] or 0)
    if expected_classes and actual_classes not in (0, expected_classes):
        raise ApiError(
            "E_CLASS_COUNT_MISMATCH",
            f"权重类别数 {actual_classes} / 当前步骤数 {expected_classes} 不一致",
            detail="步骤已增删，请重新构建并重训",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="时序模型训练 Web 服务")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"时序模型训练服务: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
