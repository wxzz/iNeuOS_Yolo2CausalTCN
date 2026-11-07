"""全局任务管理器：单任务锁 + 子进程 + SSE 广播（文档 §16.5 / §16.6）。

要点：
- 同一时刻只允许一个 build / train / validate；
- 子进程用**参数数组**启动，``shell=False``，注入 ``PYTHONUTF8=1``；
- stdout/stderr 合并读取；text 模式会把 tqdm 的 ``\\r`` 也当作换行；
- 每个任务保留最近 10000 条事件，供刷新后的页面回放。
"""

from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Sequence

from .errors import ApiError
from .paths import project_root

# 训练 80 epoch 约几百行；10000 条能覆盖长训练 + 逐帧构建日志，内存占用可忽
MAX_EVENTS = 10000
_LEVEL_RE = re.compile(
    r"(?:^|\|\s*)(?P<level>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\s*(?:\||$)"
)


@dataclass
class Task:
    """一次长任务。"""

    kind: str
    argv: List[str]
    label: str = ""
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: str = "running"  # running | done | error | cancelled
    result: Dict[str, Any] = field(default_factory=dict)
    events: Deque[tuple] = field(default_factory=lambda: deque(maxlen=MAX_EVENTS))
    subscribers: List[queue.Queue] = field(default_factory=list)
    scratch: Dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    proc: Optional[subprocess.Popen] = None
    stop_requested: bool = False

    @property
    def running(self) -> bool:
        return self.state == "running"


def _format(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


IS_WINDOWS = os.name == "nt"


def _spawn_kwargs() -> Dict[str, Any]:
    """让子进程自成一个进程组，停止时才能把整棵树一起终止。"""
    if IS_WINDOWS:
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def terminate_process_tree(proc: subprocess.Popen, *, force: bool = False) -> bool:
    """终止子进程**及其派生进程**（dataloader worker / ultralytics 内部进程）。

    只按 **pid 树** 操作：Windows 用 ``taskkill /T /PID``，POSIX 用 ``killpg``。
    严禁 ``taskkill /IM python.exe`` 这类按镜像名的全局杀（会误杀 web_server 自身）。
    """
    if proc.poll() is not None:
        return True
    pid = int(proc.pid)
    if IS_WINDOWS:
        try:
            # /T = 终止整棵进程树；/F = 强制（Windows 上没有可用的温和停止）
            result = subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return True
        except OSError:
            pass
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL if force else signal.SIGTERM)
            return True
        except (OSError, ProcessLookupError):
            pass
    try:
        if force:
            proc.kill()
        else:
            proc.terminate()
        return True
    except OSError:
        return False


def _level_of(text: str) -> str:
    """识别日志级别：兼容 ``LEVEL | msg``、logging 时间戳格式与 traceback。

    子进程日志形如 ``2026-11-06 16:23:00,031 | INFO    | __main__ | ...``，
    旧实现只看行首，导致 ERROR/WARNING 全部按 info 展示。
    """
    head = text.lstrip()[:200]
    upper = head.upper()
    if upper.startswith("TRACEBACK"):
        return "error"
    if upper.startswith(("ERROR", "CRITICAL")):
        return "error"
    if upper.startswith(("WARNING", "WARN")):
        return "warn"
    match = _LEVEL_RE.search(upper)
    if match:
        level = match.group("level")
        if level in ("ERROR", "CRITICAL"):
            return "error"
        if level in ("WARNING", "WARN"):
            return "warn"
    return "info"


class TaskManager:
    """进程内单例（Flask 单进程）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._task: Optional[Task] = None

    # -- 查询 ---------------------------------------------------------------
    @property
    def current(self) -> Optional[Task]:
        return self._task

    def status(self) -> Dict[str, Any]:
        task = self._task
        if task is None:
            return {"running": False, "task": None}
        return {
            "running": task.running,
            "task": {
                "id": task.task_id,
                "kind": task.kind,
                "label": task.label,
                "state": task.state,
            },
        }

    # -- 启动 / 停止 --------------------------------------------------------
    def ensure_idle(self, action: str = "执行该操作") -> None:
        """长任务运行期间禁止修改配置/数据集，避免产物与配置版本错位。"""
        task = self._task
        if task is None or not task.running:
            return
        raise ApiError(
            "E_TASK_BUSY",
            f"任务运行中（{task.label or task.kind}），无法{action}",
            status=409,
            detail=f"当前任务：{task.kind}（{task.task_id}）",
        )

    def start(
        self,
        kind: str,
        argv: Sequence[str],
        label: str = "",
        on_line: Optional[Callable[[Task, str], None]] = None,
        on_exit: Optional[Callable[[Task, int], Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Task:
        with self._lock:
            if self._task is not None and self._task.running:
                raise ApiError(
                    "E_TASK_BUSY",
                    "已有任务在运行，请先停止",
                    status=409,
                    detail=f"当前任务：{self._task.kind}（{self._task.task_id}）",
                )

            task = Task(kind=kind, argv=[str(item) for item in argv], label=label)
            if context:
                task.scratch.update(context)
            env = dict(os.environ)
            env.update(
                {
                    "PYTHONUTF8": "1",
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            try:
                task.proc = subprocess.Popen(
                    task.argv,
                    cwd=str(project_root()),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=env,
                    **_spawn_kwargs(),
                )
            except OSError as exc:
                raise ApiError(
                    "E_SPAWN_FAILED",
                    f"子进程启动失败：{exc}",
                    detail=" ".join(task.argv),
                ) from exc

            self._task = task
            self._emit(
                task, "log", {"level": "info", "text": f"$ {' '.join(task.argv)}"}
            )

        threading.Thread(
            target=self._reader, args=(task, on_line, on_exit), daemon=True
        ).start()
        return task

    def stop(self) -> bool:
        task = self._task
        if task is None or not task.running or task.proc is None:
            return False
        task.stop_requested = True
        self._emit(
            task,
            "log",
            {"level": "warn", "text": "收到停止请求，正在终止子进程树…"},
        )
        terminate_process_tree(task.proc, force=False)
        threading.Thread(target=self._force_kill, args=(task,), daemon=True).start()
        return True

    def _force_kill(self, task: Task) -> None:
        proc = task.proc
        if proc is None:
            return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            terminate_process_tree(proc, force=True)
            self._emit(
                task,
                "log",
                {"level": "warn", "text": f"已强制终止进程树 pid={proc.pid}"},
            )
        else:
            self._emit(
                task,
                "log",
                {"level": "warn", "text": f"进程已退出 pid={proc.pid}"},
            )

    # -- 读取子进程输出 ------------------------------------------------------
    def _reader(
        self,
        task: Task,
        on_line: Optional[Callable[[Task, str], None]],
        on_exit: Optional[Callable[[Task, int], Dict[str, Any]]],
    ) -> None:
        proc = task.proc
        assert proc is not None and proc.stdout is not None

        for raw in proc.stdout:
            text = raw.rstrip("\n")
            if not text.strip():
                continue
            self._emit(task, "log", {"level": _level_of(text), "text": text})
            if on_line is not None:
                try:
                    on_line(task, text)
                except Exception:  # noqa: BLE001 - 解析失败绝不能中断任务
                    pass

        code = proc.wait()
        result: Dict[str, Any] = {}
        if on_exit is not None:
            try:
                result = on_exit(task, code) or {}
            except Exception as exc:  # noqa: BLE001
                result = {"finalize_error": f"{type(exc).__name__}: {exc}"}
        task.result = result

        if task.stop_requested:
            task.state = "cancelled"
        elif code == 0:
            task.state = "done"
        else:
            task.state = "error"

        if task.state == "done":
            self._emit(task, "done", {"ok": True, "result": result})
        elif task.state == "cancelled":
            self._emit(task, "done", {"ok": False, "cancelled": True, "result": result})
        else:
            tail = "\n".join(
                text
                for _, data in list(task.events)[-20:]
                for text in [data.get("text", "")]
            )
            self._emit(
                task,
                "error",
                {
                    "code": "E_TASK_FAILED",
                    "message": f"{task.kind} 失败（退出码 {code}）",
                    "detail": tail,
                    "result": result,
                },
            )

    # -- 事件与 SSE --------------------------------------------------------
    def emit(self, task: Task, event: str, data: Dict[str, Any]) -> None:
        """公开的事件推送（供各 service 上报 progress / metric）。"""
        self._emit(task, event, data)

    def _emit(self, task: Task, event: str, data: Dict[str, Any]) -> None:
        with task.lock:
            task.events.append((event, data))
            payload = _format(event, data)
            for subscriber in list(task.subscribers):
                subscriber.put(payload)
            if event in ("done", "error"):
                for subscriber in list(task.subscribers):
                    subscriber.put(None)
                task.subscribers.clear()

    def stream(self, task: Task) -> Iterator[str]:
        """SSE 生成器：先回放已产生的事件，再持续推送新事件。"""
        subscriber: "queue.Queue[Optional[str]]" = queue.Queue()
        with task.lock:
            backlog = [_format(event, data) for event, data in task.events]
            finished = not task.running
            if not finished:
                task.subscribers.append(subscriber)

        for payload in backlog:
            yield payload
        if finished:
            return

        try:
            while True:
                try:
                    item = subscriber.get(timeout=15)
                except queue.Empty:
                    yield ": ping\n\n"
                    continue
                if item is None:
                    break
                yield item
        finally:
            with task.lock:
                if subscriber in task.subscribers:
                    task.subscribers.remove(subscriber)


MANAGER = TaskManager()
