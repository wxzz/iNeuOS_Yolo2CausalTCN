"""统一错误码与异常类型（文档 §18）。"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ApiError(Exception):
    """带错误码的接口异常，由 web_server 统一转成 JSON 响应。"""

    def __init__(
        self,
        code: str,
        message: str,
        status: int = 400,
        detail: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.detail = detail

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.detail:
            payload["detail"] = self.detail
        return payload


def warn(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    """构造一条 warning 记录（不阻断流程）。"""
    item: Dict[str, Any] = {"code": code, "message": message}
    item.update(extra)
    return item


def err(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    """构造一条 error 记录（阻断流程，但不抛异常，用于批量校验）。"""
    item: Dict[str, Any] = {"code": code, "message": message}
    item.update(extra)
    return item
