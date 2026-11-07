"""验证来源缩略图接口的回归测试（Tab4 图片列表）。"""

from __future__ import annotations

from pathlib import Path

import pytest

import web_server
from web.services import validate_service
from web.services.errors import ApiError


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff\xe0jpeg-bytes")


def test_resolve_image_inside_source_dir(tmp_path: Path):
    image = tmp_path / "images" / "a.jpg"
    _write_image(image)

    assert validate_service.resolve_image(tmp_path / "images", "a.jpg") == image.resolve()


def test_resolve_image_rejects_traversal(tmp_path: Path):
    _write_image(tmp_path / "images" / "a.jpg")
    _write_image(tmp_path / "secret.jpg")

    with pytest.raises(ApiError):
        validate_service.resolve_image(tmp_path / "images", "../secret.jpg")


def test_resolve_image_rejects_non_image(tmp_path: Path):
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ApiError) as exc:
        validate_service.resolve_image(tmp_path, "note.txt")
    assert exc.value.status == 404


def test_thumb_endpoint_serves_image(tmp_path: Path):
    image = tmp_path / "images" / "a.jpg"
    _write_image(image)

    client = web_server.app.test_client()
    response = client.get("/api/images/thumb", query_string={"path": str(tmp_path / "images"), "name": "a.jpg"})

    assert response.status_code == 200
    assert response.data == image.read_bytes()
    response.close()


def test_thumb_endpoint_rejects_missing_param(tmp_path: Path):
    client = web_server.app.test_client()
    response = client.get("/api/images/thumb", query_string={"path": str(tmp_path)})

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "E_SOURCE_REQUIRED"
