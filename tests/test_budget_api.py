"""POST /redact/budget 端到端测试。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def post(payload: object) -> object:
    return client.post("/redact/budget", json=payload)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 正常流程
# ---------------------------------------------------------------------------


def test_budget_end_to_end() -> None:
    resp = post(
        {
            "text": "你好，世界！",
            "fragments": [
                {"id": 1, "weight": 5, "start": 0, "end": 3},
                {"id": 2, "weight": 3, "start": 0, "end": 2},
                {"id": 3, "weight": 1, "start": 2, "end": 5},
            ],
            "max_masked_chars": 3,
        }
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # 预算 3：{1}（权重 5，用 3 码点）与 {2,3}（权重 4，用 3 码点）均可，
    # 权重总和最大者入选。
    assert data["selected_ids"] == [1]
    assert data["rejected_ids"] == [2, 3]
    assert data["masked_text"] == "███世界！"
    assert data["used_chars"] == 3


def test_budget_tight_picks_smaller_fragment() -> None:
    resp = post(
        {
            "text": "你好，世界！",
            "fragments": [
                {"id": 1, "weight": 5, "start": 0, "end": 3},
                {"id": 2, "weight": 3, "start": 0, "end": 2},
                {"id": 3, "weight": 1, "start": 2, "end": 5},
            ],
            "max_masked_chars": 2,
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    # 只有片段 2（长 2）符合预算。
    assert data["selected_ids"] == [2]
    assert data["rejected_ids"] == [1, 3]
    assert data["masked_text"] == "██，世界！"
    assert data["used_chars"] == 2


def test_budget_zero_returns_unmasked_text() -> None:
    resp = post(
        {
            "text": "abc",
            "fragments": [{"id": 1, "weight": 1, "start": 0, "end": 1}],
            "max_masked_chars": 0,
        }
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "selected_ids": [],
        "rejected_ids": [1],
        "masked_text": "abc",
        "used_chars": 0,
    }


def test_budget_zero_with_empty_fragments() -> None:
    resp = post({"text": "abc", "fragments": [], "max_masked_chars": 0})
    assert resp.status_code == 200
    assert resp.json() == {
        "selected_ids": [],
        "rejected_ids": [],
        "masked_text": "abc",
        "used_chars": 0,
    }


def test_budget_adjacent_fragments_both_selected() -> None:
    resp = post(
        {
            "text": "abcdef",
            "fragments": [
                {"id": 2, "weight": 1, "start": 3, "end": 6},
                {"id": 1, "weight": 1, "start": 0, "end": 3},
            ],
            "max_masked_chars": 6,
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["selected_ids"] == [1, 2]
    assert data["rejected_ids"] == []
    assert data["masked_text"] == "██████"
    assert data["used_chars"] == 6


def test_budget_weight_tie_prefers_more_chars() -> None:
    resp = post(
        {
            "text": "abcdef",
            "fragments": [
                {"id": 1, "weight": 2, "start": 0, "end": 1},
                {"id": 2, "weight": 2, "start": 1, "end": 3},
            ],
            "max_masked_chars": 2,
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    # 预算 2 下只能二选一；权重并列，取遮盖码点更多的片段 2。
    assert data["selected_ids"] == [2]
    assert data["used_chars"] == 2
    assert data["masked_text"] == "a██def"


def test_budget_astral_code_points_counted_per_code_point() -> None:
    resp = post(
        {
            "text": "a😀b",  # 3 个码点
            "fragments": [{"id": 1, "weight": 1, "start": 1, "end": 2}],
            "max_masked_chars": 1,
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["selected_ids"] == [1]
    assert data["masked_text"] == "a█b"
    assert data["used_chars"] == 1


def test_budget_larger_than_text_works_like_redact() -> None:
    payload = {
        "text": "你好，世界！",
        "fragments": [
            {"id": 1, "weight": 5, "start": 0, "end": 3},
            {"id": 2, "weight": 3, "start": 0, "end": 2},
            {"id": 3, "weight": 1, "start": 2, "end": 5},
        ],
    }
    redact = client.post("/redact", json=payload).json()
    resp = post({**payload, "max_masked_chars": 10**6})
    assert resp.status_code == 200
    data = resp.json()
    assert data["selected_ids"] == redact["selected_ids"]
    assert data["rejected_ids"] == redact["rejected_ids"]
    assert data["masked_text"] == redact["masked_text"]
    assert data["used_chars"] == 3


def test_budget_extra_fields_are_ignored() -> None:
    resp = post(
        {
            "text": "abc",
            "fragments": [{"id": 1, "weight": 1, "start": 0, "end": 1, "note": "x"}],
            "max_masked_chars": 1,
            "request_id": "r-1",
        }
    )
    assert resp.status_code == 200
    assert resp.json()["selected_ids"] == [1]


def test_openapi_lists_budget_path() -> None:
    schema = client.get("/openapi.json").json()
    assert "/redact/budget" in schema["paths"]
    assert "/redact" in schema["paths"]
    assert "/redact/revise" in schema["paths"]


# ---------------------------------------------------------------------------
# 拒绝类输入：全部 400
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "budget",
    [
        -1,
        -100,
        1.0,
        0.5,
        True,
        "1",
        None,
        [1],
    ],
)
def test_invalid_budget_rejected_with_400(budget: object) -> None:
    resp = post(
        {
            "text": "abc",
            "fragments": [{"id": 1, "weight": 1, "start": 0, "end": 1}],
            "max_masked_chars": budget,
        }
    )
    assert resp.status_code == 400, budget
    assert "detail" in resp.json()


def test_missing_budget_field_rejected_with_400() -> None:
    resp = post(
        {"text": "abc", "fragments": [{"id": 1, "weight": 1, "start": 0, "end": 1}]}
    )
    assert resp.status_code == 400
    assert "max_masked_chars" in resp.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        # 缺字段
        {"fragments": [], "max_masked_chars": 0},
        {"text": "abc", "max_masked_chars": 0},
        # text / fragments 校验沿用 /redact
        {"text": 123, "fragments": [], "max_masked_chars": 0},
        {"text": "abc", "fragments": {}, "max_masked_chars": 0},
        # 片段字段与语义错误
        {"text": "abc", "fragments": [{"id": 1, "weight": 0, "start": 0, "end": 1}], "max_masked_chars": 1},
        {"text": "abc", "fragments": [{"id": 1, "weight": 1, "start": 0, "end": 4}], "max_masked_chars": 1},
        {"text": "abc", "fragments": [{"id": 1, "weight": 1, "start": 1, "end": 1}], "max_masked_chars": 1},
        {"text": "abc", "fragments": [
            {"id": 1, "weight": 1, "start": 0, "end": 1},
            {"id": 1, "weight": 1, "start": 1, "end": 2},
        ], "max_masked_chars": 2},
        # 请求体不是对象
        [1, 2, 3],
    ],
)
def test_invalid_text_or_fragments_rejected_with_400(payload: object) -> None:
    resp = post(payload)
    assert resp.status_code == 400, payload
    assert "detail" in resp.json()


def test_budget_invalid_json_body_rejected_with_400() -> None:
    resp = client.post(
        "/redact/budget", content="{not json", headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 400


def test_budget_empty_body_rejected_with_400() -> None:
    resp = client.post(
        "/redact/budget", content=b"", headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 400
