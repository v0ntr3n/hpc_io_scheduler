"""HTTP advisor smoke tests (mock httpx, no real server needed)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from hpc_io_scheduler.config import Config
from hpc_io_scheduler.models.lora_advisor import QwenAdvisor, _map_action


def test_map_action_keywords():
    assert _map_action("HOLD") == "HOLD"
    assert _map_action("please throttle this") == "THROTTLE"
    assert _map_action("submit it") == "SUBMIT"
    assert _map_action("nonsense") is None


def test_advise_heuristic_fallback_when_unreachable():
    cfg = Config()
    cfg.llm.backend = "http"
    cfg.llm.api_base = "http://127.0.0.1:1"  # no server
    adv = QwenAdvisor(cfg)
    assert adv.ok is False
    assert adv.advise({"bw_bound": 1.0, "rpc_bound": 1.0,
                       "bw_hard": 1.0, "rpc_hard": 1.0, "priority": 50}) == (None, [])
    assert adv.advise_batch([{"bw_bound": 1.0, "rpc_bound": 1.0,
                              "bw_hard": 1.0, "rpc_hard": 1.0, "priority": 50}]) == [(None, [])]


def test_advise_calls_http_and_parses_action():
    cfg = Config()
    cfg.llm.backend = "http"
    cfg.llm.api_base = "http://fake"
    adv = QwenAdvisor.__new__(QwenAdvisor)
    adv.cfg = cfg
    adv.llm = cfg.llm
    adv._memo = {}
    adv.ok = True
    adv.mode = "http"

    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": json.dumps({
            "action": "THROTTLE",
            "feedback_loop": {"retrain_trigger": True, "hitl_escalation": False},
        })}}]
    }
    fake_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=fake_response) as post:
        action, codes = adv.advise({"bw_bound": 1.0, "rpc_bound": 1.0,
                                    "bw_hard": 2.0, "rpc_hard": 2.0, "priority": 50})
    assert action == "THROTTLE"
    assert "RETRAIN" in codes
    assert "HITL" not in codes
    post.assert_called_once()
    body = post.call_args.kwargs["json"]
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == cfg.llm.max_new_tokens
    assert body["model"] == cfg.llm.api_model


def test_memoization_dedupes_repeated_calls():
    cfg = Config()
    adv = QwenAdvisor.__new__(QwenAdvisor)
    adv.cfg = cfg
    adv.llm = cfg.llm
    adv._memo = {}
    adv.ok = True
    adv.mode = "http"

    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": '{"action": "SUBMIT"}'}}]
    }
    fake_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=fake_response) as post:
        ctx = {"bw_bound": 1.0, "rpc_bound": 2.0, "bw_hard": 5.0, "rpc_hard": 5.0, "priority": 50}
        adv.advise_batch([ctx, ctx, ctx])
    assert post.call_count == 1
