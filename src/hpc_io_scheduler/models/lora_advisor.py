"""LLM advisor (Layer 3) for gray-zone decisions.

Calls an external llama.cpp server (OpenAI-compatible HTTP API) — no
in-process model load. Memoizes by context hash to dedupe identical calls.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from hpc_io_scheduler.config import Config, LLMConfig


HEURISTIC_REASONS = ("BG_HIGH_BW", "BG_HIGH_RPC", "NEAR_HARD", "HIGH_PRIO", "LOW_PRIO")


def heuristic_advise(ctx: dict[str, Any]) -> tuple[str, list[str]]:
    bw = float(ctx["bw_bound"]) / max(float(ctx["bw_hard"]), 1e-9)
    rpc = float(ctx["rpc_bound"]) / max(float(ctx["rpc_hard"]), 1e-9)
    frac = max(bw, rpc)
    codes: list[str] = []
    if bw > 0.85:
        codes.append("BG_HIGH_BW")
    if rpc > 0.85:
        codes.append("BG_HIGH_RPC")
    if frac > 0.95:
        codes.append("NEAR_HARD")
    codes.append("HIGH_PRIO" if ctx.get("priority", 100) >= 90 else "LOW_PRIO")
    if frac > 0.98:
        return "HOLD", codes
    return "THROTTLE", codes


def _map_action(raw: Any) -> str | None:
    a = str(raw).upper()
    if "HOLD" in a:
        return "HOLD"
    if "THROTTLE" in a:
        return "THROTTLE"
    if "SUBMIT" in a:
        return "SUBMIT"
    return None


def _build_user_msg(ctx: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_context": {
            "job_id": str(ctx.get("job_id", "NA")),
            "task_type": str(ctx.get("task_type", "unlabeled")),
            "cpus_req": int(ctx.get("cpus_req", 0)),
        },
        "status_packet": {
            "sys_status": str(ctx.get("sys_status", "GRAY")),
            "predicted_bw_bound": round(float(ctx["bw_bound"]), 1),
            "predicted_rpc_bound": round(float(ctx["rpc_bound"]), 1),
        },
        "telemetry_packet": {
            "bw_mu_actual": round(float(ctx.get("bw_mu_actual", ctx["bw_bound"])), 1),
            "rpc_mu_actual": round(float(ctx.get("rpc_mu_actual", ctx["rpc_bound"])), 1),
        },
    }


class QwenAdvisor:
    def __init__(self, cfg: Config, llm_cfg: LLMConfig | None = None):
        self.cfg = cfg
        self.llm = llm_cfg or cfg.llm
        self.ok = False
        self.mode = "heuristic"
        self._memo: dict[tuple, tuple[str | None, list[str]]] = {}
        if not self.llm.use_llm:
            return
        if self.llm.backend != "http":
            print(f"[warn] backend={self.llm.backend} not supported; use backend=http")
            return
        self._check_server()

    def _check_server(self) -> None:
        try:
            import httpx

            r = httpx.get(f"{self.llm.api_base}/health", timeout=5.0)
            r.raise_for_status()
            self.ok = True
            self.mode = f"http ({self.llm.api_base})"
        except Exception as e:
            print(f"[warn] llama-server unreachable at {self.llm.api_base}: {e}")

    def _http_chat(self, ctx: dict[str, Any]) -> str:
        import httpx

        msgs = [
            {"role": "system", "content": self.llm.sys_prompt},
            {"role": "user", "content": json.dumps(_build_user_msg(ctx))},
        ]
        body = {
            "model": self.llm.api_model,
            "messages": msgs,
            "max_tokens": self.llm.max_new_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.llm.api_key:
            headers["Authorization"] = f"Bearer {self.llm.api_key}"
        r = httpx.post(
            f"{self.llm.api_base}/v1/chat/completions",
            json=body, headers=headers, timeout=self.llm.api_timeout_sec,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _ctx_key(ctx: dict[str, Any]) -> tuple:
        return (
            str(ctx.get("task_type", "NA")),
            int(ctx.get("cpus_req", 0)),
            round(float(ctx["bw_bound"]), 1),
            round(float(ctx["rpc_bound"]), 1),
            round(float(ctx.get("priority", 100)), 0),
        )

    def advise(self, ctx: dict[str, Any]) -> tuple[str | None, list[str]]:
        return self.advise_batch([ctx])[0]

    def advise_batch(
        self, contexts: list[dict[str, Any]], batch_size: int | None = None
    ) -> list[tuple[str | None, list[str]]]:
        if not self.ok or not contexts:
            return [(None, [])] * len(contexts)
        keys = [self._ctx_key(c) for c in contexts]
        uniq_idx: dict[tuple, int] = {}
        todo_ctx: list[dict] = []
        todo_keys: list[tuple] = []
        for k, c in zip(keys, contexts):
            if k not in self._memo and k not in uniq_idx:
                uniq_idx[k] = len(todo_ctx)
                todo_ctx.append(c)
                todo_keys.append(k)
        if todo_ctx:
            fresh = self._infer(todo_ctx)
            for k, r in zip(todo_keys, fresh):
                self._memo[k] = r if r[0] is not None else (None, [])
        return [self._memo.get(k, (None, [])) for k in keys]

    def _infer(self, contexts: list[dict]) -> list[tuple[str | None, list[str]]]:
        results: list[tuple[str | None, list[str]]] = []
        for ctx in contexts:
            try:
                ans = self._http_chat(ctx)
                action, codes = None, ["llama-server"]
                try:
                    obj = json.loads(ans[ans.index("{"): ans.rindex("}") + 1])
                    action = _map_action(obj.get("action", ""))
                    fb = obj.get("feedback_loop", {})
                    if fb.get("retrain_trigger"):
                        codes.append("RETRAIN")
                    if fb.get("hitl_escalation"):
                        codes.append("HITL")
                except Exception:
                    action = _map_action(ans)
                results.append((action, codes))
            except Exception as e:
                print(f"[warn] http inference: {e}")
                results.append((None, []))
        return results


class DistilledAdvisor:
    def __init__(self, model_path: str):
        import joblib

        self.model = joblib.load(model_path)
        self.ok = True
        self.mode = "distilled"

    def _featurize(self, ctx: dict[str, Any]) -> np.ndarray:
        return np.array(
            [
                int(ctx.get("cpus_req", 0)),
                float(ctx["bw_bound"]),
                float(ctx["rpc_bound"]),
                float(ctx.get("priority", 100)),
                int(ctx.get("task_type", "NA") == "training"),
                float(ctx.get("bw_frac", 0.0)),
                float(ctx.get("rpc_frac", 0.0)),
            ],
            dtype=np.float32,
        )

    def advise(self, ctx: dict[str, Any]) -> tuple[str | None, list[str]]:
        x = self._featurize(ctx).reshape(1, -1)
        return str(self.model.predict(x)[0]), ["distilled"]

    def advise_batch(
        self, contexts: list[dict], batch_size: int | None = None
    ) -> list[tuple[str | None, list[str]]]:
        X = np.vstack([self._featurize(c) for c in contexts])
        return [(str(a), ["distilled"]) for a in self.model.predict(X)]
