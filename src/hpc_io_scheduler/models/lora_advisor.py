"""Fine-tuned Qwen LoRA advisor (Layer 3) for gray-zone decisions.

Currently configured for GGUF Gemma-3-4B via llama-cpp-python. No LoRA
adapter applied; the model is used zero-shot with the sys_prompt.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
import torch

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
    """Batched Qwen + LoRA inference. Memoizes by context hash."""

    def __init__(self, cfg: Config, llm_cfg: LLMConfig | None = None):
        self.cfg = cfg
        self.llm = llm_cfg or cfg.llm
        self.ok = False
        self.mode = "heuristic"
        self._memo: dict[tuple, tuple[str | None, list[str]]] = {}
        if not self.llm.use_llm:
            return
        if self.llm.backend != "gguf":
            print(f"[warn] backend={self.llm.backend} not supported in this build; use backend=gguf")
            return
        self._init_gguf()

    def _init_hf(self) -> None:
        raise NotImplementedError("HF backend removed; use backend=gguf")

    def _init_gguf(self) -> None:
        """Load GGUF via llama-cpp-python. No LoRA (Gemma base)."""
        try:
            from llama_cpp import Llama

            if not os.path.isfile(self.llm.gguf_path):
                print(f"[warn] GGUF not found: {self.llm.gguf_path}")
                return
            print(f"Loading GGUF: {self.llm.gguf_path}")
            kw = dict(
                model_path=self.llm.gguf_path,
                n_ctx=self.llm.gguf_n_ctx,
                n_gpu_layers=self.llm.gguf_n_gpu_layers,
                verbose=False,
            )
            if self.llm.gguf_n_threads:
                kw["n_threads"] = self.llm.gguf_n_threads
            self.llm_model = Llama(**kw)
            self.ok = True
            self.mode = f"gguf ({os.path.basename(self.llm.gguf_path)})"
        except Exception as e:  # pragma: no cover
            print(f"[warn] GGUF init failed: {e}")

    def _gguf_chat(self, ctx: dict[str, Any]) -> str:
        """Single-context chat completion via llama-cpp chat handler."""
        msgs = [
            {"role": "system", "content": self.llm.sys_prompt},
            {"role": "user", "content": json.dumps(_build_user_msg(ctx))},
        ]
        out = self.llm_model.create_chat_completion(
            messages=msgs,
            max_tokens=self.llm.max_new_tokens,
            temperature=0.0,
        )
        return out["choices"][0]["message"]["content"]

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
        return self.advise_batch([ctx], batch_size=1)[0]

    def advise_batch(
        self, contexts: list[dict[str, Any]], batch_size: int | None = None
    ) -> list[tuple[str | None, list[str]]]:
        if not self.ok or not contexts:
            return [(None, [])] * len(contexts)
        bs = batch_size or self.llm.batch_size
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
            fresh = self._infer_batch(todo_ctx, bs)
            for k, r in zip(todo_keys, fresh):
                self._memo[k] = r if r[0] is not None else (None, [])
        return [self._memo.get(k, (None, [])) for k in keys]

    def _infer_batch(
        self, contexts: list[dict], batch_size: int
    ) -> list[tuple[str | None, list[str]]]:
        results: list[tuple[str | None, list[str]]] = []
        if self.llm.backend == "gguf":
            for ctx in contexts:
                try:
                    ans = self._gguf_chat(ctx)
                    action, codes = None, ["gguf"]
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
                    print(f"[warn] gguf inference: {e}")
                    results.append((None, []))
            return results

        try:
            texts = []
            for ctx in contexts:
                msgs = [
                    {"role": "system", "content": self.llm.sys_prompt},
                    {"role": "user", "content": json.dumps(_build_user_msg(ctx))},
                ]
                texts.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            for s in range(0, len(texts), batch_size):
                batch = texts[s : s + batch_size]
                ids = self.tok(batch, return_tensors="pt", padding=True).to(self.model.device)
                prompt_len = ids["input_ids"].shape[1]
                with torch.no_grad():
                    out = self.model.generate(
                        **ids,
                        max_new_tokens=self.llm.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tok.eos_token_id,
                        eos_token_id=self.tok.eos_token_id,
                    )
                for tokens in out:
                    ans = self.tok.decode(tokens[prompt_len:], skip_special_tokens=True)
                    action, codes = None, ["LLM_finetuned"]
                    try:
                        obj = json.loads(ans[ans.index("{") : ans.rindex("}") + 1])
                        action = _map_action(obj.get("action", ""))
                        fb = obj.get("feedback_loop", {})
                        if fb.get("retrain_trigger"):
                            codes.append("RETRAIN")
                        if fb.get("hitl_escalation"):
                            codes.append("HITL")
                    except Exception:
                        action = _map_action(ans)
                    results.append((action, codes))
            while len(results) < len(contexts):
                results.append((None, []))
            return results
        except Exception as e:  # pragma: no cover
            print(f"[warn] LLM batch failed: {e}")
            return [(None, [])] * len(contexts)


# ---- Distilled surrogate (Tier-2 I) ----------------------------------------


class DistilledAdvisor:
    """Tree-based surrogate trained on logged LLM decisions.

    Drop-in for QwenAdvisor in production. Same interface.
    """

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
            ]
        )

    def advise(self, ctx: dict[str, Any]) -> tuple[str | None, list[str]]:
        x = self._featurize(ctx).reshape(1, -1)
        return str(self.model.predict(x)[0]), ["distilled"]

    def advise_batch(
        self, contexts: list[dict], batch_size: int | None = None
    ) -> list[tuple[str | None, list[str]]]:
        X = np.vstack([self._featurize(c) for c in contexts])
        return [(str(a), ["distilled"]) for a in self.model.predict(X)]
