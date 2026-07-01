"""Centralized config. YAML-backed, pydantic-validated."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

XGBParamValue = int | float | str | bool
XGBParams = dict[str, XGBParamValue]

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"


class DataConfig(BaseModel):
    sys_csv: str = "system_state_5min.csv"
    job_csv: str = "job_impact_dataset.csv"
    node_csv: str = "node-data.csv"

    past_bins: int = 12       # 1 hour @ 5 min/bin
    future_bins: int = 6      # 30 min horizon
    bin_seconds: int = 300
    z_upper: float = 1.645    # 95% one-sided


class ModelConfig(BaseModel):
    # DLinear
    dlin_epochs: int = 30
    dlin_bs: int = 128
    dlin_lr: float = 5e-3
    dlin_kernel: int = 7      # trend MA kernel

    # XGBoost delta-IO
    xgb_params: XGBParams = Field(
        default_factory=lambda: {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.05,
            "tree_method": "hist",
        }
    )
    xgb_rounds: int = 400
    xgb_early_stop: int = 20
    xgb_n_splits: int = 3
    xgb_tune: bool = False
    xgb_tune_grid: list[XGBParams] = Field(default_factory=list)

    # Congestion classifier
    congestion_use_lstm: bool = False  # dropped: LSTM degenerates
    congestion_beta: float = 2.0
    congestion_min_recall: float = 0.80
    congestion_max_recall: float = 0.98
    congestion_min_lift: float = 1.25


class GuardrailConfig(BaseModel):
    bw_quantile_soft: float = 0.95
    bw_quantile_hard: float = 0.99
    rpc_quantile_soft: float = 0.95
    rpc_quantile_hard: float = 0.99

    # Sustained-violation policy
    sustained_bins: int = 2
    z: float = 1.645  # 95% upper
    hard_buffer: float = 1.1  # HOLD if bound > hard * 1.1

    # Rescue
    rescue_thr_max: float = 0.15

    # Causal Kalman residual
    kf_sigma_w2_bw: float = 1e-4
    kf_sigma_v2_bw: float = 1e-2
    kf_p0_bw: float = 1.0
    kf_sigma_w2_rpc: float = 1e12
    kf_sigma_v2_rpc: float = 1e14
    kf_p0_rpc: float = 1e14

    # Pareto auto-tune (Tier-2 enhancement F)
    auto_tune: bool = True


class MLOpsConfig(BaseModel):
    k_nis: float = 6.63        # chi2 99% with 1 dof
    conf_min: float = 0.5
    recent_window: int = 20


class LLMConfig(BaseModel):
    use_llm: bool = True
    backend: Literal["hf", "gguf"] = "gguf"
    gguf_path: str = "models/gemma-3-4b-it-IQ4_XS.gguf"
    gguf_n_ctx: int = 4096
    gguf_n_gpu_layers: int = -1
    gguf_n_threads: int | None = None
    sys_prompt: str = (
        "You are an HPC Orchestrator. Output strictly JSON with keys: action, "
        "throttle_config(bw_limit, rpc_limit), feedback_loop(tuning_offset, "
        "retrain_trigger, hitl_escalation)."
    )
    batch_size: int = 32
    max_new_tokens: int = 64


class SimConfig(BaseModel):
    capacity_quantile: float = 0.95
    capacity_min: int = 50
    duration_logn_mean_sec: float = 7200
    duration_logn_sigma: float = 1.0
    duration_min_sec: int = 60
    duration_max_sec: int = 7 * 24 * 3600
    seed: int = 42


class Config(BaseModel):
    seed: int = 42
    data_dir: str = "data"
    artifact_dir: str = "artifacts"
    report_dir: str = "reports"

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    guardrail: GuardrailConfig = Field(default_factory=GuardrailConfig)
    mlops: MLOpsConfig = Field(default_factory=MLOpsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    sim: SimConfig = Field(default_factory=SimConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        import yaml

        with open(path) as f:
            return cls(**yaml.safe_load(f))

    def to_yaml(self, path: str | Path) -> None:
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, sort_keys=False)


def get_config(name: Literal["default", "strict", "permissive"] = "default") -> Config:
    p = CONFIG_DIR / f"{name}.yaml"
    return Config.from_yaml(p) if p.exists() else Config()
