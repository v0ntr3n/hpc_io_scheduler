from __future__ import annotations

from hpc_io_scheduler.config import Config
from hpc_io_scheduler.models.xgb_delta import candidate_xgb_params


def test_xgb_tuning_candidates_default_to_single_base_params() -> None:
    cfg = Config()

    candidates = candidate_xgb_params(cfg)

    assert candidates == [dict(cfg.model.xgb_params, seed=cfg.seed)]


def test_xgb_tuning_candidates_require_explicit_gate() -> None:
    cfg = Config()
    cfg.model.xgb_tune = True
    cfg.model.xgb_tune_grid = [
        {"max_depth": 3, "learning_rate": 0.03, "subsample": 0.8},
        {"max_depth": 5, "learning_rate": 0.05, "reg_lambda": 2.0},
    ]

    candidates = candidate_xgb_params(cfg)

    assert candidates[0]["max_depth"] == 3
    assert candidates[0]["subsample"] == 0.8
    assert candidates[1]["max_depth"] == 5
    assert candidates[1]["reg_lambda"] == 2.0
    assert all(candidate["seed"] == cfg.seed for candidate in candidates)
