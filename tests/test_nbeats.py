"""Smoke tests for NBeatsProbabilistic."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hpc_io_scheduler.config import Config
from hpc_io_scheduler.models.nbeats import (
    NBeatsProbabilistic,
    predict_nbeats,
    train_nbeats,
)


def test_nbeats_forward_shape():
    cfg = Config()
    model = NBeatsProbabilistic(
        seq_len=cfg.data.past_bins,
        pred_len=cfg.data.future_bins,
        n_features=7,
        n_targets=2,
        target_idx=(0, 4),
        n_blocks=2, hidden=32,
    )
    x = torch.randn(4, cfg.data.past_bins, 7)
    mean, std = model(x)
    assert mean.shape == (4, cfg.data.future_bins, 2)
    assert std.shape == (4, cfg.data.future_bins, 2)
    assert (std > 0).all()


def test_nbeats_train_predict_roundtrip():
    cfg = Config()
    cfg.model.dlin_epochs = 2
    cfg.model.dlin_bs = 16
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, cfg.data.past_bins, 7)).astype(np.float32)
    Y = rng.normal(0, 1, (40, cfg.data.future_bins, 2)).astype(np.float32)
    model, sec = train_nbeats(X, Y, cfg, device="cpu", n_blocks=2, hidden=32)
    assert sec >= 0
    m, s = predict_nbeats(model, X[:5], device="cpu")
    assert m.shape == (5, cfg.data.future_bins, 2)
    assert s.shape == (5, cfg.data.future_bins, 2)
    assert (s > 0).all()


def test_nbeats_output_is_finite_on_ones():
    cfg = Config()
    model = NBeatsProbabilistic(
        seq_len=cfg.data.past_bins,
        pred_len=cfg.data.future_bins,
        n_features=7, n_targets=2, target_idx=(0, 4),
        n_blocks=2, hidden=32,
    )
    x = torch.ones(2, cfg.data.past_bins, 7)
    m, s = model(x)
    assert torch.isfinite(m).all()
    assert torch.isfinite(s).all()
