"""Probabilistic DLinear (Zeng AAAI 2023) — pure temporal, no feature mixing.

Outputs (mean, std) for 2 target channels (BW, RPC).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hpc_io_scheduler.config import Config
from hpc_io_scheduler.data.windowing import make_windows


class ProbabilisticDLinear(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_features: int,
        n_targets: int = 2,
        kernel_size: int = 7,
        target_idx: tuple[int, int] = (0, 4),
    ):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReplicationPad1d((pad, pad))
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1)
        self.lin_trend_mean = nn.Linear(seq_len, pred_len)
        self.lin_seas_mean = nn.Linear(seq_len, pred_len)
        self.lin_trend_std = nn.Linear(seq_len, pred_len)
        self.lin_seas_std = nn.Linear(seq_len, pred_len)
        self.bw_idx, self.rpc_idx = target_idx
        self.n_targets = n_targets

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xp = x.permute(0, 2, 1)
        trend = self.avg(self.pad(xp))
        seasonal = xp - trend
        out_mean = self.lin_trend_mean(trend) + self.lin_seas_mean(seasonal)
        out_std = self.lin_trend_std(trend) + self.lin_seas_std(seasonal)
        out_mean = out_mean.permute(0, 2, 1)
        out_std = out_std.permute(0, 2, 1)
        mean = out_mean[:, :, [self.bw_idx, self.rpc_idx]]
        std = out_std[:, :, [self.bw_idx, self.rpc_idx]]
        std = torch.clamp(torch.nn.functional.softplus(std), 1e-3, 1e5)
        return mean, std


class MultiHorizonDLinear(nn.Module):
    """Shared trend/seasonal decomposition, separate linear heads per horizon.

    Output: dict horizon_idx -> (mean, std) each (B, S_h, T).
    """

    def __init__(
        self,
        seq_len: int,
        horizons: list[int],
        n_features: int,
        n_targets: int = 2,
        kernel_size: int = 7,
        target_idx: tuple[int, int] = (0, 4),
    ):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReplicationPad1d((pad, pad))
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1)
        self.seq_len = seq_len
        self.horizons = list(horizons)
        self.bw_idx, self.rpc_idx = target_idx
        self.n_targets = n_targets
        self.heads = nn.ModuleDict()
        for h in self.horizons:
            self.heads[str(h)] = nn.ModuleDict({
                "trend_mean": nn.Linear(seq_len, h),
                "seas_mean": nn.Linear(seq_len, h),
                "trend_std": nn.Linear(seq_len, h),
                "seas_std": nn.Linear(seq_len, h),
            })

    def forward(self, x: torch.Tensor) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        xp = x.permute(0, 2, 1)
        trend = self.avg(self.pad(xp))
        seasonal = xp - trend
        out = {}
        for h, head in zip(self.horizons, self.heads.values()):
            mean = (head["trend_mean"](trend) + head["seas_mean"](seasonal)).permute(0, 2, 1)
            std = (head["trend_std"](trend) + head["seas_std"](seasonal)).permute(0, 2, 1)
            mean = mean[:, :, [self.bw_idx, self.rpc_idx]]
            std = std[:, :, [self.bw_idx, self.rpc_idx]]
            std = torch.clamp(torch.nn.functional.softplus(std), 1e-3, 1e5)
            out[h] = (mean, std)


def train_multi_horizon(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    horizons: list[int],
    cfg: Config,
    device: str | torch.device = "cpu",
) -> tuple[MultiHorizonDLinear, float]:
    """Y_train shape (N, max_h, T); shorter horizons slice Y_train[:, :h, :]."""
    n_features = X_train.shape[2]
    model = MultiHorizonDLinear(
        cfg.data.past_bins, horizons, n_features, target_idx=(0, 4),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.model.dlin_lr)
    nll = nn.GaussianNLLLoss()
    max_h = max(horizons)

    ds = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(Y_train[:, :max_h, :]),
    )
    dl = DataLoader(ds, batch_size=cfg.model.dlin_bs, shuffle=True)
    model.train()
    t0 = time.time()
    for _ in range(cfg.model.dlin_epochs):
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            preds = model(bx)
            loss = sum(
                nll(preds[h][0], by[:, :h, :], preds[h][1].pow(2))
                for h in horizons
            )
            loss.backward()
            opt.step()
    return model, time.time() - t0


def train_dlinear(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    cfg: Config,
    device: str | torch.device = "cpu",
) -> tuple[ProbabilisticDLinear, float]:
    """Gaussian NLL training. Returns (model, train_sec)."""
    n_features = X_train.shape[2]
    model = ProbabilisticDLinear(
        cfg.data.past_bins,
        cfg.data.future_bins,
        n_features,
        target_idx=(0, 4),
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg.model.dlin_lr
    )
    nll = nn.GaussianNLLLoss()

    ds = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    dl = DataLoader(ds, batch_size=cfg.model.dlin_bs, shuffle=True)
    model.train()
    t0 = time.time()
    for _ in range(cfg.model.dlin_epochs):
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            mean, std = model(bx)
            loss = nll(mean, by, std.pow(2))
            loss.backward()
            opt.step()
    return model, time.time() - t0


@torch.no_grad()
def predict_dlinear(
    model: ProbabilisticDLinear,
    X: np.ndarray,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    mean, std = model(torch.tensor(X, device=device))
    return mean.cpu().numpy(), std.cpu().numpy()
