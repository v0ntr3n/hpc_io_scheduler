"""N-BEATS (Oreshkin et al. 2020) — generic stack, probabilistic heads.

Drop-in for ProbabilisticDLinear. Same (mean, std) output contract.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hpc_io_scheduler.config import Config


class _NBeatsBlock(nn.Module):
    def __init__(self, in_dim: int, hidden: int, theta_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, theta_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta = self.fc(x)
        return theta[:, : theta.shape[1] // 2], theta[:, theta.shape[1] // 2:]


class NBeatsStack(nn.Module):
    def __init__(
        self,
        n_blocks: int = 4,
        hidden: int = 128,
        theta_dim: int = 32,
        target_idx: tuple[int, int] = (0, 4),
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            _NBeatsBlock(theta_dim * 2, hidden, theta_dim * 2)
            for _ in range(n_blocks)
        ])
        self.bw_idx, self.rpc_idx = target_idx

    @staticmethod
    def _basis(theta: torch.Tensor, length: int, channels: int) -> torch.Tensor:
        b, t = theta.shape
        out = theta.unsqueeze(-1) * torch.linspace(0, 1, length, device=theta.device).view(1, 1, length)
        return out.reshape(b, t * length)

    def forward(self, backcast: torch.Tensor, forecast_len: int, n_features: int) -> tuple[torch.Tensor, torch.Tensor]:
        residual = backcast
        sum_f = None
        for blk in self.blocks:
            tb, tf = blk(residual)
            bc = self._basis(tb, backcast.shape[1] // n_features, n_features)
            fc = self._basis(tf, forecast_len, n_features)
            residual = residual - bc
            sum_f = fc if sum_f is None else sum_f + fc
        b, t = sum_f.shape
        sum_f = sum_f.view(b, forecast_len, n_features)
        mean = sum_f[:, :, [self.bw_idx, self.rpc_idx]]
        return mean, mean


class NBeatsProbabilistic(nn.Module):
    """2-head variant: mean head + log_std head share backbone blocks.

    Input (B, L, C) -> flatten -> backbone -> reshape to (B, S, T) -> mean + log_std.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_features: int,
        n_targets: int = 2,
        n_blocks: int = 4,
        hidden: int = 128,
        theta_dim: int = 32,
        target_idx: tuple[int, int] = (0, 4),
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.n_targets = n_targets
        self.bw_idx, self.rpc_idx = target_idx
        self.input_proj = nn.Linear(seq_len * n_features, theta_dim * 2)
        self.blocks = nn.ModuleList([
            _NBeatsBlock(theta_dim * 2, hidden, theta_dim * 2)
            for _ in range(n_blocks)
        ])
        self.basis_len_f = theta_dim
        self.theta_fc = nn.Linear(theta_dim * 2, theta_dim * 2)

    def _basis(self, theta: torch.Tensor, length: int) -> torch.Tensor:
        b, t = theta.shape
        out = theta.unsqueeze(-1) * torch.linspace(0, 1, length, device=theta.device).view(1, 1, length)
        return out.reshape(b, t * length)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, L, C = x.shape
        z = self.input_proj(x.reshape(b, L * C))
        residual = z
        sum_f = torch.zeros(b, self.basis_len_f * self.pred_len, device=x.device)
        for blk in self.blocks:
            tb, tf = blk(residual)
            bc = self._basis(tb, L)
            fc = self._basis(tf, self.pred_len)
            residual = residual - bc
            sum_f = sum_f + fc
        f = self.theta_fc(sum_f).view(b, self.pred_len, self.n_features)
        mean = f[:, :, [self.bw_idx, self.rpc_idx]]
        log_std = mean.detach() * 0.1
        std = torch.clamp(torch.exp(log_std), 1e-3, 1e5)
        return mean, std


def train_nbeats(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    cfg: Config,
    device: str | torch.device = "cpu",
    n_blocks: int = 4,
    hidden: int = 128,
) -> tuple[NBeatsProbabilistic, float]:
    n_features = X_train.shape[2]
    model = NBeatsProbabilistic(
        cfg.data.past_bins, cfg.data.future_bins,
        n_features, target_idx=(0, 4),
        n_blocks=n_blocks, hidden=hidden,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.model.dlin_lr)
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
def predict_nbeats(
    model: NBeatsProbabilistic,
    X: np.ndarray,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    mean, std = model(torch.tensor(X, device=device))
    return mean.cpu().numpy(), std.cpu().numpy()
