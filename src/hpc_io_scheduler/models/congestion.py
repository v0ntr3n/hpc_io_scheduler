"""Congestion classifier: XGBoost + (optional) LSTM on rich features.

Tier-2 E: LSTM degenerates after epoch 30 in original notebook. Default
keeps XGBoost-only; LSTM behind flag for ablation only.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc, precision_recall_curve

from hpc_io_scheduler.config import Config


# ---------- Threshold policy (Tier-2 F: Pareto auto-tune) ----------

def fbeta_curve(p: np.ndarray, r: np.ndarray, beta: float = 2.0) -> np.ndarray:
    pp, rr = p[:-1], r[:-1]
    b2 = beta ** 2
    return (1 + b2) * pp * rr / (b2 * pp + rr + 1e-9)


def is_trivial(
    P: float, R: float, base_rate: float, cfg: Config
) -> bool:
    """Trivial = recall-saturated OR precision no better than naive baseline."""
    return (R >= cfg.model.congestion_max_recall) or (
        P < base_rate * cfg.model.congestion_min_lift
    )


def select_best_idx(
    p: np.ndarray, r: np.ndarray, base_rate: float, cfg: Config
) -> tuple[int, np.ndarray]:
    """Pick threshold via F-beta within [min_recall, max_recall) and lift band."""
    fb = fbeta_curve(p, r, cfg.model.congestion_beta)
    pp, rr = p[:-1], r[:-1]
    nontrivial = (pp >= base_rate * cfg.model.congestion_min_lift) & (
        rr < cfg.model.congestion_max_recall
    )
    band = nontrivial & (rr >= cfg.model.congestion_min_recall)
    if band.any():
        cand = np.where(band)[0]
    elif nontrivial.any():
        cand = np.where(nontrivial)[0]
    elif (rr < cfg.model.congestion_max_recall).any():
        cand = np.where(rr < cfg.model.congestion_max_recall)[0]
    else:
        cand = np.arange(len(fb))
    return int(cand[np.argmax(fb[cand])]), fb


# ---------- Feature engineering (kept verbatim from notebook) ----------

BW_IDX, RPC_IDX, ACT_IDX = 0, 4, 5


def build_lag_features(win_3d: np.ndarray) -> np.ndarray:
    win_bw = win_3d[:, :, BW_IDX]
    win_rpc = win_3d[:, :, RPC_IDX]
    win_act = win_3d[:, :, ACT_IDX]
    return np.column_stack(
        [
            win_bw.mean(1), win_bw.std(1),
            win_rpc.mean(1), win_rpc.std(1),
            win_bw.max(1), win_rpc.max(1),
            win_bw[:, -3:].mean(1) - win_bw[:, :3].mean(1),
            win_rpc[:, -3:].mean(1) - win_rpc[:, :3].mean(1),
            win_bw[:, -1] - win_bw[:, -2],
            win_rpc[:, -1] - win_rpc[:, -2],
            np.percentile(win_bw, 90, 1) - np.percentile(win_bw, 10, 1),
            np.percentile(win_rpc, 90, 1) - np.percentile(win_rpc, 10, 1),
            win_bw[:, -1] / (win_bw.mean(1) + 1e-9),
            win_rpc[:, -1] / (win_rpc.mean(1) + 1e-9),
            win_act.mean(1), win_act.max(1),
        ]
    )


def build_time_features(t: np.ndarray) -> np.ndarray:
    t_pd = np.array(t, dtype="datetime64[ns]")
    hour = t_pd.astype("datetime64[h]").astype(int) % 24
    dow = (t_pd.astype("datetime64[D]").astype(int) % 7)
    return np.column_stack(
        [
            hour, dow,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
        ]
    )


def build_history_features(
    proxy_cong: np.ndarray, sig_bw: np.ndarray, win: int = 50
) -> np.ndarray:
    n = len(proxy_cong)
    out_rate = np.zeros(n, dtype=np.float32)
    out_sig = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = max(0, i - win)
        out_rate[i] = proxy_cong[s:i].mean() if i > 0 else 0.0
        out_sig[i] = sig_bw[s:i].mean() if i > 0 else sig_bw[0]
    return np.column_stack([out_rate, out_sig])


def build_rich_features(
    X_test_sorted: np.ndarray,
    pos_valid: np.ndarray,
    m_real_sorted: np.ndarray,
    s_real_sorted: np.ndarray,
    job_dbw: np.ndarray,
    job_drpc: np.ndarray,
    t_win_sorted: np.ndarray,
    Xj_test_valid: np.ndarray,
    bw_hard: float,
    rpc_hard: float,
    node_features: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns (X_cong_rich, y_cong_proxy_unused, tab_start).

    If `node_features` provided (shape (n_jobs, F_node)), concatenated to feature matrix.
    """
    sig_bw = s_real_sorted[pos_valid, :, 0].max(1) + 1e-6
    sig_rpc = s_real_sorted[pos_valid, :, 1].max(1) + 1e-6
    pred_bw_max = m_real_sorted[pos_valid, :, 0].max(1)
    pred_rpc_max = m_real_sorted[pos_valid, :, 1].max(1)
    predU_bw = (m_real_sorted[pos_valid, :, 0] + 1.645 * s_real_sorted[pos_valid, :, 0]).max(1)
    predU_rpc = (m_real_sorted[pos_valid, :, 1] + 1.645 * s_real_sorted[pos_valid, :, 1]).max(1)

    fc = np.column_stack(
        [
            pred_bw_max, pred_rpc_max, predU_bw, predU_rpc,
            sig_bw, sig_rpc,
            job_dbw, job_drpc,
            pred_bw_max + job_dbw, pred_rpc_max + job_drpc,
        ]
    )
    sys_flat = X_test_sorted[pos_valid].reshape(len(pos_valid), -1)
    lag = build_lag_features(X_test_sorted[pos_valid])
    time_f = build_time_features(t_win_sorted[pos_valid])

    proxy = ((predU_bw + job_dbw) > bw_hard) | ((predU_rpc + job_drpc) > rpc_hard)
    hist = build_history_features(proxy, sig_bw)

    parts = [sys_flat, Xj_test_valid, fc, lag, time_f, hist]
    if node_features is not None:
        parts.append(np.asarray(node_features, dtype=np.float32))
    X_cong = np.hstack(parts).astype(np.float32)
    return X_cong, proxy.astype(int), sys_flat.shape[1]


# ---------- LSTM (optional, ablation only) ----------

class CongestionLSTM(nn.Module):
    def __init__(self, n_sys: int = 7, lstm_h: int = 64, tab_dim: int = 0, fc_h: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_sys, lstm_h, num_layers=2, batch_first=True, dropout=0.2)
        self.bn_lstm = nn.BatchNorm1d(lstm_h)
        self.tab_fc = (
            nn.Sequential(
                nn.Linear(tab_dim, fc_h), nn.ReLU(), nn.Dropout(0.2), nn.BatchNorm1d(fc_h)
            )
            if tab_dim > 0
            else None
        )
        combined = lstm_h + (fc_h if tab_dim > 0 else 0)
        self.head = nn.Sequential(
            nn.Linear(combined, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
        )

    def forward(self, x_seq: torch.Tensor, x_tab: torch.Tensor | None = None) -> torch.Tensor:
        _, (h, _) = self.lstm(x_seq)
        h = self.bn_lstm(h[-1])
        if self.tab_fc is not None and x_tab is not None:
            h = torch.cat([h, self.tab_fc(x_tab)], dim=1)
        return self.head(h).squeeze(1)
