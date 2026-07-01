# HPC IO Scheduler — FIAC

**Forecast-Informed Adaptive Control** for HPC job I/O scheduling on MIT Supercloud.

3-layer hybrid pipeline: probabilistic DLinear → causal Kalman residual correction → XGBoost delta-IO → Qwen-LoRA gray-zone advisor, wrapped in a safety-overriding guardrail.

> Migrated from `hpc_final (5).ipynb`. Original notebook is the source of truth for the model architecture; this package is the production-ready refactor.

---

## Why this exists

Naive FCFS submission lets concurrent jobs saturate Lustre bandwidth and metadata RPCs, causing system-wide congestion. FIAC forecasts background I/O 30 min ahead, adds predicted job delta-IO, applies a causal Kalman correction, and decides `SUBMIT | THROTTLE | HOLD` per job.

**Result (test split, MIT Supercloud)**: Decision Acc 0.872 vs FCFS 0.683, False-Submit-Rate 0.082 vs 0.317. Per ablation, the reasoning layer (LLM) is load-bearing — without it, accuracy collapses to 0.317.

---

## Structure

```
hpc_io_scheduler/
├── pyproject.toml          # PEP 621, src layout, entry points
├── README.md
├── configs/
│   ├── default.yaml        # balanced policy
│   ├── strict.yaml         # safety-first (tighter thresholds, 1-bin trip)
│   └── permissive.yaml     # throughput-first
├── src/hpc_io_scheduler/
│   ├── config.py           # pydantic-validated YAML config
│   ├── pipeline.py         # one-shot fit_all + save_bundle
│   ├── data/
│   │   ├── loader.py       # CSV loaders + schema validation
│   │   ├── windowing.py    # sliding windows + scaler fit (train-only)
│   │   └── splits.py       # GroupKFold, chronological split, alignment
│   ├── models/
│   │   ├── dlinear.py      # Probabilistic DLinear (Zeng AAAI'23)
│   │   ├── conformal.py    # distribution-free uncertainty band
│   │   ├── xgb_delta.py    # Layer 2: per-job delta-IO
│   │   ├── congestion.py   # XGB+LSTM classifier + F2-Pareto policy
│   │   └── lora_advisor.py # Qwen-1.5B + LoRA + distilled surrogate
│   ├── guardrail/
│   │   ├── thresholds.py   # empirical quantiles, train-only
│   │   ├── kalman.py       # online causal residual correction
│   │   └── policy.py       # ForecastGuardrail + safety override
│   ├── evaluation/
│   │   ├── metrics.py      # forecast, classifier, decision metrics
│   │   ├── baselines.py    # FCFS / forecast / LLM-only (no oracle)
│   │   ├── simulator.py    # closed-loop queue sim
│   │   ├── ablations.py    # drop-uncertainty / drop-dio / drop-reasoning
│   │   ├── audit.py        # XGBoost delta-IO leakage audit
│   │   └── shap_explain.py # SHAP for congestion classifier
│   └── mlops/
│       └── monitor.py      # KF NIS retrain trigger
├── scripts/
│   ├── train.py            # hpc-train
│   ├── backtest.py         # hpc-backtest [--leakage-audit]
│   └── serve.py            # hpc-serve (skeleton)
├── tests/                  # pytest, no GPU required
└── data/                   # user-provided CSVs (gitignored)
```

---

## Install

```bash
pip install -e ".[advisor,dev]"
```

GPU strongly recommended for the LLM advisor; CPU-only runs still work but use the heuristic fallback for gray-zone cases.

## Data

Drop these into `data/` (or set `data_dir` in config):
- `system_state_5min.csv` — columns: `bin, bw_recon_mbps, io_p95, io_max, io_fano, lustre_rpc, active_jobs, sum_file_rw`
- `job_impact_dataset.csv` — must contain `t_start, id_job_norm, split, dnn_label` + numeric/cat features + `delta_bw_p90, delta_rpc_p90`
- `node-data.csv` (optional) — for node-locality features

## Usage

```bash
# Train
hpc-train
# or: python -m hpc_io_scheduler.pipeline train

# Backtest (shadow mode) + write reports
hpc-backtest --leakage-audit

# Backtest with strict policy
hpc-backtest --config configs/strict.yaml

# Serve (skeleton — wire FastAPI app)
hpc-serve --port 8000
```

## Tests

```bash
pytest -q
```

Smoke tests cover:
- Chronological split integrity
- Causal KF prior-independence (mutating obs[i] must not change bias[i])
- Conformal coverage ≥ 1 - alpha
- Guardrail safety override
- Simulator determinism (same seed → same metrics)
- Pipeline fit+save round-trip on synthetic data

---

## Migration from the notebook

| Notebook cell | New location |
|---|---|
| CELL 0-1 (config + load) | `config.py`, `data/loader.py` |
| CELL 2 (windowing + scalers) | `data/windowing.py` |
| CELL 3 (job preprocessing) | `data/splits.py:fit_transform_jobs` |
| CELL 4 (DLinear) | `models/dlinear.py` |
| CELL 5 (causal KF) | `guardrail/kalman.py` |
| CELL 6 (XGB delta) | `models/xgb_delta.py` |
| CELL 7 (guardrail) | `guardrail/policy.py` |
| CELL 8 (Qwen advisor) | `models/lora_advisor.py` |
| CELL 9 (backtest) | `scripts/backtest.py` |
| CELL 10 (baselines) | `evaluation/baselines.py` |
| CELL 11 (simulator) | `evaluation/simulator.py` |
| CELL 12 (MLOps) | `mlops/monitor.py` |
| CELL 12b (congestion classifier) | `models/congestion.py` |
| CELL 12c (rescue) | (TBD) `guardrail/policy.py:rescue_hold_with_low_risk` |
| CELL 13 (eval suite) | `evaluation/*` |
| CELL 14 (artifacts) | `pipeline.py:save_bundle` |

---

## Honest evaluation

- All baselines use **predicted** delta-IO, never ground-truth.
- GT congestion computed only for evaluation, never for decisions.
- Causal KF uses **prior** for correction, posterior for next step.
- Train-only scaler, threshold, and conformal calibration fit.

### Run the XGBoost leakage audit

The original notebook reported `BW RMSE=0.0008` — suspiciously low. This is the first thing to verify on your data:

```bash
hpc-backtest --leakage-audit
```

If the audit returns `CRITICAL: group leakage in fold`, the paper claim is invalid; investigate `hist_*` features and `id_job_norm` group definition.

---

## Enhancements baked in (vs notebook)

| # | Enhancement | Where |
|---|---|---|
| 1 | Conformal prediction replaces Gaussian NLL std | `models/conformal.py` |
| 2 | LSTM behind flag (degenerates after epoch 30) | `config.model.congestion_use_lstm` |
| 3 | XGBoost leakage audit | `evaluation/audit.py` |
| 4 | Distilled tree advisor (drop-in for Qwen) | `models/lora_advisor.py:DistilledAdvisor` |
| 5 | Pareto-front threshold policy (config-tunable) | `models/congestion.py:select_best_idx` |
| 6 | Adaptive thresholds from KF bias | `guardrail/thresholds.py:adaptive_thresholds` |
| 7 | SHAP explainability for classifier | `evaluation/shap_explain.py` |
| 8 | YAML config with 3 presets | `configs/{default,strict,permissive}.yaml` |
| 9 | Pytest smoke tests | `tests/` |
| 10 | Pydantic-validated config | `config.py` |

Skipped: per-job utility weighting (Tier-2 G — wire when priority policy changes), rolling-window backtest (Tier-3 K — add when evaluating in prod).
