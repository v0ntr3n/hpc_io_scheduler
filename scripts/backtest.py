"""Backtest CLI. Thin wrapper over hpc_io_scheduler.backtest.run_backtest."""
from __future__ import annotations

import argparse

from hpc_io_scheduler.backtest import run_backtest
from hpc_io_scheduler.config import Config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML config path")
    ap.add_argument("--artifact-dir", default=None)
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--leakage-audit", action="store_true",
                    help="Run XGBoost delta-IO leakage audit before backtest")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    run_backtest(
        cfg,
        artifact_dir=args.artifact_dir or cfg.artifact_dir,
        report_dir=args.report_dir or cfg.report_dir,
        run_audit=args.leakage_audit,
    )


if __name__ == "__main__":
    main()
