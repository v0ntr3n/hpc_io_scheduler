"""Train CLI: fit all models + scalers + thresholds, persist to artifacts/."""
from __future__ import annotations

from hpc_io_scheduler.pipeline import main_train


if __name__ == "__main__":
    main_train()
