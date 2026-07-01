"""Serve CLI: load bundle, expose POST /decide (job features -> decision)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hpc_io_scheduler.config import Config
from hpc_io_scheduler.pipeline import cfg_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", default="artifacts")
    ap.add_argument("--config", default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    print(f"Loading bundle from {args.artifact_dir} on device={cfg_device()}")
    # Wire FastAPI / Flask here. Skeleton intentionally minimal.
    print(json.dumps({
        "status": "skeleton",
        "next": "wire FastAPI app with POST /decide endpoint",
        "artifact_dir": args.artifact_dir,
        "port": args.port,
    }, indent=2))


if __name__ == "__main__":
    main()
