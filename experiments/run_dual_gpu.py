"""Use both Kaggle T4 GPUs by running disjoint condition shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    config = json.loads(args.config.read_text(encoding="utf-8"))
    condition_count = len(config.get("conditions", [])) or (
        len(config.get("models", []))
        * len(config.get("precisions", []))
        * len(config.get("datasets", []))
    )
    gpu_count = min(2, torch.cuda.device_count(), max(1, condition_count))
    if gpu_count < 1:
        raise RuntimeError("No CUDA GPU detected")

    processes: list[subprocess.Popen] = []
    for shard in range(gpu_count):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(shard)
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("run_panel.py")),
            "--config",
            str(args.config),
            "--output-dir",
            str(args.output_dir),
            "--shard-index",
            str(shard),
            "--num-shards",
            str(gpu_count),
        ]
        processes.append(subprocess.Popen(command, env=env))

    return_codes = [process.wait() for process in processes]
    if any(code != 0 for code in return_codes):
        raise SystemExit(f"GPU shard failure(s): {return_codes}")

    if gpu_count == 1:
        # run_panel already wrote summary.json in single-shard mode.
        return

    merged: list[dict] = []
    for shard in range(gpu_count):
        shard_path = args.output_dir / f"summary_shard_{shard}.json"
        merged.extend(json.loads(shard_path.read_text(encoding="utf-8")))
    merged.sort(key=lambda item: item["condition"])
    atomic_json(args.output_dir / "summary.json", merged)


if __name__ == "__main__":
    main()
