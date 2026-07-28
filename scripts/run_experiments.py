#!/usr/bin/env python
"""跑三组不同数据配置的完整复现实验，并写日志。"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wpdb.experiments import RunConfig, run_one_experiment  # noqa: E402


EXPERIMENTS = [
    # 小网格、少量物体：对齐论文负结果 + 基础三层
    RunConfig(
        run_id="run01_grid5_obj6_seed7",
        seed=7,
        grid_size=5,
        n_objects=6,
        gumbel_epochs=60,
        pretrain_steps=800,
        interaction_steps=400,
    ),
    # 更大网格、仍 6 物体
    RunConfig(
        run_id="run02_grid7_obj6_seed42",
        seed=42,
        grid_size=7,
        n_objects=6,
        gumbel_epochs=60,
        pretrain_steps=1000,
        interaction_steps=500,
    ),
    # 多物体制造符号冲突，突出 DP-Means
    RunConfig(
        run_id="run03_grid7_obj24_seed123",
        seed=123,
        grid_size=7,
        n_objects=24,
        gumbel_epochs=50,
        pretrain_steps=1200,
        interaction_steps=800,
    ),
]


def setup_logger(run_id: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_dir / f"{run_id}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main() -> int:
    results_dir = ROOT / "results"
    log_dir = ROOT / "logs"
    summary = []
    master = logging.getLogger("master")
    master.setLevel(logging.INFO)
    master.handlers.clear()
    mf = logging.FileHandler(log_dir / "MASTER_RUN.md", encoding="utf-8")
    mf.setFormatter(logging.Formatter("%(message)s"))
    master.addHandler(mf)
    master.addHandler(logging.StreamHandler(sys.stdout))

    master.info("# 写保护离散瓶颈 — 三组实验主日志\n")
    master.info(f"工作目录: `{ROOT}`\n")

    for cfg in EXPERIMENTS:
        log = setup_logger(cfg.run_id, log_dir)
        payload = run_one_experiment(cfg, out_dir=results_dir, log=log)
        s = payload["summary"]
        summary.append(
            {
                "run_id": cfg.run_id,
                "seed": cfg.seed,
                "grid_size": cfg.grid_size,
                "n_objects": cfg.n_objects,
                "summary": s,
                "elapsed_sec": payload["elapsed_sec"],
            }
        )
        master.info(f"## {cfg.run_id}\n")
        master.info("```json")
        master.info(json.dumps(summary[-1], ensure_ascii=False, indent=2))
        master.info("```\n")

    summary_path = results_dir / "summary_three_runs.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    master.info(f"\n三组实验汇总: `{summary_path}`\n")
    print(f"\n完成。汇总: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
