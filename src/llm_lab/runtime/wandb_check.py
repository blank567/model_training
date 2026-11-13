"""W&B 写入自检；合成指标不代表模型训练效果，不下载权重或启动训练。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from llm_lab.common.io import read_json, write_json
from llm_lab.common.tracking import WANDB_VERSION, start_tracker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--project", default="bilingual-training-lab")
    parser.add_argument("--entity")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    output = root / "runs/wandb_smoke" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:6])
    cfg = {"wandb": True, "wandb_mode": args.mode, "wandb_project": args.project,
           "wandb_entity": args.entity, "wandb_run_name": "sdk-smoke-synthetic-metrics"}
    tracker = start_tracker(cfg, output, root, job_type="sdk_smoke", run_config={"synthetic_metrics_only": True})
    try:
        for step, loss in enumerate((3.0, 2.5, 2.0), 1):
            tracker.log({"smoke/loss": loss, "smoke/learning_rate": 0.001}, step)
    finally:
        tracker.finish()
    state = read_json(output / "wandb_run.json")
    event_files = sorted(str(path) for path in output.rglob("*.wandb"))
    report = {"status": "passed" if not tracker.failed and event_files else "failed", "mode": args.mode,
              "sdk_version": WANDB_VERSION, "run_id": state["id"], "url": state["url"],
              "output_dir": str(output), "event_files": event_files, "synthetic_metrics_only": True}
    write_json(output / "smoke_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
