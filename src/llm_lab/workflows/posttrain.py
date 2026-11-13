"""薄 CLI：配置校验不加载 HF 模型，训练只允许远程 CUDA 单卡。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .posttrain_contracts import *  # 兼容公共校验接口；全部为标准库。
from .posttrain_runtime import configure_environment, load_saved_model, run_training

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", nargs="?", const="latest", help="完整 checkpoint 路径；省略值时选择 output_dir 最新 checkpoint")
    parser.add_argument("--validate-only", action="store_true", help="只用标准库校验配置/数据/恢复，不加载模型、不创建输出")
    args = parser.parse_args(argv)
    try:
        run = validate_config(args.config, args.resume)
        if args.validate_only:
            print(json.dumps({"valid": True, "stage": run.config["stage"], "method": run.config["method"],
                              "train_examples": len(run.train_rows), "validation_examples": len(run.validation_rows),
                              "identity": run.identity, "resume": str(run.resume_path) if run.resume_path else None,
                              "note": "未加载 HF/tokenizer；Token长度、显存、吞吐和训练效果尚未验证。"}, ensure_ascii=False, indent=2))
        else:
            run_training(run)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(2, f"错误: {exc}\n")


if __name__ == "__main__":
    main()
