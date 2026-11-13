"""轻量 CLI；数据与后训练入口独立，避免导入时加载大型框架。"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Single-GPU bilingual training laboratory")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("pretrain")
    train.add_argument("--config", required=True)
    train.add_argument("--resume")
    train.add_argument("--hourly-rate-cny", type=float)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--output-dir")
    train.add_argument("--validate-only", action="store_true")
    for command in ("evaluate", "generate"):
        sub = commands.add_parser(command)
        sub.add_argument("--checkpoint", required=True)
        sub.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
        if command == "evaluate":
            sub.add_argument("--split", choices=["validation", "test"], default="validation")
            sub.add_argument("--allow-test", action="store_true")
            sub.add_argument("--output", required=True)
        else:
            sub.add_argument("--prompt", required=True)
            sub.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args(argv)
    if args.command == "pretrain":
        from .config import load_config
        config = load_config(args.config)
        if args.hourly_rate_cny is not None:
            config.budget["hourly_rate_cny"] = args.hourly_rate_cny
        if args.max_steps is not None:
            config.train.max_steps = args.max_steps
            config.train.warmup_steps = min(config.train.warmup_steps, args.max_steps)
        if args.output_dir:
            config.output_dir = str((Path(config.project_root) / args.output_dir).resolve())
        config.validate()
        if args.validate_only:
            print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
            return
        from .engine.pretrain import run_pretrain
        print(json.dumps(run_pretrain(config, args.resume), ensure_ascii=False, indent=2))
        return
    import torch
    from .common.io import sha256_file, write_json
    from .config import ExperimentConfig, ModelConfig, TrainConfig
    from .engine.checkpoint import load_checkpoint
    from .engine.corpus import BinaryCorpus
    from .engine.pretrain import evaluate_model
    from .models import Decoder
    checkpoint = load_checkpoint(args.checkpoint)
    raw = checkpoint["config"]
    config = ExperimentConfig(raw["data_manifest"], raw["output_dir"], ModelConfig(**raw["model"]), TrainConfig(**raw["train"]), raw["budget"], raw.get("project_root", "."))
    if sha256_file(config.data_manifest) != checkpoint["identity"]["manifest_sha256"]:
        raise ValueError("Evaluation/generation manifest differs from the checkpoint's original dataset")
    config.train = replace(config.train, device=args.device, dtype="fp32" if args.device == "cpu" else config.train.dtype)
    model = Decoder(config.model).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if args.command == "evaluate":
        if args.split == "test" and not args.allow_test:
            raise ValueError("Final test requires --allow-test; do not use test metrics for model selection")
        corpus = BinaryCorpus(config.data_manifest, args.split, config.train.seq_len)
        result = {"split": args.split, "checkpoint": str(Path(args.checkpoint).resolve()),
                  "manifest_sha256": checkpoint["identity"]["manifest_sha256"],
                  "tokenizer_sha256": checkpoint["identity"]["tokenizer_sha256"],
                  "metrics": evaluate_model(model, corpus, config)}
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        from tokenizers import Tokenizer
        manifest = BinaryCorpus(config.data_manifest, "validation", config.train.seq_len)
        tokenizer = Tokenizer.from_file(str(Path(config.data_manifest).resolve().parent / manifest.manifest["tokenizer"]["path"]))
        ids = tokenizer.encode(args.prompt).ids or [manifest.manifest["tokenizer"]["eos_id"]]
        with torch.no_grad():
            for _ in range(args.max_new_tokens):
                # 超过上下文后重新计算窗口；首版不引入 KV cache 的恢复复杂度。
                x = torch.tensor([ids[-config.model.max_seq_len:]], dtype=torch.long, device=args.device)
                next_id = int(model(x).logits[0, -1].argmax())
                ids.append(next_id)
                if next_id == manifest.manifest["tokenizer"]["eos_id"]:
                    break
        print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()

