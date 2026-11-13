"""受限的数学开发集采样；没有训练更新，不读取最终测试集。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
from llm_lab.common.budget import BudgetGuard
from llm_lab.common.io import read_json, write_json, sha256_file
from llm_lab.tokenization.bpe import load_tokenizer
from llm_lora.inference import load_model
from llm_align.core import MathCorpus, development_subset, generate, score_outputs
from train_sft import training_lock, gpu_idle, require_linux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--checkpoint-type', choices=['lora', 'full'], default='lora')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--per-stratum', type=int, default=8)
    ap.add_argument('--group-size', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    require_linux()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    corpus = MathCorpus(args.manifest, 'validation')
    rows = development_subset(corpus.rows, args.per_stratum)
    settings = {'hourly_rate_cny': 3.03, 'run_limit_cny': 1, 'project_limit_cny': 500,
                'reserve_cny': 100, 'max_runtime_seconds': 600}
    started = time.monotonic()
    with training_lock(ROOT):
        gpu_idle()
        torch.set_num_threads(8)
        torch.manual_seed(4201)
        if args.checkpoint_type == 'full':
            from chat_sft import load_model as load_full
            model, tokenizer, identity = load_full(args.checkpoint, 'cuda', threads=8)
        else:
            model, tokenizer, identity = load_model(args.checkpoint, 'cuda')
        if sha256_file(corpus.tokenizer_path) != '6040c90c045836982a47bc0544942f122d0446bf9ed4d2ac1929befd0d4b86b3':
            raise ValueError('Wrong tokenizer')
        guard = BudgetGuard(settings, str(output), ROOT / 'runs/budget_ledger.json')
        results = {}
        for mode, size in [('greedy', 1), ('sample', args.group_size)]:
            expanded = [row for row in rows for _ in range(size)]
            outputs = {'answers': [], 'texts': [], 'eos': []}
            for start in range(0, len(expanded), args.batch_size):
                batch = expanded[start:start + args.batch_size]
                generated = generate(model, tokenizer, corpus.prompts(tokenizer, batch), 'cuda', sample=mode == 'sample')
                for key in outputs:
                    outputs[key].extend(generated[key])
                print(json.dumps({'mode': mode, 'done': min(start + args.batch_size, len(expanded)), 'total': len(expanded)}), flush=True)
                if guard.should_stop():
                    raise RuntimeError('Calibration exceeded bounded budget')
            summary = score_outputs(expanded, outputs)
            if size > 1:
                mixed = sum(len(set(summary['rewards'][i:i + size])) > 1 for i in range(0, len(expanded), size))
                summary['mixed_reward_groups'] = mixed
                summary['mixed_reward_fraction'] = mixed / len(rows)
            examples = [{'id': row['id'], 'prompt': row['prompt'], 'expected': row['ground_truth'],
                         'completion': text, 'reward': r} for row, text, r in zip(expanded, outputs['texts'], summary.pop('rewards'))]
            write_json(output / (mode + '_examples.json'), examples)
            results[mode] = summary
        report = {'status': 'completed', 'checkpoint_type': args.checkpoint_type,
                  'checkpoint': args.checkpoint, 'checkpoint_sha256': identity['checkpoint_sha256'],
                  'data': corpus.identity, 'development_prompts': len(rows), 'test_used': False,
                  'max_new_tokens': 16, 'sampling_temperature': 1.0, 'results': results,
                  'elapsed_seconds': time.monotonic() - started, 'budget': guard.update()}
        write_json(output / 'report.json', report)
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
