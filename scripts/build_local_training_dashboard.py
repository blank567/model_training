"""从已保存的真实训练日志和固定测试结果，生成离线中文图表报告；不训练、不联网。"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import warnings
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib_report'))
os.environ.setdefault('TEMP', 'E:/codex/tmp')
os.environ.setdefault('TMP', 'E:/codex/tmp')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, PercentFormatter

COLORS = ['#2563eb', '#ea580c', '#059669', '#9333ea', '#dc2626', '#0891b2']
METHOD_COLORS = {'ppo': '#2563eb', 'grpo': '#059669', 'dpo': '#c24140'}
ARMS = ['old_matched', 'new_matched', 'old_full_50k']
ARM_NAMES = ['旧匹配 25k', '新匹配 25k', '旧全量 50k']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def num(value, default=None):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError):
        return default


def flatten(record, prefix=''):
    result = {}
    for key, value in record.items():
        full = prefix + key
        if isinstance(value, dict):
            result.update(flatten(value, full + '/'))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result[full] = value
    return result


def csv_file(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def configure_figures():
    # 只读已安装的中文字体；字体缓存与图表都写到 E 盘。
    candidates = [Path('C:/Windows/Fonts/msyh.ttc'), Path('C:/Windows/Fonts/simhei.ttf')]
    for candidate in candidates:
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(candidate)).get_name()
            break
    else:
        raise RuntimeError('No installed Chinese font; cannot produce readable Chinese figures')
    plt.rcParams.update({'axes.unicode_minus': False, 'axes.spines.top': False,
        'axes.spines.right': False, 'font.size': 11, 'axes.titlesize': 12,
        'axes.titlepad': 12, 'figure.dpi': 130, 'savefig.dpi': 175,
        'axes.labelcolor': '#334155', 'text.color': '#17283e', 'axes.edgecolor': '#cbd5e1'})


class Dashboard:
    def __init__(self, source, output):
        self.source, self.output = source.resolve(), output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.figures = self.output / 'figures'
        self.tables = self.output / 'tables'
        self.figures.mkdir(exist_ok=True)
        self.tables.mkdir(exist_ok=True)
        self.source_tables = self.source / 'report/tables'
        self.inventory = read_csv(self.source_tables / '01_model_inventory.csv')
        self.models = {row['key']: row for row in self.inventory}
        self.generic = {row['key']: row for row in read_csv(self.source_tables / '11_instruction_1000_all_models.csv')}
        self.reading = {row['key']: row for row in read_csv(self.source_tables / '03_reading_all_checkpoints.csv')}
        self.cross = {row['key']: row for row in read_csv(self.source_tables / '17_cross_domain_instruction_loss.csv')}
        self.primary = {row['key']: row for row in read_csv(self.source_tables / '02_reading_primary_comparisons.csv')}
        self.logs, self.configs, self.sources, self.series_stats = {}, {}, {}, []
        self.figure_info, self.sections = [], []
        self.nonfinite = 0
        raw_export = []
        for key, model in self.models.items():
            folder = self.source / 'sources_local/training' / Path(model['checkpoint']).parent.name
            path = folder / 'metrics.jsonl'
            records = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
            self.logs[key] = []
            self.configs[key] = read_json(folder / 'config.json')
            self.sources[str(path)] = digest(path)
            self.sources[str(folder / 'config.json')] = digest(folder / 'config.json')
            for index, record in enumerate(records):
                flat = flatten(record)
                if 'optimizer_step' not in flat and 'step' in flat:
                    flat['optimizer_step'] = flat['step']
                self.nonfinite += sum(not math.isfinite(v) for v in flat.values())
                self.logs[key].append(flat)
                raw_export.append({'model': key, 'wandb_run_id': model['wandb_run_id'], 'record_index': index,
                    'event': record.get('event', ''), **flat})
        csv_file(self.tables / '原始训练曲线_25组.csv', raw_export)
        for path in self.source_tables.glob('*.csv'):
            if path.name.startswith(('01_', '02_', '03_', '04_', '05_', '09_', '10_', '11_', '12_', '13_', '14_', '17_', '18_', '19_', '20_')):
                shutil.copyfile(path, self.tables / path.name)
                self.sources[str(path)] = digest(path)
        self.created = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')

    def label(self, key):
        row = self.models.get(key, {})
        if key == 'pretrained_base':
            return '预训练 Base 300M'
        if key == 'reading_sft':
            return '阅读 SFT 50k'
        if key == 'sft100k_full':
            return 'Full SFT 100k / 历史2轮'
        if key == 'lora100k_r16_historical':
            return 'LoRA r16 / 历史2轮'
        if key.startswith('sft20k'):
            return f"LR={float(row['learning_rate']):g} / B{row['effective_batch']} / {float(row['actual_epochs']):g}轮"
        if key.startswith('lora100k'):
            return f"r{row['adapter_rank']} / α{row['adapter_alpha']} / d={row['adapter_dropout'] or '0'} / 1轮"
        for method in METHOD_COLORS:
            if key.startswith(method + '_'):
                arm = next(arm for arm in reversed(ARMS) if key.startswith(method + '_' + arm))
                return method.upper() + ' ' + ARM_NAMES[ARMS.index(arm)].replace(' 25k', '').replace(' 50k', '')
        return key

    def values(self, key, metric, x='optimizer_step'):
        # 同一横坐标的重记保留最后一次，原始 CSV 仍保留所有事件；不平滑、不裁剪离群点。
        points = {}
        duplicates = 0
        for record in self.logs[key]:
            if metric not in record or x not in record:
                continue
            xv, yv = num(record[x]), num(record[metric], float('nan'))
            if xv is None:
                continue
            duplicates += int(xv in points)
            points[xv] = yv
        ordered = sorted(points.items())
        return np.asarray([p[0] for p in ordered]), np.asarray([p[1] for p in ordered]), duplicates

    def save(self, fig, name, title, panels, caption):
        fig.savefig(self.figures / name, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        self.figure_info.append({'file': name, 'title': title, 'panels': panels, 'caption': caption,
                                 'sha256': digest(self.figures / name)})
        return name

    def curve_grid(self, name, title, keys, metrics, x='optimizer_step', ncols=2, caption=''):
        nrows = math.ceil(len(metrics) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, 4.0 * nrows), squeeze=False)
        fig.suptitle(title, fontsize=17, fontweight='bold', y=1.015)
        for index, (metric, metric_title, scale) in enumerate(metrics):
            ax = axes.flat[index]
            count = 0
            for ki, key in enumerate(keys):
                xv, yv, duplicates = self.values(key, metric, x)
                if not len(xv):
                    continue
                xv = xv / (1e9 if x == 'tokens_seen' else 1000 if x == 'examples_seen' else 1)
                ax.plot(xv, yv * scale, label=self.label(key), color=COLORS[ki], linewidth=1.3,
                        marker='o' if len(xv) < 35 else None, markersize=3, alpha=.9)
                finite = yv[np.isfinite(yv)]
                if len(finite):
                    k = max(1, math.ceil(len(finite) * .1))
                    self.series_stats.append({'figure': name, 'model': key, 'metric': metric, 'x_axis': x,
                        'points': len(xv), 'duplicate_x_last_record_used': duplicates,
                        'first': float(finite[0]), 'last': float(finite[-1]), 'minimum': float(np.min(finite)),
                        'early_10pct_median': float(np.median(finite[:k])), 'late_10pct_median': float(np.median(finite[-k:]))})
                count += 1
            ax.set_title(metric_title, loc='left')
            ax.set_xlabel({'examples_seen': '累计见过的训练样本（千条）', 'tokens_seen': '累计训练 Token（十亿）', 'optimizer_step': 'Optimizer step'}[x])
            ax.grid(alpha=.16)
            if scale == 100:
                ax.set_ylabel('%')
            if count:
                ax.legend(fontsize=9, loc='best', framealpha=.9)
            else:
                ax.text(.5, .5, '该指标未记录，不补零、不推测', ha='center', va='center', transform=ax.transAxes)
        for ax in list(axes.flat)[len(metrics):]:
            ax.set_visible(False)
        fig.tight_layout(pad=1.8)
        return self.save(fig, name, title, len(metrics), caption)

    def flow(self):
        fig, ax = plt.subplots(figsize=(16, 6.0))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 6)
        ax.axis('off')
        def box(x, y, w, h, text, color='#eaf1ff'):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.13,rounding_size=.12',
                facecolor=color, edgecolor='#adc1d8', linewidth=1.2))
            ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=12, linespacing=1.7)
        def arrow(a, b):
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=17, color='#526981', linewidth=1.5))
        box(.3, 2.3, 2.5, 2, '自研 Decoder-only\n303.35M 参数\n约 30 亿 Token\n24层 · GQA · RoPE')
        box(3.5, 3.7, 3.3, 1.55, '通用 Full SFT\n20k 控制变量 → 100k\n历史100k为2轮')
        box(3.5, .7, 3.3, 1.6, 'LoRA SFT（独立分支）\nr8 / r32；dropout对照\n新实验均为1轮', '#e8f6f0')
        box(7.6, 3.7, 3.2, 1.55, '阅读专项 SFT 50k\n由通用 Full SFT 继续\n1轮；作为共同RL起点')
        box(11.6, 4.4, 3.7, 1, 'PPO：每批内部更新2次\ncritic + GAE + clipping', '#eaf1ff')
        box(11.6, 2.65, 3.7, 1, 'GRPO：每题采样4个回答\n组内相对优势 + KL', '#e8f6f0')
        box(11.6, .9, 3.7, 1, 'DPO：离线 chosen/rejected\n偏好对过滤后再训练', '#fff0e6')
        arrow((2.95, 3.65), (3.32, 4.3));arrow((2.95, 2.7), (3.32, 1.55))
        arrow((6.95, 4.5), (7.42, 4.5))
        for y in [4.9, 3.15, 1.4]:
            arrow((10.96, 4.4), (11.42, y))
        ax.text(8, .1, '各 RL 分支从同一阅读 SFT 独立初始化；旧匹配 / 新匹配 / 旧全量，每组只遍历一轮训练题',
                ha='center', fontsize=11, color='#526981')
        return self.save(fig, '01_实验流程.png', '实验流程与模型关系', 1, 'LoRA 没有接到 RL 起点；RL 共同起点是全参数阅读专项 SFT。')

    def training_figures(self):
        self.flow()
        self.curve_grid('02_300M预训练.png', '300M 预训练：从语言建模到双语表示', ['pretrained_base'],
            [('train_loss', '训练 CE ↓（记录批次）', 1), ('validation/macro_loss', '验证 Macro CE ↓', 1),
             ('validation/zh/loss', '中文验证 CE ↓', 1), ('validation/en/loss', '英文验证 CE ↓', 1)], x='tokens_seen',
             caption='只有本地完整的 300M/3B 日志；不把缺少完整本地日志的 100M 画成推测曲线。')
        lrkeys = [f'sft20k_lr{lr}_mb8_ga4_b32_ep1' for lr in ['1e-5', '2e-5', '5e-5', '1e-4']]
        self.curve_grid('03_SFT学习率曲线.png', 'SFT 学习率：任务收敛与原有能力的取舍', lrkeys,
            [('train_loss', '训练 CE ↓', 1), ('validation/macro_loss', '验证 Macro CE ↓', 1),
             ('retention_validation/macro_loss', '原有能力验证 CE ↓', 1), ('learning_rate', '真实学习率调度', 1)],
             caption='同一 Base、约2万条数据、有效 Batch32、一轮。原有能力验证 CE 越高，表示在该保留集合上表现变差。')
        batchkeys = [f'sft20k_lr5e-5_mb8_ga{ga}_b{b}_ep1' for ga, b in [(1, 8), (2, 16), (4, 32)]]
        self.curve_grid('04_SFT_Batch曲线.png', 'SFT Batch：按同一轮见过的数据量对齐', batchkeys,
            [('train_loss', '训练 CE ↓', 1), ('validation/macro_loss', '验证 Macro CE ↓', 1),
             ('grad_norm_before_clip', '裁剪前梯度范数', 1), ('supervised_tokens_per_second', '监督 Token 吞吐', 1)], x='examples_seen',
             caption='LR=5e-5；有效 Batch=micro batch×梯度累积。相同一轮数据，较小 Batch 会产生更多更新。')
        epkeys = ['sft20k_lr5e-5_mb8_ga4_b32_ep1', 'sft20k_lr5e-5_mb8_ga4_b32_ep3_stop2', 'sft20k_lr5e-5_mb8_ga4_b32_ep3_stop3']
        self.curve_grid('05_历史训练轮数.png', '历史轮数对照：调度和 best 的位置也要一起看', epkeys,
            [('validation/macro_loss', '验证 Macro CE ↓', 1), ('retention_validation/macro_loss', '原有能力验证 CE ↓', 1)], x='examples_seen',
             caption='仅回顾历史实验；两轮停止组共享三轮调度，和真正的一轮调度不同，不能仅归因于 epochs。')
        self.curve_grid('06_Full与历史LoRA.png', 'Full SFT 与 LoRA r16：同数据、同两轮的历史比较', ['sft100k_full', 'lora100k_r16_historical'],
            [('train_loss', '训练 CE ↓', 1), ('validation/macro_loss', '验证 Macro CE ↓', 1)], x='examples_seen',
             caption='二者同 Base/100k 数据/两轮，学习率分别是5e-5和2e-4；属于已选配置的整体比较。')
        rankkeys = ['lora100k_r8_ld0', 'lora100k_r32_ld0']
        self.curve_grid('07_LoRA_Rank曲线.png', 'LoRA Rank：保持 alpha/r=2，比较更新容量', rankkeys,
            [('train_loss', '训练 CE ↓', 1), ('development_validation/macro_loss', '独立开发集 Macro CE ↓', 1),
             ('retention_validation/macro_loss', '原有能力验证 CE ↓', 1), ('supervised_tokens_per_second', '监督 Token 吞吐', 1)],
             caption='同一100k数据、LR=2e-4、dropout=0、有效Batch32、一轮；r8/alpha16对比r32/alpha64。')
        dropoutkeys = ['lora100k_r32_ld0', 'lora100k_r32_ld0.05', 'lora100k_r32_ld0.1']
        self.curve_grid('08_LoRA_Dropout曲线.png', 'LoRA Dropout：小幅差异也要结合固定测试复核', dropoutkeys,
            [('train_loss', '训练 CE ↓', 1), ('development_validation/macro_loss', '独立开发集 Macro CE ↓', 1),
             ('retention_validation/macro_loss', '原有能力验证 CE ↓', 1), ('supervised_tokens_per_second', '监督 Token 吞吐', 1)],
             caption='固定r32/alpha64和其余训练参数，一轮。开发集原选中的r32/dropout0保持不变。')
        self.curve_grid('09_阅读专项SFT.png', '阅读专项 SFT：建立短回答格式与抽取能力', ['reading_sft'],
            [('train_loss', '训练 CE ↓', 1), ('validation/macro_loss', '阅读验证 Macro CE ↓', 1),
             ('validation/zh/loss', '中文验证 CE ↓', 1), ('validation/en/loss', '英文验证 CE ↓', 1)], x='examples_seen',
             caption='50k训练样本，一轮；这里的CE不能与通用指令数据上的CE直接比较绝对高低。')
        for method, index in [('ppo', 10), ('grpo', 11), ('dpo', 12)]:
            keys = [f'{method}_{arm}_best' for arm in ARMS]
            metrics = [('loss', method.upper() + ' 自身训练目标 Loss', 1),
                       ('dpo_margin' if method == 'dpo' else 'reward_mean', '偏好相对间隔' if method == 'dpo' else '训练采样平均奖励', 1),
                       ('validation/macro_exact_match', '256题开发集 Macro EM ↑', 100), ('validation/macro_f1', '256题开发集自定义 F1 ↑', 100)]
            if method != 'dpo':
                metrics += [('kl', '采样 KL 估计（各算法定义不同）', 1), ('clip_fraction', '裁剪比例', 100)]
            self.curve_grid(f'{index:02d}_{method.upper()}训练.png', method.upper() + '：旧匹配 / 新匹配 / 旧全量', keys, metrics,
                x='examples_seen', caption='训练 Loss 只在同一算法内部解释。EM/F1 使用共同的256题开发集；最终测试另列。')
        csv_file(self.tables / '逐曲线统计.csv', self.series_stats)

    def final_figures(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8.7))
        lrkeys = [f'sft20k_lr{lr}_mb8_ga4_b32_ep1' for lr in ['1e-5', '2e-5', '5e-5', '1e-4']]
        lrs = [num(self.models[k]['learning_rate']) for k in lrkeys]
        axes[0, 0].plot(lrs, [num(self.generic[k]['macro_ce']) for k in lrkeys], 'o-', color=COLORS[0])
        axes[0, 0].set(xscale='log', title='学习率 vs 指令测试 CE ↓（1000题）', xlabel='学习率', ylabel='Macro CE')
        axes[0, 1].plot(lrs, [num(self.models[k]['retention_macro_loss']) for k in lrkeys], 'o-', color=COLORS[1])
        axes[0, 1].set(xscale='log', title='学习率 vs 原有能力验证 CE ↓', xlabel='学习率', ylabel='保留集合 Macro CE')
        for ax in axes[0]:
            ax.set_xticks(lrs, ['1e-5', '2e-5', '5e-5', '1e-4'])
        batchkeys = [f'sft20k_lr5e-5_mb8_ga{ga}_b{b}_ep1' for ga, b in [(1, 8), (2, 16), (4, 32)]]
        axes[1, 0].plot([8, 16, 32], [num(self.generic[k]['macro_ce']) for k in batchkeys], 'o-', color=COLORS[2])
        axes[1, 0].set(title='有效 Batch vs 指令测试 CE ↓', xlabel='有效 Batch', ylabel='Macro CE', xticks=[8, 16, 32])
        durations = [num(self.models[k]['recorded_compute_seconds_all_sessions']) / 60 for k in batchkeys]
        axes[1, 1].bar(['B8', 'B16', 'B32'], durations, color=COLORS[:3])
        axes[1, 1].set(title='相同一轮数据的记录耗时 ↓', ylabel='分钟（含训练内验证）')
        for ax in axes.flat:
            ax.grid(alpha=.15)
        for b, value in zip(axes[1, 1].patches, durations):
            axes[1, 1].text(b.get_x() + b.get_width()/2, value+.05, f'{value:.2f}', ha='center', fontsize=11)
        fig.suptitle('SFT 控制变量：任务效果、能力保留与成本', fontsize=17, y=1.02)
        fig.tight_layout(pad=1.7)
        self.save(fig, '13_SFT最终对比.png', 'SFT 最终对比', 4, '测试用于复核，不据此改写原有选择；学习率和Batch两个对照分别保持其余变量。')

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
        groups = [['lora100k_r8_ld0', 'lora100k_r32_ld0'], ['lora100k_r32_ld0', 'lora100k_r32_ld0.05', 'lora100k_r32_ld0.1'], ['sft100k_full', 'lora100k_r16_historical']]
        labels = [['r8/α16', 'r32/α64'], ['d=0', 'd=0.05', 'd=0.1'], ['Full\n历史2轮', 'LoRA r16\n历史2轮']]
        for ax, keys, ticks, title in zip(axes, groups, labels, ['Rank对比（一轮）', 'Dropout对比（一轮）', 'Full与LoRA（相同两轮）']):
            values = [num(self.generic[k]['macro_ce']) for k in keys]
            bars = ax.bar(ticks, values, color=COLORS[:len(keys)], width=.65)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, value+.012, f'{value:.4f}', ha='center', fontsize=11)
            ax.set_ylim(0, 2.35)
            ax.set_title(title)
            ax.set_ylabel('指令测试 Macro CE ↓（1000题）')
            ax.grid(axis='y', alpha=.15)
        fig.suptitle('LoRA 固定测试：Rank、Dropout及历史Full对照', fontsize=17, y=1.03)
        fig.tight_layout(pad=1.7)
        self.save(fig, '14_LoRA最终对比.png', 'LoRA 最终对比', 3, '柱状图从0开始；dropout差异较小，具体区间另见统计表。历史两轮与新一轮不可视为同预算。')

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        keys = ['sft100k_full', 'lora100k_r16_historical', 'lora100k_r8_ld0', 'lora100k_r32_ld0']
        ticks = ['Full\n2轮', 'LoRA r16\n2轮', 'LoRA r8\n1轮', 'LoRA r32\n1轮']
        parameters = [num(self.models[k]['trainable_parameters'])/1e6 for k in keys]
        times = [num(self.models[k]['recorded_compute_seconds_all_sessions'])/60 for k in keys]
        for ax, values, title, ylabel in [(axes[0], parameters, '实际可训练参数', '百万参数'), (axes[1], times, '记录耗时（轮数已标明）', '分钟')]:
            bars = ax.bar(ticks, values, color=COLORS[:4]);ax.set(title=title, ylabel=ylabel)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, value+max(values)*.015, f'{value:.2f}', ha='center')
            ax.margins(y=.15);ax.grid(axis='y', alpha=.15)
        fig.tight_layout(pad=2)
        self.save(fig, '15_参数与成本.png', '参数量与实际记录成本', 2, '可训练参数少不保证每步更快；时间包含训练内验证，且这四项并非全部等轮数。')

        fig, axes = plt.subplots(1, 2, figsize=(14.5, 6))
        x = np.arange(3);width=.23
        for ax, metric, title in [(axes[0], 'macro_em', '严格 Macro EM ↑'), (axes[1], 'macro_f1', '自定义答案 Macro F1 ↑')]:
            for i, method in enumerate(METHOD_COLORS):
                values = [num(self.reading[f'{method}_{arm}_best'][metric])*100 for arm in ARMS]
                bars = ax.bar(x+(i-1)*width, values, width, label=method.upper(), color=METHOD_COLORS[method])
                ax.bar_label(bars, fmt='%.2f', fontsize=10, padding=3)
            baseline = num(self.reading['reading_sft'][metric])*100
            ax.axhline(baseline, color='#475569', linestyle='--', label=f'阅读SFT {baseline:.2f}%')
            ax.set_xticks(x, ARM_NAMES);ax.set_ylabel('%');ax.set_title(title);ax.set_ylim(0, 68 if metric=='macro_f1' else 50)
            ax.legend(fontsize=9, loc='upper left', ncol=2);ax.grid(axis='y', alpha=.15)
        fig.suptitle('统一阅读测试 2000题：按原开发集规则选出的 best', fontsize=17, y=1.02)
        fig.text(.5, .008, 'DPO偏好对实际为20,462 / 20,462 / 40,871；两个匹配组best=step0，即未做DPO更新的起点',
                 ha='center', fontsize=10, color='#526981')
        fig.tight_layout(pad=1.8, rect=(0,.035,1,1))
        self.save(fig, '16_RL最终对比.png', 'RL 固定阅读测试比较', 2, '各方法共用阅读专项SFT起点；DPO两个匹配组的best是step0，数值等于未做DPO的起点。')

        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
        for ax, method in zip(axes, METHOD_COLORS):
            rows = [next(r for r in read_csv(self.source_tables/'10_best_vs_last.csv') if r['method']==method and r['arm']==arm) for arm in ARMS]
            x=np.arange(3)
            for shift, column, color, label in [(-.19,'best_em',METHOD_COLORS[method],'best'),(.19,'last_em','#94a3b8','last')]:
                bars=ax.bar(x+shift,[num(row[column])*100 for row in rows],.36,label=label,color=color)
                ax.bar_label(bars,fmt='%.2f',fontsize=9,padding=3)
            ax.axhline(36,color='#475569',linestyle='--',linewidth=1)
            ax.set_xticks(x,['旧匹配','新匹配','旧全量']);ax.set_title(method.upper());ax.set_ylim(0,50)
            ax.legend(fontsize=10);ax.grid(axis='y',alpha=.15)
        axes[0].set_ylabel('2000题 Macro EM（%）')
        fig.suptitle('best 与 last：末期变化不能倒过来改写开发集选择',fontsize=17,y=1.03)
        fig.tight_layout(pad=1.8)
        self.save(fig,'17_best与last.png','best 与 last',3,'GRPO旧全量last为41.20%，但固定best仍报告40.40%；DPO新匹配last降至19.75%。')

        fig, ax = plt.subplots(figsize=(13,7))
        keys=[f'{method}_{arm}_best' for method in METHOD_COLORS for arm in ARMS]
        for i,key in enumerate(keys):
            row=self.primary[key];v=num(row['delta_em'])*100;lo=num(row['em_ci95_low'])*100;hi=num(row['em_ci95_high'])*100
            color=METHOD_COLORS[key.split('_')[0]]
            ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4,color=color)
            ax.text(hi+.25,i,f'{v:+.2f} pp [{lo:+.2f}, {hi:+.2f}]',va='center',fontsize=10)
        ax.set_yticks(range(len(keys)),[self.label(k) for k in keys]);ax.invert_yaxis();ax.axvline(0,color='#64748b',linestyle='--')
        ax.set_xlabel('相对阅读SFT的 Macro EM 变化（百分点）；按文章聚类的95%区间')
        ax.set_xlim(-6.5,10);ax.grid(axis='x',alpha=.15);fig.tight_layout()
        self.save(fig,'18_效果差异与区间.png','效果差异与不确定性',1,'区间只反映这批固定模型上的测试文章抽样不确定性，不含训练种子或预训练变化。')

        fig, axes=plt.subplots(1,2,figsize=(14,5.6))
        keys=['reading_sft','ppo_old_full_50k_best','grpo_old_full_50k_best','dpo_old_full_50k_best'];x=np.arange(4)
        for i,(field,label,color) in enumerate([('zh_em','中文',COLORS[0]),('en_em','英文',COLORS[1])]):
            bars=axes[0].bar(x+(i-.5)*.34,[num(self.reading[k][field])*100 for k in keys],.32,label=label,color=color)
            axes[0].bar_label(bars,fmt='%.1f',fontsize=9,padding=3)
        labels=['阅读SFT','PPO旧全量','GRPO旧全量','DPO旧全量']
        axes[0].set(xticks=x,xticklabels=labels,ylabel='EM（%）',ylim=(0,55),title='语言拆分：各1000题');axes[0].legend();axes[0].grid(axis='y',alpha=.15)
        by_source=read_csv(self.source_tables/'05_reading_by_source.csv');sources=['DRCKnowledgeTeam/DRCD','ymcui/cmrc2018','rajpurkar/squad']
        matrix=np.array([[num(next(r for r in by_source if r['key']==k and r['source']==s)['exact_match'])*100 for s in sources] for k in keys])
        im=axes[1].imshow(matrix,cmap='Blues',vmin=0,vmax=60,aspect='auto')
        axes[1].set_xticks(range(3),['DRCD\n500题','CMRC\n500题','SQuAD\n1000题']);axes[1].set_yticks(range(4),labels);axes[1].set_title('来源拆分：同一冻结测试集')
        for i in range(4):
            for j in range(3):axes[1].text(j,i,f'{matrix[i,j]:.1f}%',ha='center',va='center',color='white' if matrix[i,j]>34 else '#17283e')
        fig.colorbar(im,ax=axes[1],label='EM（%）');fig.tight_layout(pad=2)
        self.save(fig,'19_语言与题源.png','语言和题源拆分',2,'按来源和语言定位收益，不能把单一测试来源上的变化等同于所有阅读任务的变化。')

        fig,ax=plt.subplots(figsize=(12,6))
        for method,color in METHOD_COLORS.items():
            keys=[f'{method}_{arm}_best' for arm in ARMS]
            ax.scatter([num(self.cross[k]['instruction_macro_ce']) for k in keys],[num(self.reading[k]['macro_em'])*100 for k in keys],s=65,label=method.upper(),color=color)
            for key,arm in zip(keys,['旧25k','新25k','旧全量']):
                xv=num(self.cross[key]['instruction_macro_ce']);yv=num(self.reading[key]['macro_em'])*100
                if method=='dpo' and arm!='旧全量':continue
                ax.annotate(arm,(xv,yv),xytext=(5,7 if arm=='旧全量' else -12),textcoords='offset points',fontsize=10,color=color)
        ax.scatter([num(self.cross['reading_sft']['instruction_macro_ce'])],[36],marker='*',s=180,color='#334155',label='阅读SFT / DPO匹配best=step0')
        ax.set(xlabel='通用指令参考答案 Macro CE ↓（1000题）',ylabel='阅读 Macro EM ↑（2000题）',title='专项收益与通用能力：越靠左上越好')
        ax.legend(fontsize=10);ax.grid(alpha=.15);fig.tight_layout()
        self.save(fig,'20_能力取舍.png','专项收益与通用能力',1,'这张图中的CE与EM来自不同任务，不能数值相减；只用于观察两种能力的方向。')

        fig,axes=plt.subplots(1,3,figsize=(16,5.4))
        keys=['pretrained_base','sft100k_full','lora100k_r16_historical','lora100k_r32_ld0']
        labels=['Base','Full\n历史2轮','LoRA r16\n历史2轮','LoRA r32\n1轮']
        for ax,field,title in [(axes[0],'eos','遇EOS停止率 ↑'),(axes[1],'repeat_3gram','重复3-gram比例 ↓'),(axes[2],'rouge_macro_language','ROUGE-L重合 ↑（非准确率）')]:
            values=[num(self.generic[k][field])*100 for k in keys];bars=ax.bar(labels,values,color=COLORS[:4]);ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=10)
            ax.set_title(title);ax.set_ylabel('%');ax.set_ylim(0,110 if field=='eos' else 95 if field=='repeat_3gram' else 45);ax.grid(axis='y',alpha=.15)
        fig.suptitle('通用指令生成质量：会停止、少重复，与回答正确是不同指标',fontsize=16,y=1.03)
        fig.tight_layout(pad=1.8)
        self.save(fig,'21_停止与重复.png','通用指令生成行为',3,'同一1000题、最多256个新Token；这些格式和重合指标不能替代人工正确性评审。')

    def add(self, sid, title, paragraphs, figures=(), tables=(), points=()):
        self.sections.append({'id': sid, 'title': title, 'paragraphs': paragraphs,
                              'figures': list(figures), 'tables': list(tables), 'points': list(points)})

    def content(self):
        for path in [self.source/'report/case_studies.json', self.source/'report/statistical_analysis.json',
                     self.source/'sources_local/plan.json', ROOT/'src/llm_rl/objectives.py',
                     ROOT/'src/llm_rl/engine.py', ROOT/'src/llm_reading_data/verifier.py']:
            self.sources[str(path)]=digest(path)
        f = lambda v, digits=4: '未记录' if num(v) is None else f'{float(v):.{digits}f}'
        p = lambda v: f'{float(v)*100:.2f}%'
        tbl = lambda headers, rows: {'headers': headers, 'rows': rows}
        self.add('overview', '先看结论与数据范围', [
            '**本地可独立阅读。** 本报告使用25组已保存的真实训练日志和既有固定测试结果，提供实验流程图、训练曲线、控制变量比较、效果差异区间和逐项解释。生成报告只使用CPU绘图，不启动训练、推理或上传W&B。',
            '**通用指令任务：** 历史Full SFT 100k在1000题参考答案Macro CE上为1.8440，低于同为历史两轮的LoRA r16的1.9209。新一轮LoRA内部，r32/dropout0优于r8；增加dropout没有改善本次CE。',
            '**阅读任务：** 阅读专项SFT为36.00% Macro EM；固定开发集规则选出的GRPO旧全量best为40.40%，提升4.40个百分点，即2000题中多答对88题。两个匹配组DPO的best都是step0，不能解释为DPO提升。',
            '**覆盖边界：** 本地25组日志不含完整100M/1B轨迹，也不含历史r16的另外两种学习率完整日志，因此不补画这三条缺失曲线。本地报告中的图直接取自日志与结果文件，不依赖云端面板是否成功加载。',
            '**测试题数：** 15个模型的通用指令测试各1000题，22个阅读模型/Checkpoint条目各2000题，共59000个逻辑题次；实际只有3000个唯一测试题，不能把重复评测次数当成独立样本数。另外10个模型各有1000题通用指令参考答案CE，用于能力保留诊断。',
        ], tables=[tbl(['层次','作用','本报告如何使用'],[
            ['训练集','更新参数','训练Loss、奖励、margin、KL、吞吐；不能独立判断最终效果'],
            ['验证/开发集','选配置与Checkpoint','SFT验证CE、LoRA独立开发CE、RL共同256题EM/F1'],
            ['最终指令测试1000题','复核指令表现','参考答案CE、生成EOS、重复和ROUGE-L'],
            ['最终阅读测试2000题','复核阅读表现','中文1000＋英文1000；严格EM和自定义F1'],
        ])], points=['图表曲线不平滑，不忽略离群点；原始CSV保留所有记录事件。',
                     '下面的“更好”均限定在本次模型、数据、指标和训练实现中，不是算法普遍排名。'])

        self.add('pipeline','模型与训练路线：各分支是什么关系',[
            'Base为自研Decoder-only模型：303,350,784个参数，24层，隐藏维度1024，16个注意力头、4个KV头，词表32000，上下文1024。GQA和RoPE等结构由已有配置记录确认。',
            '**SFT是训练目标/数据使用方式，LoRA是参数更新方式。** 这里比较的是“全参数SFT”与“LoRA SFT”，两者都用指令监督和assistant回答Loss Mask；不是两种互斥的指令学习目标。LoRA把更新限制为低秩增量，W′=W+(α/r)BA。',
            '通用Full SFT和LoRA分支都从同一预训练Base开始。阅读专项SFT由通用Full SFT继续训练，再作为PPO、GRPO和DPO三条独立路线的共同起点；没有把LoRA模型作为本次RL起点。',
            'PPO和GRPO属于在线采样再优化；本项目DPO使用离线偏好对。可验证奖励RLVR描述的是奖励来源，不是与PPO/GRPO互斥的第四个优化器。',
        ], figures=['01_实验流程.png'])

        self.add('metrics','先理解指标：Loss、Macro EM、F1各能说明什么',[
            '**Macro EM = (中文EM + 英文EM)/2。** 阅读SFT中文394/1000正确，英文326/1000正确，所以Macro EM=(39.4%+32.6%)/2=36.00%；本次两种语言题数相同，恰好也等于720/2000。',
            'EM先按项目规则做NFKC、大小写折叠、空白、成对引号和部分句尾符号规范化，再匹配任一参考答案别名。它保留小数点、负号及单位等有意义的符号。答案多解释一句、缺一个限定词或别名不完整都可能得0分；EM=0不能直接推出语义完全错误。',
            '项目自定义F1使用中文字符/词片段、英文空白分词的重合，在多个参考答案中取最大值。它能区分部分重合与完全不重合，但不是语义正确率，也不冒充官方SQuAD排行榜实现。',
            'CE在给定参考答案时计算。较低CE表示更容易给参考答案Token高概率，不保证自由生成答案一定正确。不同语料、模板、语言分词分布和Loss Mask会改变CE，跨任务不能直接比较绝对数值。',
        ], tables=[tbl(['指标','优先看什么','常见误读'],[
            ['Train Loss / CE ↓','一段时间的趋势，而非单个batch','有起伏就等于没学好；训练CE最低就最好'],
            ['Validation / Development CE ↓','同一数据与相同计算方式','不同数据集CE可直接排序'],
            ['Retention CE ↓','在原有能力验证集合上的变化','某集合变差就等于丢失全部知识'],
            ['Perplexity ↓','同一Token化和口径下 exp(CE)','把宏平均CE的指数误当成所有口径的宏平均PPL'],
            ['Macro EM / F1 ↑','固定测试、相同模板与生成上限','F1重合就是完整语义正确'],
            ['Reward ↑ / DPO margin ↑','训练目标是否在被优化','奖励或margin上涨必然带来独立测试提升'],
            ['KL / Clip fraction','各自算法内部的更新幅度','跨算法直接比较不同定义的KL数值'],
            ['EOS / 重复率','输出停止和重复行为','会停止、低重复就等于内容准确'],
        ])])

        pre = self.models['pretrained_base']
        token_count=max(row.get('tokens_seen',0) for row in self.logs['pretrained_base'])
        self.add('pretraining','预训练：学习趋势与能力边界',[
            f"这组300M实验记录到{token_count/1e9:.4f}十亿Token，最终更新{pre['final_step']}步，记录的训练与内置验证总时长约{float(pre['recorded_compute_seconds_all_sessions'])/3600:.2f}小时。图中按累计Token对齐，避免把更新步数误当成训练数据量。",
            '训练CE反映当时记录批次的难度，验证CE反映固定验证集合。训练曲线有尖峰时，应结合验证走势、梯度、学习率以及数据批次变化判断；仅凭某次batch变难不能判断模型失效。这里显示全部已记录点，没有额外平滑。',
            '中文和英文验证CE分开观察，用来定位语言方向的收敛与退化；两种语言的Token分布不同，不能仅根据中文CE和英文CE的高低断言哪种语言理解更强。',
            '预训练Base并未直接针对短回答模板或指令结束行为训练。它在当前指令生成测试中重复较高、EOS较少，说明后续对格式和任务的适配很必要，不等同于预训练完全没有形成语言能力。',
        ], figures=['02_300M预训练.png'])

        lrkeys=[f'sft20k_lr{lr}_mb8_ga4_b32_ep1' for lr in ['1e-5','2e-5','5e-5','1e-4']]
        self.add('sft_lr','SFT学习率：更快适应与原有能力保留',[
            '四组均从同一Base、约19991条样本、有效Batch32开始，只训练一轮；这个对照主要改变学习率。图的上排展示任务收敛，下排展示原有能力保留集合与实际学习率调度。',
            '本次1e-4的指令测试CE为2.1095，低于5e-5的2.1213，差0.0118；但保留集合CE也从约3.6014升至3.8516。可以说任务参考答案拟合更好、在这份保留集合上代价更明显，不能只报一个“学习率越大越好”。',
            '训练曲线的抖动大小受batch难度、有效Token量和当前学习率影响。应优先观察固定开发集走势及相同预算的结果。下表最终1000题仅做复核，本报告不据此修改历史选择或重新挑模型。',
        ], figures=['03_SFT学习率曲线.png','13_SFT最终对比.png'], tables=[tbl(['学习率','最终指令CE ↓','保留集合CE ↓','EOS ↑','重复3-gram ↓'],[
            [f(self.models[k]['learning_rate'],5),f(self.generic[k]['macro_ce']),f(self.models[k]['retention_macro_loss']),p(self.generic[k]['eos']),p(self.generic[k]['repeat_3gram'])] for k in lrkeys])])

        bkeys=[f'sft20k_lr5e-5_mb8_ga{ga}_b{b}_ep1' for ga,b in [(1,8),(2,16),(4,32)]]
        self.add('sft_batch','有效Batch：相同一轮数据，不是相同更新步数',[
            '有效Batch=micro batch×gradient accumulation。此处micro batch固定8，累积1/2/4次对应有效Batch8/16/32，学习率固定5e-5。按examples_seen绘图更适合比较同一轮数据的进展。',
            'B8有2499次更新，B16有1250次，B32有625次；这些不是训练轮数不同。一次更新使用的样本数不同，保持同一轮样本数就必然改变更新次数。',
            '本次指令测试CE从B8的2.2277改善到B16的2.1420和B32的2.1213。记录耗时接近，因此这份数据与配方下B32具有更好的效果/成本表现；不能据此推导任何更大Batch都更优。',
            '裁剪前梯度范数大于阈值表示裁剪会发生，不等于裁剪后的梯度仍然越界。日志中的Token吞吐只代表当时实际记录，不把它改写成显卡通用性能。',
        ], figures=['04_SFT_Batch曲线.png'], tables=[tbl(['有效Batch','累积次数','一轮步数','指令CE ↓','记录耗时/分钟'],[
            [self.models[k]['effective_batch'],self.models[k]['gradient_accumulation'],self.models[k]['final_step'],f(self.generic[k]['macro_ce']),f(float(self.models[k]['recorded_compute_seconds_all_sessions'])/60,2)] for k in bkeys])])

        self.add('historical_epochs','历史轮数：区分计划终点、best与调度',[
            '历史一轮、两轮停止和三轮实验都保留原配置。两轮停止组使用三轮总调度，与真正的一轮Warmup/Cosine调度不同，因此这里不是只改epochs的完全纯净对照。',
            '三轮组的训练终点为1875步，但best在1200步，早于第二轮结束。最终测试使用该best，而不是第三轮末的last。不能把“3轮实验的测试分数”简单解释成“训练到第三轮末的权重分数”。',
            '历史两轮组测试CE约2.0924，三轮实验best约2.0975；结合best提前和能力保留变化，可以看到收益递减/过拟合风险信号，但不能单凭这些曲线锁定唯一原因。',
            '**未来约束保持不变：新实验epochs=1，stop_epochs省略或null。** 本报告只回顾历史，不添加第二轮、不改调度，也不启动延长训练。',
        ], figures=['05_历史训练轮数.png'])

        fullkeys=['sft100k_full','lora100k_r16_historical','lora100k_r8_ld0','lora100k_r32_ld0']
        self.add('full_lora','全参数SFT与LoRA：效果、更新容量和真实成本',[
            '历史Full SFT 100k与r16 LoRA使用同一Base、同一100k数据清单，实际均训练两轮；两者采用各自选定学习率5e-5与2e-4。这个比较支持当前配方整体效果的判断，不是只改参数更新方式的所有因素完全相同实验。',
            'Full最终指令CE=1.84399，r16=1.92093；配对差值LoRA−Full为+0.07694，95%区间[+0.06062,+0.09273]，支持本次固定测试上Full在参考答案CE更好。',
            'r16只更新6.98M参数，约为全部303.35M参数的2.30%，但记录耗时76.63分钟，高于Full的56.26分钟。冻结参数会减少可训练状态，不保证本实现每步更快，LoRA额外算子、前向计算和验证成本仍存在。',
            'r8/r32新实验只有一轮，不能用其总耗时与Full历史两轮作等预算速度结论。r32一轮CE=1.96197应作为当前高效适配分支的结果，而不是纯粹的“方法能力上限”比较。',
        ], figures=['06_Full与历史LoRA.png','15_参数与成本.png'], tables=[tbl(['配置','实际轮数','可训练参数/M','指令CE ↓','记录分钟'],[
            [self.label(k),f(self.models[k]['actual_epochs'],0),f(float(self.models[k]['trainable_parameters'])/1e6,3),f(self.generic[k]['macro_ce']),f(float(self.models[k]['recorded_compute_seconds_all_sessions'])/60,2)] for k in fullkeys])])

        lorakeys=['lora100k_r8_ld0','lora100k_r32_ld0','lora100k_r32_ld0.05','lora100k_r32_ld0.1']
        self.add('lora_controls','LoRA控制变量：Rank与Dropout分别看',[
            '**Rank对照：** 固定LR=2e-4、dropout=0、100k数据、有效Batch32、一轮；比较r8/alpha16与r32/alpha64。alpha/r都为2，主要改变可训练增量的容量。r32指令CE比r8降低约0.08897，固定测试的配对区间为[0.08458,0.09364]。',
            '**Dropout对照：** 固定r32/alpha64，比较0、0.05、0.1。测试CE依次1.96197、1.96421、1.96644；0最好，但差异远小于r8到r32的差异。不能因为曲线接近就说完全等价，也不能把很小的CE差异夸大为明显语义能力差距。',
            'dropout0.05的EOS和ROUGE-L等生成行为并非每项都比0差，说明指标存在取舍。本报告保持原独立开发集选择r32/dropout0，不用最终测试重新选配置；也没有多seed复验来估计训练随机性。',
        ], figures=['07_LoRA_Rank曲线.png','08_LoRA_Dropout曲线.png','14_LoRA最终对比.png'], tables=[tbl(['配置','开发集CE ↓','最终指令CE ↓','EOS ↑','重复3-gram ↓'],[
            [self.label(k),f(self.models[k]['validation_best_loss']),f(self.generic[k]['macro_ce']),p(self.generic[k]['eos']),p(self.generic[k]['repeat_3gram'])] for k in lorakeys])])

        self.add('reading_start','阅读专项SFT：为什么先有这一步',[
            '阅读专项使用50k条训练数据，一轮，LR=5e-5、有效Batch32、上下文1024，完成1563次更新。它让通用指令模型适应“根据给定材料，只输出简短答案”的任务和结束格式。',
            '固定阅读测试中，预训练Base为0%，通用Full SFT和所测r32 LoRA均约0.05%，而阅读专项SFT达到36%。这说明任务与短回答格式适配非常关键；不能据此断言通用SFT在所有任务上没有效果。',
            '后续九组RL/偏好实验都从这一个阅读SFT Checkpoint独立开始。共同开发集为256题；最终阅读测试2000题，生成上限64个新Token。开发集与测试集的绝对分数不同属于不同集合结果，不把两种曲线拼到同一训练轨迹。',
        ], figures=['09_阅读专项SFT.png'], tables=[tbl(['起点','阅读EM','阅读F1','含义'],[
            ['预训练Base',p(self.reading['pretrained_base']['macro_em']),p(self.reading['pretrained_base']['macro_f1']),'当前短回答任务/模板不适配'],
            ['通用Full SFT',p(self.reading['sft100k_full']['macro_em']),p(self.reading['sft100k_full']['macro_f1']),'通用任务表现不等于专项阅读表现'],
            ['阅读专项SFT',p(self.reading['reading_sft']['macro_em']),p(self.reading['reading_sft']['macro_f1']),'共同RL起点；720/2000正确'],
        ])])

        rl_rows=[]
        for method in METHOD_COLORS:
            for arm in ARMS:
                key=f'{method}_{arm}_best';row=self.models[key]
                rl_rows.append([self.label(key),row['train_examples'],row['final_step'],row['best_step'],f(float(row['recorded_compute_seconds_all_sessions'])/60,2)])
        self.add('rl_training','RL训练曲线：同样一轮，优化过程并不相同',[
            '**PPO：** 每批采样回答，按答案别名严格匹配得到0/1奖励，加入相对参考策略的KL shaping，使用GAE和独立线性价值头，再做裁剪策略/价值更新。每批内部更新2次，所以25k题有3126次optimizer step；不是把数据训练了两轮。',
            '**GRPO：** 每题采样4个回答，组内奖励减均值并除以标准差得到相对优势，再优化带裁剪与KL项的目标。四个回答奖励都相同，尤其全部为0时，组内任务优势为0；大量这类组会减少有效学习信号。KL正则本身仍可能产生梯度。',
            '**DPO：** 使用离线chosen/rejected偏好对，优化相对于固定参考模型的回答偏好差；损失为−log sigmoid(β×相对偏好margin)。本项目β=0.1。偏好对过滤后，两个匹配组各20462对、旧全量40871对；不能把它们写成完整25k/50k偏好对。',
            '**数据关系：** 旧组来自已有阅读训练题，新匹配组来自独立候选数据；旧组与专项SFT题目有复用。最终测试独立，但“旧组优于新组”不等于证明新语料质量更差，题源、长度、难度和对旧训练题的熟悉度都可能影响结果。',
            '**为何Loss不可跨算法比较：** PPO Loss含策略和价值项，GRPO由组内相对优势和KL构成，DPO是离线偏好分类型目标。它们的符号和尺度不同。PPO日志KL是采样old−reference的估计，可能为负；GRPO使用另一种非负估计，因此本报告分图展示。',
            '本项目的实现选择（如PPO线性价值头、GRPO回答聚合方式、严格答案奖励）也限定了结果外推范围。训练目标被优化，只说明该目标下降；是否得到有效泛化要看相同开发集及独立最终测试。',
        ], figures=['10_PPO训练.png','11_GRPO训练.png','12_DPO训练.png'], tables=[tbl(['运行','训练题/偏好对','最终更新步数','原选best步数','记录分钟'],rl_rows)])

        rows=[]
        for method in METHOD_COLORS:
            for arm in ARMS:
                key=f'{method}_{arm}_best';r=self.reading[key];delta=num(r['macro_em'])-.36
                rows.append([self.label(key),self.models[key]['best_step'],p(r['macro_em']),p(r['macro_f1']),f'{delta*100:+.2f} pp',str(round(num(r['macro_em'])*2000))+'/2000'])
        self.add('rl_final','统一2000题最终测试：本次谁提升，谁退化',[
            '**按原开发集规则选出的Checkpoint：** PPO旧匹配38.80%，GRPO旧匹配39.55%；PPO旧全量39.50%，GRPO旧全量40.40%。新匹配PPO36.20%、GRPO36.50%，比起点只略高。',
            'GRPO旧全量best在本次固定测试上最高，较阅读SFT增加4.40个百分点，95%文章聚类区间约[+3.18,+5.66]个百分点。但相对于PPO的微小领先不能直接上升为普遍算法结论；没有多seed和等计算预算的全面复验。',
            '两个DPO匹配组best=0，恰好等于阅读SFT的36.00%，因为保留的是训练前起点；旧全量DPO best为33.35%，低于起点。这些结果不支持当前DPO配方成功提升阅读任务。',
            '“旧全量”题数比匹配组更多，一轮意味着更多样本、Token和采样成本。题数相同也不意味着答案长度、组采样数、优化步数和算力完全相同，必须把数据与计算预算同时看。',
        ], figures=['16_RL最终对比.png','18_效果差异与区间.png'], tables=[tbl(['模型/数据组','best步数','Macro EM ↑','自定义F1 ↑','相对阅读SFT','严格答对题数'],rows)])

        bl=read_csv(self.source_tables/'10_best_vs_last.csv')
        self.add('best_last','best与last：DPO为什么不能只看训练Loss',[
            'best是训练时根据开发集既定规则选出的权重；last是这一轮结束时的权重。两者不同不一定表示保存出错。最终测试只是复核，不反过来重新定义“best”。',
            'GRPO旧全量last达到41.20%，高于原best的40.40%；这说明开发集选点存在有限样本波动。不能因此直接把last改成默认胜出模型，否则最终测试变成新的调参依据。',
            'DPO旧匹配last降到30.40%，新匹配降到19.75%，旧全量降到31.30%。即使训练Loss下降、偏好margin扩大，也没有转化成这份独立阅读测试的收益。',
            '可以优先审查偏好对的chosen/rejected构造、负例质量、长度偏置、学习率/β和任务匹配度，但当前曲线不能单独证明其中哪一项是原因。规则构造的负例不等于人工比较标注；具体修正需要新的验证证据。',
        ], figures=['17_best与last.png'], tables=[tbl(['模型','best步数','last步数','best EM','last EM','末期变化'],[
            [self.label(r['best']),r['best_step'],r['last_step'],p(r['best_em']),p(r['last_em']),f"{num(r['delta_em_last_minus_best'])*100:+.2f} pp"] for r in bl])])

        self.add('generalization','语言、题源与通用能力：提升发生在哪里',[
            '阅读SFT中文EM39.4%、英文32.6%；GRPO旧全量best分别为43.1%和37.7%，两种语言均有提升，英文提升5.1个百分点、中文3.7个百分点。按DRCD、CMRC、SQuAD拆分可以进一步检查收益是否集中在某个题源。',
            '这份阅读测试的中文由DRCD500题和CMRC500题组成，英文为SQuAD1000题。题目共享文章，统计记录中英文仅20个文章簇，中文210个；不能把全部2000题当作完全独立的2000篇材料。置信区间因此使用按语言分层、按文章聚类的配对bootstrap。',
            '能力取舍图横轴是1000题通用指令参考答案CE，纵轴是2000题阅读EM；越左上越理想。阅读SFT通用CE=2.34065，GRPO旧全量为2.38275，意味着阅读收益伴随这份通用参考答案集合上的小幅退化。两个轴不是同一种指标，不能相减或合成未经定义的总分。',
            '当前置信区间反映固定权重、固定任务上的测试材料抽样不确定性，不包含训练seed、预训练随机性或换一个领域后的变化。因此结论应限定为本次公开阅读理解实验，不扩展成数学推理或所有任务的算法排名。',
        ], figures=['19_语言与题源.png','20_能力取舍.png'])

        self.add('generation','输出会不会停止、会不会重复：必须单独评测',[
            '在同一1000题指令生成测试、最多256个新Token的条件下，Base的EOS停止率仅1.8%，重复3-gram比例约79.99%；Full SFT 100k为86.0%和16.15%。这直接展示后训练对输出格式和重复行为的改善。',
            'LoRA r32/dropout0一轮的EOS约89.1%，重复3-gram约22.17%；历史r16的EOS约88.4%。它们某项格式指标超过Full并不表示内容整体更准确，Full的参考答案CE和ROUGE-L仍有自己的优势。',
            '这里重复3-gram是既有评测器按其Token规则计算的重复比例，ROUGE-L是与参考答案的最长公共子序列重合。它们都不是开放式回答的人工正确率。限制max_new_tokens能保证程序结束，但模型本身何时预测EOS，需要数据、模板和目标共同学会。',
        ], figures=['21_停止与重复.png'])

        cases=read_json(self.source/'report/case_studies.json')
        selected=[]
        for index in [4,9,12]:
            case=cases[index]
            selected.append({'selection':case['selection'],'question':case.get('question',''),
                'context':case.get('context','')[:450], 'references':case.get('references',[]),
                'models':[{k:m.get(k) for k in ['model','answer','completion','text','prediction','generated_text','exact_match','f1'] if k in m} for m in case.get('models',[])]})
        write_json(self.output/'精选样例.json',selected)
        case_tables=[]
        case_paragraphs=['只列既有逐题评测中的实际输入与输出。以下是有目的挑选的解释性样例，不是随机抽样，不能据几个例子推算整体准确率。完整逐题记录仍以原测试报告和本报告表格为准。']
        for case in selected:
            case_rows=[]
            for m in case['models']:
                text=next((m[k] for k in ['answer','completion','prediction','generated_text','text'] if isinstance(m.get(k),str)),None)
                if text is None:
                    raise ValueError('Case prediction text is missing')
                case_rows.append([self.label(m['model']),text,f(m.get('exact_match'),0),f(m.get('f1'))])
            item=tbl(['模型','真实输出','EM','自定义F1'],case_rows)
            item['caption']='问题：'+case['question']
            item['intro']=['材料节选：'+case['context'],'参考答案：'+'；'.join(case['references'])]
            case_tables.append(item)
        self.add('cases','具体样例：改善、退化与严格匹配的边界',case_paragraphs,tables=case_tables,
            points=['GRPO改善样例说明部分题目上的答案定位变准，不代表所有题都改善。',
                    'DPO退化样例帮助定位回答偏移；最终应回到全部2000题的整体统计。',
                    '部分重合样例中EM为0而F1可能较高，说明严格EM与语义评价仍需区分。'])

        self.add('conclusions','研究总结与后续优先级',[
            '**SFT最明确的收益是任务适配与生成行为改善。** 学习率和Batch实验显示效果、原有能力保留与成本之间有取舍；不要只用train loss选模型。',
            '**LoRA的优势在可训练参数和状态规模。** 本次Full在同两轮的通用CE更好；LoRA内部r32相对r8的收益，比轻微dropout变化更明显。新一轮LoRA与历史两轮Full的总成本差异不能直接作为同预算结论。',
            '**当前成功的专项路线是先阅读SFT，再进行受约束的在线优化。** 在这批模型上，PPO和GRPO旧数据组有明确阅读收益，GRPO旧全量best为40.40%。旧数据复用、题源差异和采样预算限制了算法间的因果解释。',
            '**当前DPO配方需要先诊断，不值得因Loss下降而直接扩大训练。** 两组best停在step0，last明显退化；应先检验偏好对和独立验证表现，而不是盲目加数据或训练轮数。',
            '**下一步若继续研究：** 先按目标任务确定主指标和通用能力下限，再预注册新开发/测试边界；固定Base、训练数据与预算做必要的多seed复验。新训练仍只跑一轮，不自动加stop_epochs=2，不重新利用当前最终测试选参数。',
            '**本报告已完成的是离线重画与解释。** 没有新增训练，没有重新跑GPU推理，没有修改原run/config/Checkpoint，也没有把本地最终测试结果上传到W&B。',
        ])

        model_rows=[]
        for key,r in self.models.items():
            model_rows.append([self.label(key),r['wandb_run_id'],r['train_examples'] or '预训练Token',r['actual_epochs'] or '按Token预算',
                r['learning_rate'],r['effective_batch'] or '见配置',r['final_step'],r['best_step'] or '见原summary'])
        self.add('provenance','实验清单、数据来源与重建方式',[
            f"生成时间：{self.created}（UTC+08:00）。训练曲线来源为sources_local/training下25组metrics.jsonl；固定测试来自原完整测试报告的01/02/03/04/05/09/10/11/13/17/19/20号表。所有输入文件路径和SHA256保存在source_manifest.json。",
            '本地交互限于报告导航、目录、搜索实验与放大查看图像，不需要登录、CDN或服务器。便携HTML内嵌全部21张PNG；图像另存于figures，可用于PPT。tables保存原始训练曲线、逐曲线统计及测试汇总CSV。',
            '重复横坐标绘图使用最后一次记录，原始CSV保留全部事件；缺失指标不补零，不平滑，不静默删除异常值。曲线是实际日志记录点，并非每个未记录优化步的重建。',
            '重建命令：E:/conda_envs/llm/python.exe scripts/build_local_training_dashboard.py --source runs/assistant/inference/posttrain_comprehensive_20261104_v3 --output <E盘输出目录>。从同一输入重建不会访问远程GPU。',
        ], tables=[tbl(['实验','W&B run ID','训练样本','实际轮数','学习率','有效Batch','最终步','best步'],model_rows)])

    def render(self):
        from PIL import Image
        figure_lookup={row['file']:row for row in self.figure_info}
        def rich(text):
            value=html.escape(str(text))
            value=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',value)
            return re.sub(r'`(.+?)`',r'<code>\1</code>',value)
        def md_table(table):
            def cell(value):return str(value).replace('|','\\|').replace('\n','<br>')
            rows=['| '+' | '.join(map(cell,table['headers']))+' |','| '+' | '.join(['---']*len(table['headers']))+' |']
            rows += ['| '+' | '.join(map(cell,row))+' |' for row in table['rows']]
            return '\n'.join(rows)+'\n'
        document=[];markdown=['# 300M模型本地图表、比较与研究总结','',f'生成时间：{self.created}。离线报告，所有图取自既有日志和测试结果。','']
        nav=[];used_figures=[]
        for section in self.sections:
            sid,title=section['id'],section['title'];nav.append(f'<a href="#{sid}">{html.escape(title)}</a>')
            document.append(f'<section id="{sid}"><h2>{html.escape(title)}</h2>')
            markdown += [f'## {title}','']
            for paragraph in section['paragraphs']:
                document.append('<p>'+rich(paragraph)+'</p>');markdown += [paragraph,'']
            for filename in section['figures']:
                info=figure_lookup[filename];used_figures.append(filename)
                data=base64.b64encode((self.figures/filename).read_bytes()).decode('ascii')
                with Image.open(self.figures/filename) as im:w,h=im.size
                document.append(f'<figure><img class="zoomable" loading="lazy" width="{w}" height="{h}" src="data:image/png;base64,{data}" alt="{html.escape(info["title"])}"><figcaption><strong>{html.escape(info["title"])}</strong>　{rich(info["caption"])}　<span class="hint">点击放大</span></figcaption></figure>')
                markdown += [f'![{info["title"]}](figures/{filename})','',info['caption'],'']
            for table in section['tables']:
                if table.get('caption'):
                    document.append('<h3>'+rich(table['caption'])+'</h3>');markdown += ['### '+table['caption'],'']
                for paragraph in table.get('intro',[]):
                    document.append('<p>'+rich(paragraph)+'</p>');markdown += [paragraph,'']
                if sid=='provenance':
                    document.append('<label class="search-label">查找实验 <input id="model-search" type="search" placeholder="输入 run ID、r32、学习率等"></label>')
                table_id=' id="inventory-table"' if sid=='provenance' else ''
                document.append(f'<div class="table-wrap"><table{table_id}><thead><tr>'+''.join('<th>'+rich(h)+'</th>' for h in table['headers'])+'</tr></thead><tbody>')
                for row in table['rows']:
                    if len(row)!=len(table['headers']):raise ValueError('Mismatched table columns')
                    document.append('<tr>'+''.join('<td>'+rich(v)+'</td>' for v in row)+'</tr>')
                document.append('</tbody></table></div>');markdown += [md_table(table),'']
            if section['points']:
                document.append('<ul>'+''.join('<li>'+rich(point)+'</li>' for point in section['points'])+'</ul>')
                markdown += ['- '+point for point in section['points']]+['']
            document.append('</section>')
        assert len(used_figures)==len(set(used_figures))==len(self.figure_info)
        chart_panels=sum(row['panels'] for row in self.figure_info)
        page='''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'">
<title>300M模型 · 本地图表与详细研究总结</title>
<style>
:root{color-scheme:light;--ink:#203149;--muted:#5e6c80;--line:#dce4ed;--blue:#174c8f}
*{box-sizing:border-box}html{scroll-behavior:auto;scroll-padding-top:25px}body{margin:0;color:var(--ink);background:#f4f6f9;font:16px/1.85 "Microsoft YaHei", "PingFang SC",sans-serif}
a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}.layout{display:grid;align-items:start;grid-template-columns:245px minmax(0,1fr);max-width:1700px;margin:0 auto}
nav{position:sticky;top:0;height:100vh;overflow:auto;padding:32px 20px;background:#f8fafc;border-right:1px solid var(--line)}nav .brand{font-size:20px;font-weight:700;margin-bottom:6px}nav .caption{font-size:12px;color:var(--muted);line-height:1.6;margin-bottom:20px}nav a{display:block;margin:5px 0;padding:6px 8px;border-radius:4px;font-size:13px;line-height:1.6;color:#3a526b}nav a:hover{background:#e6edf6}
main{min-width:0;padding:40px 44px 80px;background:white}header{padding-bottom:28px;border-bottom:2px solid #244f7d;margin-bottom:32px}header .eyebrow{font-size:12px;letter-spacing:2px;color:#52718e}h1{font-size:32px;line-height:1.45;letter-spacing:-.4px;margin:12px 0 14px}header p{color:var(--muted);margin:5px 0}header .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:25px}.stat{padding:15px;background:#f5f8fc;border:1px solid var(--line);border-radius:4px}.stat b{display:block;font-size:27px;line-height:1.3;color:#173c67}.stat span{font-size:12px;color:var(--muted)}
section{padding:16px 0 32px;border-bottom:1px solid var(--line);margin-bottom:22px;scroll-margin-top:20px}h2{font-size:24px;line-height:1.5;margin:10px 0 20px;color:#193b61}h3{font-size:18px;margin:26px 0 12px}p{margin:15px 0}strong{font-weight:650}code{font:13px/1.7 Consolas,monospace;background:#f1f5f9;padding:2px 5px;border-radius:3px}
figure{margin:25px 0;background:white;border:1px solid var(--line);border-radius:4px;overflow:hidden}figure img{display:block;width:100%;height:auto;cursor:zoom-in}figcaption{padding:12px 16px;background:#f7f9fc;font-size:13px;color:#50647c;line-height:1.8}.hint{white-space:nowrap;color:#174c8f}.table-wrap{overflow-x:auto;margin:22px 0;border:1px solid var(--line);border-radius:4px}table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.7}th,td{padding:11px 13px;text-align:left;vertical-align:top;border-bottom:1px solid #e5ebf1}th{background:#edf3fa;font-weight:650;white-space:nowrap}tr:last-child td{border-bottom:0}tbody tr:nth-child(even){background:#f8fafc}td:first-child{font-weight:500}ul{padding-left:24px;color:#3d536d}li{margin:8px 0}.search-label{display:flex;align-items:center;gap:12px;color:var(--muted);font-size:14px}input{padding:9px 12px;width:350px;max-width:100%;border:1px solid #becddb;border-radius:4px;font:inherit}.footer{font-size:12px;color:var(--muted);padding-top:30px}
#lightbox{display:none;position:fixed;inset:0;background:#122033ed;z-index:10;overflow:auto;padding:60px 24px 24px}#lightbox.open{display:block}#lightbox img{display:block;max-width:none;width:1600px;min-width:100%;margin:auto;background:white}#close-box{position:fixed;right:24px;top:16px;background:white;color:#17283e;border:0;padding:8px 20px;font-size:16px;border-radius:4px;cursor:pointer}
@media(max-width:1000px){.layout{display:block}nav{position:relative;height:auto;border-right:0;border-bottom:1px solid var(--line);padding:20px}nav a{display:inline-block;margin:3px;font-size:12px}main{padding:25px 18px}.stats{gap:6px!important}.stat{padding:10px}h1{font-size:27px}h2{font-size:21px}#lightbox img{width:1200px}}
@media print{body{background:white}.layout{display:block}nav,#lightbox,.search-label{display:none}main{padding:0}section{break-before:auto}figure{break-inside:avoid}h2{break-after:avoid}th{background:#eef3f8!important}header .stats{grid-template-columns:repeat(4,1fr)}}
</style></head><body><div class="layout"><nav><div class="brand">300M 实验研究</div><div class="caption">本地离线 · 原始训练曲线<br>固定测试结果 · 比较与解释</div>__NAV__</nav><main><header><div class="eyebrow">LOCAL EXPERIMENT REVIEW / 2026.11.04</div><h1>从预训练到后训练：<br>本地图表、比较与研究总结</h1><p>按“训练目标 → 控制变量 → 固定测试 → 能力取舍”阅读。每组图均配解释，点击图像可放大。</p><div class="stats"><div class="stat"><b>25</b><span>组真实训练日志</span></div><div class="stat"><b>__FIGURES__</b><span>张PNG图示</span></div><div class="stat"><b>__PANELS__</b><span>个图表面板（含流程图）</span></div><div class="stat"><b>3,000</b><span>唯一固定测试题</span></div></div></header>__CONTENT__<div class="footer">生成时间：__CREATED__。已有记录的离线分析；未运行新训练、GPU评测或云端写入。原始数据与SHA256另存于同目录。</div></main></div><div id="lightbox" role="dialog" aria-label="放大图表"><button id="close-box">关闭 / Esc</button><img alt="放大图表"></div>
<script>
const box=document.getElementById('lightbox'),big=box.querySelector('img');
document.querySelectorAll('.zoomable').forEach(img=>img.addEventListener('click',()=>{big.src=img.src;big.alt=img.alt;box.classList.add('open');document.body.style.overflow='hidden';}));
function closeBox(){box.classList.remove('open');document.body.style.overflow='';}
document.getElementById('close-box').addEventListener('click',closeBox);box.addEventListener('click',e=>{if(e.target===box)closeBox();});document.addEventListener('keydown',e=>{if(e.key==='Escape')closeBox();});
const search=document.getElementById('model-search');search.addEventListener('input',()=>{const q=search.value.toLowerCase();document.querySelectorAll('#inventory-table tbody tr').forEach(row=>row.hidden=!row.textContent.toLowerCase().includes(q));});
</script></body></html>'''
        page=page.replace('__NAV__',''.join(nav)).replace('__CONTENT__','\n'.join(document)).replace('__CREATED__',html.escape(self.created))
        page=page.replace('__FIGURES__',str(len(self.figure_info))).replace('__PANELS__',str(chart_panels))
        (self.output/'本地图表与详细总结.html').write_text(page,encoding='utf-8')
        (self.output/'本地图表与详细总结.md').write_text('\n'.join(markdown),encoding='utf-8')
        csv_file(self.tables/'模型参数与成绩.csv',[{
            'model':key,'中文标签':self.label(key),'wandb_run_id':r['wandb_run_id'],'learning_rate':r['learning_rate'],
            'batch':r['effective_batch'],'实际轮数':r['actual_epochs'],'best_step':r['best_step'],
            '指令测试MacroCE':self.generic.get(key,self.cross.get(key,{})).get('macro_ce',self.cross.get(key,{}).get('instruction_macro_ce','')),
            '阅读测试MacroEM':self.reading.get(key,{}).get('macro_em',''),'阅读测试MacroF1':self.reading.get(key,{}).get('macro_f1',''),
        } for key,r in self.models.items()])
        write_json(self.output/'source_manifest.json',{'source_root':str(self.source),'created':self.created,'files':self.sources})
        write_json(self.output/'analysis.json',{'created':self.created,'training_runs':len(self.models),'figures':self.figure_info,
            'series_statistics':self.series_stats,'nonfinite_logged_values':self.nonfinite,
            'test':{'instruction_model_entries':len(self.generic),'reading_checkpoint_entries':len(self.reading),
                    'instruction_questions':1000,'reading_questions':2000,'unique_test_questions':3000,
                    'logical_generated_question_evaluations':len(self.generic)*1000+len(self.reading)*2000},
            'no_training_or_new_inference':True,'no_cloud_writes':True,
            'local_curve_coverage_exclusions':['100M/1B完整曲线','历史r16的5e-5和1e-4两组完整曲线']})
        self.verify()

    def verify(self):
        from PIL import Image, ImageStat
        assert len(self.models)==25 and len(self.generic)==15 and len(self.reading)==22
        assert all(int(row['examples'])==1000 for row in self.generic.values())
        assert all(int(row['examples'])==2000 for row in self.reading.values())
        assert self.reading['dpo_old_matched_best']['macro_em']==self.reading['reading_sft']['macro_em']
        assert self.models['dpo_old_matched_best']['best_step']=='0' and self.models['dpo_new_matched_best']['best_step']=='0'
        full=self.configs['sft100k_full'];lora=self.configs['lora100k_r16_historical']
        assert full['base_checkpoint_sha256']==lora['base_checkpoint_sha256']
        assert full['data_manifest_sha256']==lora['data_manifest_sha256']
        assert self.models['sft100k_full']['actual_epochs']==self.models['lora100k_r16_historical']['actual_epochs']
        rl_keys=[f'{method}_{arm}_best' for method in METHOD_COLORS for arm in ARMS]
        assert len({self.configs[key]['initial_checkpoint_sha256'] for key in rl_keys})==1
        assert all(float(self.configs[key]['train']['epochs'])==1 for key in rl_keys)
        plan=read_json(self.source/'sources_local/plan.json')
        assert plan['suites']['reading_test_2000']['max_new_tokens']==64
        assert plan['suites']['instruction_test_1000']['max_new_tokens']==256
        groups=read_json(self.source/'report/statistical_analysis.json')['primary'][0]['em_interval']['groups']
        assert groups['zh']['examples']==groups['en']['examples']==1000
        assert groups['zh']['clusters']==210 and groups['en']['clusters']==20
        files=[]
        for figure in self.figure_info:
            path=self.figures/figure['file']
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width,height=image.size
                assert width>=1000 and height>=600
                assert max(ImageStat.Stat(image.convert('RGB')).stddev)>12,'Blank figure'
            assert digest(path)==figure['sha256']
            files.append({'file':'figures/'+figure['file'],'width':width,'height':height,'sha256':digest(path)})
        page=(self.output/'本地图表与详细总结.html').read_text(encoding='utf-8')
        assert page.count('class="zoomable"')==len(self.figure_info)
        assert not re.search(r'<(?:script|img)[^>]+src=["\']https?://',page)
        assert len(self.sources)>=50
        assert all(digest(path)==expected for path,expected in self.sources.items())
        assert '0.404' in (self.tables/'模型参数与成绩.csv').read_text(encoding='utf-8-sig')
        result={'passed':True,'training_runs':25,'png_figures':len(files),'panels':sum(row['panels'] for row in self.figure_info),
            'instruction_questions':1000,'reading_questions':2000,'all_images_embedded':True,'external_runtime_dependencies':False,
            'input_hashes_rechecked':len(self.sources),'original_sources_unchanged':True,'no_training_or_inference':True,
            'full_vs_historical_lora_base_data_epoch_checked':True,'all_nine_rl_same_initial_checkpoint_checked':True,'files':files}
        write_json(self.output/'verification.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reuse-figures',action='store_true',help='输入与图像哈希一致时复用已生成的图')
    args=parser.parse_args()
    if args.output.resolve().drive.upper()!='E:':
        raise ValueError('Generated artifacts must stay on E:')
    configure_figures();dashboard=Dashboard(args.source,args.output)
    cache=args.output/'figures_manifest.json'
    if args.reuse_figures and cache.is_file():
        cached=read_json(cache)
        assert cached['sources']==dashboard.sources,'Figure source data changed'
        assert all(digest(dashboard.figures/row['file'])==row['sha256'] for row in cached['figures'])
        dashboard.figure_info=cached['figures'];dashboard.series_stats=cached['series_stats']
    else:
        dashboard.training_figures();dashboard.final_figures()
        write_json(cache,{'figures':dashboard.figure_info,'series_stats':dashboard.series_stats,'sources':dashboard.sources,'created':dashboard.created})
    dashboard.content();dashboard.render()
    archive=dashboard.output/'本地图表与总结_含PNG与CSV.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
        for path in sorted(dashboard.output.rglob('*')):
            if path.is_file() and path!=archive and path.suffix in {'.html','.md','.json','.csv','.png'} and 'preview' not in path.name and path.name!='archive_verification.json':
                zf.write(path,path.relative_to(dashboard.output))
    write_json(dashboard.output/'archive_verification.json',{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)})
    result=read_json(dashboard.output/'verification.json')
    print(json.dumps({'output':str(dashboard.output),'verified':result['passed'],'figures':result['png_figures'],
        'panels':result['panels'],'source_hashes':result['input_hashes_rechecked'],'zip_mib':archive.stat().st_size/2**20},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()


