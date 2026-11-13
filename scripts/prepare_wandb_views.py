"""离线构建 W&B 图表布局；不启动训练，不改实验配置，不包含认证信息。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 图表 SDK 隔离安装，避免改动训练环境。
sys.path.insert(0, str(ROOT / '.cache/wandb_ui_sdk'))
import wandb_workspaces.expr as expr
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from wandb_workspaces.reports.v2 import internal

ENTITY = 'mblankm-peking-university'
PROJECT = 'bilingual-training-lab'
COLORS = ['#2563eb', '#ea580c', '#059669', '#9333ea', '#dc2626', '#0891b2']


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def build(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cloud = json.loads(args.inventory.read_text(encoding='utf-8-sig'))
    assert cloud['hasMore'] is False, 'Cloud inventory is incomplete'
    cloud_ids = {row['id'] for row in cloud['runs']}
    table_dir = args.report_root / 'report/tables'
    inventory = read_csv(table_dir / '01_model_inventory.csv')
    models = {row['key']: row for row in inventory}
    assert len(models) == len(inventory)
    assert all(row['wandb_run_id'] in cloud_ids for row in inventory)
    ns = f'plot_{args.date}'
    labels = {}
    for row in inventory:
        key, rid = row['key'], row['wandb_run_id']
        if key == 'pretrained_base': label = 'Base 300M / 3B tokens'
        elif key == 'reading_sft': label = 'Reading SFT / 50k / ep1'
        elif key == 'sft100k_full': label = 'Full SFT / 100k / ep2 [history]'
        elif key.startswith('sft20k'):
            label = f"SFT 20k / lr{float(row['learning_rate']):g} / B{row['effective_batch']} / ep{float(row['actual_epochs']):g}"
        elif key.startswith('lora'):
            label = f"LoRA r{row['adapter_rank']} / a{row['adapter_alpha']} / d{row['adapter_dropout'] or '0'} / ep{float(row['actual_epochs']):g}"
        else:
            algorithm = key.split('_')[0].upper()
            arm = 'new25k' if '_new_' in key else ('old50k' if '_full_50k_' in key else 'old25k')
            label = f'{algorithm} / {arm}'
            if algorithm == 'DPO': label += f" / n{row['train_examples']}"
        labels[rid] = label
    labels['c114c0dc5e9c'] = 'Base 100M / 1B tokens'
    for rid, lr in [('b42623376a7c', '5e-5'), ('67fcad7c81cd', '1e-4'), ('9f2a23bb0e2a', '2e-4')]:
        labels[rid] = f'LoRA r16 / lr{lr} / ep2 [history]'

    def ids(keys: list[str]) -> list[str]:
        return [models[key]['wandb_run_id'] for key in keys]

    def palette(run_ids: list[str]) -> dict[str, str]:
        return {rid: COLORS[i % len(COLORS)] for i, rid in enumerate(run_ids)}

    def line(title: str, metric: str, run_ids: list[str], index: int, x: str = 'optimizer_step'):
        return wr.LinePlot(
            title=title, x=x, y=[metric], title_x=x, title_y=metric,
            smoothing_type='none', smoothing_factor=0, smoothing_show_original=True,
            ignore_outliers=False, aggregate=False, max_runs_to_show=max(10, len(run_ids)),
            point_visualization_method='bucketing-gorilla', legend_position='south',
            line_titles={f'{rid}:{metric}': labels[rid] for rid in run_ids},
            line_colors={f'{rid}:{metric}': color for rid, color in palette(run_ids).items()},
            layout=wr.Layout(x=(index % 2) * 12, y=(index // 2) * 9, w=12, h=9),
        )

    def bar(title: str, metrics: list[str], run_ids: list[str], index: int = 0):
        return wr.BarPlot(
            title=title, metrics=[wr.SummaryMetric(metric) for metric in metrics],
            orientation='h', aggregate=False, max_runs_to_show=40, max_bars_to_show=60,
            line_titles={f'{rid}:{metric}': labels[rid] for rid in run_ids for metric in metrics},
            line_colors={f'{rid}:{metric}': color for rid, color in palette(run_ids).items() for metric in metrics},
            layout=wr.Layout(x=0, y=index * 13, w=24, h=13),
        )

    sections = []
    blocks = [
        wr.H1('300M 双语模型实验总览'),
        wr.P('按预训练、SFT、LoRA、阅读专项和 RL 分组。保留原始曲线，不更名原 run、不修改 config/group，不启动训练。'),
        wr.P('新实验均为 epochs=1；历史 ep2/ep3 仅用于解释已有结果。训练曲线与最终测试分开，最终测试不能继续用于挑选配置。'),
        wr.TableOfContents(),
    ]

    def section(title: str, run_ids: list[str], note: str, metrics: list[tuple[str, str]], x='optimizer_step'):
        assert run_ids and all(rid in cloud_ids for rid in run_ids)
        panels = [line(t, metric, run_ids, i, x) for i, (t, metric) in enumerate(metrics)]
        runset = wr.Runset(name=title, filters=[expr.Metric('ID').isin(run_ids)], custom_run_colors=palette(run_ids))
        blocks.extend([wr.H2(title), wr.P(note), wr.PanelGrid(runsets=[runset], panels=panels)])
        sections.append({'title': title, 'run_ids': run_ids, 'note': note, 'panels': panels})

    section('01 预训练 · 100M 与 300M', ['c114c0dc5e9c', models['pretrained_base']['wandb_run_id']],
            '模型规模与数据量都不同，仅观察学习过程，不能视为单变量实验。横轴为实际累计训练 Token。',
            [('训练 CE ↓', 'train_loss'), ('验证 Macro CE ↓', 'validation/macro_loss'),
             ('中文验证 CE ↓', 'validation/zh/loss'), ('英文验证 CE ↓', 'validation/en/loss')], x='tokens_seen')
    sft_lr = ids([f'sft20k_lr{lr}_mb8_ga4_b32_ep1' for lr in ['1e-5', '2e-5', '5e-5', '1e-4']])
    section('02 SFT · 学习率', sft_lr, '同一 Base、约 2 万训练样本、B32、一轮。仅学习率不同。',
            [('验证 Macro CE ↓', 'validation/macro_loss'), ('训练 CE ↓', 'train_loss'),
             ('原有能力验证 CE ↓', 'retention_validation/macro_loss'), ('学习率调度', 'learning_rate')])
    sft_batch = ids([f'sft20k_lr5e-5_mb8_ga{ga}_b{batch}_ep1' for ga, batch in [(1, 8), (2, 16), (4, 32)]])
    section('03 SFT · 有效 Batch', sft_batch, 'LR=5e-5，一轮相同数据量；横轴 examples_seen，避免更新步数差异造成误读。',
            [('验证 Macro CE ↓', 'validation/macro_loss'), ('训练 CE ↓', 'train_loss'),
             ('梯度裁剪前范数', 'grad_norm_before_clip'), ('监督 Token 吞吐', 'supervised_tokens_per_second')], x='examples_seen')
    hist = ids(['sft20k_lr5e-5_mb8_ga4_b32_ep1', 'sft20k_lr5e-5_mb8_ga4_b32_ep3_stop2', 'sft20k_lr5e-5_mb8_ga4_b32_ep3_stop3', 'sft100k_full'])
    section('04 SFT · 历史训练量', hist, '历史多轮配置保留原样。数据规模和学习率调度有差异，不能据此单独归因于 epochs。',
            [('验证 Macro CE ↓', 'validation/macro_loss'), ('原有能力验证 CE ↓', 'retention_validation/macro_loss')], x='examples_seen')
    lora_lr = ['b42623376a7c', '67fcad7c81cd', '9f2a23bb0e2a']
    section('05 LoRA · 历史学习率', lora_lr, '历史 r16/alpha32、10 万样本、实际两轮；与新的一轮实验分开。',
            [('验证 Macro CE ↓', 'validation/macro_loss'), ('训练 CE ↓', 'train_loss')])
    lora_rank = ids(['lora100k_r8_ld0', 'lora100k_r32_ld0'])
    section('06 LoRA · Rank', lora_rank, '固定 LR=2e-4、dropout=0、B32、一轮；比较 r8/alpha16 与 r32/alpha64。',
            [('开发集 Macro CE ↓', 'development_validation/macro_loss'), ('训练 CE ↓', 'train_loss'),
             ('原有能力验证 CE ↓', 'retention_validation/macro_loss'), ('监督 Token 吞吐', 'supervised_tokens_per_second')])
    lora_dropout = ids(['lora100k_r32_ld0', 'lora100k_r32_ld0.05', 'lora100k_r32_ld0.1'])
    section('07 LoRA · Dropout', lora_dropout, '固定 r32/alpha64、LR=2e-4、B32、一轮；dropout=0/0.05/0.1。',
            [('开发集 Macro CE ↓', 'development_validation/macro_loss'), ('训练 CE ↓', 'train_loss')])
    reading = ids(['reading_sft'])
    section('08 阅读专项 SFT · RL 共同起点', reading, '公开阅读理解 5 万条，一轮；后续 PPO、GRPO、DPO 都从此模型独立开始。',
            [('验证 Macro CE ↓', 'validation/macro_loss'), ('训练 CE ↓', 'train_loss')])
    algorithm_ids = {}
    for algorithm, title in [('ppo', '09 PPO'), ('grpo', '10 GRPO'), ('dpo', '11 DPO')]:
        run_ids = ids([f'{algorithm}_{arm}_best' for arm in ['old_matched', 'new_matched', 'old_full_50k']])
        algorithm_ids[algorithm] = run_ids
        extra = 'PPO 数据一轮，每批内部两次优化。' if algorithm == 'ppo' else '训练题只遍历一轮。'
        if algorithm == 'dpo':
            extra += '偏好过滤后样本数不同。'
            if not args.history_only:
                extra += '两组匹配集的 best=step0，即初始 SFT，不能宣称 DPO 带来提升。'
        metrics = [('开发集 Macro EM ↑', 'validation/macro_exact_match'), ('开发集 Macro F1 ↑', 'validation/macro_f1'),
                   (f'{algorithm.upper()} 训练目标 Loss（仅本算法可比）', 'loss'),
                   ('偏好 margin' if algorithm == 'dpo' else '采样平均奖励', 'dpo_margin' if algorithm == 'dpo' else 'reward_mean')]
        if algorithm != 'dpo': metrics += [('相对参考策略 KL', 'kl'), ('裁剪比例', 'clip_fraction')]
        section(title, run_ids, '旧匹配/新匹配/旧全量分别展示。开发集 256 题，非最终测试。' + extra, metrics, x='examples_seen')

    updates = {}
    def add(key: str, name: str, value: object) -> None:
        rid = models[key]['wandb_run_id']
        if isinstance(value, float): assert math.isfinite(value)
        updates.setdefault(rid, {})[f'{ns}/{name}'] = value

    reading_rows = read_csv(table_dir / '03_reading_all_checkpoints.csv')
    for row in reading_rows:
        evalkey = row['key']
        modelkey = evalkey if evalkey in models else evalkey.removesuffix('_last') + '_best'
        if modelkey not in models or not (modelkey == 'reading_sft' or modelkey.startswith(('ppo_', 'grpo_', 'dpo_'))): continue
        kind = 'last' if evalkey.endswith('_last') else 'best'
        assert int(row['examples']) == 2000
        for metric in ['macro_em', 'macro_f1', 'zh_em', 'en_em', 'eos', 'repeat_3gram']:
            add(modelkey, f'reading/{kind}/{metric}', float(row[metric]))
        add(modelkey, f'reading/{kind}/examples', 2000)
        add(modelkey, f'reading/{kind}/checkpoint_step', int(row['checkpoint_step']))
    for row in read_csv(table_dir / '11_instruction_1000_all_models.csv'):
        assert int(row['examples']) == 1000
        for metric in ['macro_ce', 'eos', 'repeat_3gram', 'rouge_macro_language']:
            add(row['key'], f'instruction/{metric}', float(row[metric]))
        add(row['key'], 'instruction/examples', 1000)
    for row in read_csv(table_dir / '17_cross_domain_instruction_loss.csv'):
        assert int(row['examples']) == 1000
        add(row['key'], 'instruction/macro_ce', float(row['instruction_macro_ce']))
        add(row['key'], 'instruction/examples', 1000)

    rl_ids = reading + [rid for group in algorithm_ids.values() for rid in group]
    all_ids = [row['wandb_run_id'] for row in inventory]
    final_panels = [
        bar('阅读测试 2000 题 · best Macro EM ↑（0–1）', [f'{ns}/reading/best/macro_em'], rl_ids),
        bar('阅读测试 2000 题 · best Macro F1 ↑（0–1）', [f'{ns}/reading/best/macro_f1'], rl_ids, 1),
        bar('阅读测试 · best 与 last（两组 DPO best=step0）', [f'{ns}/reading/best/macro_em', f'{ns}/reading/last/macro_em'], rl_ids, 2),
        bar('通用指令测试 1000 题 · 参考答案 Macro CE ↓', [f'{ns}/instruction/macro_ce'], all_ids, 3),
    ]
    if not args.history_only:
      blocks.extend([
        wr.H2('12 最终测试 · 与训练过程分开'),
        wr.P('best 按各实验原开发集规则选择，last 仅诊断训练末期变化。阅读 EM/F1 为中英文各 1000 题的宏平均；EM 严格匹配，F1 是 Token 重合。通用指令 CE 来自 1000 题参考答案，不能解释为生成准确率。'),
        wr.P('阅读专项 SFT Macro EM=36.00%；GRPO 旧全量 best=40.40%。DPO 两组匹配实验 best=step0。全参数 SFT 100k 和历史 r16 为两轮，新 Rank/Dropout 为一轮，不能声称完全等预算。'),
        wr.PanelGrid(runsets=[wr.Runset(name='最终测试 · 固定模型', filters=[expr.Metric('ID').isin(all_ids)])], panels=final_panels),
        wr.H2('13 查看说明与诊断记录'),
        wr.P('100M 早期基线、300M 中止重跑、SDK smoke 和 probe 保留在原项目，不进入正式对比。原工作区保留。未删除运行、历史、模型或数据。统一无平滑、不忽略离群点，避免掩盖波动。'),
      ])
    else:
      blocks.extend([wr.H2('12 最终测试说明'), wr.P('本报告展示已有训练与开发集曲线。固定最终测试结果见独立测试报告，与训练曲线分开。'),
                     wr.H2('13 诊断记录'), wr.P('原工作区、早期基线、中止重跑、SDK smoke 和 probe 均保留。正式图表不混入助手诊断记录。曲线无平滑，不忽略离群点。')])
    description = '已有云端训练曲线与开发集指标，按控制变量分组；原实验身份保持不变。' if args.history_only else '真实运行曲线、控制变量分组及固定测试集结果；原实验身份保持不变。'
    report = wr.Report(entity=ENTITY, project=PROJECT, title=f'双语模型实验总览 · 预训练 SFT LoRA RL · {args.date}',
                       description=description, width='fluid', blocks=blocks)
    report_model = report._to_model()
    # 同一项目的离线布局使用已验证的 internalId；不触发 SDK 登录或网络调用。
    for block in report_model.spec.blocks:
        if getattr(block, 'type', None) == 'panel-grid':
            block.metadata.panel_bank_section_config.is_open = True
            for runset in block.metadata.run_sets:
                runset.project = internal.Project(name=PROJECT, entityName=ENTITY, id=cloud['project']['internalId'])
    report_raw = report_model.model_dump(mode='json', by_alias=True, exclude_none=True)
    internal.ReportViewspec.model_validate(report_raw)
    grids = [block['metadata'] for block in report_raw['spec']['blocks'] if block['type'] == 'panel-grid']
    assert all(runset['project']['id'] == cloud['project']['internalId'] for grid in grids for runset in grid['runSets'])
    expected_charts = 40 if args.history_only else 44
    assert len(grids) == len(sections) + (0 if args.history_only else 1)
    assert sum(len(grid['panelBankSectionConfig']['panels']) for grid in grids) == expected_charts

    selected_views = [
        ('pretrain', '01 预训练', sections[0]),
        ('sft_lr', '02 SFT 学习率', sections[1]),
        ('sft_batch', '03 SFT Batch', sections[2]),
        ('lora_rank', '04 LoRA Rank', sections[5]),
        ('lora_dropout', '05 LoRA Dropout', sections[6]),
    ]
    workspaces = []
    for slug, title, item in selected_views:
        view = ws.Workspace(entity=ENTITY, project=PROJECT, name=title,
            sections=[ws.Section(name=title, panels=item['panels'], is_open=True)],
            runset_settings=ws.RunsetSettings(filters=[expr.Metric('ID').isin(item['run_ids'])],
                run_settings={rid: ws.RunSettings(color=color) for rid, color in palette(item['run_ids']).items()}),
            auto_generate_panels=False)
        workspaces.append({'slug': slug, 'displayName': title, 'spec': view._to_model().spec.model_dump(mode='json', by_alias=True, exclude_none=True)})
    dev_panels = [line('共同开发集 256 题 · Macro EM ↑', 'validation/macro_exact_match', rl_ids[1:], 0, x='examples_seen'),
                  line('共同开发集 256 题 · Macro F1 ↑', 'validation/macro_f1', rl_ids[1:], 1, x='examples_seen')]
    rl_sections = [ws.Section(name='开发集 · 训练过程', panels=dev_panels, is_open=True)]
    if not args.history_only:
        rl_sections.append(ws.Section(name='最终测试 · best 与 last', panels=final_panels[:3], is_open=True))
    rl_view = ws.Workspace(entity=ENTITY, project=PROJECT, name='06 RL 统一评测',
        sections=rl_sections,
        runset_settings=ws.RunsetSettings(filters=[expr.Metric('ID').isin(rl_ids)]), auto_generate_panels=False)
    workspaces.append({'slug': 'rl_evaluation', 'displayName': '06 RL 统一评测', 'spec': rl_view._to_model().spec.model_dump(mode='json', by_alias=True, exclude_none=True)})

    payload = {'entity': ENTITY, 'project': PROJECT, 'date': args.date, 'namespace': ns,
               'report': {'displayName': report.title, 'description': report.description, 'spec': report_raw['spec']},
               'workspaces': workspaces}
    if not args.history_only:
        payload['summary_updates'] = updates
    write_json(out / 'layout_payload.json', payload)
    write_json(out / 'run_labels.json', labels)
    manifest = {'date': args.date, 'cloud_runs': len(cloud_ids), 'formal_models': len(inventory),
                'report_sections': len(sections) + 2, 'report_chart_count': expected_charts,
                'history_only': args.history_only,
                'named_workspaces': len(workspaces), 'summary_runs': 0 if args.history_only else len(updates),
                'summary_metric_count': 0 if args.history_only else sum(len(value) for value in updates.values()),
                'payload_sha256': hashlib.sha256((out / 'layout_payload.json').read_bytes()).hexdigest(),
                'sources': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                    [table_dir / name for name in ['01_model_inventory.csv', '03_reading_all_checkpoints.csv', '11_instruction_1000_all_models.csv', '17_cross_domain_instruction_loss.csv']]}}
    write_json(out / 'layout_validation.json', manifest)
    change_note = '只创建引用现有云端指标的视图，不修改原 run 摘要，不上传本地最终测试结果。' if args.history_only else f'拟新增视图和测试指标命名空间 `{ns}/`；上传评测指标另需明确授权。'
    notes = ['# W&B 图表整理', '', f'构建日期：{args.date}。{change_note}', '',
             '原始 run ID、名称、group、config、历史曲线及工作区保持不变。', '',
             '## 报告分区', '']
    notes.extend(f"- {s['title']}：{len(s['run_ids'])} 条运行，{len(s['panels'])} 张图。{s['note']}" for s in sections)
    notes.extend(['', '- 最终测试：阅读 2000 题、通用指令 1000 题，best 与 last 分开。', '',
                  '曲线无平滑，保留离群点。各算法训练 Loss 分开；跨算法主要比较同一开发集 EM/F1 和固定最终测试。', '',
                  'SDK 布局已通过 Pydantic 校验。`layout_payload.json` 不含认证信息，可复查，云端创建后再做回读校验。'])
    (out / 'README.md').write_text('\n'.join(notes) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-root', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--date', required=True)
    parser.add_argument('--history-only', action='store_true', help='仅构建已有云端训练曲线，不上传最终评测或写原 run 摘要')
    build(parser.parse_args())
