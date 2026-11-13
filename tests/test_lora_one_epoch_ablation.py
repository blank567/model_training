"""验证单轮约束、独立命名和先 rank 后 dropout 的调参顺序。"""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from llm_lab.common.io import sha256_file, write_json
import train_lora
import train_lora_one_epoch as entry
from run_lora_ablation import completed_result, drive, select, validate_rank_configs
from test_sft_runtime_data import corpus
from test_sft_engine import fixture_config


def config_for(corpus):
    config, _ = fixture_config(corpus, output_name='one_epoch')
    config['method'] = 'lora'
    config['train'].update(epochs=1., stop_epochs=None, learning_rate=2e-4)
    config['adapter'] = {'rank': 8, 'alpha': 16, 'dropout': 0.05, 'target_scope': 'all', 'bias': 'none'}
    config['experiment']['one_epoch_entry_sha256'] = sha256_file(entry.ENTRY)
    return config


@pytest.mark.parametrize('epochs,stop', [(3, 1), (1, 1), (2, None), (1, 2)])
def test_multi_epoch_schedules_and_stop_overrides_are_rejected(corpus, epochs, stop):
    config = config_for(corpus)
    config['train'].update(epochs=epochs, stop_epochs=stop)
    with pytest.raises(ValueError, match='Only epochs=1'):
        entry.validate_one_epoch(config)


def test_real_fixture_freeze_and_training_use_one_epoch_and_dropout_name(corpus, monkeypatch):
    import torch
    before = train_lora.name_for
    config = config_for(corpus)
    path = Path(config['project_root']) / 'template_one_epoch.json'
    write_json(path, config)
    ready = entry.prepare(path, '20261101_120000')
    assert '_r8_a16_all_ld0p05_' in ready['RUN_NAME']
    assert '_ep1_' in ready['RUN_NAME'] and '_stop' not in ready['RUN_NAME']
    assert ready['plan']['epochs'] == 1 and ready['plan']['stop_epochs'] is None
    assert ready['plan']['stop_step'] == ready['plan']['max_steps'] == 1
    assert train_lora.name_for is before
    # Windows 的 CPU fixture 不具备 Linux /proc；只替换进程枚举，不替换训练。
    monkeypatch.setattr(train_lora.legacy, 'training_processes', lambda: [])
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        summary = entry.execute('start', ready['RUN_CONFIG'])
    finally:
        torch.set_num_threads(previous_threads)
    assert summary['status'] == 'completed' and summary['step'] == 1
    assert summary['progress']['examples_seen'] == ready['plan']['train_examples']
    assert train_lora.name_for is before
    assert Path(summary['best_checkpoint']).is_file() and Path(summary['last_checkpoint']).is_file()


def test_rank_control_rejects_other_variable_changes_and_entry_tampering(corpus):
    first = config_for(corpus)
    first['adapter']['dropout'] = 0.
    second = deepcopy(first)
    second['adapter'].update(rank=32, alpha=64)
    validate_rank_configs([first, second])
    second['train']['seed'] += 1
    with pytest.raises(ValueError, match='control variable'):
        validate_rank_configs([first, second])
    first['experiment']['one_epoch_entry_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='entry source'):
        entry.validate_one_epoch(first)


def test_rank_finishes_before_dropout_and_only_winner_is_used():
    trace = []
    def candidate(item, phase):
        trace.append((phase, item['key']))
        return {**item, 'development_loss': {'r8': 2., 'r32': 1., 'd0.05': .9, 'd0.1': 1.1}[item['key']]}
    def prepare(winner, value):
        assert winner['adapter']['rank'] == 32
        assert trace[:2] == [('rank', 'r8'), ('rank', 'r32')]
        return {'key': f'd{value:g}', 'adapter': {'rank': 32, 'dropout': value}}
    result = drive([{'key': f'r{rank}', 'adapter': {'rank': rank, 'dropout': 0.}} for rank in (8, 32)],
                   candidate, prepare, lambda winner, rows: trace.append(('selection', winner['key'])))
    assert result['selected_rank'] == 32 and result['selected']['key'] == 'd0.05'
    assert trace == [('rank', 'r8'), ('rank', 'r32'), ('selection', 'r32'),
                     ('dropout', 'd0.05'), ('dropout', 'd0.1')]


def test_failed_rank_prevents_followup_and_incomplete_results_cannot_win():
    trace = []
    def candidate(item, phase):
        trace.append(item['key'])
        raise RuntimeError('candidate failed')
    with pytest.raises(RuntimeError, match='candidate failed'):
        drive([{'key': 'r8'}, {'key': 'r32'}], candidate,
              lambda *args: pytest.fail('Dropout started before ranks completed'), lambda *args: None)
    assert trace == ['r8']
    with pytest.raises(ValueError):
        select([{'key': 'r8', 'development_loss': 1.}], 2)
    with pytest.raises(ValueError):
        completed_result({'status': 'paused', 'step': 3125}, 3125)
    with pytest.raises(ValueError):
        select([{'key': 'a', 'development_loss': float('nan')}, {'key': 'b', 'development_loss': 1.}], 2)
