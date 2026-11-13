import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
# fcntl是Linux运行依赖；Windows仅以测试stub导入，生产文件仍使用真实flock。
if sys.platform=='win32':
    import types
    sys.modules.setdefault('fcntl',types.SimpleNamespace())
from run_reading_rl_queue import case_finished,free_required


def test_queue_requires_all_one_epoch_updates_and_both_weight_files(tmp_path):
    out=tmp_path/'case';out.mkdir();spec={'output':str(out)}
    assert not case_finished(spec)
    (out/'best.pt').write_bytes(b'fixture');(out/'last.pt').write_bytes(b'fixture')
    result={'status':'completed','examples_seen':16,'data_batch_cursor':1,'optimizer_step':2,
        'plan':{'examples':16,'data_batches':1,'optimizer_steps':2},'best_checkpoint':str(out/'best.pt'),'last_checkpoint':str(out/'last.pt')}
    (out/'summary.json').write_text(json.dumps(result));assert case_finished(spec)
    result['optimizer_step']=1;(out/'summary.json').write_text(json.dumps(result))
    with pytest.raises(RuntimeError,match='Optimizer'):case_finished(spec)


def test_queue_does_not_treat_user_stop_as_success(tmp_path):
    (tmp_path/'summary.json').write_text(json.dumps({'status':'user_stop'}))
    with pytest.raises(RuntimeError,match='before one epoch'):case_finished({'output':str(tmp_path)})


def test_disk_gate_includes_retention_and_atomic_temporary():
    assert free_required('train')==8*2**30+7*303350784*4
