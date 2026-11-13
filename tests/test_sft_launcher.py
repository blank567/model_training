"""冻结/恢复与控制变量计划的 CPU 测试；不下载基座，不启动 GPU。"""
import copy
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_lab.common.io import canonical_hash, read_json, sha256_file, write_json
from llm_sft.config import GIB, PARAMETER_COUNT, plan_details, read_config

SPEC = importlib.util.spec_from_file_location("train_sft_launcher", Path(__file__).parents[1]/"scripts/train_sft.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture
def project(tmp_path, monkeypatch):
    base = tmp_path / "base/best.pt"; base.parent.mkdir(); base.write_bytes(b"trusted fixture checkpoint")
    data = tmp_path / "data"; data.mkdir()
    token = data/"tokenizer.json"; token.write_text("{}")
    splits = {}
    for split, count in (("train",5),("pilot",3),("validation",2),("test",2)):
        path=data/(split+".jsonl"); path.write_text('{}\n'*count)
        splits[split]={"path":path.name,"sha256":sha256_file(path),"examples":count}
    manifest={"schema_version":1,"kind":"sft_tokenized_conversations","fixture":True,"splits":splits,
              "tokenizer":{"path":token.name,"sha256":sha256_file(token)},"tokenization":{},"chinese_normalization":{}}
    manifest['data_identity']=canonical_hash({key:manifest[key] for key in ('schema_version','kind','splits','tokenizer','tokenization','chinese_normalization')})
    report={'status':'passed','data_identity':manifest['data_identity']};write_json(data/'acceptance.json',report)
    manifest['acceptance']={'status':'passed','report':'acceptance.json','sha256':sha256_file(data/'acceptance.json')}
    write_json(data/'manifest.json',manifest)
    raw={"schema_version":1,"project_root":".","base_checkpoint":"base/best.pt","data_manifest":"data/manifest.json", "output_dir":"runs/unprepared",
         "fixture":True,"min_free_bytes":0,"train":{"device":"cpu","dtype":"fp32","micro_batch_size":2,"gradient_accumulation_steps":2,
         "tensorboard":False,"wandb":False},"budget":{"hourly_rate_cny":0}}
    path=tmp_path/'template.json';write_json(path,raw)
    monkeypatch.setattr(launcher,'runtime_fingerprint',lambda root: 'a'*64)
    return tmp_path,path,raw,manifest


def test_read_config_is_strict_and_resolves_paths(project):
    root,path,raw,_=project;config=read_config(path)
    assert config['project_root']==str(root)
    assert Path(config['base_checkpoint'])==root/'base/best.pt'
    assert config['train']['retention_eval_batches']==0
    raw['train']['learning_ratte']=1e-5;write_json(path,raw)
    with pytest.raises(ValueError,match='train'):
        read_config(path)


@pytest.mark.parametrize('field,value',[('learning_rate',True),('epochs',0),('stop_epochs',2),('warmup_ratio',-1),('betas',[1,.95])])
def test_invalid_optimizer_or_plan_values_fail(project,field,value):
    _,path,raw,_=project;raw['train'][field]=value;write_json(path,raw)
    with pytest.raises(ValueError):read_config(path)


def test_fractional_plan_has_epoch_boundaries_and_shared_schedule(project):
    _,path,raw,_=project
    raw['train'].update(epochs=2,stop_epochs=.5);write_json(path,raw)
    details=plan_details(read_config(path))
    assert details['max_steps']==4 and details['epoch_sample_counts']==[5,5]
    assert details['stop_step']==1 and details['actual_examples_at_stop']==4
    raw['train']['stop_epochs']=1;write_json(path,raw)
    assert plan_details(read_config(path))['stop_step']==2


def test_prepare_binds_data_base_and_code_without_creating_output(project):
    root,path,_,_=project
    prepared=launcher.prepare(path,'20261027_010000')
    assert not Path(prepared['RUN_DIR']).exists()
    assert Path(prepared['env']).is_file()
    frozen=launcher.frozen_config(prepared['RUN_CONFIG'])
    assert frozen['base_checkpoint_sha256']==sha256_file(root/'base/best.pt')
    assert frozen['runtime_sha256']=='a'*64
    assert frozen['train']['epochs']==1
    with pytest.raises(FileExistsError):launcher.prepare(path,'20261027_010000')


def test_parameter_name_is_shared_by_directory_config_log_env_and_wandb(project):
    root,path,raw,_=project
    raw['train'].update(micro_batch_size=8,gradient_accumulation_steps=4)
    write_json(path,raw)
    prepared=launcher.prepare(path,'20261027_010010')
    name='sft_300m_full_lr2e-5_mb8_ga4_b32_ep1_wu0p03_wd0p01_seed42_20261027_010010'
    assert prepared['RUN_NAME']==name
    assert Path(prepared['RUN_DIR'])==root/'runs'/name
    assert Path(prepared['RUN_CONFIG'])==root/'runs/configs'/(name+'.json')
    assert Path(prepared['RUN_LOG'])==root/'runs/logs'/(name+'.log')
    assert Path(prepared['env']).name==name+'.env'
    identity=read_json(prepared['identity'])
    config=launcher.frozen_config(prepared['RUN_CONFIG'])
    assert identity['name']==config['train']['wandb_run_name']==name
    assert 'export RUN_NAME='+name in Path(prepared['env']).read_text(encoding='utf-8')
    assert launcher.status(prepared['RUN_CONFIG'])['name']==name
    assert launcher.status(prepared['RUN_CONFIG'])['legacy_naming'] is False


@pytest.mark.parametrize('field,value,token',[
    ('learning_rate',1e-5,'lr1e-5'),
    ('micro_batch_size',4,'mb4_ga2_b8'),
    ('gradient_accumulation_steps',1,'mb2_ga1_b2'),
    ('epochs',.5,'ep0p5'),
    ('stop_epochs',.5,'ep1_stop0p5'),
    ('warmup_ratio',.01,'wu0p01'),
    ('weight_decay',0,'wd0'),
    ('seed',43,'seed43'),
])
def test_parameter_name_changes_for_each_control_setting(project,field,value,token):
    _,path,_,_=project;config=read_config(path);stamp='20261027_010011'
    baseline=launcher.experiment_name(config,stamp)
    config['train'][field]=value
    changed=launcher.experiment_name(config,stamp)
    assert changed!=baseline and token in changed
    assert '.' not in changed and 'e-05' not in changed


def test_names_distinguish_schedule_horizon_stop_amount_and_micro_decomposition(project):
    _,path,_,_=project;config=read_config(path);stamp='20261027_010012'
    config['train'].update(epochs=2,stop_epochs=.5)
    assert '_ep2_stop0p5_' in launcher.experiment_name(config,stamp)
    second=copy.deepcopy(config);second['train']['micro_batch_size']=1;second['train']['gradient_accumulation_steps']=4
    assert launcher.experiment_name(config,stamp)!=launcher.experiment_name(second,stamp)
    assert '_b4_' in launcher.experiment_name(config,stamp)
    assert '_b4_' in launcher.experiment_name(second,stamp)


def test_tampered_parameter_name_is_rejected(project):
    _,path,_,_=project;prepared=launcher.prepare(path,'20261027_010013')
    identity=read_json(prepared['identity']);identity['name']='another_experiment'
    write_json(prepared['identity'],identity)
    with pytest.raises(ValueError,match='名称与参数'):
        launcher.frozen_config(prepared['RUN_CONFIG'])


def test_legacy_names_remain_read_only_and_superseded_configs_cannot_execute(project):
    root,path,_,_=project;stamp='20261027_010014';prepared=launcher.prepare(path,stamp)
    config=read_json(prepared['RUN_CONFIG']);identity=read_json(prepared['identity'])
    legacy=root/'runs/configs'/('sft_300m_full_'+stamp+'.json')
    config['output_dir']=str(root/'runs'/('sft_300m_full_'+stamp))
    config['train']['wandb_run_name']='sft-300m-full-'+stamp
    write_json(legacy,config);identity.pop('name')
    identity.update(config_sha256=sha256_file(legacy),output_dir=config['output_dir'])
    write_json(legacy.with_suffix('.identity.json'),identity)
    assert launcher.status(legacy)['legacy_naming'] is True
    with pytest.raises(ValueError,match='重新 prepare'):
        launcher.frozen_config(legacy)
    write_json(Path(prepared['RUN_CONFIG']).with_suffix('.state.json'),{'status':'superseded'})
    with pytest.raises(ValueError,match='已被替代'):
        launcher.frozen_config(prepared['RUN_CONFIG'])


def test_tampered_frozen_config_and_code_are_rejected(project,monkeypatch):
    _,path,_,_=project;prepared=launcher.prepare(path,'20261027_010001')
    monkeypatch.setattr(launcher,'runtime_fingerprint',lambda root:'b'*64)
    with pytest.raises(ValueError,match='身份'):
        launcher.frozen_config(prepared['RUN_CONFIG'])
    monkeypatch.setattr(launcher,'runtime_fingerprint',lambda root:'a'*64)
    frozen=read_json(prepared['RUN_CONFIG']);frozen['train']['learning_rate']=3e-5;write_json(prepared['RUN_CONFIG'],frozen)
    with pytest.raises(ValueError,match='配置'):
        launcher.frozen_config(prepared['RUN_CONFIG'])


def test_changed_base_or_training_data_are_rejected(project):
    root,path,_,_=project;prepared=launcher.prepare(path,'20261027_010002')
    (root/'base/best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='身份'):
        launcher.frozen_config(prepared['RUN_CONFIG'])
    (root/'base/best.pt').write_bytes(b'trusted fixture checkpoint')
    (root/'data/train.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='SHA'):
        launcher.frozen_config(prepared['RUN_CONFIG'])


def test_report_binding_and_superseded_dataset_are_rejected(project):
    root,path,_,_=project;config=read_config(path)
    (root/'data/SUPERSEDED.json').write_text('{}')
    with pytest.raises(ValueError,match='废弃'):launcher.dataset_identity(config)
    (root/'data/SUPERSEDED.json').unlink()
    (root/'data/acceptance.json').write_text('{}')
    with pytest.raises(ValueError,match='报告 SHA'):launcher.dataset_identity(config)


def test_production_base_must_be_completed_pretraining(project):
    root,path,_,_=project;config=read_config(path);config['fixture']=False
    write_json(root/'base/summary.json',{'status':'signal_stop','step':10})
    write_json(root/'base/run_manifest.json',{'stage':'pretrain','parameters':PARAMETER_COUNT})
    write_json(root/'base/config.json',{'train':{'max_steps':10}})
    with pytest.raises(ValueError,match='已完成'):launcher.base_identity(config)
    write_json(root/'base/summary.json',{'status':'completed','step':9})
    with pytest.raises(ValueError,match='冻结步数'):launcher.base_identity(config)


def test_check_never_calls_engine_and_probe_keeps_plan(project,monkeypatch):
    _,path,_,_=project;prepared=launcher.prepare(path,'20261027_010003')
    calls=[]
    monkeypatch.setattr(launcher,'require_linux',lambda:None)
    monkeypatch.setattr(launcher,'training_lock',lambda root:nullcontext())
    monkeypatch.setattr(launcher,'training_processes',lambda:[])
    monkeypatch.setattr(launcher,'run_engine',lambda cfg,**kwargs:calls.append((cfg,kwargs)) or {'step':1})
    assert launcher.execute('check',prepared['RUN_CONFIG'])['gpu_training_started'] is False
    assert not calls
    launcher.execute('probe',prepared['RUN_CONFIG'],probe_steps=1)
    assert calls[0][1]=={'stop_after_steps':1}
    assert plan_details(calls[0][0])['max_steps']==2
    assert read_config(prepared['RUN_CONFIG'])['train']['epochs']==1


def test_resume_does_not_use_best_model_only_checkpoint(project,monkeypatch):
    _,path,_,_=project;prepared=launcher.prepare(path,'20261027_010004');config=launcher.frozen_config(prepared['RUN_CONFIG'])
    output=Path(config['output_dir']);output.mkdir();(output/'best.pt').write_bytes(b'model only')
    monkeypatch.setattr(launcher,'training_processes',lambda:[])
    with pytest.raises(ValueError,match='完整 last.pt'):
        launcher.preflight(config,'resume')


def test_busy_gpu_and_sft_process_are_detected(tmp_path,monkeypatch):
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,stdout='777\n'))
    with pytest.raises(RuntimeError,match='已有'):launcher.gpu_idle()
    proc=tmp_path/'proc';p=proc/'999999';p.mkdir(parents=True)
    (p/'cmdline').write_bytes(b'python\0scripts/train_sft.py\0start\0--config\0x\0')
    assert launcher.training_processes(proc)[0]['pid']==999999


def test_required_space_accounts_for_owned_checkpoint_inodes(project):
    root,path,_,_=project;config=read_config(path);config['fixture']=False;config['min_free_bytes']=8*GIB
    output=root/'runs/unprepared';output.mkdir(parents=True)
    (output/'last.pt').write_bytes(b'abc')
    os.link(output/'last.pt',output/'best.pt')
    assert launcher.required_free_bytes(config)==8*GIB+PARAMETER_COUNT*32-3


def test_lr_matrix_is_three_candidates_not_cartesian_and_never_runs(project,monkeypatch):
    root,path,_,_=project
    monkeypatch.setattr(launcher,'run_engine',lambda *a,**kw:pytest.fail('plan cannot run training'))
    (root/'base/best.pt').unlink()
    result=launcher.plan(path,'lr','20261027_020000')
    assert len(result['candidates'])==3 and result['gpu_training_started'] is False
    values=[read_config(row['template'])['train']['learning_rate'] for row in result['candidates']]
    assert values==[1e-5,2e-5,5e-5]
    for row in result['candidates']:
        config=read_config(row['template']);name=launcher.experiment_name(config,row['proposed_stamp'])
        assert row['RUN_NAME']==name==Path(config['output_dir']).name
        assert row['proposed_output_dir']==config['output_dir']


def test_epochs_matrix_changes_stop_amount_not_cosine_horizon(project):
    _,path,_,_=project;result=launcher.plan(path,'epochs','20261027_020001')
    configs=[read_config(row['template']) for row in result['candidates']]
    assert [cfg['train']['epochs'] for cfg in configs]==[2,2,2]
    assert [cfg['train']['stop_epochs'] for cfg in configs]==[.5,1,2]
    assert {plan_details(cfg)['max_steps'] for cfg in configs}=={4}


def test_batch_matrix_keeps_micro_and_matches_sample_eval_intervals(project):
    _,path,raw,_=project
    raw['train'].update(micro_batch_size=8,gradient_accumulation_steps=4);write_json(path,raw)
    result=launcher.plan(path,'batch','20261027_020002')
    configs=[read_config(row['template']) for row in result['candidates']]
    assert [cfg['train']['micro_batch_size'] for cfg in configs]==[8,8,8]
    assert [cfg['train']['gradient_accumulation_steps'] for cfg in configs]==[1,2,4]
    assert [cfg['train']['eval_interval'] for cfg in configs]==[400,200,100]


def test_plan_refuses_existing_directory(project):
    _,path,_,_=project;launcher.plan(path,'weight_decay','20261027_020003')
    with pytest.raises(FileExistsError):launcher.plan(path,'weight_decay','20261027_020003')
