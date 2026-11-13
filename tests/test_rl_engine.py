"""真实tiny Decoder的完整优化、参考策略固定与恢复测试；全部为CPU fixture。"""
from copy import deepcopy
from pathlib import Path
import json
import pytest,torch
from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json,write_jsonl
from llm_rl.config import DEFAULT_TRAIN,runtime_identity,validate
from llm_rl import engine
from llm_rl.inference import load_model
from test_sft_runtime_data import corpus
from test_rl_comparison_data import setup
from test_math_sft import config_for
from llm_reading_independent.comparison import write_arm

ROOT=Path(__file__).resolve().parents[1]


def config(corpus,name,algorithm):
    directory,manifest,rows,tokenizer=corpus
    arm_path=directory/'comparison/old.json'
    if not arm_path.exists():arm_path=setup(corpus)
    _,parent=config_for(corpus,'sft_parent_config')
    # config_for补全分词协议并重新固定源清单；比较视图随后绑定最终清单。
    if read_json(arm_path)['source_manifest_sha256']!=sha256_file(directory/'manifest.json'):
        arm_path.unlink()  # 仅删除本测试创建且已失效的微型比较视图。
        write_arm(directory/'manifest.json',arm_path,[1,0],'old_seen')
    t=deepcopy(DEFAULT_TRAIN);t.update(device='cpu',dtype='fp32',learning_rate=.002,critic_learning_rate=.003,
      prompt_batch_size=1,score_micro_batch_size=2,eval_size=2,eval_batch_size=2,eval_interval=1,save_interval=1,log_interval=1,
      warmup_ratio=0.,group_size=4 if algorithm=='grpo' else 1,tensorboard=False,wandb=False)
    preference=None
    if algorithm=='dpo':
        preference=directory/'preferences/manifest.json';preference.parent.mkdir(exist_ok=True)
        examples=[]
        for row in rows['train']:
            p=row['input_ids'][:row['prompt_token_count']];chosen=row['input_ids'][row['prompt_token_count']:]
            examples.append({'id':row['id'],'prefix_sha256':canonical_hash(p),'chosen_ids':chosen,'rejected_ids':[3,2]})
        write_jsonl(preference.parent/'pairs.jsonl',examples)
        write_json(preference,{'status':'published','kind':'reading_policy_preferences','initial_checkpoint_sha256':sha256_file(parent),
           'source_manifest_sha256':sha256_file(directory/'manifest.json'),'records':{'path':'pairs.jsonl','sha256':sha256_file(preference.parent/'pairs.jsonl')}})
    return {'schema_version':1,'stage':'reading_rl','algorithm':algorithm,'project_root':str(ROOT),
      'initial_checkpoint':str(parent),'initial_checkpoint_sha256':sha256_file(parent),
      'arm_manifest':str(arm_path),'arm_manifest_sha256':sha256_file(arm_path),'validation_manifest':str(directory/'manifest.json'),
      'output_dir':str(directory/name),'preference_manifest':str(preference) if preference else None,'train_examples':2,
      'fixture':True,'min_free_bytes':0,'train':t,'budget':{'hourly_rate_cny':0,'run_limit_cny':1,'project_limit_cny':500,
         'reserve_cny':100,'max_runtime_seconds':120,'ledger_path':str(directory/'rl_ledger.json')},
      'runtime_sha256':canonical_hash(runtime_identity(ROOT)),'experiment':{},'calibration_report':None,'calibration_report_sha256':None}


def scripted_rollout(corpus):
    _,_,rows,tokenizer=corpus
    lookup={tuple(r['input_ids'][:r['prompt_token_count']]):r['input_ids'][r['prompt_token_count']:] for r in rows['train']}
    def generate(model,tok,prompts,device,**kwargs):
        answers=[lookup[tuple(p)] if i%4==0 else [3,2] for i,p in enumerate(prompts)]
        return {'answers':answers,'texts':[tokenizer.decode(a[:-1],skip_special_tokens=False) for a in answers],
                'eos':[True]*len(answers),'generated_tokens':sum(map(len,answers))}
    return generate


@pytest.mark.parametrize('algorithm',['dpo','ppo','grpo'])
def test_cpu_continuous_equals_one_batch_then_resume(corpus,monkeypatch,algorithm):
    monkeypatch.setattr(engine,'generate',scripted_rollout(corpus))
    a=config(corpus,'continuous_'+algorithm,algorithm);parent=sha256_file(a['initial_checkpoint'])
    first=engine.run(a)
    b=config(corpus,'resumed_'+algorithm,algorithm);pause=engine.run(b,stop_after_batches=1)
    assert pause['data_batch_cursor']==1 and pause['status']=='bounded_smoke_stop'
    second=engine.run(b,resume=Path(b['output_dir'])/'last.pt')
    one=torch.load(Path(a['output_dir'])/'last.pt',weights_only=False,map_location='cpu')
    two=torch.load(Path(b['output_dir'])/'last.pt',weights_only=False,map_location='cpu')
    assert all(torch.equal(one['model'][key],two['model'][key]) for key in one['model'])
    assert one['scheduler']==two['scheduler'] and torch.equal(one['rng']['torch'],two['rng']['torch'])
    assert first['examples_seen']==second['examples_seen']==2 and first['plan']['epochs']==1
    assert first['optimizer_step']==(4 if algorithm=='ppo' else 2)
    assert sha256_file(a['initial_checkpoint'])==parent
    model,tok,identity=load_model(Path(a['output_dir'])/'best.pt','cpu')
    assert identity['algorithm']==algorithm


def test_rl_does_not_add_another_data_epoch(corpus):
    cfg=config(corpus,'bad','ppo');cfg['train']['epochs']=2
    with pytest.raises(ValueError,match='epochs=1'):validate(cfg)
