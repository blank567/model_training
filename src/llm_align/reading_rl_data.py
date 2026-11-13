"""阅读任务的只读RL数据视图：复用已验收分区，不复制上下文、不伪造DPO负例。"""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import os

from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json
from llm_sft.data import SFTCorpus
from llm_reading_data.verifier import reward

ROOT=Path(__file__).resolve().parents[2]
VERIFIER=ROOT/'src/llm_reading_data/verifier.py'


def _identity(manifest):
    return canonical_hash({k:v for k,v in manifest.items() if k not in {'data_identity','created_at'}})


def _parent(path,manifest):
    reference=Path(manifest['source']['manifest'])
    if reference.is_absolute():raise ValueError('RL source reference must be relative')
    parent=(Path(path).parent/reference).resolve()
    if not parent.is_relative_to(Path(path).parent.parent.resolve()):raise ValueError('RL source escapes sibling data releases')
    if sha256_file(parent)!=manifest['source']['manifest_sha256']:raise ValueError('Frozen RL source manifest changed')
    return parent


class ReadingRLCorpus:
    """只向策略提供prefix_ids；标准答案单独保留给奖励计算。"""
    def __init__(self,path,split='train',*,allow_test=False,allow_fixture=False):
        if split not in {'train','validation'} and not (split=='test' and allow_test):
            raise ValueError('Final test needs explicit allow_test=True and must not be used to train')
        path=Path(path);manifest=read_json(path)
        if manifest.get('kind')!='reading_rl_view' or manifest.get('data_identity')!=_identity(manifest):
            raise ValueError('Invalid RL view identity')
        if manifest['reader_sha256']!=sha256_file(Path(__file__)):raise ValueError('Frozen RL reader changed')
        if manifest['reward']['code_sha256']!=sha256_file(VERIFIER):raise ValueError('Frozen RL reward changed')
        self.source=SFTCorpus(_parent(path,manifest),split,allow_test=allow_test)
        if self.source.fixture and not allow_fixture:raise ValueError('Fixture is not production RL data')
        if self.source.identity['data_identity']!=manifest['source']['data_identity']:raise ValueError('Wrong source data identity')
        spec=manifest['splits'][split]
        if len(self.source)!=spec['examples'] or self.source.identity['split_sha256']!=spec['sha256']:
            raise ValueError('RL split identity mismatch')
        for row in self.source.rows:
            if not row.get('human_annotated') or row.get('reward_type')!='human_alias_exact':
                raise ValueError('RL requires public human answer annotations')
            if not row.get('answers') or row['response'] not in row['answers']:raise ValueError('Unbound chosen/gold answer')
            if row['prompt_token_count']+manifest['generation']['max_new_tokens']>1024:
                raise ValueError('RL prompt plus generation exceeds context')
        self.manifest=manifest;self.split=split
        self.identity={'data_identity':manifest['data_identity'],'manifest_sha256':sha256_file(path),
                       'source':self.source.identity,'reader_sha256':manifest['reader_sha256'],
                       'reward_code_sha256':manifest['reward']['code_sha256']}

    def __len__(self):return len(self.source)

    def __getitem__(self,index):
        row=self.source.rows[index]
        return {'id':row['id'],'language':row['language'],'prompt':row['prompt'],
                'prefix_ids':list(row['input_ids'][:row['prompt_token_count']]),
                'reference_answers':list(row['answers']),'chosen':row['response'],
                'source':row.get('source'),'group_id':row['group_id']}

    def score(self,index,completion):
        row=self.source.rows[index]
        return reward(completion,row['answers'],row['language'])


def prepare(parent,target,*,allow_fixture=False):
    """发布索引清单，不复制大JSONL，也不使用任何测试答案生成偏好对。"""
    parent=Path(parent).resolve();target=Path(target).resolve();data=read_json(parent)
    if target.exists():raise ValueError('Never overwrite a prior RL release')
    if data.get('task')!='extractive_reading_qa' or data.get('status')!='published':
        raise ValueError('RL needs audited and published reading QA data')
    if data.get('fixture') and not allow_fixture:raise ValueError('Fixture is not production data')
    source=SFTCorpus(parent,'train');validation=SFTCorpus(parent,'validation')
    if not data.get('fixture') and (len(source)!=50000 or len(validation)!=2000):raise ValueError('Wrong production split sizes')
    if not parent.is_relative_to(target.parent.parent):raise ValueError('Keep RL view beside its immutable source release')
    manifest={'schema_version':1,'kind':'reading_rl_view','task':'extractive_reading_qa',
        'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'source':{'manifest':Path(os.path.relpath(parent,target.parent)).as_posix(),
                  'manifest_sha256':sha256_file(parent),'data_identity':source.identity['data_identity'],
                  'tokenizer_sha256':source.identity['tokenizer_sha256']},
        'splits':{name:{'source_split':name,'examples':data['splits'][name]['examples'],
                        'sha256':data['splits'][name]['sha256']} for name in ['train','validation','test']},
        'reader_sha256':sha256_file(Path(__file__)),
        'reward':{'name':'strict_human_alias_exact_v1','code_sha256':sha256_file(VERIFIER),'values':[0,1]},
        'generation':{'max_new_tokens':64,'max_context_tokens':1024,'eos_id':2},
        'training':{'epochs':1,'stop_epochs':None,'sft_training_prompts_reused':True,
                    'selection':'all source training records once; no resampling or evaluation promotion'},
        'ppo_grpo':{'input_status':'ready','policy_status':'awaiting_this_reading_sft',
                    'launch_requires':'development reward calibration and checked algorithm implementation'},
        'dpo':{'status':'awaiting_policy_generated_rejected','chosen':'unaltered public human answer',
               'rejected_count':0,'negative_policy_checkpoint_sha256':None,
               'rule':'Generate from the same selected reading SFT as PPO/GRPO; retain reward-0 candidates, review and deduplicate'},
        'evaluation':{'validation_for_selection':True,'test_for_training':False,'final_test_examples':data['splits']['test']['examples']}}
    manifest['data_identity']=_identity(manifest);write_json(target,manifest)
    for split in ['train','validation']:ReadingRLCorpus(target,split,allow_fixture=allow_fixture)
    return manifest
