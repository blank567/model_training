"""旧SFT已见题与新独立题的等量RL对照；仅发布数据索引，不启动训练。"""
from pathlib import Path
from collections import Counter
import os
from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json
from llm_sft.data import SFTCorpus
from llm_reading_data.build import token_balanced
from llm_reading_data.verifier import reward
from .isolation import compact

ROOT=Path(__file__).resolve().parents[2]


def identity(value):return canonical_hash({k:v for k,v in value.items() if k!='data_identity'})


def write_arm(source,path,indices,condition):
    source=Path(source).resolve();path=Path(path).resolve();corpus=SFTCorpus(source,'train')
    if path.exists():raise ValueError('Never overwrite a comparison arm')
    if condition not in {'old_seen','new_independent'}:raise ValueError('Unknown condition')
    rows=[corpus.rows[i] for i in indices];tokens=Counter();languages=Counter();input_tokens=0
    for row in rows:
        tokens[row['language']]+=row['answer_token_count'];languages[row['language']]+=1;input_tokens+=len(row['input_ids'])-1
    manifest={'schema_version':1,'kind':'reading_rl_comparison_arm','condition':condition,'source_split':'train',
        'source_manifest':Path(os.path.relpath(source,path.parent)).as_posix(),'source_manifest_sha256':sha256_file(source),
        'source_data_identity':corpus.identity['data_identity'],'tokenizer_sha256':corpus.identity['tokenizer_sha256'],
        'reader_sha256':sha256_file(Path(__file__)),'reward_code_sha256':sha256_file(ROOT/'src/llm_reading_data/verifier.py'),
        'selection':[{'source_index':i,'id':corpus.rows[i]['id']} for i in indices],
        'statistics':{'examples':len(rows),'languages':dict(languages),'answer_tokens':dict(tokens),'input_tokens':input_tokens,
                      'zh_answer_token_fraction':tokens['zh']/sum(tokens.values())},
        'sft_training_prompts_reused':condition=='old_seen','purpose':'explicit equal-size data comparison approved by user',
        'epochs':1,'stop_epochs':None,'training_started':False}
    manifest['data_identity']=identity(manifest);write_json(path,manifest);return manifest


class ComparisonRLCorpus:
    def __init__(self,path,*,allow_fixture=False):
        path=Path(path);self.manifest=read_json(path);m=self.manifest
        if m.get('kind')!='reading_rl_comparison_arm' or m.get('data_identity')!=identity(m):raise ValueError('Arm identity changed')
        if m['source_split']!='train':raise ValueError('Comparison training can only use train')
        if m['reader_sha256']!=sha256_file(Path(__file__)):raise ValueError('Frozen comparison reader changed')
        if m['reward_code_sha256']!=sha256_file(ROOT/'src/llm_reading_data/verifier.py'):raise ValueError('Frozen reward changed')
        reference=Path(m['source_manifest'])
        if reference.is_absolute():raise ValueError('Source reference must be relative')
        source=(path.parent/reference).resolve()
        if not source.is_relative_to(path.parent.parent.resolve()) or sha256_file(source)!=m['source_manifest_sha256']:
            raise ValueError('Source manifest changed or escapes data releases')
        self.source=SFTCorpus(source,'train')
        if self.source.fixture and not allow_fixture:raise ValueError('Fixture is not real RL training data')
        if self.source.identity['data_identity']!=m['source_data_identity']:raise ValueError('Wrong source data')
        self.indices=[];seen=set()
        for spec in m['selection']:
            i=spec['source_index']
            if type(i) is not int or not 0<=i<len(self.source) or i in seen or self.source.rows[i]['id']!=spec['id']:
                raise ValueError('Invalid, duplicated or redirected selected row')
            self.indices.append(i);seen.add(i)
        if len(self.indices)!=m['statistics']['examples']:raise ValueError('Arm count mismatch')
        if not self.source.fixture and not (20000<=len(self.indices)<=30000 or m['condition']=='old_seen' and len(self.indices)==50000):
            raise ValueError('Production arms must use the approved matched size or the original 50000 examples')
        if m['condition']=='new_independent':
            data=self.source.manifest;report=read_json(source.parent/data['acceptance']['report'])
            if data.get('training_policy',{}).get('sft_training_prompts_reused') is not False:
                raise ValueError('New arm must be independently prepared')
            if not all(report['checks'].get(name) for name in ['cross_stage_question_isolation','cross_stage_passage_isolation','known_article_isolation']):
                raise ValueError('New arm lacks independent isolation evidence')
        elif m['condition']!='old_seen':raise ValueError('Unknown data arm')

    def __len__(self):return len(self.indices)

    def __getitem__(self,index):
        row=self.source.rows[self.indices[index]]
        return {'id':row['id'],'language':row['language'],'prompt':row['prompt'],
            'prefix_ids':list(row['input_ids'][:row['prompt_token_count']]),'reference_answers':list(row['answers']),
            'chosen':row['response'],'group_id':row['group_id']}

    def score(self,index,completion):
        row=self.source.rows[self.indices[index]];return reward(completion,row['answers'],row['language'])


def prepare(old_source,new_source,out,checkpoint,authorized=False):
    out=Path(out);old_source=Path(old_source).resolve();new_source=Path(new_source).resolve()
    if out.exists():raise ValueError('Never overwrite a comparison plan')
    old=SFTCorpus(old_source,'train');new=SFTCorpus(new_source,'train')
    if not 20000<=len(new)<=30000:raise ValueError('Expected the approved 20000-30000-example independent dataset')
    selected,balance=token_balanced(old.rows,total=len(new),seed=20261101)
    positions={row['id']:i for i,row in enumerate(old.rows)};indices=[positions[r['id']] for r in selected]
    old_questions={compact(r['question']) for r in old.rows};old_passages={compact(r['context']) for r in old.rows}
    if any(compact(r['question']) in old_questions or compact(r['context']) in old_passages for r in new.rows):
        raise ValueError('New arm overlaps full SFT training source')
    for split in ['validation','test']:
        if old.manifest['splits'][split]['sha256']!=new.manifest['splits'][split]['sha256']:raise ValueError('Evaluation is not identical')
    out.mkdir(parents=True)
    arms={}
    for name,condition,source,selection in [('old_matched','old_seen',old_source,indices),
        ('new_matched','new_independent',new_source,list(range(len(new)))),('old_full_50k','old_seen',old_source,list(range(len(old))))]:
        path=out/(name+'.json');manifest=write_arm(source,path,selection,condition);view=ComparisonRLCorpus(path)
        if len(view)!=len(selection):raise ValueError('Wrong arm length')
        arms[name]={'manifest':path.name,'sha256':sha256_file(path),'data_identity':manifest['data_identity'],
                         'statistics':manifest['statistics'],'sft_training_prompts_reused':manifest['sft_training_prompts_reused']}
    plan={'schema_version':1,'status':'data_ready_training_authorized' if authorized else 'prepared_training_paused_by_user','launch_enabled':authorized,'training_started':False,
        'initial_checkpoint':checkpoint,'arms':arms,'algorithms':['DPO','PPO+RLVR','GRPO+RLVR'],
        'planned_runs':9,'shared_settings':{'epochs':1,'stop_epochs':None,'matched_train_examples':len(new),'extended_old_train_examples':len(old),
          'same_initial_weights_per_run':True,'same_hyperparameters_between_arms_for_each_algorithm':True,
          'ppo_optimization_passes_per_rollout':1,'grpo_iterations':1,'max_new_tokens':64},
        'evaluation':{name:{'examples':old.manifest['splits'][name]['examples'],'sha256':old.manifest['splits'][name]['sha256']}
                       for name in ['validation','test']},
        'readiness':{'data_arms':'ready','dpo_completed_pairs_per_arm':0,'policy_generation':'not_started',
                     'algorithm_implementations':'not_yet_validated_for_reading_comparison'},
        'limitations':['Source, difficulty and sequence length differ; this measures data-regime outcomes, not a causal reuse-only effect',
                       'Equal examples and one epoch do not imply equal input/output tokens or wall time; report both',
                       'Primary final test is from original reading sources; do not claim source-neutral evaluation'],
        'resume_requires_user_instruction':not authorized}
    plan['plan_identity']=canonical_hash(plan);write_json(out/'plan.json',plan);return plan
