"""扩充数据的真实边界：首轮关系、旧评测隔离、监督Token平衡和冻结评测字节。"""
from collections import Counter
import copy
import json
from pathlib import Path
import sys

import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import prepare_sft_expansion as expansion
from test_sft_data_audit import corpus,make_row


def cn_source():
    return {'source':'smoltalk_chinese','file':'rewrite.parquet','language':'zh'}


def cn_row(score=4):
    return {'score':score,'difficulty':'容易','conversations':[
        {'role':'user','content':'请改写这句话：小猫在公园里晒太阳。'},
        {'role':'assistant','content':'小猫正在公园里享受阳光。'}]}


def test_chinese_adapter_quarantines_missing_or_low_instruction_score():
    for value in (None,3,float('nan')):
        row,reason=expansion.adapt(cn_row(value),cn_source())
        assert row is None and reason in {'missing_instruction_score','low_instruction_score'}


def test_first_turn_does_not_turn_followup_into_a_standalone_question():
    messages=[{'role':'system','content':'Be concise.'},{'role':'user','content':'Name a fruit.'},
              {'role':'assistant','content':'An apple.'},{'role':'user','content':'Why?'},
              {'role':'assistant','content':'It grows on trees.'}]
    row,reason=expansion.first_turn(messages)
    assert reason is None and row['prompt']=='Name a fruit.' and row['response']=='An apple.'
    assert expansion.first_turn(messages[2:])[1]=='not_self_contained_first_turn'


def test_explicit_task_files_define_task_instead_of_generic_qa_bucket():
    row,reason=expansion.adapt(cn_row(),cn_source())
    assert reason is None and row['task']=='rewriting'
    source={**cn_source(),'file':'summary.parquet'}
    raw=cn_row();raw['conversations'][0]['content']='请总结以下内容。\n小猫在公园里晒太阳。'
    row,reason=expansion.adapt(raw,source)
    assert reason is None and row['input_context']=='小猫在公园里晒太阳。'


def test_existing_smol_source_namespace_is_preserved_across_shards():
    source={'source':'smol_smoltalk','file':'data/train-00002-of-00004.parquet','language':'en',
            'selection_policy':{'source_allowlist':['smollm-rewrite-30k']}}
    raw={'source':'smollm-rewrite-30k','messages':[{'role':'user','content':'Original input.'},
        {'role':'assistant','content':'Edited input.'}]}
    row,reason=expansion.adapt(raw,source)
    assert reason is None and row['dataset']=='smol_smoltalk_2' and row['task']=='rewriting'


def test_oasst_quality_is_not_inferred_from_rank_zero():
    row={'review_result':True,'deleted':False,'rank':0,
         'labels':{'name':['quality','spam'],'value':[.2,0.]}}
    assert not expansion.reviewed_message(row,answer=True)
    row['labels']['value'][0]=.9
    assert expansion.reviewed_message(row,answer=True)
    row['deleted']=True
    assert not expansion.reviewed_message(row,answer=True)


def test_oasst_pairs_require_root_parent_and_choose_reviewed_sibling(tmp_path):
    arrow=pytest.importorskip('pyarrow');pq=pytest.importorskip('pyarrow.parquet')
    def message(identifier,parent,role,quality,rank=None):
        return {'message_id':identifier,'parent_id':parent,'role':role,'text':'A complete question or answer.',
                'lang':'en','message_tree_id':'tree','rank':rank,'review_result':True,'deleted':False,
                'labels':{'name':['quality'],'value':[quality]}}
    rows=[message('root',None,'prompter',.9),message('bad','root','assistant',.2,0),
          message('good','root','assistant',.9,1),message('follow','good','prompter',.9),
          message('reply','follow','assistant',.9,0)]
    path=tmp_path/'train.parquet';pq.write_table(arrow.Table.from_pylist(rows),path)
    source={'source':'oasst2','format':'oasst2_parquet'}
    selected=[row for _,_,row,reason in expansion.source_records(path,source) if reason is None]
    assert len(selected)==1 and selected[0]['answer_message_id']=='good' and selected[0]['tree_id']=='tree'


@pytest.mark.parametrize('prompt,answer,language,reason',[
    ('请用一句话回答。','第一句。第二句。','zh','chinese_one_sentence_violation'),
    ('总结不超过两句话。','甲。乙。丙。','zh','chinese_sentence_limit_violation'),
    ('只输出JSON。','说明：{"a":1}','zh','invalid_required_json'),
    ('Return only valid JSON.','{bad}','en','invalid_required_json'),
    ('Use exactly 2 bullet points.','- one\n- two\n- three','en','exact_bullet_count_violation'),
])
def test_hard_constraints_are_checked(prompt,answer,language,reason):
    assert expansion.constraint_reason(prompt,answer,language)==reason


def test_valid_short_json_and_chinese_sentence_limit_are_kept():
    assert expansion.constraint_reason('只输出JSON。','{"标签":"正面"}','zh') is None
    assert expansion.constraint_reason('不超过三句话。','小猫晒太阳。小狗在睡觉。','zh') is None


def protected_row(prompt='A frozen question?',response='A frozen answer.',context='',split='validation'):
    return {'source':{'revision':'a'*40,'file':'source.json','row_index':1},
            'prompt':prompt,'response':response,'input_context':context,'system':'Common instruction.',
            'split':split}


def test_existing_prompt_and_origin_are_blocked_even_with_new_response():
    guard=expansion.ProtectedKeys([protected_row()])
    source={'revision':'b'*40,'file':'other.json'}
    assert guard.reason(protected_row(response='Different answer.'),source,4)=='existing_sft_prompt'
    row=protected_row(prompt='Unrelated question?')
    assert guard.reason(row,{'revision':'a'*40,'file':'source.json'},1)=='existing_sft_source_row'


def test_frozen_material_fragment_is_detected_at_any_offset():
    text='A distinct passage about apples and rivers with carefully chosen details. '*4
    guard=expansion.ProtectedKeys([protected_row(context=text)])
    row=protected_row(prompt='New question?',response='New answer.',context='Prefix '+text[13:143]+' suffix')
    assert guard.reason(row,{'revision':'b'*40,'file':'other.json'},2)=='existing_sft_holdout_fragment'


def test_rolling_hash_collision_without_real_fragment_does_not_reject(monkeypatch):
    guard=expansion.ProtectedKeys([protected_row(response='Old protected material. '*8)])
    guard.fragments.add(123)
    monkeypatch.setattr(expansion.base,'rolling',lambda text:iter([(0,123)]))
    row=protected_row(prompt='A new unrelated question about a different topic. '*8,response='Entirely different answer.',context='')
    assert guard.reason(row,{'revision':'b'*40,'file':'other.json'},2) is None


def test_common_system_instruction_is_not_treated_as_heldout_material():
    guard=expansion.ProtectedKeys([protected_row()])
    row=protected_row(prompt='A completely new question?',response='A new answer.')
    assert guard.reason(row,{'revision':'b'*40,'file':'other.json'},2) is None


def selection_row(identifier,language,tokens=10,task='rewriting'):
    return {'id':identifier,'language':language,'task':task,'answer_token_count':tokens,'input_ids':[3]*tokens}


def test_expansions_are_nested_retain_baseline_and_are_order_independent():
    baseline=[selection_row('old-zh','zh'),selection_row('old-en','en')]
    candidates=[selection_row(f'{lang}-{index}',lang) for lang in ('zh','en') for index in range(10)]
    result,shortfall=expansion.nested_selection(baseline,candidates,[6,12],{'rewriting':1},42)
    reverse,_=expansion.nested_selection(baseline,list(reversed(candidates)),[6,12],{'rewriting':1},42)
    assert not shortfall and {row['id'] for row in result[6]}<={row['id'] for row in result[12]}
    assert {row['id'] for row in baseline}<={row['id'] for row in result[6]}
    assert result==reverse


def test_language_balance_uses_answer_tokens_not_equal_example_counts():
    baseline=[selection_row('old-zh','zh',30),selection_row('old-en','en',30)]
    candidates=[selection_row(f'{lang}-{i}',lang,10 if lang=='zh' else 30) for lang in ('zh','en') for i in range(30)]
    results,_=expansion.nested_selection(baseline,candidates,[10,18],{'rewriting':1},42)
    for rows in results.values():
        tokens=Counter();counts=Counter()
        for row in rows:tokens[row['language']]+=row['answer_token_count'];counts[row['language']]+=1
        assert .45<=tokens['zh']/sum(tokens.values())<=.55 and counts['zh']>counts['en']


def test_shortfall_is_reported_without_repeating_or_fabricating_samples():
    baseline=[selection_row('old-zh','zh'),selection_row('old-en','en')]
    result,shortfall=expansion.nested_selection(baseline,[selection_row('only-zh','zh')],[10],{'rewriting':1},42)
    assert not result and shortfall=={10:'insufficient_unique_balanced_candidates'}


def test_publication_retokenizes_and_keeps_frozen_evaluation_bytes(corpus):
    root,donor,rows,tokenizer=corpus
    reserve=root/'artifacts/posttrain/reserve';reserve.mkdir(parents=True)
    expansion.base.atomic_json(reserve/'reserve_config.json',{'files':[]})
    output=root/'artifacts/posttrain/expansion';output.mkdir()
    candidates=[]
    for lang in ('zh','en'):
        for index in range(4):
            row=make_row(tokenizer,f'extra-{lang}-{index}','candidate',lang);candidates.append(row)
    expansion.base.write_jsonl(output/'candidates.jsonl',candidates)
    expansion.base.write_jsonl(output/'review_sample.jsonl',[candidates[0],candidates[4]])
    pool_sha=expansion.base.sha256(output/'candidates.jsonl')
    expansion.base.atomic_json(output/'pool_manifest.json',{'candidates_sha256':pool_sha,'deduplication_limits':'fixture'})
    settings={'project_root':'../..','donor_manifest':'manifest.json','donor_manifest_sha256':expansion.base.sha256(root/'manifest.json'),
        'output_dir':'artifacts/posttrain/expansion','work_dir':'runs/assistant/data_prepare/fixture',
        'reserve_dir':'artifacts/posttrain/reserve','train_targets':[6,10],'task_weights':{'fixture':1},
        'max_sequence_tokens':1024,'max_response_tokens':512,'seed':42}
    config=root/'configs/data/expansion.json';config.parent.mkdir(parents=True);expansion.base.atomic_json(config,settings)
    (root/settings['work_dir']).mkdir(parents=True)
    review=root/'review.json';expansion.base.atomic_json(review,{'pool_sha256':pool_sha,'test_content_used':False,
        'reviewed':[{'id':row['id'],'decision':'accept','reason':'Fixture content checked.'} for row in (candidates[0],candidates[4])]})
    frozen={split:(root/f'{split}.jsonl').read_bytes() for split in ('validation','test','pilot')}
    result=expansion.publish(config,review)
    assert [item['train_examples'] for item in result['datasets']]==[6,10]
    for item in result['datasets']:
        folder=Path(item['manifest']).parent
        assert expansion.read_json(folder/'acceptance.json')['status']=='passed'
        for split,body in frozen.items():assert (folder/f'{split}.jsonl').read_bytes()==body
