"""RL视图隔离最终测试、绑定来源身份，且策略输入不包含SFT目标后缀。"""
from pathlib import Path
import json
import pytest
from llm_align.reading_rl_data import ReadingRLCorpus,prepare
from test_sft_runtime_data import corpus,repin


def make(corpus):
    directory,manifest,rows,tokenizer=corpus
    manifest.update(task='extractive_reading_qa',status='published')
    for selected in rows.values():
        for row in selected:row.update(answers=[row['response']],human_annotated=True,reward_type='human_alias_exact')
    repin(directory,manifest,rows)
    path=directory/'rl/manifest.json';prepare(directory/'manifest.json',path,allow_fixture=True)
    return directory,path,rows


def test_policy_input_excludes_sft_answer_and_reward_remains_separate(corpus):
    directory,path,rows=make(corpus);view=ReadingRLCorpus(path,allow_fixture=True)
    item=view[0];raw=rows['train'][0]
    assert item['prefix_ids']==raw['input_ids'][:raw['prompt_token_count']]
    assert len(item['prefix_ids'])<len(raw['input_ids']) and view.score(0,raw['response'])==1
    assert view.score(0,'This entire unrelated paragraph is not a gold answer')==0
    assert view.manifest['dpo']['rejected_count']==0


def test_test_split_cannot_silently_become_training_input(corpus):
    _,path,_=make(corpus)
    with pytest.raises(ValueError,match='explicit'):ReadingRLCorpus(path,'test',allow_fixture=True)
    test=ReadingRLCorpus(path,'test',allow_test=True,allow_fixture=True)
    assert all(test[i]['id'].startswith('test') for i in range(len(test)))


def test_source_mutation_invalidates_the_entire_rl_view(corpus):
    directory,path,_=make(corpus)
    with (directory/'manifest.json').open('a',encoding='utf-8') as stream:stream.write('\n')
    with pytest.raises(ValueError,match='source manifest changed'):ReadingRLCorpus(path,allow_fixture=True)


def test_fixture_cannot_be_mistaken_for_real_rl_data(corpus):
    _,path,_=make(corpus)
    with pytest.raises(ValueError,match='Fixture'):ReadingRLCorpus(path)
