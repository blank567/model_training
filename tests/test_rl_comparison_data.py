from pathlib import Path
import pytest
from llm_lab.common.io import read_json,write_json
from llm_reading_independent.comparison import write_arm,ComparisonRLCorpus,identity
from test_sft_runtime_data import corpus,repin


def setup(corpus):
    directory,m,rows,_=corpus;m.update(task='extractive_reading_qa',status='published')
    for selected in rows.values():
        for row in selected:row.update(answers=[row['response']],human_annotated=True,reward_type='human_alias_exact')
    repin(directory,m,rows);target=directory/'comparison/old.json';target.parent.mkdir()
    write_arm(directory/'manifest.json',target,[1,0],'old_seen');return target


def test_controlled_reuse_arm_keeps_exact_selected_prefix_and_gold(corpus):
    path=setup(corpus);view=ComparisonRLCorpus(path,allow_fixture=True)
    assert len(view)==2 and view[0]['id'].endswith('en') and view[1]['id'].endswith('zh')
    assert view.score(0,view[0]['chosen'])==1
    assert view[0]['prefix_ids']==view.source.rows[1]['input_ids'][:view.source.rows[1]['prompt_token_count']]


def test_selected_id_cannot_silently_point_at_another_row(corpus):
    path=setup(corpus);m=read_json(path);m['selection'][0]['id']='different';m['data_identity']=identity(m);write_json(path,m)
    with pytest.raises(ValueError,match='redirected'):ComparisonRLCorpus(path,allow_fixture=True)


def test_test_data_cannot_be_promoted_by_modifying_a_selection_manifest(corpus):
    path=setup(corpus);m=read_json(path);m['source_split']='test';m['data_identity']=identity(m);write_json(path,m)
    with pytest.raises(ValueError,match='only use train'):ComparisonRLCorpus(path,allow_fixture=True)
