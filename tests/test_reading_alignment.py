from types import SimpleNamespace
import pytest
from llm_reading_data.source import convert_record
from llm_reading_data.verifier import normalize_answer,reward,answer_f1
from llm_reading_data.build import token_balanced,UnionFind,reserve_articles,unique_inputs


def raw():
    return {'source':'DRCKnowledgeTeam/DRCD','source_file':'train.json','source_role':'official_train','source_row':0,
        'source_id':'q1','source_record_sha256':'bound','article_id':'a','paragraph_id':'p','title':'示例',
        'context':'开头的材料足够长，接下来提供关键字㊙以及后面的乙，其余文字用于组成完整的阅读段落。',
        'question':'关键字后面的字是什么？','answers':[]}


def test_original_offsets_must_be_valid_before_any_conversion():
    value=raw();value['answers']=[{'text':'乙','answer_start':0}]
    assert convert_record(value,SimpleNamespace(convert=lambda t:t))[1]=='original_answer_span_mismatch'


def test_simplification_rebuilds_answer_positions_after_length_change():
    value=raw();offset=value['context'].index('乙');value['answers']=[{'text':'乙','answer_start':offset}]
    converted,reason=convert_record(value,SimpleNamespace(convert=lambda t:t.replace('㊙','秘密')))
    assert reason is None and converted['answer_occurrences']['乙']==[offset+1]


def test_context_sensitive_conversion_must_not_invent_a_matching_answer():
    value=raw();value['context']+='乾坤';value['answers']=[{'text':'乾','answer_start':len(value['context'])-2}]
    converter=SimpleNamespace(convert=lambda t:t if '乾坤' in t else t.replace('乾','干'))
    assert convert_record(value,converter)[1]=='converted_answer_not_in_passage'


def test_reward_accepts_human_aliases_but_not_long_explanations_or_context_copy():
    assert reward('「台北」',['台北','台北市'],'zh')==1
    assert reward('答案是台北，因为原文说台北。',['台北'],'zh')==0
    assert reward('The full passage mentions Taipei and many other cities.',['Taipei'],'en')==0
    assert reward('3.14',['314'],'zh')==0
    assert reward('C++',['C'],'en')==0


def test_f1_is_secondary_and_does_not_replace_exact_match():
    assert answer_f1('New York city',['New York'],'en')>.5
    assert reward('New York city',['New York'],'en')==0


def test_balance_uses_supervised_tokens_instead_of_equal_row_counts():
    rows=[{'id':'zh'+str(i),'language':'zh','answer_token_count':9} for i in range(100)]
    rows += [{'id':'en'+str(i),'language':'en','answer_token_count':5} for i in range(100)]
    chosen,report=token_balanced(rows,total=100)
    assert len(chosen)==100 and len({r['id'] for r in chosen})==100
    assert abs(report['zh_answer_token_fraction']-.5)<=.005
    assert report['zh_examples']!=report['en_examples']


def test_balance_never_duplicates_insufficient_data():
    with pytest.raises(ValueError):token_balanced([{'language':'zh','answer_token_count':5}],total=100)


def test_article_or_shared_passage_alias_connects_all_questions():
    uf=UnionFind();uf.union('docA','passage1');uf.union('docB','passage1');uf.union('docB','passage2')
    assert uf.find('docA')==uf.find('docB')==uf.find('passage2')


def test_reserved_articles_are_wholly_identified_for_train_exclusion():
    rows=[{'id':str(i),'group_id':'a'+str(i//3)} for i in range(30)]
    chosen,groups=reserve_articles(rows,8,42,3)
    assert len(chosen)==8 and all(r['group_id'] in groups for r in chosen)
    remaining=[r for r in rows if r['group_id'] not in groups]
    assert not {r['group_id'] for r in chosen}&{r['group_id'] for r in remaining}


def test_official_distinct_question_ids_cannot_duplicate_an_evaluation_input():
    rows=[{'source_id':'a','context':'passage','question':'same question'},
          {'source_id':'b','context':'passage','question':'same question'},
          {'source_id':'c','context':'passage','question':'different question'}]
    assert [r['source_id'] for r in unique_inputs(rows)]==['a','c']
