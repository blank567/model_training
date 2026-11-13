import json,zipfile
from llm_reading_independent.isolation import IsolationIndex,article_names,compact
from llm_reading_independent.sources import records,raw_target
from llm_reading_independent.quality import reason as quality_reason


def test_question_and_article_exclusion_independent_of_formatting():
    index=IsolationIndex();index.add_row({'question':'Where is Paris?','title':'Paris','context':'An unrelated sufficiently long background passage about a French city.'})
    assert index.reason('WHERE IS PARIS !','different material')=='sft_or_evaluation_question_overlap'
    assert index.reason('New question','different material',url='https://en.wikipedia.org/wiki/Paris')=='sft_or_evaluation_article_overlap'
    assert compact('台 北，是什么？')==compact('台北是什么')


def test_paragraph_and_near_duplicate_exclusion():
    source=' '.join('This is the original evidence about city number '+str(i)+'.' for i in range(15))
    index=IsolationIndex();index.add_passage(source)
    assert index.reason('new',source.upper())=='sft_or_evaluation_passage_overlap'
    assert index.reason('new',source+' An additional short sentence.')=='sft_or_evaluation_near_passage'
    assert index.reason('new','A different topic describes the microscopic structure of volcanic glass and the crystalline formation of minerals.') is None


def test_wikipedia_url_canonicalizes_encoded_article_title():
    assert article_names(url='https://en.wikipedia.org/wiki/2007%E2%80%932008_Nazko_earthquakes') & article_names('2007–2008 Nazko earthquakes')
    assert not article_names(url='not a valid url')


def test_multispan_quoref_gold_is_not_mistaken_for_alternative_aliases(tmp_path):
    context='Alice met Bob in the library while Carol wrote the project notes.'
    value={'data':[{'title':'Example','paragraphs':[{'context':context,'qas':[
        {'id':'multi','question':'Who met in the library?','answers':[{'text':'Alice','answer_start':0},{'text':'Bob','answer_start':10}]},
        {'id':'single','question':'Who wrote the notes?','answers':[{'text':'Carol','answer_start':35}]}]}]}]}
    # 使用实际跨度，确保测试关注多段答案的语义而非错误偏移。
    value['data'][0]['paragraphs'][0]['qas'][1]['answers'][0]['answer_start']=context.index('Carol')
    (tmp_path/'q.json').write_text(json.dumps(value),encoding='utf-8')
    got=list(records(tmp_path,{'source':'allenai/quoref','path':'q.json','family':'quoref'}))
    assert len(got)==1 and got[0]['source_id']=='single' and got[0]['original_answers']==['Carol']


def test_raw_dureader_uses_actual_human_answer_not_fake_span(tmp_path):
    row={'question_id':1,'question':'What is the name?','question_type':'ENTITY','fact_or_opinion':'FACT','answers':['Alice'],
         'fake_answers':['Bob'],'documents':[{'is_selected':True,'title':'Sample','paragraphs':['Alice met Bob at the historical city library.']}]}
    archive=tmp_path/'raw.zip'
    with zipfile.ZipFile(archive,'w') as stream:stream.writestr('raw/trainset/search.train.json',json.dumps(row)+'\n')
    got=list(records(tmp_path,{'source':'baidu/DuReader-2.0','path':'raw.zip','family':'dureader_raw','member':'raw/trainset/search.train.json'}))
    assert got[0]['original_answers']==['Alice'] and got[0]['original_answer_start']==0


def test_single_entity_annotation_is_allowed_only_when_all_human_references_agree():
    row={'question':'哪一个城市？','question_type':'ENTITY','fact_or_opinion':'FACT',
         'answers':['城市是北京。','北京为答案。'],'entity_answers':[['北京'],['北京']]}
    assert raw_target(row)==('北京','entity_answers')
    row['entity_answers']=[['北京','上海']]
    assert raw_target(row)==(None,None)
    row['entity_answers']=[['北京']];row['answers']=['北京','上海']
    assert raw_target(row)==(None,None)


def test_opinion_and_yes_no_cannot_be_relabelled_as_extractive_entity_questions():
    row={'question':'哪个品牌最好？','question_type':'ENTITY','fact_or_opinion':'OPINION',
         'answers':['品牌A'],'entity_answers':[['品牌A']]}
    assert raw_target(row)==(None,None)


def test_contradictory_weekday_is_filtered_without_using_a_model():
    row={'language':'zh','question':'历史事件日期是什么？','context':'记载为2017年7月12日(星期六)，随后继续发生其他事情。'}
    assert quality_reason(row)=='contradictory_weekday'
    row['context']='记载为2017年7月12日(星期三)，随后继续发生其他事情。'
    assert quality_reason(row) is None


def test_advertising_and_health_opinions_are_not_used_as_factual_training_targets():
    assert quality_reason({'language':'zh','question':'喝什么水最健康','context':'正文'})
    assert quality_reason({'language':'zh','question':'在哪里工作？','context':'底薪很高，包吃包住，关注公众平台有礼。'})
