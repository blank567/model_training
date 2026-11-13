"""只读官方训练文件；保留人工答案，绝不读取fake_answers或合成新答案。"""
from pathlib import Path
import json,re,zipfile
from llm_lab.common.io import canonical_hash


def _view(spec,ordinal,raw,question,context,answer,title='',url='',location=None):
    return {'source':spec['source'],'source_file':spec['path'],'source_family':spec['family'],
       'source_member':spec.get('member'),'source_role':'official_train','source_row':ordinal,
       'source_id':str(raw.get('query_id',raw.get('question_id',raw.get('id',ordinal)))),
       'source_record_sha256':canonical_hash(raw),'question':question,'context':context,
       'original_answers':[answer],'title':title,'source_url':url,'source_location':location or {},
       'original_answer_start':context.index(answer),'source_answer_field':'answers'}


def raw_target(raw):
    """事实题可用发布方人工标注的单实体；多实体答案绝不拆成别名。"""
    if raw.get('fact_or_opinion')!='FACT' or raw.get('question_type')=='YES_NO':return None,None
    answers=list(dict.fromkeys(a.strip() for a in raw.get('answers',[]) if a.strip()))
    if not answers:return None,None
    if raw.get('question_type')=='ENTITY' and not re.search(r'哪些|有哪几|分别|列举|名单|都有什么',raw['question']):
        entities=raw.get('entity_answers') or []
        if entities and all(len(items)==1 and isinstance(items[0],str) and items[0].strip() for items in entities):
            values={items[0].strip() for items in entities}
            if len(values)==1:
                answer=next(iter(values))
                if all(answer in reference for reference in answers):return answer,'entity_answers'
    return (answers[0],'answers') if len(answers)==1 else (None,None)


def records(cache,spec):
    path=Path(cache)/spec['path'];family=spec['family']
    if family in {'robust','quoref'}:
        data=json.loads(path.read_text(encoding='utf-8'));ordinal=0
        for doc in data['data']:
            for paragraph_index,paragraph in enumerate(doc['paragraphs']):
                for qa in paragraph['qas']:
                    current=ordinal;ordinal+=1
                    answers=qa.get('answers') or [];texts=list(dict.fromkeys(a.get('text','').strip() for a in answers))
                    if len(texts)!=1 or not texts[0] or qa.get('is_impossible'):continue
                    context=paragraph['context']
                    if any(type(a.get('answer_start')) is not int or a['answer_start']<0 or context[a['answer_start']:a['answer_start']+len(a['text'])]!=a['text'] for a in answers):continue
                    raw={'title':doc.get('title',''),'url':doc.get('url',''),'paragraph':paragraph_index,'context':context,'qa':qa}
                    result=_view(spec,current,raw,qa['question'],context,texts[0],paragraph.get('title') or doc.get('title',''),doc.get('url',''))
                    result['source_id']=qa['id'];yield result
    elif family=='marco':
        import pyarrow.parquet as pq
        ordinal=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            for raw in batch.to_pylist():
                current=ordinal;ordinal+=1;answers=list(dict.fromkeys(a.strip() for a in raw['answers'] if a.strip()))
                if len(answers)!=1 or answers[0].casefold()=='no answer present.':continue
                p=raw['passages'];n=len(p['passage_text'])
                if len(p['is_selected'])!=n or len(p['url'])!=n:raise ValueError('Mismatched official passage arrays')
                for i in range(n):
                    context=p['passage_text'][i]
                    if p['is_selected'][i]==1 and answers[0] in context:
                        yield _view(spec,current,raw,raw['query'],context,answers[0],url=p['url'][i],location={'passage':i});break
    elif family=='dureader_raw':
        with zipfile.ZipFile(path) as archive,archive.open(spec['member']) as stream:
            for ordinal,line in enumerate(stream):
                raw=json.loads(line);answer,field=raw_target(raw)
                if answer is None:continue
                done=False
                for doc_index,doc in enumerate(raw.get('documents',[])):
                    if not doc.get('is_selected'):continue
                    for paragraph_index,context in enumerate(doc.get('paragraphs',[])):
                        if answer in context:
                            row=_view(spec,ordinal,raw,raw['question'],context,answer,doc.get('title',''),
                                doc.get('url',''),{'document':doc_index,'paragraph':paragraph_index})
                            row.update(source_answer_field=field,publisher_reference_answers=raw['answers'],
                                       publisher_entity_answers=raw.get('entity_answers',[]),fact_or_opinion=raw['fact_or_opinion'])
                            yield row;done=True;break
                    if done:break
    else:raise ValueError('Unsupported public training source')
