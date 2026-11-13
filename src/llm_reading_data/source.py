"""把官方JSON/Parquet变成保留原始文章、题目、跨度及来源行的统一只读视图。"""
from pathlib import Path
import hashlib,json,re,unicodedata


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def text_key(text):
    return hashlib.sha256(re.sub(r'\s+','',unicodedata.normalize('NFKC',text).casefold()).encode()).hexdigest()


def fixed_point(text,converter):
    for _ in range(8):
        result=converter.convert(text)
        if result==text:return text
        text=result
    raise ValueError('Chinese conversion failed fixed-point check')


def source_rows(cache,spec):
    path=Path(cache)/spec['local_path'];source=spec['source'];ordinal=0
    if path.suffix=='.parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            for raw in batch.to_pylist():
                answers=raw['answers']
                if len(answers['text'])!=len(answers['answer_start']):raise ValueError('Mismatched official answer arrays')
                values=[{'text':text,'answer_start':start} for text,start in zip(answers['text'],answers['answer_start'])]
                yield {'source':source,'source_file':spec['local_path'],'source_role':spec['role'],'source_row':ordinal,
                    'source_id':raw['id'],'article_id':raw['title'],'title':raw['title'],'paragraph_id':text_key(raw['context']),
                    'context':raw['context'],'question':raw['question'],'answers':values,'source_record_sha256':digest(raw)}
                ordinal+=1
    else:
        data=json.loads(path.read_text(encoding='utf-8'))
        for doc_index,doc in enumerate(data['data']):
            for para_index,para in enumerate(doc['paragraphs']):
                for qa in para['qas']:
                    raw={'article_id':doc.get('id',doc_index),'title':doc.get('title',''),
                         'paragraph_id':para.get('id',para_index),'context':para['context'],'qa':qa}
                    yield {'source':source,'source_file':spec['local_path'],'source_role':spec['role'],'source_row':ordinal,
                        'source_id':qa['id'],'article_id':str(doc.get('id',doc_index)),'title':doc.get('title',''),
                        'paragraph_id':str(para.get('id',para_index)),'context':para['context'],'question':qa['question'],
                        'answers':qa.get('answers',[]),'source_record_sha256':digest(raw)}
                    ordinal+=1


def convert_record(raw,converter):
    language='en' if raw['source']=='rajpurkar/squad' else 'zh'
    context=raw['context'];question=raw['question'];answers=raw['answers']
    if not isinstance(context,str) or not isinstance(question,str) or len(context)<30 or len(question.strip())<2:
        return None,'invalid_context_or_question'
    if not isinstance(answers,list) or not answers:return None,'no_public_answer_annotation'
    original=[]
    for annotation in answers:
        answer=annotation.get('text');start=annotation.get('answer_start')
        if not isinstance(answer,str) or not answer.strip() or type(start) is not int or start<0:
            return None,'invalid_human_annotation'
        if context[start:start+len(answer)]!=answer:return None,'original_answer_span_mismatch'
        original.append(answer)
    title=raw['title']
    if language=='zh':
        context=fixed_point(context,converter);question=fixed_point(question,converter);title=fixed_point(title,converter)
        aliases=[fixed_point(answer,converter).strip() for answer in original]
    else:aliases=[answer.strip() for answer in original]
    aliases=list(dict.fromkeys(aliases))
    if any(not answer or answer not in context for answer in aliases):return None,'converted_answer_not_in_passage'
    if any(answer==context.strip() for answer in aliases):return None,'answer_is_entire_passage'
    spans={answer:[m.start() for m in re.finditer(re.escape(answer),context)] for answer in aliases}
    article=text_key(title.replace('_',' ')) if title.strip() else digest([raw['source'],raw['article_id']])
    return {'source':raw['source'],'source_file':raw['source_file'],'source_role':raw['source_role'],
        'source_row':raw['source_row'],'source_id':raw['source_id'],'source_record_sha256':raw['source_record_sha256'],
        'source_article_id':raw['article_id'],'source_paragraph_id':raw['paragraph_id'],
        'title':title,'language':language,'context':context,'question':question,'answers':aliases,
        'answer_occurrences':spans,'article_key':article,'passage_key':text_key(context),
        'question_key':text_key(question),'group_id':article},None


def make_prompt(context,question,language):
    if language=='zh':return '请根据阅读材料回答问题，只输出原文中的答案，不要解释。\n\n阅读材料：\n'+context+'\n\n问题：'+question
    return 'Answer the question using the passage. Output only the answer span, without explanation.\n\nPassage:\n'+context+'\n\nQuestion: '+question
