"""保守筛选策略生成的负回答；区分错误材料片段与不符合抽取格式的回答。"""
from difflib import SequenceMatcher
import re,unicodedata
from llm_reading_data.verifier import normalize_answer,reward,answer_f1


def compact(text):
    value=unicodedata.normalize('NFKD',text).casefold()
    return ''.join(c for c in value if c.isalnum() and not unicodedata.combining(c))


def material_span(text,source):
    language=source['language'];normalized=normalize_answer(text,language)
    if language=='en':return bool(re.search(r'(?<!\w)'+re.escape(normalized)+r'(?!\w)',normalize_answer(source['context'],language)))
    return bool(compact(text)) and compact(text) in compact(source['context'])


def acceptable_negative(ids,text,source,*,require_grounded=False):
    if not 2<=len(ids)<=64 or ids[-1]!=2 or any(t in {0,1,2} for t in ids[:-1]):return False
    language=source['language'];answers=source['reference_answers']
    if reward(text,answers,language):return False
    value=compact(text)
    if not value or '�' in text or any(symbol in text for symbol in ['<','>','\n']):return False
    if language=='zh' and not re.search(r'[\u4e00-\u9fff]',text) and not text.strip().replace('.','').isdigit():return False
    # 去除大小写、重音、标点与空白后仍排除包含关系及近似写法，减少漏标别名负例。
    for answer in answers:
        target=compact(answer)
        if value in target or target in value or SequenceMatcher(None,value,target).ratio()>=.8:return False
    if answer_f1(text,answers,language)>.25:return False
    # 任务要求原文片段：材料外回答仅标记为格式负例，不宣称其已被人工判为语义错误。
    if require_grounded and not material_span(text,source):return False
    return True
