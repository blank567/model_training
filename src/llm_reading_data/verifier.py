"""答案别名的保守精确匹配；保留数字中的标点，不奖励复制整段材料。"""
from collections import Counter
import re
import unicodedata


def normalize_answer(text,language):
    if not isinstance(text,str):return None
    text=unicodedata.normalize('NFKC',text).casefold().strip()
    text=re.sub(r'\s+',' ',text)
    if language=='zh':text=re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])','',text)
    pairs=[('"','"'),("'","'"),('“','”'),('‘','’'),('「','」'),('『','』'),('《','》')]
    for _ in range(3):
        for left,right in pairs:
            if len(text)>2 and text.startswith(left) and text.endswith(right):text=text[len(left):-len(right)].strip();break
        else:break
    # 不去掉小数点、数值负号、C++或单位中的标点。
    return text.rstrip('。!?！？').strip()


def reward(completion,answers,language):
    actual=normalize_answer(completion,language)
    if not isinstance(answers,(list,tuple)) or not answers:raise ValueError('Nonempty human answer aliases required')
    expected={normalize_answer(text,language) for text in answers}
    if None in expected or '' in expected:raise ValueError('Invalid human answer alias')
    return float(bool(actual) and actual in expected)


def answer_f1(completion,answers,language):
    """自定义字符/词F1，仅作次要诊断，不冒充官方SQuAD排行榜指标。"""
    def tokens(text):
        value=normalize_answer(text,language) or ''
        return re.findall(r'[\u4e00-\u9fff]|[a-z0-9]+(?:[.\-/][a-z0-9]+)*|[^\w\s]',value) if language=='zh' else value.split()
    predicted=tokens(completion);best=0.
    for answer in answers:
        gold=tokens(answer);overlap=sum((Counter(predicted)&Counter(gold)).values())
        if overlap:best=max(best,2*overlap/(len(predicted)+len(gold)))
    return best
