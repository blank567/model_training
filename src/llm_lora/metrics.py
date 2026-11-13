"""只评分可明确解析的格式约束；不把未覆盖要求计作已通过。"""
import json
import re


def supported_constraints(prompt,answer):
    checks=[]
    def add(name,passed):checks.append({'name':name,'passed':bool(passed)})
    if re.search(r'all lowercase|no capital letters|全部小写|全文小写',prompt,re.I):add('lowercase',not re.search('[A-Z]',answer))
    if re.search(r'all uppercase|全部大写|全文大写',prompt,re.I):add('uppercase',not re.search('[a-z]',answer))
    if re.search(r'return\s+only\s+(?:valid\s+)?json|(?:只|仅)(?:返回|输出).{0,6}JSON',prompt,re.I):
        try:json.loads(answer);valid=True
        except (ValueError,TypeError):valid=False
        add('json_only',valid)
    for match in re.finditer(r'\b(at least|at most|exactly|less than|no more than)\s+(\d+)\s+(bullet points|words|placeholders)\b',prompt,re.I):
        comparator,limit,unit=match.group(1).lower(),int(match.group(2)),match.group(3).lower()
        if unit=='words':count=len(re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*",answer))
        elif unit=='placeholders':count=len(re.findall(r'\[[^\]\n]+\]',answer))
        else:count=len(re.findall(r'(?m)^\s*(?:[-*•]|\d+[.)])\s+',answer))
        passed=count>=limit if comparator=='at least' else count==limit if comparator=='exactly' else count<limit if comparator=='less than' else count<=limit
        add(unit.replace(' ','_')+'_'+comparator.replace(' ','_'),passed)
    quoted=r"(?:'([^']{1,60})'|\"([^\"]{1,60})\"|\[([^\]\n]{1,60})\])"
    for match in re.finditer(r'\bword\s+'+quoted+r'\s+(?:should|must)\s+(?:appear|occur)\s+(at least|at most|exactly)\s+(\d+)\s+times',prompt,re.I):
        word=next(value for value in match.groups()[:3] if value is not None)
        count=len(re.findall(r'(?<!\w)'+re.escape(word)+r'(?!\w)',answer,re.I));limit=int(match.group(5));comparison=match.group(4).lower()
        add('keyword_frequency',count>=limit if comparison=='at least' else count<=limit if comparison=='at most' else count==limit)
    return {'supported_check_count':len(checks),'checks':checks,'all_supported_passed':all(item['passed'] for item in checks) if checks else None,
            'does_not_cover_all_instruction_requirements':True}


def summarize_constraints(predictions):
    covered=[row['constraint_checks'] for row in predictions if row['constraint_checks']['supported_check_count']]
    return {'covered_examples':len(covered),'total_examples':len(predictions),
        'all_supported_constraints_pass_fraction':sum(row['all_supported_passed'] for row in covered)/len(covered) if covered else None,
        'not_general_task_accuracy':True,'limits':'Only recognized explicit lexical/format checks; no semantic correctness or unsupported constraints implied'}
