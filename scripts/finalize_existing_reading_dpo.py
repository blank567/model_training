"""按用户指定直接使用原合格候选，主对照等量并匹配语言题数，记录实际Token配比。"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,random,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from llm_lab.common.io import canonical_hash,read_json,read_jsonl,sha256_file,write_json
from llm_reading_independent.comparison import ComparisonRLCorpus
from prepare_reading_dpo_pairs import generator_identity,publish


def select_existing(old_matched,new,seed=20261104):
    """旧主对照保留全部合格题；新主对照按相同语言题数确定性抽取。"""
    counts=Counter(r['language'] for r in old_matched)
    if len({r['id'] for r in old_matched})!=len(old_matched):raise ValueError('Duplicated old question')
    if len({r['id'] for r in new})!=len(new):raise ValueError('Duplicated new question')
    selected=[]
    for language,count in sorted(counts.items()):
        values=[r for r in new if r['language']==language]
        if len(values)<count:raise ValueError('Insufficient existing new candidates in language '+language)
        random.Random(str(seed)+'/'+language).shuffle(values);selected.extend(values[:count])
    random.Random(seed).shuffle(selected)
    if len(selected)!=len(old_matched):raise ValueError('Matched counts differ')
    return list(old_matched),selected


def statistics(rows):
    languages=Counter(r['language'] for r in rows);tokens=Counter()
    for row in rows:tokens[row['language']]+=len(row['chosen_ids'])
    return {'examples':len(rows),'languages':dict(languages),'answer_tokens':dict(tokens),
        'zh_answer_token_fraction':tokens['zh']/sum(tokens.values())}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    q=read_json(a.state);plan=read_json(q['plan']);base=Path(q['preference_work']);initial=plan['initial_checkpoint']
    if not a.output.resolve().is_relative_to(ROOT/'artifacts/posttrain'):raise ValueError('Output escapes published artifacts')
    if (a.output/'preference_index.json').exists():raise ValueError('Already published; do not overwrite')
    original=generator_identity(initial);arms={name:ComparisonRLCorpus(Path(q['plan']).parent/spec['manifest']) for name,spec in plan['arms'].items()}
    candidates={}
    for label,name in [('old50k','old_full_50k'),('new25k','new_matched')]:
        state=read_json(base/label/'state.json');file=base/label/'candidates.jsonl'
        if state['cursor']!=len(arms[name]) or file.stat().st_size!=state['file_bytes']:raise ValueError('Candidate journal is incomplete')
        if state['identity']!={**original,'source_arm_identity':arms[name].manifest['data_identity']}:raise ValueError('Frozen generation identity differs')
        rows=read_jsonl(file)
        if len(rows)!=state['pairs']:raise ValueError('Candidate row count differs')
        candidates[label]=rows
    ids={arms['old_matched'][i]['id'] for i in range(len(arms['old_matched']))}
    old=[r for r in candidates['old50k'] if r['id'] in ids];left,right=select_existing(old,candidates['new25k'])
    provenance={'base_generator':original,'publisher_sha256':sha256_file(__file__),'additional_generation_used':False,
        'selection':'All existing valid old matched questions; same number of zh/en questions sampled from existing new candidates',
        'seed':20261104,'strict_50_50_token_gate':False,'user_direction':'不用补齐数据了，就按当前的数据跑吧',
        'old_candidates_sha256':sha256_file(base/'old50k/candidates.jsonl'),'new_candidates_sha256':sha256_file(base/'new25k/candidates.jsonl')}
    if sha256_file(initial['path'])!=initial['sha256']:raise ValueError('Initial policy weights changed')
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(arms['old_matched'].source.tokenizer_path))
    results={};stats={}
    for name,rows in [('old_matched',left),('new_matched',right),('old_full_50k',candidates['old50k'])]:
        results[name]=publish(rows,arms[name],tokenizer,initial,provenance,a.output/name)
        if results[name]['examples']!=len(rows):raise ValueError('Published partial arm differs')
        stats[name]=statistics(rows)
    receipt={'status':'published','plan_sha256':sha256_file(q['plan']),'initial_checkpoint_sha256':initial['sha256'],
        'arms':results,'matched_pair_count':len(left),'source_prompt_counts':{'old_full_50k':50000,'new_matched':25000},
        'statistics':stats,'selection':provenance,'final_test_used':False}
    write_json(a.output/'preference_index.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
