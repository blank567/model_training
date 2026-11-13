"""独立重读所有分区、源文本和分词结果；不进行模型测试或训练。"""
from collections import Counter
from importlib import metadata
from pathlib import Path
import json

from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json
from .rules import question_hash


def audit(manifest_path,cache,publish=False):
    if publish and (Path(manifest_path).parent/'DO_NOT_TRAIN.json').exists():
        raise ValueError('Quarantined dataset cannot be published')
    import pyarrow.parquet as pq
    from opencc import OpenCC
    from tokenizers import Tokenizer
    from .build import clean_text,training_text,IDENTITY_FIELDS
    from .rules import classify,reject_reason
    path=Path(manifest_path);directory=path.parent;m=read_json(path);cache=Path(cache)
    if metadata.version('OpenCC')!='1.1.9':raise ValueError('Wrong normalization implementation')
    if canonical_hash({k:m[k] for k in IDENTITY_FIELDS})!=m['data_identity']:raise ValueError('Manifest identity mismatch')
    lock=read_json(cache/'sources.lock.json')
    if sha256_file(cache/'sources.lock.json')!=m['origin']['source_lock_sha256']:raise ValueError('Source lock changed')
    for spec in lock['sources']:
        if sha256_file(cache/spec['local_path'])!=spec['sha256']:raise ValueError('Source file changed')
    en=read_json(cache/'TIGER-Lab__MathInstruct/MathInstruct.json')
    zh={r['id']:r for r in pq.read_table(cache/'ALmonster__MathInstruct-Chinese/mathinstruct-zh.parquet').to_pylist()}
    converter=OpenCC('t2s');tokenizer=Tokenizer.from_file(str(directory/m['tokenizer']['path']))
    if sha256_file(directory/m['tokenizer']['path'])!=m['tokenizer']['sha256']:raise ValueError('Tokenizer changed')
    if m['tokenizer']['sha256']!='6040c90c045836982a47bc0544942f122d0446bf9ed4d2ac1929befd0d4b86b3':raise ValueError('Wrong production tokenizer')
    groups={};questions={};records={};specs={}
    for split,spec in m['splits'].items():
        source=directory/spec['path']
        if sha256_file(source)!=spec['sha256']:raise ValueError('Split checksum changed')
        lang=Counter();tokens=Counter();topics=Counter();ids=set();splitgroups=set();qset=set();pilotrefs={}
        with source.open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);rid=row['id'];language=row['language'];provenance=row['provenance']
                if rid in ids or row['split']!=split or row['schema_version']!=1 or language not in {'zh','en'}:raise ValueError('Invalid row identity')
                index=int(provenance['original_row_one_based'])-1;original=en[index]
                original_q=clean_text(original['instruction'])
                if row['group_id']!=question_hash(original_q) or row['topic']!=classify(original_q,original['source']):raise ValueError('Original problem group/topic mismatch')
                if language=='zh':
                    translated=zh[provenance['original_row_one_based']]['messages']
                    q=clean_text(translated[0]['content'],converter);a=clean_text(translated[1]['content'],converter)
                    if converter.convert(row['prompt'])!=row['prompt'] or converter.convert(row['response'])!=row['response']:raise ValueError('Chinese not simplified fixed point')
                else:q,a=original_q,clean_text(original['output'])
                if row['question']!=q or reject_reason(q,a,original['source'],language):raise ValueError('Source identity/quality changed')
                expected_prompt,expected_response=training_text(q,a,language)
                if row['prompt']!=expected_prompt or row['response']!=expected_response:raise ValueError('Training text is not traceable to source')
                before=tokenizer.encode('User:\n'+row['prompt']+'\n\nAssistant:\n',add_special_tokens=False).ids
                after=tokenizer.encode(row['response'],add_special_tokens=False).ids+[2]
                if row['input_ids']!=before+after or row['loss_mask']!=[0]*len(before)+[1]*len(after):raise ValueError('Retokenization or answer loss mask mismatch')
                if not 2<=len(row['input_ids'])<=1024 or row['input_ids'][-1]!=2 or any(v in {0,1,2} for v in row['input_ids'][:-1]):raise ValueError('Context/special token invalid')
                if row['prompt_token_count']!=len(before) or row['answer_token_count']!=len(after):raise ValueError('Token counts invalid')
                if row['normalized_text_sha256']!=canonical_hash([language,'',row['prompt'],row['response']]):raise ValueError('Normalized text identity invalid')
                ids.add(rid);splitgroups.add(row['group_id']);qset.add(question_hash(row['question']))
                pilotrefs[rid]=canonical_hash({k:v for k,v in row.items() if k!='split'})
                lang[language]+=1;tokens[language]+=len(after);topics[row['topic']]+=1
        if len(ids)!=spec['examples'] or lang['en']!=lang['zh'] or len(splitgroups)*2!=len(ids):raise ValueError('Parallel record count/group/language mismatch')
        if any(topics[k]!=m['statistics'][split]['topics'][k] for k in topics):raise ValueError('Coverage report does not match rows')
        groups[split]=splitgroups;questions[split]=qset;records[split]=pilotrefs
        specs[split]={'sha256':spec['sha256'],'examples':len(ids),'groups':len(splitgroups),
                      'answer_tokens':dict(tokens),'language_examples':dict(lang),'topics':dict(topics),
                      'zh_answer_token_fraction':tokens['zh']/sum(tokens.values())}
        print(json.dumps({'audit_split':split,'records':len(ids),'status':'passed'}),flush=True)
    for i,a in enumerate(['train','validation','test']):
        for b in ['train','validation','test'][i+1:]:
            if groups[a]&groups[b] or questions[a]&questions[b]:raise ValueError('Cross-split original or exact question leakage')
    if any(records['train'].get(key)!=value for key,value in records['pilot'].items()):raise ValueError('Pilot not an exact subset of train')
    report={'schema_version':1,'status':'passed','fixture':False,'structurally_ready':True,
        'data_identity':m['data_identity'],'tokenizer_sha256':m['tokenizer']['sha256'],'splits':specs,
        'language_balance_basis':'Equal language record counts; token fractions are measured, not assumed equal',
        'mathematical_correctness_certified':False,'subject_labels':'heuristic, not gold difficulty or taxonomy',
        'checks':{k:True for k in ['file_hashes','retokenization','answer_mask_and_eos','normalized_text_identity',
            'declared_group_and_exact_split_isolation','pilot_subset','language_token_balance','explicit_quality_rules',
            'chinese_fixed_point','upstream_source_text_identity']}}
    target=directory/'acceptance.json'
    if publish:
        if m.get('status')=='published':raise ValueError('Refusing to rewrite a published dataset')
        write_json(target,report);m['acceptance']={'status':'passed','report':target.name,'sha256':sha256_file(target)}
        m['status']='published';write_json(path,m)
    return report
