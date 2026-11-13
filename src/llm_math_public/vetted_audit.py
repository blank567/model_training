"""独立核对新公开数学候选的源行、监督Token、分区和抽检回执。"""
from collections import Counter,defaultdict
from pathlib import Path
import json

from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json
from .build import IDENTITY_FIELDS
from .rules import question_hash


def audit(path,cache,publish=False,review=None):
    import pyarrow.parquet as pq
    from opencc import OpenCC
    from tokenizers import Tokenizer
    from .vetted import parse_row,formula_key
    path=Path(path);cache=Path(cache);directory=path.parent;m=read_json(path)
    if (directory/'DO_NOT_TRAIN.json').exists():raise ValueError('Dataset is quarantined')
    if m['data_identity']!=canonical_hash({k:m[k] for k in IDENTITY_FIELDS}):raise ValueError('Dataset identity changed')
    if sha256_file(cache/'sources.lock.json')!=m['origin']['source_lock_sha256']:raise ValueError('Source lock changed')
    wanted=defaultdict(dict);allrows={};groups={};questions={};formulas={};stats={}
    for split,spec in m['splits'].items():
        file=directory/spec['path']
        if sha256_file(file)!=spec['sha256']:raise ValueError('Split checksum changed')
        rows=[json.loads(line) for line in file.open(encoding='utf-8')]
        allrows[split]=rows
        for row in rows:wanted[row['upstream_file']][row['upstream_row']]=row['source_record_sha256']
    references={}
    for name,indices in wanted.items():
        original=cache/name
        spec=next(s for s in m['origin']['public_sources'] if s['local_path']==name)
        if sha256_file(original)!=spec['sha256']:raise ValueError('Raw source changed')
        cursor=0
        for batch in pq.ParquetFile(original).iter_batches(batch_size=4096):
            for raw in batch.to_pylist():
                if cursor in indices:
                    if canonical_hash(raw)!=indices[cursor]:raise ValueError('Candidate does not match original source record')
                    references[(name,cursor)]=raw
                cursor+=1
        if sum(key[0]==name for key in references)!=len(indices):raise ValueError('Missing original source rows')
        print(json.dumps({'source_audit':name,'selected_unique_records':len(indices),'status':'passed'}),flush=True)
    converter=OpenCC('t2s');tokenizer=Tokenizer.from_file(str(directory/'tokenizer.json'))
    if sha256_file(directory/'tokenizer.json')!=m['tokenizer']['sha256']:raise ValueError('Tokenizer changed')
    rows_by_id={}
    for split,rows in allrows.items():
        ids=set();gset=set();qset=set();fset=set();langs=Counter();topics=Counter();tokens=Counter()
        for row in rows:
            if row['id'] in ids or row['split']!=split:raise ValueError('Row ID/split invalid')
            raw=references[(row['upstream_file'],row['upstream_row'])]
            kind='numina' if row['upstream_file'].startswith('project-numina__') else 'cmm'
            expected,reason=parse_row(raw,kind,converter)
            if reason:raise ValueError('Quality rule rejected selected upstream row: '+reason)
            for key in ['question','prompt','response','final_answer','source','publisher_topic','publisher_grade']:
                if row[key]!=expected[key]:raise ValueError('Source-to-training text mismatch: '+key)
            if row['group_id']!=question_hash(row['question']):raise ValueError('Group identity mismatch')
            p=tokenizer.encode('User:\n'+row['prompt']+'\n\nAssistant:\n',add_special_tokens=False).ids
            a=tokenizer.encode(row['response'],add_special_tokens=False).ids+[2]
            if row['input_ids']!=p+a or row['loss_mask']!=[0]*len(p)+[1]*len(a):raise ValueError('Retokenization/mask invalid')
            if not 2<=len(p+a)<=1024 or any(t in {0,1,2} for t in (p+a)[:-1]):raise ValueError('Token/context invalid')
            if row['prompt_token_count']!=len(p) or row['answer_token_count']!=len(a):raise ValueError('Declared token counts invalid')
            if row['language']=='zh' and (converter.convert(row['prompt'])!=row['prompt'] or converter.convert(row['response'])!=row['response']):raise ValueError('Chinese fixed point invalid')
            if row['normalized_text_sha256']!=canonical_hash([row['language'],'',row['prompt'],row['response']]):raise ValueError('Text identity invalid')
            ids.add(row['id']);gset.add(row['group_id']);qset.add(question_hash(row['question']))
            signature=formula_key(row['question'])
            if signature:fset.add(signature)
            langs[row['language']]+=1;topics[row['topic']]+=1;tokens[row['language']]+=len(a)
            if split=='train':rows_by_id[row['id']]=canonical_hash({k:v for k,v in row.items() if k!='split'})
            if split=='pilot' and rows_by_id.get(row['id'])!=canonical_hash({k:v for k,v in row.items() if k!='split'}):raise ValueError('Pilot is not an exact train subset')
        if len(ids)!=m['splits'][split]['examples'] or len(gset)!=len(ids):raise ValueError('Row/group count mismatch')
        if dict(langs)!=m['statistics'][split]['languages'] or dict(topics)!=m['statistics'][split]['topics']:raise ValueError('Coverage/language statistics mismatch')
        if min(langs['zh'],langs['en'])<1:raise ValueError('Bilingual split is empty')
        groups[split]=gset;questions[split]=qset;formulas[split]=fset
        stats[split]={'sha256':m['splits'][split]['sha256'],'examples':len(ids),'language_examples':dict(langs),
                      'answer_tokens':dict(tokens),'topics':dict(topics),'zh_answer_token_fraction':tokens['zh']/sum(tokens.values())}
        print(json.dumps({'split_audit':split,'records':len(ids),'status':'passed'}),flush=True)
    for i,a in enumerate(['train','validation','test']):
        for b in ['train','validation','test'][i+1:]:
            if groups[a]&groups[b] or questions[a]&questions[b] or formulas[a]&formulas[b]:raise ValueError('Cross-split question/formula leakage')
    checks={key:True for key in ['file_hashes','retokenization','answer_mask_and_eos','normalized_text_identity',
        'declared_group_and_exact_split_isolation','pilot_subset','language_token_balance','explicit_quality_rules',
        'chinese_fixed_point','upstream_source_text_identity']}
    result={'schema_version':1,'status':'passed','fixture':False,'structurally_ready':True,'data_identity':m['data_identity'],
        'tokenizer_sha256':m['tokenizer']['sha256'],'checks':checks,'splits':stats,
        'language_balance_basis':'Measured source availability and declared counts; no synthetic duplication to force a ratio',
        'mathematical_correctness_certified':False,'verification_scope':'Raw provenance, publisher validity labels, exact encoding, isolation; no blanket proof certification'}
    if publish:
        if review is None:raise ValueError('Quality spotcheck receipt required before publishing')
        evidence=read_json(review)
        if evidence.get('status')!='passed_spotcheck' or evidence.get('train_sha256')!=m['splits']['train']['sha256'] or evidence.get('data_identity')!=m['data_identity']:
            raise ValueError('Quality receipt not bound to this exact candidate')
        target=directory/'acceptance.json';write_json(target,result)
        m['acceptance']={'status':'passed','report':target.name,'sha256':sha256_file(target)}
        m['quality_review']={'path':'quality_review.json','sha256':sha256_file(review)}
        if Path(review).resolve()!=(directory/'quality_review.json').resolve():
            (directory/'quality_review.json').write_bytes(Path(review).read_bytes())
        m['status']='published';write_json(path,m)
    return result
