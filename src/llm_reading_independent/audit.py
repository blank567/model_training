"""重读官方人工跨度、重算编码并核验跨SFT隔离；通过后才发布RL语料。"""
from collections import Counter
from importlib import metadata
from pathlib import Path
import json,re
from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json
from llm_reading_data.build import IDENTITY_FIELDS
from llm_reading_data.source import fixed_point,make_prompt
from llm_reading_data.verifier import reward
from .isolation import load_reference,compact
from .sources import records
from .quality import reason as quality_reason

ROOT=Path(__file__).resolve().parents[2]


def audit(path,cache,lock_path,publish=False,review=None):
    from opencc import OpenCC
    from tokenizers import Tokenizer
    path=Path(path);directory=path.parent;cache=Path(cache);m=read_json(path)
    if any((directory/name).exists() for name in ['DO_NOT_TRAIN.json','SUPERSEDED.json']):raise ValueError('Blocked candidate')
    if canonical_hash({k:m[k] for k in IDENTITY_FIELDS})!=m['data_identity']:raise ValueError('Dataset identity changed')
    if metadata.version('OpenCC')!='1.1.9':raise ValueError('OpenCC 1.1.9 required')
    if sha256_file(lock_path)!=m['origin']['source_lock_sha256']:raise ValueError('Official source lock changed')
    if sha256_file(directory/m['tokenizer']['path'])!=m['tokenizer']['sha256']:raise ValueError('Tokenizer changed')
    reference=directory/m['exclusion_reference']['path']
    if sha256_file(reference)!=m['exclusion_reference']['sha256']:raise ValueError('Cross-stage exclusion reference changed')
    index=load_reference(reference)
    if index.bound_sources!=m['exclusion_manifests']:raise ValueError('Exclusion source identities changed')
    # 不改写历史绝对路径；读取时映射当前机器的项目根。
    original=str(index.bound_sources[0]['path']).replace('\\','/').split('/artifacts/',1)[1]
    source_manifest=ROOT/'artifacts'/original;source=read_json(source_manifest)
    if sha256_file(source_manifest)!=index.bound_sources[0]['manifest_sha256']:raise ValueError('Selected SFT source identity changed')
    tokenizer=Tokenizer.from_file(str(directory/m['tokenizer']['path']));converter=OpenCC('t2s')
    summaries={};training={};questions=set();groups={};inputs={};selected_originals={}
    for split in ['train','validation','test','pilot']:
        spec=m['splits'][split];file=directory/spec['path']
        if sha256_file(file)!=spec['sha256']:raise ValueError('Split file changed')
        if split in {'validation','test'} and spec['sha256']!=source['splits'][split]['sha256']:
            raise ValueError('Shared evaluation was resampled or rewritten')
        ids=set();lang=Counter();tokens=Counter();gset=set();iset=set()
        for line in file.open(encoding='utf-8'):
            row=json.loads(line);language=row['language']
            if row['schema_version']!=1 or row['split']!=split or row['id'] in ids:raise ValueError('Invalid/duplicate example')
            p=tokenizer.encode('User:\n'+row['prompt']+'\n\nAssistant:\n',add_special_tokens=False).ids
            a=tokenizer.encode(row['response'],add_special_tokens=False).ids+[2]
            if row['prompt']!=make_prompt(row['context'],row['question'],language):raise ValueError('Full context/question was changed')
            if row['input_ids']!=p+a or row['loss_mask']!=[0]*len(p)+[1]*len(a):raise ValueError('Tokenization/mask mismatch')
            if row['prompt_token_count']!=len(p) or row['answer_token_count']!=len(a):raise ValueError('Token counts mismatch')
            if len(p)>960 or len(a)>64 or len(p+a)>1024 or any(t in {0,1,2} for t in p+a[:-1]):raise ValueError('Token budget invalid')
            if row['normalized_text_sha256']!=canonical_hash([language,'',row['prompt'],row['response']]):raise ValueError('Text identity mismatch')
            for answer in row['answers']:
                if answer not in row['context'] or reward(answer,row['answers'],language)!=1:raise ValueError('Gold answer not supported')
                if row['answer_occurrences'][answer]!=[match.start() for match in re.finditer(re.escape(answer),row['context'])]:raise ValueError('Converted spans mismatch')
            if reward(row['context'],row['answers'],language)!=0:raise ValueError('Whole passage earns reward')
            if language=='zh' and any(converter.convert(t)!=t for t in [row['question'],row['context'],row['response']]):raise ValueError('Chinese is not simplified')
            if split=='train':
                if quality_reason(row):raise ValueError('Quality exclusion rule failed')
                if row['source_family'] not in {'robust','quoref'} or row['source_role']!='official_train':raise ValueError('Unapproved training source')
                reason=index.reason(row['question'],row['context'],row['title'],row['source_url'])
                if reason:raise ValueError('SFT/evaluation reuse detected: '+reason)
                qkey=compact(row['question'])
                if qkey in questions:raise ValueError('Duplicate RL training question')
                questions.add(qkey);training[row['id']]=canonical_hash({k:v for k,v in row.items() if k!='split'})
                key=(row['source_file'],row['source_row'])
                if key in selected_originals:raise ValueError('Duplicate official source row')
                selected_originals[key]=row
            if split=='pilot' and training.get(row['id'])!=canonical_hash({k:v for k,v in row.items() if k!='split'}):raise ValueError('Pilot not exact train subset')
            ids.add(row['id']);lang[language]+=1;tokens[language]+=len(a);gset.add(row['group_id']);iset.add(canonical_hash([row['context'],row['question']]))
        if len(ids)!=spec['examples']:raise ValueError('Example count mismatch')
        if dict(lang)!=m['statistics'][split]['languages'] or dict(tokens)!=m['statistics'][split]['answer_tokens']:raise ValueError('Language/token statistics mismatch')
        groups[split]=gset;inputs[split]=iset
        summaries[split]={'sha256':spec['sha256'],'examples':len(ids),'languages':dict(lang),'answer_tokens':dict(tokens)}
        print(json.dumps({'audit_split':split,'records':len(ids),'status':'passed'}),flush=True)
    if not 20000<=summaries['train']['examples']<=30000:raise ValueError('Expected the approved 20000-30000-example release')
    tokens=summaries['train']['answer_tokens']
    if abs(tokens['zh']/sum(tokens.values())-.5)>.005:raise ValueError('Answer-token balance failed')
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if groups[a]&groups[b] or inputs[a]&inputs[b]:raise ValueError('Split grouping/input overlap')
    matched=set()
    for spec in read_json(lock_path)['sources']:
        if sha256_file(cache/spec['path'])!=spec['sha256']:raise ValueError('Original official training source changed')
        for raw in records(cache,spec):
            key=(raw['source_file'],raw['source_row'])
            if key not in selected_originals:continue
            row=selected_originals[key];language=row['language']
            if raw['source_id']!=row['source_id'] or raw['source_record_sha256']!=row['source_record_sha256']:raise ValueError('Source annotation identity mismatch')
            answer=raw['original_answers'][0];start=raw['original_answer_start']
            if raw['context'][start:start+len(answer)]!=answer:raise ValueError('Original human span invalid')
            texts=[raw['context'],raw['question'],raw['title'],answer]
            if language=='zh':texts=[fixed_point(t,converter) for t in texts]
            if texts!=[row['context'],row['question'],row['title'],row['response']] or row['answers']!=[texts[-1]]:
                raise ValueError('Original human answer or passage was rewritten')
            matched.add(key)
    if matched!=set(selected_originals):raise ValueError('Cannot find every selected example in original official source')
    checks={name:True for name in ['file_hashes','retokenization','answer_mask_and_eos','normalized_text_identity',
        'declared_group_and_exact_split_isolation','pilot_subset','language_token_balance','explicit_quality_rules',
        'chinese_fixed_point','original_human_offsets','cross_stage_question_isolation','cross_stage_passage_isolation',
        'known_article_isolation','near_duplicate_screening','original_training_sources_only','no_fake_answers']}
    result={'schema_version':1,'status':'passed','fixture':False,'structurally_ready':True,'data_identity':m['data_identity'],
        'tokenizer_sha256':m['tokenizer']['sha256'],'splits':summaries,'checks':checks,
        'exclusion_identity':canonical_hash(m['exclusion_manifests']),'exclusion_reference_sha256':m['exclusion_reference']['sha256'],
        'original_source_records_verified':len(matched),'reward_code_sha256':sha256_file(ROOT/'src/llm_reading_data/verifier.py'),
        'recipe_files':{p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py')},
        'scope':'Question/passage disjoint; known titles/URLs excluded, near-duplicate screening; not all semantic or pretraining decontamination'}
    if publish:
        if m.get('status')=='published':raise ValueError('Never rewrite published data')
        review_data=read_json(review)
        if review_data.get('status')!='passed_spotcheck' or review_data.get('data_identity')!=m['data_identity'] or review_data.get('train_sha256')!=m['splits']['train']['sha256']:
            raise ValueError('Representative quality review not bound to current candidate')
        write_json(directory/'acceptance.json',result);write_json(directory/'quality_review.json',review_data)
        m.update(status='published',acceptance={'status':'passed','report':'acceptance.json','sha256':sha256_file(directory/'acceptance.json')},
            quality_review={'path':'quality_review.json','sha256':sha256_file(directory/'quality_review.json')},
            reward_protocol={'name':'strict_human_alias_exact_v1','code_sha256':result['reward_code_sha256']})
        write_json(path,m)
    return result
