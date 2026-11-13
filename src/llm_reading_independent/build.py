"""独立RL训练题候选：原始人工答案、SFT排除索引、固定中英监督Token配比。"""
from collections import Counter
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json,re,shutil
from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json,write_jsonl
from llm_reading_data.source import fixed_point,text_key,make_prompt,digest
from llm_reading_data.build import PROTOCOL,IDENTITY_FIELDS,token_balanced,question_type,length_bucket
from .isolation import exclusion_index,compact,article_names,export_reference
from .sources import records

AD_QUESTION=re.compile(r'哪家(?:医院)?好|哪家医院|哪个(?:牌子|品牌)|什么牌子|哪种牌子|好不好|靠谱吗|口碑|包皮|男科|丰胸|壮阳|人流医院|减肥药|整形医院')
AD_CONTEXT=re.compile(r'在线咨询|免费咨询|点击咨询|咨询热线|联系我[们]?|加微信|官方热线|优惠券|团购价')


def make_row(raw,converter,tokenizer):
    lang='zh' if raw['source_family'] in {'robust','dureader_raw'} else 'en'
    context=raw['context'];question=raw['question'];title=raw['title'];answer=raw['original_answers'][0]
    if len(context)<40 or len(question.strip())<3:return None,'too_short'
    if lang=='zh':
        context=fixed_point(context,converter);question=fixed_point(question,converter);title=fixed_point(title,converter);answer=fixed_point(answer,converter)
        if AD_QUESTION.search(question) or AD_CONTEXT.search(context):return None,'advertising_or_commercial_medical_query'
    if answer not in context or len(compact(answer))<1 or compact(answer)==compact(context):return None,'missing_or_whole_passage_answer'
    if any(x in question for x in ['如图','见图','如下图','上图','图中所示']):return None,'requires_missing_image'
    prompt=make_prompt(context,question,lang);prefix='User:\n'+prompt+'\n\nAssistant:\n'
    p=tokenizer.encode(prefix,add_special_tokens=False).ids;a=tokenizer.encode(answer,add_special_tokens=False).ids+[2]
    if len(p)>960 or len(a)>64:return None,'full_context_or_answer_too_long'
    if any(t in {0,1,2} for t in p+a[:-1]):return None,'invalid_special_token'
    names=article_names(title,raw['source_url']);article=min(names) if names else 'unknown:'+text_key(context)
    row={**raw,'language':lang,'context':context,'question':question,'title':title,'answers':[answer],
        'answer_occurrences':{answer:[m.start() for m in re.finditer(re.escape(answer),context)]},
        'schema_version':1,'id':'reading_independent_'+digest([context,question])[:24],'system':'','prompt':prompt,'response':answer,
        'input_ids':p+a,'loss_mask':[0]*len(p)+[1]*len(a),'prompt_token_count':len(p),'answer_token_count':len(a),
        'article_key':article,'article_identity_available':bool(names),'passage_key':text_key(context),
        'question_key':text_key(question),'group_id':digest(article),'human_annotated':True,'reward_type':'human_alias_exact',
        'question_type':question_type(question,lang),'answer_length_bucket':length_bucket(len(a)-1),
        'normalized_text_sha256':canonical_hash([lang,'',prompt,answer])}
    return row,None


def build(cache,lock_path,excluded,output,work,total=50000):
    from opencc import OpenCC
    from tokenizers import Tokenizer
    cache=Path(cache);output=Path(output);work=Path(work);work.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('Never overwrite a candidate')
    lock=read_json(lock_path);base=Path(excluded[0]);parent=read_json(base)
    tokenizer_path=base.parent/parent['tokenizer']['path'];tokenizer=Tokenizer.from_file(str(tokenizer_path));converter=OpenCC('t2s')
    index=exclusion_index(excluded);print(json.dumps({'phase':'exclusion_index_ready','passages':len(index.passages),'questions':len(index.questions),'articles':len(index.articles)}),flush=True)
    chosen=[];rejected=Counter();seen_questions=set();seen_inputs=set();eligible=Counter()
    for spec in lock['sources']:
        if sha256_file(cache/spec['path'])!=spec['sha256']:raise ValueError('Official source checksum changed')
        raw_count=0
        for raw in records(cache,spec):
            raw_count+=1;row,reason=make_row(raw,converter,tokenizer)
            if reason:rejected[spec['source']+'/'+reason]+=1;continue
            reason=index.reason(row['question'],row['context'],row['title'],row['source_url'])
            if reason:rejected[spec['source']+'/'+reason]+=1;continue
            qkey=compact(row['question']);pkey=digest([row['context'],row['question']])
            if qkey in seen_questions or pkey in seen_inputs:rejected['duplicate_rl_question']+=1;continue
            seen_questions.add(qkey);seen_inputs.add(pkey);chosen.append(row);eligible[row['language']]+=1
            if raw_count%10000==0:print(json.dumps({'phase':'candidate_progress','source':spec['source'],'human_span_candidates_seen':raw_count,'eligible':dict(eligible)}),flush=True)
        print(json.dumps({'phase':'source_complete','source':spec['source'],'path':spec['path'],'member':spec.get('member'),'human_span_candidates_seen':raw_count,'eligible':dict(eligible)}),flush=True)
    answer_tokens=Counter()
    for row in chosen:answer_tokens[row['language']]+=row['answer_token_count']
    report={'eligible':dict(eligible),'answer_tokens':dict(answer_tokens),'sources':dict(Counter(r['source'] for r in chosen)),'rejections':dict(rejected),'exclusions':index.bound_sources}
    write_json(work/'candidate_capacity.json',report)
    write_jsonl(work/'eligible_pool.jsonl',chosen)
    write_json(work/'eligible_pool.identity.json',{'source_lock_sha256':sha256_file(lock_path),
        'code':{p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py')},'exclusions':index.bound_sources,
        'pool_sha256':sha256_file(work/'eligible_pool.jsonl')})
    training,balance=token_balanced(chosen,total=total,seed=20261101)
    output.mkdir(parents=True);shutil.copyfile(tokenizer_path,output/'tokenizer.json');specs={};stats={}
    export_reference(index,output/'exclusion_reference.json')
    for split in ['train','validation','test','pilot']:
        if split=='train':rows=[{**r,'split':'train'} for r in training]
        elif split=='pilot':rows=[{**r,'split':'pilot'} for r in training[:2000]]
        else:rows=[json.loads(line) for line in (base.parent/parent['splits'][split]['path']).open(encoding='utf-8')]
        file=output/(split+'.jsonl');write_jsonl(file,rows)
        specs[split]={'path':file.name,'sha256':sha256_file(file),'examples':len(rows)}
        lang=Counter(r['language'] for r in rows);tokens=Counter()
        for row in rows:tokens[row['language']]+=row['answer_token_count']
        stats[split]={'records':len(rows),'languages':dict(lang),'answer_tokens':dict(tokens),
            'sources':dict(Counter(r['source'] for r in rows)),'zh_answer_token_fraction':tokens['zh']/sum(tokens.values())}
    manifest={'schema_version':1,'kind':'sft_tokenized_conversations','task':'extractive_reading_qa','fixture':False,
       'status':'awaiting_independent_audit','purpose':'independent RL training; not new SFT',
       'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'splits':specs,'tokenizer':{**parent['tokenizer'],'path':'tokenizer.json'},
       'tokenization':PROTOCOL,'chinese_normalization':parent['chinese_normalization'],
       'origin':{'sources':lock['sources'],'source_lock_sha256':sha256_file(lock_path),'human_answers':True,
                 'raw_human_answers_only':True,'fake_answers_used':False},
       'exclusion_manifests':index.bound_sources,'statistics':stats,'language_balance':balance,
       'exclusion_reference':{'path':'exclusion_reference.json','sha256':sha256_file(output/'exclusion_reference.json')},
       'training_policy':{'epochs':1,'stop_epochs':None,'sft_training_prompts_reused':False},
       'generation':parent['generation'],'evaluation':{**parent['evaluation'],'shared_frozen_evaluation_from':parent['data_identity']},
       'isolation_policy':{'question':'normalized exact','passage':'normalized exact plus 13-character shingle/8-window winnowing at containment .85',
          'article':'known article title/Wikipedia URL excluded; unavailable article identities use passage grouping',
          'limits':'Not proof of all cross-language semantic duplicates or pretraining contamination'},
       'dpo':{'status':'awaiting_policy_generated_rejected','rejected_count':0},'training_started':False}
    manifest['data_identity']=canonical_hash({k:manifest[k] for k in IDENTITY_FIELDS});write_json(output/'manifest.json',manifest)
    write_json(work/'build_report.json',{'manifest':str(output/'manifest.json'),'statistics':stats,'balance':balance})
    print(json.dumps({'phase':'candidate_built','manifest':str(output/'manifest.json'),'statistics':stats,'balance':balance},ensure_ascii=False),flush=True)
    return output/'manifest.json'
