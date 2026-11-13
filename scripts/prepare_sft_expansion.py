"""将固定中英储备整理为去污染候选池，再发布嵌套的SFT扩充版本。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import time

import prepare_sft_data as base
from sft_data_quality import fingerprint, group_candidates, normalize, quality_reason

TEXT_FIELDS = ('system','prompt','response','input_context')
GIB = 1024**3
NUMBER = {'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10}
CN_TASKS = {'format-constrain.parquet':'format_following','rewrite.parquet':'rewriting',
            'summary.parquet':'summarization','document-qa.parquet':'grounded_qa',
            'editing.parquet':'rewriting'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def row_stream(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def checked(root,name):
    path = base.under(Path(root),name)
    for current in (path,*path.parents):
        if current.is_symlink():raise ValueError('Data paths cannot traverse symlinks')
    return path


def human_labels(row):
    """OASST的Arrow字段是列式字典；也支持公开JSON导出的字典列表。"""
    labels = row.get('labels') or {}
    if isinstance(labels,dict):
        return {name:value for name,value in zip(labels.get('name',[]),labels.get('value',[]))
                if isinstance(value,(int,float)) and math.isfinite(value)}
    if isinstance(labels,list):
        return {item['name']:item['value'] for item in labels
                if isinstance(item,dict) and isinstance(item.get('value'),(int,float)) and math.isfinite(item['value'])}
    return {}


def reviewed_message(row,*,answer=False):
    if row.get('review_result') is not True or row.get('deleted') is not False:
        return False
    labels=human_labels(row)
    if labels.get('quality',0) < .70:return False
    unwanted=('spam','lang_mismatch','pii','not_appropriate','hate_speech','sexual_content','violence','fails_task')
    if any(labels.get(name,0) > .10 for name in unwanted):return False
    if labels.get('toxicity',0) > .25:return False
    if answer and (type(row.get('rank')) is not int or row['rank'] < 0):return False
    return True


def first_turn(messages):
    if not isinstance(messages,list):return None,'invalid_conversations'
    offset=0;system=''
    if messages and messages[0].get('role')=='system':
        system=messages[0].get('content');offset=1
    if (len(messages)<offset+2 or messages[offset].get('role')!='user'
            or messages[offset+1].get('role')!='assistant'):
        return None,'not_self_contained_first_turn'
    prompt=messages[offset].get('content');response=messages[offset+1].get('content')
    if not all(isinstance(value,str) for value in (system,prompt,response)):
        return None,'non_string_fields'
    return {'system':system,'prompt':prompt,'response':response,'input_context':''},None


def material_from_prompt(prompt,task):
    """只抽取明显分隔的原材料，不编造缺失上下文。"""
    if task not in {'summarization','rewriting','grounded_qa'}:return ''
    parts=[part.strip() for part in prompt.splitlines() if part.strip()]
    if len(parts)>1:
        if re.search(r'请|总结|概括|改写|润色|summari|rewrite',parts[0],re.I):return '\n'.join(parts[1:])
        if re.search(r'[?？]|请问|which|what|according',parts[-1],re.I):return '\n'.join(parts[:-1])
    return prompt if task in {'summarization','rewriting'} else ''


def adapt(raw,source):
    family=source['source'];name=Path(source['file']).name
    if family=='smoltalk_chinese':
        score=raw.get('score')
        if not isinstance(score,(int,float)) or not math.isfinite(score):return None,'missing_instruction_score'
        if score < 4:return None,'low_instruction_score'
        if raw.get('difficulty') in {'困难','非常困难','hard','very hard'}:return None,'outside_small_model_curriculum'
        row,reason=first_turn(raw.get('conversations'))
        if reason:return None,reason
        task=CN_TASKS.get(name,base.task_kind(row['prompt'],name))
        row.update(subset=name,task=task,language='zh',dataset=family,
                   input_context=material_from_prompt(row['prompt'],task),instruction_score=score)
        return row,None
    if family=='alpaca_gpt4_zh':
        row,reason=base.extract(raw,{'name':family,'format':'alpaca_json'})
    elif family=='dolly':
        row,reason=base.extract(raw,{'name':family,'format':'dolly_jsonl',
            'category_allowlist':source['selection_policy']['allowed_categories'],'require_context':True})
    elif family=='smol_smoltalk':
        row,reason=base.extract(raw,{'name':family,'format':'messages_parquet',
            'source_allowlist':source['selection_policy']['source_allowlist']})
    else:
        raise ValueError('Unsupported adapter source')
    if reason:return None,reason
    subset=row['subset'];task=base.task_kind(row['system']+'\n'+row['prompt'],subset)
    if subset in {'smol-summarize-20k','summarization'}:task='summarization'
    elif subset=='smollm-rewrite-30k':task='rewriting'
    elif subset=='smol-contraints':task='format_following'
    elif subset=='closed_qa':task='grounded_qa'
    elif subset=='information_extraction':task='extraction'
    dataset=family
    if family=='smol_smoltalk':dataset='smol_smoltalk_'+str(int(re.search(r'train-(\d+)-',name).group(1)))
    row.update(language=source['language'],task=task,dataset=dataset)
    return row,None


def source_records(path,source):
    """返回原始行号、可追溯原行、抽取结果及拒收原因；只读取上游训练文件。"""
    if source['format']=='alpaca_json':
        with path.open(encoding='utf-8') as stream:rows=json.load(stream)
        iterator=enumerate(rows)
    elif source['format']=='dolly_jsonl':
        iterator=enumerate(row_stream(path))
    else:
        import pyarrow.parquet as pq
        def parquet_rows():
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128,use_threads=False):
                yield from batch.to_pylist()
        if source['source']=='oasst2':
            rows=list(parquet_rows());parents={row['message_id']:row for row in rows}
            eligible=defaultdict(list)
            for index,answer in enumerate(rows):
                parent=parents.get(answer.get('parent_id'))
                if (answer.get('role')=='assistant' and parent and parent.get('role')=='prompter'
                    and parent.get('parent_id') is None and answer.get('lang') in {'en','zh'}
                    and parent.get('lang')==answer['lang'] and parent.get('message_tree_id')==answer.get('message_tree_id')
                    and reviewed_message(parent) and reviewed_message(answer,answer=True)):
                    eligible[parent['message_id']].append((answer['rank'],-human_labels(answer).get('quality',0),answer['message_id'],index))
            chosen={min(values)[-1] for values in eligible.values()}
            for index,answer in enumerate(rows):
                if index not in chosen:
                    yield index,answer,None,'not_selected_reviewed_root_answer';continue
                parent=parents[answer['parent_id']]
                item={'system':'','prompt':parent['text'],'response':answer['text'],'input_context':'',
                      'subset':'reviewed_root_pair','task':base.task_kind(parent['text'],'oasst2'),
                      'language':answer['lang'],'dataset':'oasst2','tree_id':answer['message_tree_id'],
                      'answer_message_id':answer['message_id'],'parent_message_id':parent['message_id'],
                      'human_quality':human_labels(answer)['quality']}
                yield index,{'parent':parent,'answer':answer},item,None
            return
        iterator=enumerate(parquet_rows())
    policy=source['selection_policy']
    blocked=set(policy.get('known_rejected_row_indices',[]))
    if source['source']=='smol_smoltalk' and 'train-00000-' in source['file']:
        blocked.update(policy.get('known_rejected_first_shard_row_indices',[]))
    for index,raw in iterator:
        if index in blocked:yield index,raw,None,'known_research_rejection';continue
        row,reason=adapt(raw,source)
        yield index,raw,row,reason


def constraint_reason(prompt,response,language):
    """仅检查可明确解析的硬约束；未触发不代表语义已经通过。"""
    number=r'(\d+|一|二|两|三|四|五|六|七|八|九|十)'
    if language=='zh':
        match=re.search(r'(?:不超过|最多|至多)\s*'+number+r'\s*(?:句|句话)',prompt)
        if match:
            value=match.group(1);limit=int(value) if value.isdigit() else NUMBER[value]
            count=len([part for part in re.split(r'[。！？!?]+',response) if part.strip()])
            if count>limit:return 'chinese_sentence_limit_violation'
        if re.search(r'(?:仅|只|恰好|正好)(?:用|使用)?一句话|请(?:你)?(?:只|仅)?用一句话',prompt):
            if len([part for part in re.split(r'[。！？!?]+',response) if part.strip()])>1:
                return 'chinese_one_sentence_violation'
    only_json=re.search(r'(?:只|仅)(?:能|需|需要|允许)?(?:返回|输出).{0,8}JSON|(?:只|仅)(?:以|用)JSON|return\s+only\s+(?:a\s+)?(?:valid\s+)?json',prompt,re.I)
    if only_json:
        try:json.loads(response)
        except (ValueError,TypeError):return 'invalid_required_json'
    if re.search(r'全部小写|所有.{0,6}小写',prompt) and any('A'<=char<='Z' for char in response):
        return 'chinese_lowercase_constraint_violation'
    match=re.search(r'exactly\s+(\d+)\s+bullet\s+points',prompt,re.I)
    if match and len(re.findall(r'(?m)^\s*[-*•]\s+',response))!=int(match.group(1)):
        return 'exact_bullet_count_violation'
    return None


class ProtectedKeys:
    """评测题只进入自动去污染索引，绝不写入抽审清单。"""
    def __init__(self,rows):
        self.origins=set();self.prompts=set();self.contexts=set();self.responses=set()
        self.fragments=set();self.texts=[]
        for row in rows:
            source=row['source'];self.origins.add((source['revision'],source['file'],source['row_index']))
            self.prompts.add(fingerprint(row['prompt']))
            if row['input_context']:self.contexts.add(fingerprint(row['input_context']))
            if len(base.compact(row['response']))>=80:self.responses.add(fingerprint(row['response']))
            if row['split'] not in {'validation','test'}:continue
            for key in ('input_context','response'):
                text=base.compact(row[key])
                if len(text)>=100 and text not in self.texts:
                    self.texts.append(text);self.fragments.update(value for _,value in base.rolling(text))

    def reason(self,row,source,index):
        if (source['revision'],source['file'],index) in self.origins:return 'existing_sft_source_row'
        if fingerprint(row['prompt']) in self.prompts:return 'existing_sft_prompt'
        if row['input_context'] and fingerprint(row['input_context']) in self.contexts:return 'existing_sft_context'
        if len(base.compact(row['response']))>=80 and fingerprint(row['response']) in self.responses:return 'existing_sft_long_response'
        for key in ('prompt','response','input_context'):
            text=base.compact(row[key])
            if len(text)<100:continue
            for offset,value in base.rolling(text):
                if value in self.fragments and any(text[offset:offset+100] in protected for protected in self.texts):
                    return 'existing_sft_holdout_fragment'
        return None


def donor_rows(root,settings):
    path=checked(root,settings['donor_manifest']);manifest=read_json(path)
    if base.sha256(path)!=settings['donor_manifest_sha256']:raise ValueError('Donor SFT manifest changed')
    rows=[]
    for split in ('train','validation','test'):
        spec=manifest['splits'][split];file=checked(path.parent,spec['path'])
        if base.sha256(file)!=spec['sha256']:raise ValueError('Donor SFT split changed')
        rows.extend(row_stream(file))
    return path,manifest,rows


def grouping_row(row,*,protected=False):
    return {'id':row['id'],'prompt':row['prompt'],'answer':row['response'],
            'input_context':row['input_context'],
            'source_group_id':('protected:'+row['group_id']) if protected else row.get('source_group_id','')}


def scan(config_path):
    from llm_lab.data.text_normalization import build_chinese_normalizer
    from tokenizers import Tokenizer
    settings=read_json(config_path);root=(Path(config_path).resolve().parent/settings['project_root']).resolve()
    work=checked(root,settings['work_dir']);output=checked(root,settings['output_dir'])
    if output.exists() or (work/'candidates.sqlite').exists():raise ValueError('Expansion stage already exists; do not overwrite')
    work.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    def event(phase,**extra):
        if time.monotonic()-started>settings['max_runtime_seconds']:raise TimeoutError('CPU data preparation time budget reached')
        if shutil.disk_usage(root).free<settings['min_free_bytes']:raise OSError('Data preparation disk reserve reached')
        written=sum(path.stat().st_size for directory in (work,output) if directory.exists()
                    for path in directory.rglob('*') if path.is_file())
        if written>settings['max_work_and_output_bytes']:raise OSError('Bounded expansion output budget reached')
        import resource
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>settings['max_rss_bytes']:
            raise MemoryError('Bounded expansion resident-memory budget reached')
        record={'phase':phase,'elapsed_seconds':time.monotonic()-started,**extra}
        base.atomic_json(work/'progress.json',record);print(json.dumps(record,ensure_ascii=False),flush=True)
    normalizer=build_chinese_normalizer('t2s')
    donor_path,donor,protected=donor_rows(root,settings)
    if normalizer.provenance!=donor['chinese_normalization']:raise ValueError('Chinese normalization identity changed')
    tokenizer_path=checked(donor_path.parent,donor['tokenizer']['path'])
    if base.sha256(tokenizer_path)!=donor['tokenizer']['sha256']:raise ValueError('Tokenizer identity changed')
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    guard=ProtectedKeys(protected)
    parent_path=checked(root,donor['pretraining_parent']['path']);parent=read_json(parent_path)
    if base.sha256(parent_path)!=donor['pretraining_parent']['sha256']:raise ValueError('Pretraining manifest changed')
    documents=checked(parent_path.parent,parent['documents']['path'])
    if base.sha256(documents)!=donor['pretraining_parent']['documents_sha256']:raise ValueError('Pretraining holdout changed')
    pretrain_guard,pretrain_info=base.holdout_guard(documents,normalizer)
    reserve=checked(root,settings['reserve_dir']);reserve_config=read_json(reserve/'reserve_config.json')
    if base.sha256(reserve/'reserve_config.json')!=settings['reserve_config_sha256']:raise ValueError('Reserve configuration changed')
    known_rejections={(row['dataset'],row['row_index']) for row in settings['known_rejections']}
    database=sqlite3.connect(work/'candidates.sqlite')
    database.execute('CREATE TABLE candidates(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    nodes=[];rejections=Counter();source_reports=[];accepted=0
    event('guards_ready',protected_existing_examples=len(protected),pretraining_holdout=pretrain_info)
    with (work/'rejections.jsonl').open('x',encoding='utf-8') as rejected:
        for source in reserve_config['files']:
            path=checked(reserve,source['destination'])
            if path.stat().st_size!=source['size_bytes'] or base.sha256(path)!=source['sha256']:raise ValueError('Raw source identity changed')
            seen=kept=0;reasons=Counter()
            for index,raw,row,reason in source_records(path,source):
                seen+=1
                original=None
                if reason is None and (row['dataset'],index) in known_rejections:reason='known_curated_rejection'
                if reason is None:
                    original={key:row[key] for key in TEXT_FIELDS}
                    for key in TEXT_FIELDS:row[key]=normalize(normalizer.convert(normalize(row[key]),row['language']))
                    combined=(row['system']+'\n'+row['prompt']).strip()
                    if len(combined)+len(row['response'])>20000:reason='raw_text_too_long'
                    else:reason=quality_reason(combined,row['response'],row['language'],row['subset']) or constraint_reason(combined,row['response'],row['language'])
                    letters=[char for char in row['response'] if char.isalpha()]
                    if not reason and len(letters)>20:
                        han=sum('\u4e00'<=char<='\u9fff' for char in letters)/len(letters)
                        if row['language']=='zh' and han<.30:reason='chinese_response_language_mismatch'
                        if row['language']=='en' and han>.02:reason='english_response_language_mismatch'
                if reason is None:
                    ids,mask,prefix_count,answer_count=base.encode_row(row,tokenizer,2)
                    if len(ids)>settings['max_sequence_tokens']:reason='sequence_too_long'
                    elif answer_count>settings['max_response_tokens']:reason='response_too_long'
                    elif any(value in {0,1,2} for value in ids[:-1]):reason='embedded_special_token'
                    if not reason:reason=guard.reason(row,source,index)
                    if not reason and pretrain_guard(*(row[key] for key in TEXT_FIELDS)):reason='pretraining_holdout_overlap'
                if reason:
                    reasons[reason]+=1;rejections[reason]+=1
                    rejected.write(json.dumps({'source':source['source'],'file':source['file'],'row_index':index,'reason':reason})+'\n')
                else:
                    raw_sha=base.canonical(raw)
                    identifier=base.canonical({'repository':source['repository'],'revision':source['revision'],
                                                'file':source['file'],'row_index':index,'raw_sha256':raw_sha})
                    meta={'dataset':row['dataset'],'revision':source['revision'],'file':source['file'],'row_index':index,
                          'source_subset':row['subset'],'reserve_family':source['source'],'raw_row_sha256':raw_sha,
                          'original_prompt_sha256':base.text_sha(original['prompt']),'original_response_sha256':base.text_sha(original['response']),
                          'original_context_sha256':base.text_sha(original['input_context']),
                          'normalized_text_sha256':base.canonical({key:row[key] for key in TEXT_FIELDS}),'license':source['license']}
                    for name in ('instruction_score','human_quality','tree_id','answer_message_id','parent_message_id'):
                        if name in row:meta[name]=row[name]
                    entry={'schema_version':1,'id':identifier,'language':row['language'],'task':row['task'],
                           **{key:row[key] for key in TEXT_FIELDS},'input_ids':ids,'loss_mask':mask,
                           'prompt_token_count':prefix_count,'answer_token_count':answer_count,'source':meta,
                           'source_group_id':('oasst2:tree:'+row['tree_id']) if 'tree_id' in row else ''}
                    database.execute('INSERT INTO candidates VALUES(?,?)',(identifier,json.dumps(entry,ensure_ascii=False,separators=(',',':'))))
                    nodes.append(grouping_row(entry));kept+=1;accepted+=1
                if seen%5000==0:
                    database.commit();rejected.flush();event('filtering',source=source['source'],file=source['file'],rows_seen=seen,kept=kept,total_candidates=accepted)
            database.commit();source_reports.append({'source':source['source'],'file':source['file'],'rows_seen':seen,'kept_before_dedup':kept,'rejections':dict(reasons)})
            event('source_filtered',**source_reports[-1],total_candidates=accepted)
    event('grouping',new_candidates=accepted)
    protected_nodes=[grouping_row(row,protected=True) for row in protected]
    groups=group_candidates(protected_nodes+nodes,threshold=settings['near_duplicate_jaccard'],max_bucket=64)
    blocked_groups=set(groups[:len(protected_nodes)]);representatives={};blocked=0
    for node,group in zip(nodes,groups[len(protected_nodes):]):
        if group in blocked_groups:blocked+=1;continue
        entry=json.loads(database.execute('SELECT payload FROM candidates WHERE id=?',(node['id'],)).fetchone()[0])
        rank=(len(entry['input_ids'])>settings['preferred_sequence_tokens'],len(entry['input_ids']),node['id'])
        if group not in representatives or rank<representatives[group][0]:representatives[group]=(rank,node['id'])
    retained={value[1]:group for group,value in representatives.items()}
    output.mkdir(parents=True,exist_ok=False)
    counts=Counter();language_counts=Counter();families=Counter();review_buckets=defaultdict(list)
    with (output/'candidates.jsonl').open('x',encoding='utf-8') as stream:
        for identifier,payload in database.execute('SELECT id,payload FROM candidates ORDER BY id'):
            if identifier not in retained:continue
            entry=json.loads(payload);entry.pop('source_group_id',None);entry['group_id']=retained[identifier];entry['split']='candidate'
            stream.write(json.dumps(entry,ensure_ascii=False,separators=(',',':'))+'\n')
            counts[entry['task']]+=1;language_counts[entry['language']]+=1;families[entry['source']['reserve_family']]+=1
            key=(entry['language'],entry['task'],entry['source']['reserve_family'])
            if len(review_buckets[key])<3:review_buckets[key].append({key:entry[key] for key in ('id','language','task','system','prompt','response','source')})
        stream.flush();os.fsync(stream.fileno())
    database.close()
    review=[row for key in sorted(review_buckets) for row in review_buckets[key]]
    if len(review)>64:
        review=[rows[0] for _,rows in sorted(review_buckets.items())]+[row for _,rows in sorted(review_buckets.items()) for row in rows[1:]]
        review=review[:64]
    base.write_jsonl(output/'review_sample.jsonl',review)
    shutil.copy2(tokenizer_path,output/'tokenizer.json');shutil.copy2(config_path,output/'build_config.json')
    report={'status':'cleaned_pending_content_review','candidate_count':len(retained),'languages':dict(language_counts),
            'tasks':dict(counts),'families':dict(families),'before_dedup':accepted,'protected_near_duplicates_removed':blocked,
            'source_reports':source_reports,'rejections':dict(rejections),'candidates_sha256':base.sha256(output/'candidates.jsonl'),
            'donor_manifest_sha256':settings['donor_manifest_sha256'],'tokenizer_sha256':donor['tokenizer']['sha256'],
            'chinese_normalization':normalizer.provenance,'pretraining_guard':pretrain_info,
            'review_examples':len(review),'training_ready':False,'test_content_exposed_to_review':False,
            'deduplication_limits':'Exact keys plus bounded character-neighbor comparison; not semantic/translation-complete',
            'elapsed_seconds':time.monotonic()-started}
    base.atomic_json(output/'pool_manifest.json',report);event('cleaned_pending_content_review',candidate_count=len(retained),languages=dict(language_counts),tasks=dict(counts))
    return report


def nested_selection(baseline,candidates,targets,weights,seed):
    """保留基准训练集；按任务缺口与回答Token平衡追加，形成嵌套集合。"""
    if not targets or sorted(set(targets))!=targets or targets[0]<=len(baseline):raise ValueError('Invalid expansion targets')
    ids={row['id'] for row in baseline}
    if (len(ids)!=len(baseline) or len({row['id'] for row in candidates})!=len(candidates)
        or any(row['id'] in ids for row in candidates)):raise ValueError('Baseline/new candidates overlap or repeat')
    queues=defaultdict(deque)
    ordered=sorted(candidates,key=lambda row:base.canonical({'seed':seed,'id':row['id']}))
    ordered.sort(key=lambda row:len(row['input_ids'])>512)
    for row in ordered:queues[(row['language'],row['task'])].append(row)
    selected=list(baseline);tokens=Counter();tasks=Counter()
    for row in selected:tokens[row['language']]+=row['answer_token_count'];tasks[row['task']]+=1
    snapshots={};shortfalls={}
    while len(selected)<max(targets):
        language='zh' if tokens['zh']<=tokens['en'] else 'en'
        available=[task for lang,task in queues if lang==language and queues[(lang,task)]]
        if not available:break
        task=min(available,key=lambda task:(tasks[task]/weights.get(task,.05),task))
        row=queues[(language,task)].popleft();selected.append(row)
        tokens[language]+=row['answer_token_count'];tasks[task]+=1
        if len(selected) in targets:
            share=tokens['zh']/sum(tokens.values())
            if .45<=share<=.55:snapshots[len(selected)]=list(selected)
            else:shortfalls[len(selected)]='language_token_balance_not_satisfied'
    for target in targets:
        if target not in snapshots:shortfalls.setdefault(target,'insufficient_unique_balanced_candidates')
    return snapshots,shortfalls


def publish(config_path,review_path):
    import audit_sft_data
    settings=read_json(config_path);root=(Path(config_path).resolve().parent/settings['project_root']).resolve()
    output=checked(root,settings['output_dir']);work=checked(root,settings['work_dir'])
    pool=read_json(output/'pool_manifest.json');review=read_json(review_path)
    pool_sha=base.sha256(output/'candidates.jsonl')
    if pool_sha!=pool['candidates_sha256'] or review.get('pool_sha256')!=pool_sha:raise ValueError('Content review belongs to another pool')
    review_sample=list(row_stream(output/'review_sample.jsonl'))
    expected={row['id'] for row in review_sample}
    decisions=review.get('reviewed',[])
    if ({row['id'] for row in decisions}!=expected or len(decisions)!=len(expected)
        or review.get('test_content_used') is not False):raise ValueError('Every proposed review sample needs a documented decision; test stays sealed')
    if any(row.get('decision') not in {'accept','exclude'} or not row.get('reason') for row in decisions):
        raise ValueError('Invalid qualitative review decision')
    excluded={row['id'] for row in decisions if row['decision']=='exclude'}
    donor_path,donor,all_protected=donor_rows(root,settings)
    baseline=[row for row in all_protected if row['split']=='train']
    candidates=[row for row in row_stream(output/'candidates.jsonl') if row['id'] not in excluded]
    datasets,shortfalls=nested_selection(baseline,candidates,settings['train_targets'],settings['task_weights'],settings['seed'])
    if not datasets:raise ValueError('No requested expansion size has enough balanced unique candidates')
    base.atomic_json(output/'quality_review.json',review)
    inventory={'status':'published','datasets':[],'requested_targets':settings['train_targets'],'shortfalls':shortfalls,
               'novel_candidates_after_review':len(candidates),'donor_train_retained':len(baseline),
               'validation_and_test_reused_by_bytes':True,'test_content_used_for_review':False,
               'training_started':False,'quality_review_exclusions':len(excluded)}
    for target,rows in sorted(datasets.items()):
        directory=output/(f'sft_{target//1000}k' if target%1000==0 else f'sft_{target}')
        if directory.exists():raise ValueError('Refusing to overwrite a published expansion')
        directory.mkdir(exist_ok=False)
        train=[{**row,'split':'train'} for row in rows]
        random.Random(settings['seed']).shuffle(train)
        base.write_jsonl(directory/'train.jsonl',train)
        manifest=copy.deepcopy(donor)
        manifest.update(created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                        source_files=read_json(checked(root,settings['reserve_dir'])/'reserve_config.json')['files'],
                        source_config_sha256=base.canonical(settings))
        manifest.pop('curation',None);manifest.pop('elapsed_seconds',None)
        manifest['splits']['train']={'path':'train.jsonl','sha256':base.sha256(directory/'train.jsonl'),'examples':len(train)}
        for split in ('validation','test','pilot'):
            spec=donor['splits'][split];path=directory/(split+'.jsonl')
            shutil.copy2(donor_path.parent/spec['path'],path)
            manifest['splits'][split]={**spec,'path':path.name}
            if base.sha256(path)!=spec['sha256']:raise ValueError('Frozen evaluation/pilot bytes changed')
        shutil.copy2(donor_path.parent/donor['tokenizer']['path'],directory/'tokenizer.json')
        shutil.copy2(config_path,directory/'build_config.json')
        shutil.copy2(output/'quality_review.json',directory/'quality_review.json')
        manifest['quality']={'status':'automatic_expansion_filtering_plus_documented_llm_spot_checks',
            'semantic_gold':False,'review_examples':len(decisions),'review_rejections':len(excluded),
            'near_dedup':pool['deduplication_limits'],'selection':'baseline retained; balanced new tasks and supervised tokens',
            'review_scope':'new candidate pool only; finite qualitative review, not expert gold or an error-rate estimate'}
        manifest['augmentation']={'donor_manifest':settings['donor_manifest'],'donor_manifest_sha256':settings['donor_manifest_sha256'],
            'donor_train_retained':len(baseline),'novel_train_added':target-len(baseline),'pool_sha256':pool_sha,
            'review_sha256':base.sha256(directory/'quality_review.json'),'task_weights':settings['task_weights'],
            'evaluation_frozen':True,'nested_train_targets':settings['train_targets']}
        manifest['build_code']={name:base.sha256(Path(__file__).parent/name) for name in
            ('prepare_sft_expansion.py','prepare_sft_data.py','sft_data_quality.py','audit_sft_data.py')}
        manifest['acceptance']={'status':'pending_independent_audit'}
        manifest['data_identity']=audit_sft_data.dataset_identity(manifest)
        base.atomic_json(directory/'manifest.json',manifest)
        base.atomic_json(directory/'statistics.json',{'train':base.split_statistics(train),
            'donor_train_retained':len(baseline),'new_examples':target-len(baseline),
            'length_budget':{'max_sequence_tokens':settings['max_sequence_tokens'],'max_response_tokens':settings['max_response_tokens']}})
        # 审计器独立重编码全部分区，核验Mask/EOS、简体固定点和精确分区隔离。
        acceptance=audit_sft_data.audit_dataset(directory)
        base.atomic_json(directory/'acceptance.json',acceptance)
        manifest['acceptance']={'status':'passed','report':'acceptance.json',
                                'sha256':base.sha256(directory/'acceptance.json'),'semantic_gold':False}
        base.atomic_json(directory/'manifest.json',manifest)
        inventory['datasets'].append({'train_examples':target,'manifest':str(directory/'manifest.json'),
            'data_identity':manifest['data_identity'],'acceptance':'passed','statistics':base.split_statistics(train)})
    if len(datasets)>1:
        smaller,larger=[{row['id'] for row in rows} for _,rows in sorted(datasets.items())]
        if not smaller<=larger:raise ValueError('Expansion subsets are not nested')
    base.atomic_json(output/'release.json',inventory)
    # 使用新指针，不改变旧latest_sft.json与既有训练入口。
    base.atomic_json(root/'artifacts/posttrain/latest_sft_expansion.json',{'release':str(output/'release.json'),
        'datasets':inventory['datasets'],'current_default_sft_unchanged':True})
    base.atomic_json(work/'publication.json',inventory)
    return inventory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('scan','publish'))
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--review',type=Path)
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false')
    if hasattr(os,'sched_getaffinity'):
        allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[-2:]))
    if hasattr(os,'nice'):os.nice(10)
    result=scan(args.config) if args.action=='scan' else publish(args.config,args.review)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
