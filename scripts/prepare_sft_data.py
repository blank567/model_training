"""自研 300M 的独立 CPU SFT 数据制备，不修改预训练模块或启动 CUDA。

回答 mask 与 input_ids 同位置；训练适配层必须只执行一次因果错位。
自动过滤、字符去重和抽审都有边界，不把公开回答声称为逐条事实金标。
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict,deque
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import time

from fetch_sft_sources import fetch_file,sha256
from sft_data_quality import fingerprint,group_candidates,normalize,quality_reason

PREFIX='System:\n{system}\n\nUser:\n{prompt}\n\nAssistant:\n'
EMPTY_PREFIX='User:\n{prompt}\n\nAssistant:\n'
MASK64=(1<<64)-1
BASE=1000003


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def text_sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def atomic_json(path: Path,value) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def under(root: Path,relative: str) -> Path:
    path=(root/relative).resolve()
    if not path.is_relative_to(root) or path==root:
        raise ValueError('Data path must stay under project root')
    return path


def compact(text: str) -> str:
    return re.sub(r'\s+','',normalize(text).casefold())


def rolling(text: str,width: int=100):
    if len(text)<width:
        return
    factor=pow(BASE,width-1,1<<64)
    value=0
    for character in text[:width]:
        value=(value*BASE+ord(character)+1)&MASK64
    yield 0,value
    for index in range(width,len(text)):
        value=((value-(ord(text[index-width])+1)*factor)*BASE+ord(text[index])+1)&MASK64
        yield index-width+1,value


def holdout_guard(documents: Path,normalizer):
    from llm_lab.data.common import normalize_text
    texts=[];hashes=set();full=set()
    with documents.open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['split'] not in {'validation','test'}:
                continue
            if text_sha(normalize_text(row['text']))!=row['text_sha256']:
                raise ValueError('Frozen holdout row SHA mismatch')
            text=compact(normalizer.convert(normalize(row['text']),row['language']))
            texts.append(text);full.add(text)
            hashes.update(value for _,value in rolling(text))
    if not texts:
        raise ValueError('No pretraining holdout text available')
    def overlaps(*values):
        for original in values:
            value=compact(original)
            if value and value in full:
                return True
            for offset,digest in rolling(value):
                if digest in hashes:
                    fragment=value[offset:offset+100]
                    if any(fragment in document for document in texts):
                        return True
        return False
    return overlaps,{'documents':len(texts),'fingerprints':len(hashes),'fragment_characters':100,
                     'normalization':'NFKC + official t2s fixed point for zh + casefold + whitespace removal',
                     'collision_check':'real substring confirmed after rolling64 hit',
                     'scope':'fixed pretraining validation/test text only; no semantic or translation guarantee'}


def source_rows(path: Path,source: dict):
    if source['format']=='alpaca_json':
        for index,row in enumerate(json.loads(path.read_text(encoding='utf-8'))):
            yield index,row
    elif source['format']=='dolly_jsonl':
        with path.open(encoding='utf-8') as stream:
            for index,line in enumerate(stream):
                yield index,json.loads(line)
    elif source['format']=='messages_parquet':
        import pyarrow.parquet as pq
        index=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128,columns=['messages','source'],use_threads=False):
            for row in batch.to_pylist():
                yield index,row;index+=1
    else:
        raise ValueError('Unknown source format')


def extract(row: dict,source: dict):
    system='';context='';subset=source['name']
    if source['format']=='alpaca_json':
        context=row.get('input','')
        prompt=row['instruction']+('\n'+context if context else '')
        response=row['output']
    elif source['format']=='dolly_jsonl':
        subset=row['category']
        if subset not in source['category_allowlist']:
            return None,'source_category_excluded'
        context=row.get('context','')
        if source.get('require_context') and not context.strip():
            return None,'missing_grounding_context'
        prompt=row['instruction']+'\n'+context
        response=row['response']
    else:
        subset=row.get('source','')
        if subset not in source['source_allowlist']:
            return None,'source_subset_excluded'
        messages=row['messages']
        offset=0
        if messages and messages[0]['role']=='system':
            system=messages[0]['content'];offset=1
        if len(messages)<=offset+1 or messages[offset]['role']!='user' or messages[offset+1]['role']!='assistant':
            return None,'not_self_contained_first_turn'
        prompt=messages[offset]['content'];response=messages[offset+1]['content']
        # 这两个来源通常将原材料放在 user、任务要求放在 system；单独保留材料键。
        if subset in {'smollm-rewrite-30k','smol-summarize-20k'}:
            context=prompt
    if not all(isinstance(value,str) for value in (system,prompt,response,context)):
        return None,'non_string_fields'
    return {'system':system,'prompt':prompt,'response':response,'input_context':context,'subset':subset},None


def task_kind(prompt: str,subset: str) -> str:
    text=prompt.casefold()
    if re.search(r'summar|摘要|总结|概括',text):return 'summarization'
    if re.search(r'rewrit|rephras|改写|重写|润色|分成两句',text):return 'rewriting'
    if re.search(r'extract|提取|抽取|找出',text):return 'extraction'
    if re.search(r'json|format|格式|lowercase|大写|小写',text):return 'format_following'
    if re.search(r'classif|分类|情感',text):return 'classification'
    return 'short_qa_or_other'


def encode_row(row: dict,tokenizer,eos: int):
    """prefix 与回答分别编码，边界不会因 BPE 合并而误遮罩回答首 Token。"""
    prefix=(PREFIX if row['system'] else EMPTY_PREFIX).format(**row)
    before=tokenizer.encode(prefix,add_special_tokens=False).ids
    answer=tokenizer.encode(row['response'],add_special_tokens=False).ids+[eos]
    return before+answer,[0]*len(before)+[1]*len(answer),len(before),len(answer)


def take_balanced(queues: dict[str,deque],count: int,seed: int):
    """按监督 Token 总量选语言；条数配额与语言 Token 配额分别检查。"""
    selected=[];tokens={'zh':0,'en':0}
    for _ in range(count):
        language='zh' if tokens['zh']<=tokens['en'] else 'en'
        if not queues[language]:
            raise ValueError('Insufficient unique '+language+' candidates for requested balanced split')
        row=queues[language].popleft()
        selected.append(row);tokens[language]+=row['answer_token_count']
    share=tokens['zh']/sum(tokens.values())
    if not .45<=share<=.55:
        raise ValueError('Language supervised-token quota not satisfied')
    random.Random(seed).shuffle(selected)
    return selected


def write_jsonl(path: Path,rows):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
        stream.flush();os.fsync(stream.fileno())


def split_statistics(rows):
    languages={}
    for language in ('zh','en'):
        selected=[row for row in rows if row['language']==language]
        languages[language]={'examples':len(selected),'input_tokens':sum(len(row['input_ids']) for row in selected),
                             'answer_tokens':sum(row['answer_token_count'] for row in selected)}
    return {'examples':len(rows),'languages':languages,'sources':dict(Counter(row['source']['dataset'] for row in rows)),
            'tasks_heuristic':dict(Counter(row['task'] for row in rows)),
            'max_input_tokens':max(len(row['input_ids']) for row in rows),
            'within_preferred_512':sum(len(row['input_ids'])<=512 for row in rows)}


def build(config_path: Path):
    started=time.monotonic()
    settings=json.loads(config_path.read_text(encoding='utf-8'))
    root=(config_path.resolve().parent/settings['project_root']).resolve()
    if settings.get('language_supervised_token_share')!={'zh':.5,'en':.5} or settings.get('language_share_tolerance')!=.05:
        raise ValueError('This first SFT build supports only zh/en 50/50 with a 5 percentage-point tolerance')
    output=under(root,settings['output_dir']);work=under(root,settings['work_dir'])
    work.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise ValueError('Refusing to overwrite existing SFT dataset')
    if shutil.disk_usage(work).free<int(settings['min_free_bytes'])+512*1024**2:
        raise OSError('Insufficient SFT build disk reserve')
    def progress(phase,**extra):
        record={'phase':phase,'elapsed_seconds':time.monotonic()-started,**extra}
        atomic_json(work/'progress.json',record)
        print(json.dumps(record,ensure_ascii=False),flush=True)
    # 构建进程最多使用两个允许的 CPU，且不初始化任何 torch/CUDA 模块。
    if hasattr(os,'sched_getaffinity'):
        allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[-2:]))
    os.environ['TOKENIZERS_PARALLELISM']='false'
    from llm_lab.data.text_normalization import build_chinese_normalizer
    from tokenizers import Tokenizer
    normalizer=build_chinese_normalizer('t2s')
    parent_path=under(root,settings['parent_manifest'])
    if sha256(parent_path)!=settings['parent_manifest_sha256']:
        raise ValueError('Pretraining manifest identity changed')
    parent=json.loads(parent_path.read_text(encoding='utf-8'))
    tokenizer_path=under(parent_path.parent,parent['tokenizer']['path'])
    documents=under(parent_path.parent,parent['documents']['path'])
    if sha256(tokenizer_path)!=settings['tokenizer_sha256'] or sha256(documents)!=parent['documents']['sha256']:
        raise ValueError('Pretraining tokenizer/holdout identity changed')
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    assert tokenizer.token_to_id('<|eos|>')==2 and tokenizer.get_vocab_size()==32000
    overlaps,guard_info=holdout_guard(documents,normalizer)
    progress('holdout_index_ready',**guard_info)
    candidates=[];rejects=Counter();source_stats={}
    rejected_path=work/'rejections.jsonl'
    with rejected_path.open('w',encoding='utf-8',newline='\n') as rejection_log:
        for source in settings['sources']:
            path=fetch_file(root,source,int(settings['min_free_bytes']))
            accepted=0;seen=0;reasons=Counter()
            for index,raw in source_rows(path,source):
                seen+=1
                reason=None
                if index in source.get('exclude_row_indices',[]):
                    reason='known_research_rejection'
                data,extraction_reason=extract(raw,source)
                reason=reason or extraction_reason
                if not reason:
                    original={key:data[key] for key in ('system','prompt','response','input_context')}
                    for key in original:
                        data[key]=normalize(normalizer.convert(normalize(original[key]),source['language']))
                    combined=(data['system']+'\n'+data['prompt']).strip()
                    reason=quality_reason(combined,data['response'],source['language'],data['subset'])
                    letters=[character for character in data['response'] if character.isalpha()]
                    han=sum('\u4e00'<=character<='\u9fff' for character in letters)
                    if not reason and len(letters)>20:
                        if source['language']=='zh' and han/len(letters)<.30:reason='chinese_response_language_mismatch'
                        if source['language']=='en' and han/len(letters)>.02:reason='english_response_language_mismatch'
                if not reason:
                    ids,mask,prompt_count,answer_count=encode_row(data,tokenizer,2)
                    if len(ids)>int(settings['max_sequence_tokens']):reason='sequence_too_long'
                    elif answer_count>int(settings['max_response_tokens']):reason='response_too_long'
                    elif any(token in {0,1,2} for token in ids[:-1]):reason='embedded_special_token'
                    elif overlaps(data['system'],data['prompt'],data['response'],data['input_context']):reason='pretraining_holdout_overlap'
                if reason:
                    reasons[reason]+=1;rejects[reason]+=1
                    rejection_log.write(json.dumps({'dataset':source['name'],'row_index':index,'reason':reason},ensure_ascii=False)+'\n')
                else:
                    source_meta={'dataset':source['name'],'revision':source['revision'],'file':source['file'],
                                 'row_index':index,'source_subset':data['subset'],'raw_row_sha256':canonical(raw),
                                 'original_prompt_sha256':text_sha(original['prompt']),
                                 'original_response_sha256':text_sha(original['response']),
                                 'original_context_sha256':text_sha(original['input_context']),
                                 'normalized_text_sha256':canonical({key:data[key] for key in original}),
                                 'license':source['license']}
                    identifier=canonical({'repository':source['repository'],'revision':source['revision'],
                                          'file':source['file'],'row_index':index,'raw_sha256':source_meta['raw_row_sha256']})
                    candidates.append({'schema_version':1,'id':identifier,'language':source['language'],
                                       'task':task_kind(combined,data['subset']),**{key:data[key] for key in original},
                                       'input_ids':ids,'loss_mask':mask,'prompt_token_count':prompt_count,
                                       'answer_token_count':answer_count,'source':source_meta,
                                       'source_group_id':source['name']+':'+canonical(raw)})
                    accepted+=1
                if seen%5000==0:
                    progress('filtering',source=source['name'],rows_seen=seen,accepted=accepted,total_candidates=len(candidates))
            source_stats[source['name']]={'rows_seen':seen,'accepted_before_dedup':accepted,'rejections':dict(reasons)}
            progress('source_filtered',source=source['name'],**source_stats[source['name']])
    if not candidates:
        raise ValueError('No SFT candidates passed automatic checks')
    progress('grouping_near_duplicates',candidates=len(candidates))
    # 分组接口使用 answer 键，输出仍保留 response，避免和下游样本格式混淆。
    grouping=[{**{key:row[key] for key in ('id','prompt','input_context','source_group_id')},'answer':row['response']} for row in candidates]
    group_ids=group_candidates(grouping,threshold=float(settings['near_duplicate_jaccard']))
    grouped=defaultdict(list)
    for row,group_id in zip(candidates,group_ids):
        row['group_id']=group_id;row.pop('source_group_id',None);grouped[group_id].append(row)
    # 一个重复分量仅取一个代表样本，使相同问题/材料不会在不同分区泄漏。
    unique=[]
    for rows in grouped.values():
        unique.append(min(rows,key=lambda row:(len(row['input_ids'])>512,len(row['input_ids']),row['id'])))
    queues={}
    for language in ('zh','en'):
        ordered=sorted((row for row in unique if row['language']==language),key=lambda row:row['id'])
        random.Random(settings['seed']).shuffle(ordered)
        ordered.sort(key=lambda row:len(row['input_ids'])>int(settings['preferred_sequence_tokens']))
        queues[language]=deque(ordered)
    progress('selecting_splits',unique_groups=len(grouped),unique_languages={language:len(queue) for language,queue in queues.items()})
    selected={}
    for split in ('validation','test','train'):
        selected[split]=[{**row,'split':split} for row in take_balanced(queues,int(settings['targets'][split]),settings['seed'])]
    pilot_queues={language:deque(row for row in selected['train'] if row['language']==language) for language in ('zh','en')}
    selected['pilot']=[{**row,'split':'pilot'} for row in take_balanced(pilot_queues,int(settings['targets']['pilot']),settings['seed'])]
    output.mkdir(parents=True,exist_ok=False)
    splits={};statistics={}
    for split,rows in selected.items():
        path=output/(split+'.jsonl');write_jsonl(path,rows)
        splits[split]={'path':path.name,'sha256':sha256(path),'examples':len(rows)}
        statistics[split]=split_statistics(rows)
    shutil.copy2(tokenizer_path,output/'tokenizer.json')
    atomic_json(output/'build_config.json',settings)
    atomic_json(output/'statistics.json',{'splits':statistics,'filtering':source_stats,'rejections':dict(rejects),
                                        'before_dedup':len(candidates),'unique_groups':len(grouped)})
    # 结构化抽审清单只来自train/validation；独立test不进入调参和样本选择反馈。
    review=[]
    for split in ('train','validation'):
        strata=defaultdict(list)
        for row in selected[split]:strata[(row['language'],row['task'],row['source']['dataset'])].append(row)
        for key,rows in sorted(strata.items()):
            for row in sorted(rows,key=lambda row:row['id'])[:3]:
                review.append({key:row[key] for key in ('id','split','language','task','system','prompt','response','source')})
    write_jsonl(output/'review_sample.jsonl',review)
    manifest={'schema_version':1,'kind':'sft_tokenized_conversations','fixture':False,
              'created_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              'splits':splits,'tokenizer':{'path':'tokenizer.json','sha256':sha256(output/'tokenizer.json'),
                                         'eos_id':2,'pad_id':0,'unk_id':1,'vocab_size':32000},
              'tokenization':{'prefix_template':PREFIX,'empty_system_prefix_template':EMPTY_PREFIX,
                              'encoding':'prefix_and_response_separately','max_sequence_tokens':settings['max_sequence_tokens'],
                              'labels':'unshifted same-position loss mask; training adapter must shift exactly once',
                              'eos_supervised':True},
              'chinese_normalization':normalizer.provenance,
              'pretraining_parent':{'path':settings['parent_manifest'],'sha256':sha256(parent_path),
                                    'documents_sha256':sha256(documents),'guard':guard_info},
              'source_files':settings['sources'],'source_config_sha256':canonical(settings),
              'build_code':{name:sha256(Path(__file__).parent/name) for name in ('prepare_sft_data.py','sft_data_quality.py','fetch_sft_sources.py')},
              'quality':{'status':'automatic_heuristics_with_review_queue','semantic_gold':False,
                         'known_research_rejections_applied':True,'review_examples':len(review),
                         'near_dedup':'exact keys + bounded bottom-k candidate/Jaccard check; one representative per component',
                         'near_dedup_limits':'5-char shingles, prompt/context head+tail up to4096 chars, bounded buckets; no full semantic/translation guarantee'},
              'acceptance':{'status':'pending_independent_audit'},'elapsed_seconds':time.monotonic()-started}
    identity_keys=('schema_version','kind','splits','tokenizer','tokenization','chinese_normalization')
    manifest['data_identity']=canonical({key:manifest[key] for key in identity_keys})
    atomic_json(output/'manifest.json',manifest)
    progress('built_pending_audit',output=str(output),data_identity=manifest['data_identity'],statistics=statistics)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();build(args.config)
