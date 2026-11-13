"""以 Numina 公开有效性标记和 CMM 原生中文参考解答构建新的候选版本。"""
from collections import Counter,defaultdict
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json
import random
import re
import shutil

from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json,write_jsonl
from llm_math_data.verifier import numeric_value
from .build import PROTOCOL,IDENTITY_FIELDS,allocate,clean_text,exclusion_hashes,prefix
from .rules import classify,reject_reason,question_hash,final_answer,answer_key,boxed_values

WEIGHTS={'algebra_equations':.20,'geometry':.17,'functions_trigonometry':.12,
         'probability_statistics':.10,'combinatorics':.10,'number_theory':.10,
         'calculus':.08,'linear_algebra':.05,'inequalities':.04,'logic_puzzles':.02,'other_math':.02}
PUBLISHER_TOPICS={'Algebra':'algebra_equations','Geometry':'geometry','Number Theory':'number_theory',
                 'Combinatorics':'combinatorics','Calculus':'calculus','Inequalities':'inequalities',
                 'Logic and Puzzles':'logic_puzzles','Other':'other_math'}


def parse_row(raw,kind,converter):
    if kind=='numina':
        if raw.get('problem_is_valid')!='Yes' or raw.get('solution_is_valid')!='Yes':return None,'publisher_validity_not_yes'
        if raw.get('synthetic') is not False:return None,'publisher_synthetic_or_unknown'
        q,a=raw.get('problem'),raw.get('solution');gold=raw.get('answer')
        language='en';source=raw.get('source','');original_topic=raw.get('problem_type')
        topic=classify(q or '',source) or PUBLISHER_TOPICS.get(original_topic,'other_math')
        if original_topic in {'Inequalities','Logic and Puzzles'}:topic=PUBLISHER_TOPICS[original_topic]
        grade=None
    else:
        image=raw.get('image','unknown')
        if image not in {None,'','[]','null'}:return None,'requires_image'
        q=raw.get('question');options=raw.get('options')
        if options and options not in {'null','None','[]'}:q=(q or '')+'\n'+options
        a=raw.get('analysis')
        if not a or a in {'null','None'}:a=raw.get('solution')
        gold=raw.get('answer');language='zh';source='CMM-Math official train';original_topic=raw.get('subject');grade=raw.get('level')
        topic=classify(q or '',source) or ('inequalities' if '不等式' in (original_topic or '') else 'other_math')
    if not isinstance(q,str) or not isinstance(a,str):return None,'missing_reference_solution'
    q=clean_text(q,converter if language=='zh' else None);a=clean_text(a,converter if language=='zh' else None)
    reason=reject_reason(q,a,'CoT',language)
    if reason:return None,reason
    if re.search(r'\b(?:Theorem|Lemma|Proposition)\s+\d+\b|(?:定理|引理|命题)\s*\d+',a,re.I):
        return None,'missing_numbered_theorem_context'
    # 图中/选项中的图片必须同时排除；不能靠删除图片标记制造纯文本题。
    if re.search(r'<image>|<img|\\includegraphics|\.(?:png|jpg|jpeg)',q,re.I):return None,'requires_image'
    if len(q)<12 or len(a)<30:return None,'insufficient_problem_or_solution'
    if isinstance(gold,str):
        gold=clean_text(gold,converter if language=='zh' else None)
        gold=re.sub(r'^(?:最终)?答案\s*(?:是|为|[:：])\s*','',gold).strip()
        if gold.casefold() in {'proof','notfound','null','none',''}:gold=None
    else:gold=None
    boxes={answer_key(value) for value in boxed_values(a)}
    if gold and len(boxes)>1 and re.search(r'eigenvalues|find all|all (?:values|solutions)|\broots\b|分别|所有|全部|\((?:I|II|III)\)',q,re.I):
        target=answer_key(gold)
        if not all(value and value in target for value in boxes):
            return None,'incomplete_multi_answer_gold'
    found=final_answer(a)
    # 有明确数值/选项结论的题必须与源标准答案一致，不自动改成正确答案掩盖错误推导。
    if gold and found:
        gv,fv=numeric_value(gold),numeric_value(found)
        if gv is not None and fv is not None and gv!=fv:return None,'reference_numeric_disagrees_with_gold'
        if re.fullmatch('[A-H]',gold) and re.fullmatch('[A-H]',answer_key(found) or '') and gold!=answer_key(found):return None,'reference_choice_disagrees_with_gold'
    instruction='请给出解题过程，并在最后一行写出最终答案。' if language=='zh' else 'Show your solution and put the final answer on the last line.'
    prompt=q+'\n'+instruction
    if gold:a+='\n'+('答案：' if language=='zh' else 'Answer: ')+gold
    return {'language':language,'question':q,'prompt':prompt,'response':a,'final_answer':gold,'topic':topic,
        'publisher_topic':original_topic,'publisher_grade':grade,'source':source,
        'question_type':raw.get('question_type'),'source_record_sha256':canonical_hash(raw)},None


def group_key(question):
    """原题精确摘要；较长公式另给出跨语保守去重索引，不承诺全语义去重。"""
    return question_hash(question)


def formula_key(question):
    chunks=re.findall(r'\$([^$]+)\$',question)
    text=''.join(chunks)
    text=re.sub(r'\\(?:left|right|mathrm|mathbf|text|quad|qquad)|[\s{}]','',text)
    if len(text)<40:return None
    return canonical_hash({'math':text,'numbers':sorted(set(re.findall(r'\d+(?:\.\d+)?',question)))})


def build(root,cache,token_file,out,work):
    import pyarrow.parquet as pq
    from opencc import OpenCC
    from tokenizers import Tokenizer
    if out.exists():raise ValueError('Never overwrite an earlier candidate/release')
    lock=read_json(cache/'sources.lock.json');converter=OpenCC('t2s');tokenizer=Tokenizer.from_file(str(token_file))
    for s in lock['sources']:
        if sha256_file(cache/s['local_path'])!=s['sha256']:raise ValueError('Pinned public source changed')
    exclusions,test_hashes=exclusion_hashes(root)
    pool=defaultdict(list);seen_count=Counter();rejected=Counter();source_counts=Counter();rng=random.Random(20261101)
    files=[s for s in lock['sources'] if s['path'].endswith('.parquet') and s['source'] in {'project-numina/NuminaMath-1.5','ecnu-icalk/cmm-math'}]
    # 先载入原生中文，跨语公式重复时优先保留原生题面。
    files.sort(key=lambda s:(s['source']!='ecnu-icalk/cmm-math',s['path']))
    for spec in files:
        kind='numina' if spec['source'].startswith('project-numina/') else 'cmm'
        index=0
        for batch in pq.ParquetFile(cache/spec['local_path']).iter_batches(batch_size=4096):
            for raw in batch.to_pylist():
                record,reason=parse_row(raw,kind,converter)
                if reason:rejected[kind+'/'+reason]+=1;index+=1;continue
                record['upstream_file']=spec['local_path'];record['upstream_row']=index;index+=1
                source_counts[record['source']]+=1
                if group_key(record['question']) in exclusions:rejected['known_external_or_project_question']+=1;continue
                key=(record['language'],record['topic']);seen_count[key]+=1
                values=pool[key]
                if len(values)<12000:values.append(record)
                else:
                    slot=rng.randrange(seen_count[key])
                    if slot<12000:values[slot]=record
            print(json.dumps({'phase':'source_scan','source':spec['local_path'],'rows':index}),flush=True)
    eligible=defaultdict(list);qseen=set();fseen=set()
    for key,values in sorted(pool.items(),key=lambda item:(item[0][0]!='zh',item[0][1])):
        rng.shuffle(values)
        for row in values:
            qkey,fkey=group_key(row['question']),formula_key(row['question'])
            if qkey in qseen or fkey and fkey in fseen:rejected['duplicate_original_or_formula']+=1;continue
            p=tokenizer.encode(prefix(row['prompt']),add_special_tokens=False).ids
            a=tokenizer.encode(row['response'],add_special_tokens=False).ids+[2]
            if len(p)+len(a)>1024 or len(a)<8 or any(v in {0,1,2} for v in p+a[:-1]):rejected['context_or_tokens']+=1;continue
            qseen.add(qkey)
            if fkey:fseen.add(fkey)
            row.update(group_id=qkey,input_ids=p+a,loss_mask=[0]*len(p)+[1]*len(a),prompt_token_count=len(p),answer_token_count=len(a))
            eligible[key].append(row)
    counts={language:{topic:len(eligible[(language,topic)]) for topic in WEIGHTS} for language in ['zh','en']}
    write_json(work/'vetted_candidate_statistics.json',{'eligible':counts,'rejected':dict(rejected),'upstream_sources':dict(source_counts)})
    if min(sum(counts[language].values()) for language in ['zh','en'])<2000:raise ValueError('Insufficient bilingual coverage')
    selections={};offsets=Counter()
    for split,amounts in [('validation',{'zh':1000,'en':1000}),('test',{'zh':1000,'en':1000}),
                          ('train',{'zh':min(25000,sum(counts['zh'].values())-2000),'en':0})]:
        if split=='train':amounts['en']=50000-amounts['zh']
        selected=[]
        for language,n in amounts.items():
            capacity={topic:counts[language][topic]-offsets[(language,topic)] for topic in WEIGHTS}
            quotas=allocate(n,capacity,WEIGHTS)
            for topic,count in quotas.items():
                key=(language,topic);start=offsets[key]
                selected.extend(eligible[key][start:start+count]);offsets[key]+=count
        rng.shuffle(selected);selections[split]=selected
    selections['pilot']=selections['train'][:2000]
    out.mkdir(parents=True);shutil.copyfile(token_file,out/'tokenizer.json');specs={};stats={}
    for split,selected in selections.items():
        language_counts=Counter();topic_counts=Counter();source_counts=Counter();answer_tokens=Counter();terminal=Counter()
        def rows():
            for row in selected:
                language=row['language'];gold=row['final_answer']
                kind='numeric_exact' if numeric_value(gold) is not None else 'choice_exact' if isinstance(gold,str) and re.fullmatch('[A-H]',gold) else 'symbolic_pending' if gold else 'proof_or_unavailable'
                value={'schema_version':1,'split':split,'id':'vetted_math_'+row['group_id'][:24], 'system':'',
                    **{k:v for k,v in row.items() if k not in {'source_record_sha256'}},
                    'source_record_sha256':row['source_record_sha256'],'available_terminal_verifier':kind,
                    'normalized_text_sha256':canonical_hash([language,'',row['prompt'],row['response']]),
                    'topic_assignment':'publisher_domain_with_heuristic_subdomain','source_difficulty':row['publisher_grade']}
                language_counts[language]+=1;topic_counts[row['topic']]+=1;source_counts[row['source']]+=1;answer_tokens[language]+=row['answer_token_count'];terminal[kind]+=1
                yield value
        file=out/(split+'.jsonl');write_jsonl(file,rows())
        specs[split]={'path':file.name,'examples':len(selected),'sha256':sha256_file(file)}
        stats[split]={'records':len(selected),'languages':dict(language_counts),'topics':dict(topic_counts),
            'sources':dict(source_counts),'answer_tokens':dict(answer_tokens),'terminal_verifier_coverage':dict(terminal),
            'zh_answer_token_fraction':answer_tokens['zh']/sum(answer_tokens.values())}
    manifest={'schema_version':1,'kind':'sft_tokenized_conversations','fixture':False,'status':'awaiting_quality_audit',
        'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'splits':specs,
        'tokenizer':{'path':'tokenizer.json','sha256':sha256_file(token_file),'vocab_size':32000,'pad_id':0,'unk_id':1,'eos_id':2},
        'tokenization':PROTOCOL,'chinese_normalization':{'mode':'t2s','package':'OpenCC','package_version':'1.1.9','max_passes':8},
        'origin':{'source_lock_sha256':sha256_file(cache/'sources.lock.json'),'public_sources':lock['sources'],
            'training_sources':['NuminaMath-1.5 non-synthetic rows marked problem_is_valid=Yes and solution_is_valid=Yes',
                                'CMM-Math official train, image-free with reference analysis/solution'],
            'local_template_generation_used':False,'machine_translation_used_for_chinese':False},
        'statistics':stats,'curation':{'rejected':dict(rejected),'eligible':counts,'external_exclusion_sources':test_hashes,
            'grouping':'exact normalized question plus conservative long-formula dedup; not complete cross-language semantic decontamination'},
        'training_policy':{'epochs':1,'stop_epochs':None},'training_started':False,
        'limits':['Publisher validity flags do not constitute independent proof verification',
                  'Subdomain tags are heuristic; publisher domain and grade are preserved',
                  'Proofs and unsupported symbolic answers cannot use the existing strict numeric reward']}
    manifest['data_identity']=canonical_hash({k:manifest[k] for k in IDENTITY_FIELDS})
    write_json(out/'manifest.json',manifest)
    print(json.dumps({'status':'candidate_built','manifest':str(out/'manifest.json'),'statistics':stats},ensure_ascii=False),flush=True)
    return out/'manifest.json'
