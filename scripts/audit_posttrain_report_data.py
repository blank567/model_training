"""为最终报告复核公开数据身份和精确隔离，不读取或修改权重。"""
from collections import Counter
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from llm_lab.common.io import read_json,sha256_file,write_json
from llm_reading_independent.isolation import compact


def rows(path,split):
    path=Path(path);m=read_json(path)
    spec=m['splits'][split] if 'splits' in m else m['files'][split]
    file=(path.parent/spec['path']).resolve()
    if not file.is_relative_to(path.parent.resolve()) or sha256_file(file)!=spec['sha256']:raise ValueError('Changed/escaped split')
    values=[]
    with file.open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            values.append({k:r.get(k) for k in ['id','language','source','task','question','context','group_id','prompt','response','question_type','answer_length_bucket','prompt_token_count','answer_token_count']})
    if len(values)!=spec.get('examples',spec.get('records')):raise ValueError('Split count differs')
    return values


def overlap(train,test,reading):
    fields=['question','context','group_id'] if reading else ['prompt']
    result={}
    for field in fields:
        a={compact(r[field]) for r in train if isinstance(r.get(field),str) and r[field]}
        b={compact(r[field]) for r in test if isinstance(r.get(field),str) and r[field]}
        result[field]={'train_unique':len(a),'test_unique':len(b),'overlapping_keys':len(a&b)}
    return result


def main(work):
    work=Path(work);plan=read_json(work/'plan.json');result={'scope':'exact normalized supervised train/test isolation; not semantic or pretraining contamination proof','reading_arms':{},'instruction_training':[]}
    reading_test=rows(plan['suites']['reading_test_2000']['manifest'],'test')
    generic_test=rows(plan['suites']['instruction_test_1000']['manifest'],'test')
    result['test_statistics']={}
    def source_name(value):
        if isinstance(value,dict):return value.get('dataset') or value.get('dataset_id') or value.get('name') or value.get('source') or 'structured_source'
        return value or 'unknown'
    for key,values in [('reading_test_2000',reading_test),('instruction_test_1000',generic_test)]:
        result['test_statistics'][key]={'examples':len(values),'languages':dict(Counter(r['language'] for r in values)),
            'sources':dict(Counter(source_name(r.get('source')) for r in values)),
            'question_types':dict(Counter((r.get('language') or '')+'/'+(r.get('question_type') or r.get('task') or 'unknown') for r in values)),
            'group_count':len({r['group_id'] for r in values if r.get('group_id')}),
            'supervised_tokens':dict(Counter({lang:sum(r['answer_token_count'] for r in values if r['language']==lang) for lang in ['zh','en']}))}
    processed={}
    queue=read_json(ROOT/'runs/assistant/system/reading_rl_queue_20261102/queue.json')
    for job in queue['jobs']:
        if job.get('algorithm') not in ['dpo','ppo','grpo']:continue
        cfg=read_json(job['config']);arm=read_json(cfg['arm_manifest']);source=(Path(cfg['arm_manifest']).parent/arm['source_manifest']).resolve()
        if str(source) not in processed:processed[str(source)]=rows(source,'train')
        values=processed[str(source)]
        key=cfg['experiment']['arm']
        if key not in result['reading_arms']:
            result['reading_arms'][key]={'manifest':cfg['arm_manifest'],'manifest_sha256':sha256_file(cfg['arm_manifest']),
                'source':str(source),'statistics':arm['statistics'],'condition':arm['condition'],
                'source_vs_final_test_exact_isolation':overlap(values,reading_test,True)}
        if cfg.get('preference_manifest'):
            pref=read_json(cfg['preference_manifest'])
            result['reading_arms'][key]['dpo_preferences']=pref
    seen=set()
    for item in plan['models']:
        if item['stage'] not in ['sft_20k','sft_100k','lora']:continue
        p=Path(item['checkpoint']).parent/'config.json'
        if not p.exists():continue
        cfg=read_json(p);source=cfg['data_manifest']
        if source in seen:continue
        seen.add(source);values=rows(source,'train')
        result['instruction_training'].append({'manifest':source,'manifest_sha256':sha256_file(source),'examples':len(values),
             'vs_final_test':overlap(values,generic_test,False)})
    write_json(work/'report_data_audit.json',result)
    print(json.dumps({'status':'completed','reading_arms':len(result['reading_arms']),
        'instruction_sources':len(result['instruction_training']),
        'reading_overlap':{k:v['source_vs_final_test_exact_isolation'] for k,v in result['reading_arms'].items()},
        'instruction_overlap':result['instruction_training']},ensure_ascii=False))


if __name__=='__main__':
    try:main(sys.argv[1])
    except BaseException as e:
        write_json(Path(sys.argv[1])/'report_data_audit_failure.json',{'type':type(e).__name__,'error':str(e)});raise
