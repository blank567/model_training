"""官方人工答案、完整上下文、文章分组和中英监督Token均衡的SFT发布。"""
from __future__ import annotations
from collections import Counter,defaultdict
from datetime import datetime,timedelta,timezone
from pathlib import Path
import bisect,json,random,shutil

from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json,write_jsonl
from .source import source_rows,convert_record,make_prompt,digest

PROTOCOL={'prefix_template':'System:\n{system}\n\nUser:\n{prompt}\n\nAssistant:\n',
    'empty_system_prefix_template':'User:\n{prompt}\n\nAssistant:\n','encoding':'prefix_and_response_separately',
    'max_sequence_tokens':1024,'labels':'unshifted same-position loss mask; training adapter must shift exactly once','eos_supervised':True}
IDENTITY_FIELDS=('schema_version','kind','splits','tokenizer','tokenization','chinese_normalization')
SOURCES=['ymcui/cmrc2018','DRCKnowledgeTeam/DRCD','rajpurkar/squad']
QUALITY_EXCLUSIONS=[{'source':'ymcui/cmrc2018','title':'印度-阿拉伯数字系统',
    'reason':'Source passage/question contains an unreliable Latin-America claim about medieval Fibonacci; exclude the entire article instead of rewriting human labels'}]


def prefix(prompt):return 'User:\n'+prompt+'\n\nAssistant:\n'


class UnionFind:
    def __init__(self):self.parents={}
    def find(self,key):
        self.parents.setdefault(key,key)
        if self.parents[key]!=key:self.parents[key]=self.find(self.parents[key])
        return self.parents[key]
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parents[max(a,b)]=min(a,b)


def question_type(question,language):
    q=question.casefold()
    keys=[('person',['谁','何人','who']),('time',['何时','什么时候','哪一年','哪一天','多久','when','what year']),
          ('place',['哪里','何处','哪国','哪座','哪个城市','where']),('quantity',['多少','几个','how many','how much']),
          ('reason',['为何','为什么','why']),('manner',['如何','怎样','how'])]
    for kind,markers in keys:
        if any(marker in q for marker in markers):return kind
    return 'other'


def length_bucket(answer_tokens):return 'short' if answer_tokens<=4 else 'medium' if answer_tokens<=12 else 'long'


def unique_inputs(rows):
    """官方题号可能不同但文章与问题相同；在划分抽样前去重。"""
    seen=set();chosen=[]
    for row in rows:
        key=digest([row['context'],row['question']])
        if key not in seen:chosen.append(row);seen.add(key)
    return chosen


def reserve_articles(rows,amount,seed,cap):
    groups=defaultdict(list)
    for row in rows:groups[row['group_id']].append(row)
    rng=random.Random(seed);keys=sorted(groups);rng.shuffle(keys);chosen=[];capacity=0
    for key in keys:
        rng.shuffle(groups[key]);chosen.append(key);capacity+=min(cap,len(groups[key]))
        if capacity>=amount:break
    if capacity<amount:raise ValueError('Insufficient article-disjoint evaluation samples')
    selected=[]
    for offset in range(cap):
        for key in chosen:
            if offset<len(groups[key]):selected.append(groups[key][offset])
            if len(selected)==amount:return selected,set(chosen)
    raise AssertionError('Evaluation allocation failed')


def token_balanced(rows,total=50000,seed=42,tolerance=.005):
    values={language:[row for row in rows if row['language']==language] for language in ['zh','en']}
    for language in values:random.Random(str(seed)+language).shuffle(values[language])
    sums={language:[0] for language in values}
    for language in values:
        for row in values[language]:sums[language].append(sums[language][-1]+row['answer_token_count'])
    minimum=min(1000,max(1,total//10))
    low=max(minimum,total-len(values['en']));high=min(len(values['zh']),total-minimum)
    if low>high:raise ValueError('Insufficient unique bilingual training examples')
    best=min(range(low,high+1),key=lambda n:abs(sums['zh'][n]-sums['en'][total-n]))
    zh_tokens,en_tokens=sums['zh'][best],sums['en'][total-best]
    fraction=zh_tokens/(zh_tokens+en_tokens)
    if abs(fraction-.5)>tolerance:raise ValueError('Cannot meet answer-token language balance without duplicating data')
    chosen=values['zh'][:best]+values['en'][:total-best];random.Random(seed).shuffle(chosen)
    return chosen,{'zh_examples':best,'en_examples':total-best,'zh_answer_tokens':zh_tokens,
                   'en_answer_tokens':en_tokens,'zh_answer_token_fraction':fraction,'tolerance':tolerance}


def build(root,cache,tokenizer_path,out,work):
    from opencc import OpenCC
    from tokenizers import Tokenizer
    if out.exists():raise ValueError('Never overwrite a prior dataset')
    work.mkdir(parents=True,exist_ok=True);cache=Path(cache)
    lock=read_json(cache/'sources.lock.json');tokenizer=Tokenizer.from_file(str(tokenizer_path));converter=OpenCC('t2s')
    if sha256_file(tokenizer_path)!='6040c90c045836982a47bc0544942f122d0446bf9ed4d2ac1929befd0d4b86b3':raise ValueError('Wrong project tokenizer')
    rows=[];rejected=Counter();raw_counts=Counter();uf=UnionFind()
    for spec in lock['sources']:
        if sha256_file(cache/spec['local_path'])!=spec['sha256']:raise ValueError('Pinned source checksum mismatch')
        if spec['role']=='documentation':continue
        for raw in source_rows(cache,spec):
            raw_counts[spec['local_path']]+=1;row,reason=convert_record(raw,converter)
            if reason:rejected[raw['source']+'/'+reason]+=1;continue
            if any(row['source']==issue['source'] and row['title']==issue['title'] for issue in QUALITY_EXCLUSIONS):
                rejected[raw['source']+'/known_source_quality_issue']+=1;continue
            row['source_file_sha256']=spec['sha256'];row['source_revision']=spec['revision']
            uf.union('article:'+row['article_key'],'passage:'+row['passage_key'])
            row['prompt']=make_prompt(row['context'],row['question'],row['language'])
            pids=tokenizer.encode(prefix(row['prompt']),add_special_tokens=False).ids
            targets=[tokenizer.encode(answer,add_special_tokens=False).ids+[2] for answer in row['answers']]
            if len(pids)+64>1024:rejected[raw['source']+'/complete_prompt_exceeds_960_tokens']+=1;continue
            if any(len(ids)>64 for ids in targets):rejected[raw['source']+'/human_answer_exceeds_generation_budget']+=1;continue
            if any(t in {0,1,2} for t in pids+[t for ids in targets for t in ids[:-1]]):rejected[raw['source']+'/invalid_special_tokens']+=1;continue
            row.update(response=row['answers'][0],input_ids=pids+targets[0],loss_mask=[0]*len(pids)+[1]*len(targets[0]),
                prompt_token_count=len(pids),answer_token_count=len(targets[0]),
                question_type=question_type(row['question'],row['language']),answer_length_bucket=length_bucket(len(targets[0])-1))
            rows.append(row)
        print(json.dumps({'phase':'source_prepared','file':spec['local_path'],'raw_records':raw_counts[spec['local_path']],
                          'eligible_total':len(rows)}),flush=True)
    for row in rows:row['group_id']=digest(uf.find('article:'+row['article_key']))
    # 官方评测只按预先声明的源分区取题；不按任何模型得分选择。
    final={
        'ymcui/cmrc2018':lambda r:r['source']=='ymcui/cmrc2018' and 'dev.json' in r['source_file'],
        'DRCKnowledgeTeam/DRCD':lambda r:r['source']=='DRCKnowledgeTeam/DRCD' and 'DRCD_test.json' in r['source_file'],
        'rajpurkar/squad':lambda r:r['source']=='rajpurkar/squad' and '/validation-' in r['source_file']}
    test=[];blocked=set()
    for i,source in enumerate(SOURCES):
        pool=unique_inputs(r for r in rows if final[source](r) and r['group_id'] not in blocked)
        selected,groups=reserve_articles(pool,1000 if source=='rajpurkar/squad' else 500,4200+i,50 if source=='rajpurkar/squad' else 20)
        test.extend(selected);blocked|=groups
    validation=[];val_groups=set()
    for i,source in enumerate(SOURCES):
        pool=unique_inputs(r for r in rows if r['source']==source and r['source_role']=='official_train' and r['group_id'] not in blocked|val_groups)
        selected,groups=reserve_articles(pool,1000 if source=='rajpurkar/squad' else 500,4300+i,50 if source=='rajpurkar/squad' else 20)
        validation.extend(selected);val_groups|=groups
    train_pool=[];seen=set()
    for row in rows:
        if row['source_role']!='official_train' or row['group_id'] in blocked|val_groups:continue
        key=digest([row['context'],row['question']])
        if key in seen:rejected['duplicate_training_input']+=1;continue
        seen.add(key);train_pool.append(row)
    write_json(work/'candidate_statistics.json',{'source_records':dict(raw_counts),'rejections':dict(rejected),
        'training_pool':dict(Counter(r['language'] for r in train_pool)),
        'test_articles':len(blocked),'validation_articles':len(val_groups)})
    training,balance=token_balanced(train_pool)
    if min(Counter(r['source'] for r in training).values())<1000:raise ValueError('A declared source has insufficient coverage')
    out.mkdir(parents=True);shutil.copyfile(tokenizer_path,out/'tokenizer.json')
    selections={'train':training,'validation':validation,'test':test,'pilot':training[:2000]};specs={};stats={}
    for split,selected in selections.items():
        selected=list(selected);random.Random('reading:'+split).shuffle(selected)
        languages=Counter();sources=Counter();answer_tokens=Counter();qtypes=Counter();lengths=Counter();groups=set()
        def output_rows():
            for row in selected:
                record={'schema_version':1,'split':split,'id':'reading_'+digest([row['context'],row['question']])[:24],
                    'system':'',**row,'reward_type':'human_alias_exact','human_annotated':True,
                    'normalized_text_sha256':canonical_hash([row['language'],'',row['prompt'],row['response']])}
                languages[row['language']]+=1;sources[row['source']]+=1;answer_tokens[row['language']]+=row['answer_token_count']
                qtypes[row['language']+'/'+row['question_type']]+=1;lengths[row['language']+'/'+row['answer_length_bucket']]+=1;groups.add(row['group_id'])
                yield record
        file=out/(split+'.jsonl');write_jsonl(file,output_rows())
        specs[split]={'path':file.name,'examples':len(selected),'sha256':sha256_file(file)}
        stats[split]={'records':len(selected),'articles':len(groups),'languages':dict(languages),'sources':dict(sources),
            'answer_tokens':dict(answer_tokens),'zh_answer_token_fraction':answer_tokens['zh']/sum(answer_tokens.values()),
            'question_types':dict(qtypes),'answer_length_buckets':dict(lengths),
            'max_prompt_tokens':max(r['prompt_token_count'] for r in selected),
            'max_answer_tokens_with_eos':max(r['answer_token_count'] for r in selected)}
    manifest={'schema_version':1,'kind':'sft_tokenized_conversations','task':'extractive_reading_qa','fixture':False,
        'status':'awaiting_independent_audit','created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'splits':specs,'tokenizer':{'path':'tokenizer.json','sha256':sha256_file(tokenizer_path),'vocab_size':32000,'pad_id':0,'unk_id':1,'eos_id':2},
        'tokenization':PROTOCOL,'chinese_normalization':{'mode':'t2s','package':'OpenCC','package_version':'1.1.9','algorithm':'bounded_fixed_point',
            'max_passes':8,'answer_alignment':'verify original offsets, then rebuild converted-context occurrences; reject missing matches'},
        'origin':{'source_lock_sha256':sha256_file(cache/'sources.lock.json'),'sources':lock['sources'],'human_answers':True,
            'drcd_original_language':'traditional Chinese; context/question/answers jointly converted to simplified Chinese'},
        'quality_exclusions':QUALITY_EXCLUSIONS,
        'statistics':stats,'language_balance':balance,'training_policy':{'epochs':1,'stop_epochs':None},
        'generation':{'max_new_tokens':64,'max_prompt_tokens':960,'temperature':1.0},
        'evaluation':{'test_records':2000,'test_from_official_evaluation_only':True,'not_used_for_tuning':True,
            'selection':'source-stratified article groups with shuffled questions; no model scores',
            'primary_metric':'strict human-alias exact match','secondary_metric':'declared word/character F1, not official leaderboard F1'},
        'grouping':'connected components of normalized article title and exact normalized passage; no cross-split components',
        'limits':['Answers are correct relative to the supplied historical passage, not current-world fact verification',
            'Human labels may be incomplete; reward accepts only published aliases',
            'Exact article/passage grouping does not prove complete cross-language semantic or pretraining decontamination'],
        'training_started':False}
    manifest['data_identity']=canonical_hash({k:manifest[k] for k in IDENTITY_FIELDS})
    write_json(out/'manifest.json',manifest)
    write_json(work/'build_report.json',{'manifest':str(out/'manifest.json'),'statistics':stats,'balance':balance})
    print(json.dumps({'status':'candidate_built','manifest':str(out/'manifest.json'),'statistics':stats,'balance':balance},ensure_ascii=False),flush=True)
    return out/'manifest.json'
