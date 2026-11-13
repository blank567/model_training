"""从已隔离候选池修订新RL数据；不改写旧发布、旧实验或任何模型参数。"""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import heapq,json,random,re,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json,write_jsonl
from llm_reading_data.build import IDENTITY_FIELDS,token_balanced
from llm_reading_independent.quality import reason

CLINICAL=re.compile(r'免费提问|赞赏|擅长[:：]|妊娠|流产|怀孕|产后|复查|避孕|生化|月经|孕妇|妇科|白带|症状|肿瘤|癌症|胃炎|感冒|发烧|腹泻|痔疮|牙疼|用药|药物|服用|输液|消炎|病情|血糖|胰岛|糖尿病|手术|疼痛|后遗症|补钙|补铁')
SUBJECTIVE=re.compile(r'合适|合理|够不够|正不正常|怎么样|怎么办|值得|有用吗|有必要|推荐|好吗|好看|好吃|好玩|好用')


def extra_reason(row):
    if row['language']=='zh':
        if CLINICAL.search(row['question']+' '+row['context']):return 'clinical_advice_or_provider_aggregation'
        if SUBJECTIVE.search(row['question']):return 'subjective_recommendation'
    return None


def select_balanced(rows,total=25000):
    try:return token_balanced(rows,total,20261101)
    except ValueError:pass
    # 中文清洁候选不足时保留全部中文，从英文随机子集中交换答案长度，
    # 固定总题数并平衡监督Token；不依赖模型得分、不复制样本、不修改答案。
    zh=[r for r in rows if r['language']=='zh'];en=[r for r in rows if r['language']=='en']
    random.Random(20261101).shuffle(en);n=total-len(zh)
    if not 0<n<=len(en):raise ValueError('Insufficient independent clean candidates')
    selected=en[:n];remainder=en[n:];target=sum(r['answer_token_count'] for r in zh)
    current=sum(r['answer_token_count'] for r in selected);swaps=0
    # 交换选中长答案与未选中短答案，达到预先声明的50%±0.5%目标。
    long=[(-r['answer_token_count'],i,r) for i,r in enumerate(selected)];heapq.heapify(long)
    short=[(r['answer_token_count'],i,r) for i,r in enumerate(remainder)];heapq.heapify(short)
    while abs(target/(target+current)-.5)>.005 and long and short:
        neg,position,old=heapq.heappop(long);size,_,new=heapq.heappop(short)
        if current<=target or old['answer_token_count']<=size:raise ValueError('Cannot meet balanced count without replacement')
        current+=size-old['answer_token_count'];selected[position]=new;swaps+=1
    if abs(target/(target+current)-.5)>.005:raise ValueError('Insufficient balance capacity')
    chosen=zh+selected;random.Random(20261101).shuffle(chosen)
    return chosen,{'zh_examples':len(zh),'en_examples':n,'zh_answer_tokens':target,'en_answer_tokens':current,
        'zh_answer_token_fraction':target/(target+current),'tolerance':.005,'english_length_exchanges':swaps,
        'selection_note':'Answer length conditioned selection, never selected by model reward or evaluation scores.'}


def build():
    base=ROOT/'artifacts/posttrain/reading_rl_independent_25k_20261101_v4'
    out=ROOT/'artifacts/posttrain/reading_rl_independent_25k_20261102_v5'
    work=ROOT/'runs/assistant/data_prepare/reading_rl_refine_20261102';work.mkdir(parents=True,exist_ok=True)
    pool=ROOT/'runs/assistant/data_prepare/reading_rl_independent_20261101/curated_25k/eligible_pool.jsonl'
    if out.exists():raise ValueError('Never overwrite an existing candidate')
    kept=[];rejections=Counter()
    for line in pool.open(encoding='utf-8'):
        row=json.loads(line);why=reason(row) or extra_reason(row)
        if why:rejections[why]+=1
        else:kept.append(row)
    chosen,balance=select_balanced(kept);out.mkdir();m=deepcopy(read_json(base/'manifest.json'))
    for key in ['acceptance','quality_review','reward_protocol']:m.pop(key,None)
    for name in ['tokenizer.json','exclusion_reference.json','validation.jsonl','test.jsonl']:shutil.copyfile(base/name,out/name)
    for split,rows in [('train',chosen),('pilot',chosen[:2000])]:
        rows=[{**r,'split':split} for r in rows];file=out/(split+'.jsonl');write_jsonl(file,rows)
        m['splits'][split]={'path':file.name,'sha256':sha256_file(file),'examples':len(rows)}
        languages=Counter(r['language'] for r in rows);tokens=Counter()
        for row in rows:tokens[row['language']]+=row['answer_token_count']
        m['statistics'][split]={'records':len(rows),'languages':dict(languages),'answer_tokens':dict(tokens),
            'sources':dict(Counter(r['source'] for r in rows)),'zh_answer_token_fraction':tokens['zh']/sum(tokens.values())}
    m.update(status='awaiting_independent_audit',language_balance=balance,
        quality_refinement={'recipe_sha256':sha256_file(__file__),'rejections':dict(rejections),'pool_sha256':sha256_file(pool),
            'reason':'Additional QA review found clinical question/answer mismatch; exclude clinical advice and subjective recommendations.'})
    m['data_identity']=canonical_hash({k:m[k] for k in IDENTITY_FIELDS});write_json(out/'manifest.json',m)
    write_json(work/'build.json',{'manifest':str(out/'manifest.json'),'balance':balance,'rejections':dict(rejections)})
    # 生成确定性代表抽样以及全部新增临床过滤边界样本，随后由助手阅读再发布。
    samples=[]
    for language in ['zh','en']:
        for bucket in ['short','medium','long']:
            bucket_rows=sorted((r for r in chosen if r['language']==language and r['answer_length_bucket']==bucket),key=lambda r:r['id'])
            samples.extend(bucket_rows[:3])
    write_json(work/'review_samples.json',[{k:r[k] for k in ['id','source_id','source','language','question','response','context']} for r in samples])
    print(json.dumps({'candidate':str(out/'manifest.json'),'balance':balance,'rejections':dict(rejections),'review_samples':len(samples)}),flush=True)
if __name__=='__main__':build()
