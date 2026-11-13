"""重读官方标注、转换后的答案、完整上下文及全部监督Token，独立验收发布。"""
from collections import Counter
from importlib import metadata
from pathlib import Path
import json,re

from llm_lab.common.io import canonical_hash,read_json,sha256_file,write_json
from .source import source_rows,fixed_point,text_key,digest
from .verifier import reward


def audit(path,cache,publish=False,review=None):
    from opencc import OpenCC
    from tokenizers import Tokenizer
    path=Path(path);directory=path.parent;cache=Path(cache);m=read_json(path)
    if (directory/'DO_NOT_TRAIN.json').exists():raise ValueError('Quarantined data')
    identity_fields=('schema_version','kind','splits','tokenizer','tokenization','chinese_normalization')
    if canonical_hash({k:m[k] for k in identity_fields})!=m['data_identity']:raise ValueError('Data identity changed')
    if metadata.version('OpenCC')!=m['chinese_normalization']['package_version']:raise ValueError('Wrong Chinese normalizer version')
    if sha256_file(cache/'sources.lock.json')!=m['origin']['source_lock_sha256']:raise ValueError('Source lock changed')
    lock=read_json(cache/'sources.lock.json');converter=OpenCC('t2s');originals={}
    for spec in lock['sources']:
        if sha256_file(cache/spec['local_path'])!=spec['sha256']:raise ValueError('Official source checksum changed')
        if spec['role']=='documentation':continue
        for raw in source_rows(cache,spec):
            key=(raw['source_file'],raw['source_row'])
            if key in originals:raise ValueError('Duplicate source row key')
            originals[key]=raw
    tokenpath=directory/m['tokenizer']['path']
    if sha256_file(tokenpath)!=m['tokenizer']['sha256']:raise ValueError('Tokenizer identity changed')
    tokenizer=Tokenizer.from_file(str(tokenpath));groups={};articles={};passages={};inputs={};summaries={};trainrows={}
    for split in ['train','validation','test','pilot']:
        spec=m['splits'][split];file=directory/spec['path']
        if sha256_file(file)!=spec['sha256']:raise ValueError('Split checksum changed')
        ids=set();gset=set();aset=set();pset=set();iset=set();languages=Counter();tokens=Counter();sources=Counter();types=Counter();buckets=Counter()
        for line in file.open(encoding='utf-8'):
            row=json.loads(line);raw=originals[(row['source_file'],row['source_row'])];lang=row['language']
            if row['schema_version']!=1 or row['split']!=split or row['id'] in ids:raise ValueError('Invalid or duplicate record')
            if row['source_id']!=raw['source_id'] or row['source_record_sha256']!=raw['source_record_sha256']:raise ValueError('Raw human annotation identity changed')
            if any(row['source']==issue['source'] and row['title']==issue['title'] for issue in m.get('quality_exclusions',[])):
                raise ValueError('Known source quality issue entered a released split')
            annotations=raw['answers']
            for answer in annotations:
                start=answer['answer_start'];text=answer['text']
                if type(start) is not int or start<0 or raw['context'][start:start+len(text)]!=text:raise ValueError('Original human answer offset invalid')
            context=raw['context'];question=raw['question'];title=raw['title'];aliases=[a['text'] for a in annotations]
            if lang=='zh':
                context=fixed_point(context,converter);question=fixed_point(question,converter);title=fixed_point(title,converter)
                aliases=[fixed_point(answer,converter).strip() for answer in aliases]
            elif lang=='en':aliases=[answer.strip() for answer in aliases]
            else:raise ValueError('Unsupported language')
            aliases=list(dict.fromkeys(aliases))
            if row['context']!=context or row['question']!=question or row['title']!=title or row['answers']!=aliases:raise ValueError('Context, question or gold aliases were rewritten')
            if row['response']!=aliases[0] or not row['human_annotated'] or row['reward_type']!='human_alias_exact':raise ValueError('Target not bound to human labels')
            for answer in aliases:
                found=[match.start() for match in re.finditer(re.escape(answer),context)]
                if not found or row['answer_occurrences'][answer]!=found:raise ValueError('Converted answer span mismatch')
                if reward(answer,aliases,lang)!=1:raise ValueError('Gold answer fails reward')
            if reward(context,aliases,lang)!=0:raise ValueError('Copying whole context can obtain reward')
            expected_prompt=('请根据阅读材料回答问题，只输出原文中的答案，不要解释。\n\n阅读材料：\n'+context+'\n\n问题：'+question if lang=='zh' else
                'Answer the question using the passage. Output only the answer span, without explanation.\n\nPassage:\n'+context+'\n\nQuestion: '+question)
            if row['prompt']!=expected_prompt:raise ValueError('Prompt does not include complete original context')
            p=tokenizer.encode('User:\n'+expected_prompt+'\n\nAssistant:\n',add_special_tokens=False).ids
            a=tokenizer.encode(aliases[0],add_special_tokens=False).ids+[2]
            if len(p)>960 or len(a)>64 or len(p+a)>1024:raise ValueError('Generation or training token budget exceeded')
            if row['input_ids']!=p+a or row['loss_mask']!=[0]*len(p)+[1]*len(a):raise ValueError('Retokenization or answer/EOS mask mismatch')
            if any(t in {0,1,2} for t in (p+a)[:-1]) or row['prompt_token_count']!=len(p) or row['answer_token_count']!=len(a):raise ValueError('Special tokens or token counts invalid')
            if row['normalized_text_sha256']!=canonical_hash([lang,'',expected_prompt,aliases[0]]):raise ValueError('Normalized text identity invalid')
            article=text_key(title.replace('_',' ')) if title.strip() else digest([raw['source'],raw['article_id']])
            if row['article_key']!=article or row['passage_key']!=text_key(context):raise ValueError('Article/passage identity invalid')
            if lang=='zh' and any(converter.convert(text)!=text for text in [context,question,*aliases]):raise ValueError('Chinese is not simplified fixed point')
            allowed_test={
                'ymcui/cmrc2018':'ymcui__cmrc2018/squad-style-data/cmrc2018_dev.json',
                'DRCKnowledgeTeam/DRCD':'DRCKnowledgeTeam__DRCD/DRCD_test.json',
                'rajpurkar/squad':'rajpurkar__squad/plain_text/validation-00000-of-00001.parquet'}
            if split=='test' and (raw['source_role']!='official_evaluation_only' or raw['source_file']!=allowed_test[raw['source']]):
                raise ValueError('Final test came from an undeclared official source partition')
            if split in {'train','validation','pilot'} and raw['source_role']!='official_train':raise ValueError('Official evaluation data entered training/development')
            normalized={k:v for k,v in row.items() if k!='split'}
            if split=='train':trainrows[row['id']]=canonical_hash(normalized)
            if split=='pilot' and trainrows.get(row['id'])!=canonical_hash(normalized):raise ValueError('Pilot not an exact train subset')
            ids.add(row['id']);gset.add(row['group_id']);aset.add(article);pset.add(row['passage_key']);iset.add(digest([context,question]))
            languages[lang]+=1;tokens[lang]+=len(a);sources[row['source']]+=1;types[lang+'/'+row['question_type']]+=1;buckets[lang+'/'+row['answer_length_bucket']]+=1
        if len(ids)!=spec['examples']:raise ValueError('Record count mismatch')
        declared=m['statistics'][split]
        if dict(languages)!=declared['languages'] or dict(tokens)!=declared['answer_tokens'] or dict(sources)!=declared['sources']:raise ValueError('Declared language/source/token statistics mismatch')
        if set(languages)!={'zh','en'} or min(languages.values())<1:raise ValueError('A language is absent')
        groups[split]=gset;articles[split]=aset;passages[split]=pset;inputs[split]=iset
        summaries[split]={'sha256':spec['sha256'],'examples':len(ids),'language_examples':dict(languages),'answer_tokens':dict(tokens),
            'articles':len(gset),'sources':dict(sources),'zh_answer_token_fraction':tokens['zh']/sum(tokens.values())}
        print(json.dumps({'audit_split':split,'records':len(ids),'articles':len(gset),'status':'passed'}),flush=True)
    for i,a in enumerate(['train','validation','test']):
        for b in ['train','validation','test'][i+1:]:
            if any(mapping[a]&mapping[b] for mapping in [groups,articles,passages,inputs]):raise ValueError('Cross-split article/passage/question leakage')
    if summaries['train']['examples']!=50000 or abs(summaries['train']['zh_answer_token_fraction']-.5)>.005:raise ValueError('Main size/token-balance requirement failed')
    normalizer=metadata.distribution('OpenCC');resources={}
    for name in normalizer.files or []:
        if str(name).endswith(('.ocd2','.json')) and 'opencc' in str(name).lower():
            file=Path(normalizer.locate_file(name));resources[str(name)]=sha256_file(file)
    result={'schema_version':1,'status':'passed','fixture':False,'structurally_ready':True,'data_identity':m['data_identity'],
        'tokenizer_sha256':m['tokenizer']['sha256'],'splits':summaries,
        'checks':{key:True for key in ['file_hashes','retokenization','answer_mask_and_eos','normalized_text_identity',
            'declared_group_and_exact_split_isolation','pilot_subset','language_token_balance','explicit_quality_rules',
            'chinese_fixed_point','original_human_offsets','converted_human_answer_occurrences','complete_context',
            'official_test_excluded_from_training','whole_passage_reward_zero']},
        'normalizer_resources':resources,'reward_code_sha256':sha256_file(Path(__file__).parent/'verifier.py'),
        'recipe_files':{name:sha256_file(Path(__file__).parent/name) for name in ['source.py','build.py','verifier.py','audit.py']},
        'scope':'Human source labels, faithful conversion, exact span/encoding/isolation; not independent world-fact verification'}
    if publish:
        if review is None:raise ValueError('Representative source spotcheck receipt required')
        receipt=read_json(review)
        if receipt.get('status')!='passed_spotcheck' or receipt.get('data_identity')!=m['data_identity'] or receipt.get('train_sha256')!=m['splits']['train']['sha256']:raise ValueError('Spotcheck is not bound to this candidate')
        if m.get('status')=='published':raise ValueError('Never rewrite an existing publication')
        write_json(directory/'acceptance.json',result);(directory/'quality_review.json').write_bytes(Path(review).read_bytes())
        m['acceptance']={'status':'passed','report':'acceptance.json','sha256':sha256_file(directory/'acceptance.json')}
        m['quality_review']={'path':'quality_review.json','sha256':sha256_file(directory/'quality_review.json')}
        m['reward_protocol']={'name':'strict_human_alias_exact_v1','code_sha256':result['reward_code_sha256'],'primary_reward':'binary exact alias match; F1 diagnostic only'}
        m['status']='published';write_json(path,m)
    return result
