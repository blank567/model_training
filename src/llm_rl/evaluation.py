"""固定开发子集的生成评测；完整最终2000题只在独立评测阶段读取。"""
from collections import defaultdict
import torch
from llm_reading_data.verifier import reward,answer_f1
from .generation import generate


@torch.no_grad()
def evaluate(model,tokenizer,corpus,indices,device,batch_size=16):
    groups=defaultdict(list);outputs=[];model.eval()
    for start in range(0,len(indices),batch_size):
        rows=[corpus.rows[i] for i in indices[start:start+batch_size]]
        prompts=[r['input_ids'][:r['prompt_token_count']] for r in rows]
        generated=generate(model,tokenizer,prompts,device,max_new_tokens=64,sample=False)
        for row,text,ended in zip(rows,generated['texts'],generated['eos']):
            item={'id':row['id'],'language':row['language'],'exact_match':reward(text,row['answers'],row['language']),
                  'f1':answer_f1(text,row['answers'],row['language']),'eos':bool(ended)}
            groups[row['language']].append(item);outputs.append(item)
    by_language={language:{'examples':len(rows),'exact_match':sum(r['exact_match'] for r in rows)/len(rows),
        'f1':sum(r['f1'] for r in rows)/len(rows)} for language,rows in groups.items()}
    return {'examples':len(outputs),'by_language':by_language,
       'macro_exact_match':sum(r['exact_match'] for r in by_language.values())/len(by_language),
       'macro_f1':sum(r['f1'] for r in by_language.values())/len(by_language),
       'eos_rate':sum(r['eos'] for r in outputs)/len(outputs),'final_test_used':False}
