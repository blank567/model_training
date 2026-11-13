"""把冻结的比较组和偏好数据转成统一训练样本，最终测试不进入训练接口。"""
from pathlib import Path
import hashlib,json,random
from llm_lab.common.io import canonical_hash,read_json,sha256_file
from llm_reading_independent.comparison import ComparisonRLCorpus
from llm_reading_data.verifier import reward


class PreferenceCorpus:
    def __init__(self,manifest_path,arm,tokenizer,initial_sha256):
        path=Path(manifest_path);m=read_json(path)
        if m.get('status')!='published' or m.get('kind')!='reading_policy_preferences':raise ValueError('Accepted preferences required')
        if m['initial_checkpoint_sha256']!=initial_sha256:raise ValueError('Negative policy differs from the selected SFT')
        if m['source_manifest_sha256']!=arm.manifest['source_manifest_sha256']:raise ValueError('Preferences belong to different source prompts')
        file=path.parent/m['records']['path']
        if not file.resolve().is_relative_to(path.parent.resolve()):raise ValueError('Preference records escape publication')
        if sha256_file(file)!=m['records']['sha256']:raise ValueError('Preference records changed')
        if not arm.source.fixture:
            receipt=path.parent/m['acceptance']['path']
            if not receipt.resolve().is_relative_to(path.parent.resolve()) or sha256_file(receipt)!=m['acceptance']['sha256']:
                raise ValueError('Preference acceptance changed')
            accepted=read_json(receipt)
            if accepted.get('status')!='passed' or accepted.get('records_sha256')!=m['records']['sha256']:
                raise ValueError('Preference data have not passed acceptance')
            required=['original_human_gold','initial_sft_generated_negatives','negative_eos_and_token_checks',
                      'negative_reward_zero','ambiguous_aliases_excluded','source_and_prefix_bound']
            if not all(accepted.get('checks',{}).get(k) is True for k in required):raise ValueError('Incomplete preference checks')
        positions={arm[i]['id']:i for i in range(len(arm))};self.rows=[];seen=set()
        for line in file.open(encoding='utf-8'):
            row=json.loads(line)
            if row['id'] not in positions:continue
            if row['id'] in seen:raise ValueError('Duplicate preference question')
            source=arm[positions[row['id']]];chosen=tokenizer.encode(source['chosen'],add_special_tokens=False).ids+[2]
            rejected=row['rejected_ids'];text=tokenizer.decode(rejected[:-1],skip_special_tokens=False)
            if row['prefix_sha256']!=canonical_hash(source['prefix_ids']) or row['chosen_ids']!=chosen:raise ValueError('Preference prompt/gold mismatch')
            if not 2<=len(rejected)<=64 or rejected[-1]!=2 or any(t in {0,1,2} for t in rejected[:-1]):raise ValueError('Invalid rejected completion')
            if reward(text,source['reference_answers'],source['language'])!=0:raise ValueError('Correct answer cannot be rejected')
            self.rows.append({**source,'chosen_ids':chosen,'rejected_ids':rejected});seen.add(row['id'])
        if not self.rows:raise ValueError('No valid preferences for this arm')
        self.identity={'manifest_sha256':sha256_file(path),'examples':len(self.rows),'source_arm_identity':arm.manifest['data_identity']}

    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return self.rows[index]


def order_and_batches(corpus,batch_size,seed):
    """按样本ID随机排序：旧2.5万是旧5万子集，公共样本相对顺序保持一致。"""
    indices=sorted(range(len(corpus)),key=lambda i:hashlib.sha256((str(seed)+'/'+corpus[i]['id']).encode()).digest())
    return [indices[i:i+batch_size] for i in range(0,len(indices),batch_size)]


def validation_subset(corpus,size=256,seed=4201):
    indices=[];per_language=size//2
    for language in ['zh','en']:
        values=list(corpus.language_indices[language]);random.Random(str(seed)+language).shuffle(values);indices.extend(values[:per_language])
    return indices
