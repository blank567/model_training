"""扩充训练完成后一次性比较旧胜出模型与新模型的预留测试表现。"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch
from llm_lab.common.budget import BudgetGuard
from llm_lab.common.io import read_json, sha256_file, write_json, write_jsonl
from llm_lab.config import ModelConfig
from llm_lab.models import Decoder
from llm_lab.tokenization.bpe import load_tokenizer
from llm_sft.config import runtime_fingerprint
from llm_sft.data import SFTCorpus
import evaluate_sft_models as comparison
import train_sft as launcher


class AuxiliaryCorpus(SFTCorpus):
    """显式识别辅助评测协议；复用同一错位/Mask实现，不伪装成训练manifest。"""
    def __init__(self, path, expected_sha256, *, allow_fixture=False):
        self.manifest_path=Path(path).resolve();self.directory=self.manifest_path.parent
        if sha256_file(self.manifest_path)!=expected_sha256:raise ValueError('Auxiliary manifest changed')
        self.manifest=read_json(self.manifest_path);manifest=self.manifest
        if manifest.get('kind')!='sft_auxiliary_evaluation':raise ValueError('Not auxiliary evaluation')
        self.fixture=manifest.get('fixture') is True
        if self.fixture and not allow_fixture:raise ValueError('Fixture disabled for final GPU evaluation')
        acceptance=read_json(self.directory/'acceptance.json')
        if acceptance.get('status')!='passed' or not acceptance.get('training_review_and_split_isolation'):
            raise ValueError('Auxiliary isolation acceptance missing')
        self.split='test';spec=manifest['files']['test'];path=self.directory/spec['path']
        if path.is_symlink() or not path.resolve().is_relative_to(self.directory) or sha256_file(path)!=spec['sha256']:
            raise ValueError('Auxiliary test changed or escaped directory')
        token=manifest['tokenizer'];self.tokenizer_path=self.directory/token['path']
        if self.tokenizer_path.is_symlink() or not self.tokenizer_path.resolve().is_relative_to(self.directory):
            raise ValueError('Auxiliary tokenizer path escaped')
        if sha256_file(self.tokenizer_path)!=token['sha256']:raise ValueError('Auxiliary tokenizer changed')
        self.pad_id=token['pad_id'];self.unk_id=token['unk_id'];self.eos_id=token['eos_id'];self.vocab_size=token['vocab_size']
        self.max_sequence_tokens=manifest['tokenization']['max_sequence_tokens']
        tokenizer=load_tokenizer(self.tokenizer_path);self.rows=[];self.language_indices={'en':[],'zh':[]};seen=set()
        for number,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
            row=json.loads(line);self._validate_row(row,number)
            if row['id'] in seen:raise ValueError('Repeated auxiliary test ID')
            seen.add(row['id'])
            template=manifest['tokenization']['prefix_template' if row['system'] else 'empty_system_prefix_template']
            prefix=tokenizer.encode(template.format(**row),add_special_tokens=False).ids
            answer=tokenizer.encode(row['response'],add_special_tokens=False).ids+[self.eos_id]
            if row['input_ids']!=prefix+answer or row['loss_mask']!=[0]*len(prefix)+[1]*len(answer):
                raise ValueError('Auxiliary text/Token/Mask boundary changed')
            self.language_indices[row['language']].append(len(self.rows));self.rows.append(row)
        if len(self.rows)!=spec['examples'] or not all(self.language_indices.values()):raise ValueError('Auxiliary count/language mismatch')
        self.identity={'manifest_sha256':expected_sha256,'split':'test','split_sha256':spec['sha256'],
            'tokenizer_sha256':token['sha256'],'examples':len(self.rows),'kind':'sft_auxiliary_evaluation'}


def run(handoff_path):
    handoff=read_json(handoff_path);root=Path(handoff['project_root']);output=Path(handoff['final_evaluation_dir'])
    if output.exists():raise FileExistsError('Final evaluation already exists')
    if not output.resolve().is_relative_to(root/'runs/assistant/inference'):raise ValueError('Wrong final evaluation scope')
    if runtime_fingerprint(root)!=handoff['runtime_sha256']:raise ValueError('Frozen source changed')
    config=launcher.frozen_config(handoff['RUN_CONFIG']);summary=read_json(Path(handoff['RUN_DIR'])/'summary.json')
    if summary.get('status')!='completed' or summary['step']!=handoff['plan']['stop_step']:
        raise ValueError('Training did not finish the frozen plan')
    commitment=handoff['auxiliary_commitment'];corpus=AuxiliaryCorpus(commitment['manifest'],commitment['manifest_sha256'])
    primary=SFTCorpus(handoff['primary_data_manifest'],'test',allow_test=True)
    if corpus.identity['split_sha256']!=commitment['test_sha256']:raise ValueError('Reserved test changed')
    output.mkdir(parents=True,exist_ok=False);write_json(output/'handoff.json',handoff)
    tokenizer=load_tokenizer(corpus.tokenizer_path);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    eval_config={'train':{'device':'cuda','dtype':'bf16','eval_micro_batch_size':8}}
    indices=comparison.generation_indices(corpus.rows,64,42,1024,256)
    candidates=[{'key':'old_selected','checkpoint':handoff['selected']['checkpoint'],
                 'sha256':handoff['selected']['checkpoint_sha256']},
                {'key':'scaled_100k','checkpoint':str(Path(handoff['RUN_DIR'])/'best.pt')}]
    results=[]
    with launcher.training_lock(root):
        launcher.gpu_idle();guard=BudgetGuard({'hourly_rate_cny':3.03,'run_limit_cny':3,'project_limit_cny':500,
            'reserve_cny':100,'max_runtime_seconds':1800},str(output.resolve()),root/'runs/budget_ledger.json')
        try:
            for candidate in candidates:
                digest=sha256_file(candidate['checkpoint'])
                if candidate.get('sha256') and candidate['sha256']!=digest:raise ValueError('Selected model changed')
                payload=torch.load(candidate['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
                if payload.get('stage')!='sft' or payload['identity']['tokenizer_sha256']!=corpus.identity['tokenizer_sha256']:
                    raise ValueError('SFT checkpoint/auxiliary tokenizer incompatible')
                spec=ModelConfig(**payload['model_config']);spec.activation_checkpointing=False
                model=Decoder(spec);model.load_state_dict(payload['model'],strict=True);step=payload['step'];del payload
                model.to('cuda').eval();test,losses=comparison.evaluate_loss(model,corpus,eval_config,guard)
                original,primary_losses=comparison.evaluate_loss(model,primary,eval_config,guard)
                predictions=[]
                for offset in range(0,len(indices),8):
                    rows=[corpus.rows[index] for index in indices[offset:offset+8]]
                    generated=comparison.greedy_batch(model,[row['input_ids'][:row['prompt_token_count']] for row in rows],
                        256,device='cuda',dtype='bf16')
                    predictions.extend(comparison.generation_metrics(row,value,tokenizer) for row,value in zip(rows,generated))
                    if guard.should_stop():raise TimeoutError('Final evaluation budget reached')
                result={**candidate,'checkpoint_sha256':digest,'checkpoint_step':step,'auxiliary_test':test,
                    'primary_test_already_used_for_selection':original,'generation':comparison.summarize_generations(predictions)}
                directory=output/candidate['key'];directory.mkdir();write_json(directory/'metrics.json',result)
                write_jsonl(directory/'test_rows.jsonl',losses);write_jsonl(directory/'primary_test_rows.jsonl',primary_losses)
                write_jsonl(directory/'generations.jsonl',predictions);results.append(result)
                del model;gc.collect();torch.cuda.empty_cache()
            report={'status':'completed','results':results,'auxiliary_test_identity':corpus.identity,
                'auxiliary_test_not_used_for_model_selection_or_training':True,'primary_test_was_selection_data':True,
                'macro_loss_delta_new_minus_old':results[1]['auxiliary_test']['macro_loss']-results[0]['auxiliary_test']['macro_loss'],
                'not_general_task_accuracy':True,'budget':guard.update(),'weights_deleted':False}
            write_json(output/'comparison.json',report);return report
        finally:guard.update()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--handoff',type=Path,required=True)
    print(json.dumps(run(parser.parse_args().handoff),ensure_ascii=False),flush=True)
