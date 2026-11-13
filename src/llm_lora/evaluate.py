"""配置锁定后对Full和LoRA执行全量最终测试；开发评测不打开测试题。"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time

import torch
from llm_lab.common.budget import BudgetGuard
from llm_lab.common.io import read_json,sha256_file,write_json,write_jsonl
from llm_lora.config import read_config,runtime_fingerprint
from llm_lora.data import AuxiliaryCorpus,protocol
from llm_lora.inference import load_model
from llm_lora.metrics import supported_constraints,summarize_constraints


def final_evaluation(selection_path):
    # selection仅在三个候选完成相同筛选预算后写入，禁止测试反馈进入选择。
    selection_path=Path(selection_path);selection=read_json(selection_path)
    if selection.get('status')!='configuration_locked_before_final_test':raise ValueError('Configuration was not locked')
    cfg=read_config(selection['config']);root=Path(cfg['project_root']);output=Path(selection['final_output_dir'])
    if output.exists() or not output.resolve().is_relative_to(root/'runs/assistant/inference'):raise ValueError('Final output exists or escaped')
    if runtime_fingerprint()!=cfg['runtime_sha256']:raise ValueError('Evaluation source changed')
    summary=read_json(Path(cfg['output_dir'])/'summary.json')
    if summary['status']!='completed' or summary['step']!=selection['final_step']:raise ValueError('Selected training did not complete')
    spec=protocol(cfg)['final_test'];corpus=AuxiliaryCorpus(spec['manifest'],'test',spec['manifest_sha256'])
    if len(corpus)!=1000 or corpus.identity['split_sha256']!=spec['test_sha256']:raise ValueError('Final test protocol changed')
    if any(row['prompt_token_count']+256>1024 for row in corpus.rows):raise ValueError('Final generation cannot preserve full prompt')
    import train_sft as locking
    import evaluate_sft_models as metrics
    import chat_sft
    from llm_lab.tokenization.bpe import load_tokenizer
    tokenizer=load_tokenizer(corpus.tokenizer_path)
    eval_cfg={'train':{'device':'cuda','dtype':'bf16','eval_micro_batch_size':8}}
    candidates=[{'key':'full_100k','checkpoint':selection['full_checkpoint'],'sha256':selection['full_checkpoint_sha256']},
                {'key':'lora_100k','checkpoint':str(Path(cfg['output_dir'])/'best.pt')}]
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    output.mkdir(parents=True,exist_ok=False);write_json(output/'selection.json',selection)
    rows_by_model={};reports=[];started=time.monotonic()
    with locking.training_lock(root):
        locking.gpu_idle()
        guard=BudgetGuard({'hourly_rate_cny':3.03,'run_limit_cny':5,'project_limit_cny':500,'reserve_cny':100,'max_runtime_seconds':5400},
            str(output.resolve()),root/'runs/budget_ledger.json')
        try:
            for item in candidates:
                digest=sha256_file(item['checkpoint'])
                if item.get('sha256') and digest!=item['sha256']:raise ValueError('Full baseline changed')
                if item['key']=='lora_100k':model,_,metadata=load_model(item['checkpoint'],'cuda')
                else:model,_,metadata=chat_sft.load_model(item['checkpoint'],'cuda',threads=8)
                loss,losses=metrics.evaluate_loss(model,corpus,eval_cfg,guard)
                directory=output/item['key'];directory.mkdir();write_jsonl(directory/'test_rows.jsonl',losses)
                predictions=[]
                # 全部1000题生成；按长度组batch，不以抽样替代完整测试。
                order=sorted(range(len(corpus.rows)),key=lambda index:corpus.rows[index]['prompt_token_count'])
                for offset in range(0,len(order),8):
                    indices=order[offset:offset+8];rows=[corpus.rows[index] for index in indices]
                    prefixes=[row['input_ids'][:row['prompt_token_count']] for row in rows]
                    generated=metrics.greedy_batch(model,prefixes,256,device='cuda',dtype='bf16')
                    for row,value in zip(rows,generated):
                        prediction=metrics.generation_metrics(row,value,tokenizer)
                        prediction['constraint_checks']=supported_constraints(row['system']+'\n'+row['prompt'],prediction['answer'])
                        predictions.append(prediction)
                    write_json(output/'state.json',{'status':'generating','model':item['key'],'generated':len(predictions),'expected':1000})
                    if guard.should_stop():raise TimeoutError('Final LoRA comparison budget reached')
                if len(predictions)!=1000 or len({row['id'] for row in predictions})!=1000:raise ValueError('Final generation incomplete')
                write_jsonl(directory/'generations.jsonl',predictions)
                result={**item,'checkpoint_sha256':digest,'checkpoint_metadata':metadata,'test':loss,
                    'generation':metrics.summarize_generations(predictions),'supported_constraints':summarize_constraints(predictions),
                    'final_generated_examples':len(predictions)}
                write_json(directory/'metrics.json',result);reports.append(result);rows_by_model[item['key']]=losses
                del model;gc.collect();torch.cuda.empty_cache()
            report={'status':'completed','selection_sha256':sha256_file(selection_path),'test_identity':corpus.identity,'results':reports,
                'delta_lora_minus_full':reports[1]['test']['macro_loss']-reports[0]['test']['macro_loss'],
                'paired_loss_interval':metrics.paired_interval(rows_by_model['full_100k'],rows_by_model['lora_100k']),
                'final_test_was_not_used_for_selection':True,'generated_examples_per_model':1000,
                'not_general_task_accuracy':True,'semantic_gold':False,'elapsed_seconds':time.monotonic()-started,'budget':guard.update()}
            write_json(output/'comparison.json',report);write_json(output/'state.json',{'status':'completed'});return report
        finally:guard.update()
