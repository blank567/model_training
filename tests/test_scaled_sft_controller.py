"""扩大数据时只改数据身份，失败/中断/不足步数都不能进入最终测试。"""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_scaled_sft import scaled_template,training_complete


def test_scaling_preserves_base_and_every_training_hyperparameter():
    source={'project_root':'/project','base_checkpoint':'base.pt','base_checkpoint_sha256':'a'*64,
        'data_manifest':'old.json','data_manifest_sha256':'b'*64,'data_identity':'c'*64,
        'train':{'epochs':3,'stop_epochs':2,'learning_rate':5e-5,'micro_batch_size':8,
                 'gradient_accumulation_steps':4,'wandb_run_name':'old'}}
    selected={'source_config':'config.json','checkpoint':'best.pt','checkpoint_sha256':'d'*64}
    template=scaled_template(source,'new.json',selected,'comparison.json')
    assert template['base_checkpoint']==source['base_checkpoint']
    assert template['train']=={**source['train'],'wandb_run_name':None}
    assert template['data_manifest']=='new.json' and 'data_identity' not in template
    assert source['data_manifest']=='old.json' and source['train']['wandb_run_name']=='old'


@pytest.mark.parametrize('code,status,steps,expected',[(0,'completed',6250,True),
    (0,'completed',6249,False),(1,'completed',6250,False),(0,'interrupted',6250,False)])
def test_only_complete_expected_training_enters_final_test(code,status,steps,expected):
    assert training_complete(code,{'status':status,'step':steps},6250)==expected
