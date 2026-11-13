"""抽审发布必须保持父身份、冻结测试文件和pilot子集约束。"""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_sft_data import atomic_json,canonical,sha256,write_jsonl
from publish_sft_curated import publish


@pytest.fixture
def publication(tmp_path):
    parent=tmp_path/'parent';parent.mkdir()
    (parent/'tokenizer.json').write_text('{}',encoding='utf-8')
    (parent/'build_config.json').write_text('{}',encoding='utf-8')
    def row(index,language,split):
        return {'id':split+str(index),'split':split,'language':language,'task':'label','system':'','prompt':'Prompt '+str(index),
                'response':'answer','input_context':'','input_ids':[3,4,2],'loss_mask':[0,1,1],
                'prompt_token_count':1,'answer_token_count':2,'group_id':split+str(index),
                'source':{'dataset':'fixture','row_index':index}}
    data={'train':[row(i,'zh' if i%2==0 else 'en','train') for i in range(2002)],
          'validation':[row(3000,'zh','validation'),row(3001,'en','validation')],
          'test':[row(4000,'zh','test'),row(4001,'en','test')]}
    manifest={'schema_version':1,'kind':'sft_tokenized_conversations','splits':{},
              'tokenizer':{'path':'tokenizer.json','sha256':sha256(parent/'tokenizer.json')},'tokenization':{},
              'chinese_normalization':{},'quality':{},'acceptance':{'status':'passed'}}
    for split,rows in data.items():
        write_jsonl(parent/(split+'.jsonl'),rows)
        manifest['splits'][split]={'path':split+'.jsonl','sha256':sha256(parent/(split+'.jsonl')),'examples':len(rows)}
    manifest['data_identity']=canonical({k:manifest[k] for k in ('schema_version','kind','splits','tokenizer','tokenization','chinese_normalization')})
    atomic_json(parent/'manifest.json',manifest)
    write_jsonl(parent/'review_sample.jsonl',data['train'][:1])
    settings={'project_root':'.','parent_dataset':'parent','parent_data_identity':manifest['data_identity'],'output_dir':'published',
              'reviewed_examples':[{'id':'train0'}],'exclusions':[{'dataset':'fixture','row_index':0,'reason':'review'}]}
    config=tmp_path/'curation.json';atomic_json(config,settings)
    return config,parent,settings,manifest


def test_curated_subset_keeps_test_bytes_and_rebuilds_pilot(publication):
    config,parent,_,_=publication
    result=publish(config);target=config.parent/'published'
    assert result['splits']['train']['examples']==2001
    assert result['splits']['pilot']['examples']==2000
    assert (target/'test.jsonl').read_bytes()==(parent/'test.jsonl').read_bytes()
    train={json.loads(line)['id'] for line in (target/'train.jsonl').read_text().splitlines()}
    pilot={json.loads(line)['id'] for line in (target/'pilot.jsonl').read_text().splitlines()}
    assert pilot<=train and 'train0' not in train


def test_stale_parent_identity_is_rejected(publication):
    config,parent,_,manifest=publication
    manifest['splits']['train']['examples']=999
    atomic_json(parent/'manifest.json',manifest)
    with pytest.raises(ValueError,match='stale'):
        publish(config)
    assert not (config.parent/'published').exists()


def test_unknown_rejection_does_not_publish(publication):
    config,_,settings,_=publication
    settings['exclusions'][0]['row_index']=9999;atomic_json(config,settings)
    with pytest.raises(ValueError,match='not found'):
        publish(config)


def test_test_content_cannot_be_targeted_by_spotcheck(publication):
    config,_,settings,_=publication
    settings['exclusions'][0]['row_index']=4000;atomic_json(config,settings)
    with pytest.raises(ValueError,match='test content'):
        publish(config)


def test_existing_publication_is_not_overwritten(publication):
    config,_,_,_=publication
    target=config.parent/'published';target.mkdir();(target/'marker').write_text('keep')
    with pytest.raises(ValueError,match='overwrite'):
        publish(config)
    assert (target/'marker').read_text()=='keep'
