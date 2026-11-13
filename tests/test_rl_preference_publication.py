from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from llm_lab.common.io import canonical_hash,read_json,write_json
from llm_rl.data import PreferenceCorpus
from prepare_reading_dpo_pairs import publish
from test_sft_runtime_data import corpus


def test_publication_receipt_and_full_reader_bind_actual_prompt_and_answers(corpus):
    directory,_,_,tokenizer=corpus
    row={'id':'fixture-question','language':'en','prefix_ids':[3,4,5],
        'chosen':'John','reference_answers':['John'],'context':'John lived in London.'}
    class Arm:
        manifest={'source_manifest_sha256':'fixture-source-sha','data_identity':'fixture-arm-sha'}
        source=SimpleNamespace(fixture=False,rows=[{'context':row['context']}]);indices=[0]
        def __len__(self):return 1
        def __getitem__(self,i):return dict(row)
    arm=Arm();pair={'id':row['id'],'language':'en','prefix_sha256':canonical_hash(row['prefix_ids']),
        'chosen_ids':tokenizer.encode('John',add_special_tokens=False).ids+[2],
        'rejected_ids':tokenizer.encode('London',add_special_tokens=False).ids+[2]}
    out=directory/'published_fixture';result=publish([pair],arm,tokenizer,{'sha256':'fixture-policy-sha'},{'fixture':True},out)
    assert result['examples']==1
    got=PreferenceCorpus(result['manifest'],arm,tokenizer,'fixture-policy-sha');assert len(got)==1
    receipt=read_json(out/'acceptance.json');receipt['checks']['ambiguous_aliases_excluded']=False;write_json(out/'acceptance.json',receipt)
    with pytest.raises(ValueError,match='acceptance changed'):PreferenceCorpus(result['manifest'],arm,tokenizer,'fixture-policy-sha')
