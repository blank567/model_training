from copy import deepcopy
import torch
from llm_lab.config import ModelConfig
from llm_lab.models import Decoder
from llm_rl.generation import prefill,generate


def model():
    torch.manual_seed(901)
    return Decoder(ModelConfig(vocab_size=40,dim=16,n_layers=2,n_heads=2,n_kv_heads=1,
          ffn_dim=32,max_seq_len=64,dropout=.2)).eval()


def test_cache_matches_full_forward_for_unequal_lengths_and_overwrites_right_padding():
    m=model();prompts=[[4,5,6],[7,8,9,10,11,12]];state=prefill(m,prompts,'cpu')
    for i,p in enumerate(prompts):torch.testing.assert_close(state.logits[i],m(torch.tensor([p])).logits[0,-1],atol=2e-6,rtol=1e-5)
    for tokens in [[20,21],[22,23],[24,25]]:
        state.advance(torch.tensor(tokens),torch.tensor([True,True]))
        for i,token in enumerate(tokens):
            prompts[i].append(token);expected=m(torch.tensor([prompts[i]])).logits[0,-1]
            torch.testing.assert_close(state.logits[i],expected,atol=2e-6,rtol=1e-5)


def test_cached_greedy_generation_equals_unpadded_reference_and_preserves_weights():
    m=model();before=deepcopy(m.state_dict());prompts=[[3,4,5],[3,4,5],[6,7,8,9,10]]
    class Tokenizer:
        def decode(self,ids,**kwargs):return ','.join(map(str,ids))
    result=generate(m,Tokenizer(),prompts,'cpu',max_new_tokens=8,sample=False)
    assert result['unique_prefills']==2
    for i,prefix in enumerate(prompts):
        ids=list(prefix);answers=[]
        for _ in range(8):
            token=int(m(torch.tensor([ids])).logits[0,-1].argmax());answers.append(token);ids.append(token)
            if token==2:break
        assert result['answers'][i]==answers
    assert all(torch.equal(value,before[key]) for key,value in m.state_dict().items())
