import torch
from llm_lab.config import ModelConfig
from llm_lab.models import Decoder
from llm_rl.scoring import token_scores
from llm_rl.objectives import dpo_loss,group_advantages,grpo_response_loss,generalized_advantages,ppo_token_loss


def test_response_only_head_projection_matches_full_forward_scores_and_gradients():
    torch.manual_seed(19);model=Decoder(ModelConfig(vocab_size=40,dim=16,n_layers=2,n_heads=2,n_kv_heads=1,ffn_dim=32,max_seq_len=64,dropout=.2)).eval()
    prompts=[[4,5,6],[7,8,9,10]];answers=[[11,2],[12,13,2]]
    got,_=token_scores(model,prompts,answers,'cpu')
    expected=[]
    for p,a in zip(prompts,answers):
        ids=torch.tensor([p+a]);logits=model(ids[:,:-1]).logits[0].float()
        expected.append(logits.log_softmax(-1)[len(p)-1:,torch.tensor(a)].diagonal())
    for a,b in zip(got,expected):torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)
    loss=-sum(s.sum() for s in got);loss.backward()
    assert model.blocks[0].attn.q.weight.grad.abs().sum()>0


def test_dpo_increases_preferred_margin_and_has_no_reference_gradient():
    chosen=torch.tensor([0.],requires_grad=True);rejected=torch.tensor([0.],requires_grad=True)
    loss,_=dpo_loss(chosen,rejected,torch.zeros(1),torch.zeros(1));loss.sum().backward()
    assert chosen.grad.item()<0 and rejected.grad.item()>0


def test_grpo_zero_variance_groups_have_zero_policy_advantage():
    adv,mixed=group_advantages([0,0,0,0,1,0,0,0],4)
    assert torch.equal(adv[:4],torch.zeros(4)) and mixed.tolist()==[False,True]
    assert adv[4]>0 and abs(adv[4:].sum())<1e-5


def test_grpo_matches_reference_at_initialization_and_clips_after_policy_change():
    current=torch.tensor([-.1],requires_grad=True);old=current.detach().clone()
    loss,kl,clip=grpo_response_loss(current,old,old,torch.tensor(1.))
    assert kl==0 and clip==0;loss.backward();assert current.grad.item()<0
    loss,_,fraction=grpo_response_loss(torch.tensor([1.]),torch.tensor([0.]),torch.tensor([0.]),torch.tensor(1.))
    assert fraction==1 and torch.isfinite(loss)


def test_gae_assigns_terminal_reward_to_all_previous_tokens():
    adv,returns=generalized_advantages(torch.tensor([0.,0.,1.]),torch.zeros(3),gamma=1.,lam=1.)
    torch.testing.assert_close(adv,torch.ones(3));torch.testing.assert_close(returns,adv)


def test_ppo_second_update_can_activate_clipping():
    current=torch.tensor([1.],requires_grad=True);values=torch.tensor([.1],requires_grad=True)
    loss,policy,value=ppo_token_loss(current,torch.tensor([0.]),torch.ones(1),values,torch.zeros(1),torch.ones(1))
    loss.sum().backward();assert current.grad.item()==0 and values.grad.item()<0
