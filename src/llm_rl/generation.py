"""只读KV缓存解码适配器；与冻结Decoder共享权重，支持不同长度的右填充。"""
from contextlib import nullcontext
import torch
from torch.nn import functional as F


def precision(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if str(device).startswith('cuda') else nullcontext()


def rope(q,k,positions,theta):
    dim=q.shape[-1]
    frequency=theta**(-torch.arange(0,dim,2,device=q.device,dtype=torch.float32)/dim)
    angles=positions.float().unsqueeze(-1)*frequency
    cos,sin=angles.cos().unsqueeze(1),angles.sin().unsqueeze(1)
    def rotate(x):
        even,odd=x.float()[...,0::2],x.float()[...,1::2]
        return torch.stack((even*cos-odd*sin,even*sin+odd*cos),dim=-1).flatten(-2).to(x.dtype)
    return rotate(q),rotate(k)


def projections(attention,x,positions):
    batch,length,_=x.shape;dim=attention.head_dim
    q=attention.q(x).view(batch,length,attention.heads,dim).transpose(1,2)
    k=attention.k(x).view(batch,length,attention.kv_heads,dim).transpose(1,2)
    v=attention.v(x).view(batch,length,attention.kv_heads,dim).transpose(1,2)
    q,k=rope(q,k,positions,attention.rope_theta)
    return q,k,v


class KVState:
    def __init__(self,model,keys,values,lengths,logits):
        self.model=model;self.keys=keys;self.values=values;self.lengths=lengths;self.logits=logits

    def expand(self,indices):
        return KVState(self.model,[k.index_select(0,indices) for k in self.keys],
            [v.index_select(0,indices) for v in self.values],self.lengths.index_select(0,indices),self.logits.index_select(0,indices))

    @torch.no_grad()
    def advance(self,tokens,active):
        model=self.model;positions=self.lengths.clone();batch=len(tokens)
        if positions.max().item()>=model.config.max_seq_len:raise ValueError('Cached decode exceeds context')
        x=model.embedding(tokens[:,None]);rows=torch.arange(batch,device=tokens.device)
        size=int(positions.max().item())+1
        allowed=torch.arange(size,device=tokens.device)[None,:]<=positions[:,None]
        with precision(tokens.device):
            for i,block in enumerate(model.blocks):
                q,k,v=projections(block.attn,block.attn_norm(x),positions[:,None])
                self.keys[i][rows,:,positions,:]=k[:,:,0,:];self.values[i][rows,:,positions,:]=v[:,:,0,:]
                repeats=block.attn.heads//block.attn.kv_heads
                keys=self.keys[i][:,:,:size,:].repeat_interleave(repeats,dim=1)
                values=self.values[i][:,:,:size,:].repeat_interleave(repeats,dim=1)
                attended=F.scaled_dot_product_attention(q,keys,values,attn_mask=allowed[:,None,None,:],dropout_p=0.,is_causal=False)
                x=x+block.attn.out(attended.transpose(1,2).contiguous().view(batch,1,-1))
                normalized=block.ffn_norm(x);x=x+block.down(F.silu(block.gate(normalized))*block.up(normalized))
            self.logits=model.lm_head(model.norm(x[:,0,:])).float()
        self.lengths=positions+active.long()
        return self.logits


@torch.no_grad()
def prefill(model,prompts,device):
    if model.training:raise ValueError('Rollout requires eval mode with dropout disabled')
    if not prompts or any(not p for p in prompts):raise ValueError('Nonempty prompts required')
    batch=len(prompts);length=max(map(len,prompts));capacity=model.config.max_seq_len
    if length>capacity:raise ValueError('Prompt exceeds model context')
    tokens=torch.zeros((batch,length),dtype=torch.long,device=device)
    lengths=torch.tensor([len(p) for p in prompts],device=device)
    for i,prompt in enumerate(prompts):tokens[i,:len(prompt)]=torch.tensor(prompt,device=device)
    positions=torch.arange(length,device=device)[None,:].expand(batch,-1);keys=[];values=[]
    with precision(device):
        x=model.embedding(tokens)
        for block in model.blocks:
            q,k,v=projections(block.attn,block.attn_norm(x),positions)
            kc=torch.zeros((batch,block.attn.kv_heads,capacity,block.attn.head_dim),dtype=k.dtype,device=device)
            vc=torch.zeros_like(kc);kc[:,:,:length,:]=k;vc[:,:,:length,:]=v;keys.append(kc);values.append(vc)
            repeats=block.attn.heads//block.attn.kv_heads
            attended=F.scaled_dot_product_attention(q,k.repeat_interleave(repeats,1),v.repeat_interleave(repeats,1),dropout_p=0.,is_causal=True)
            x=x+block.attn.out(attended.transpose(1,2).contiguous().view(batch,length,-1))
            normalized=block.ffn_norm(x);x=x+block.down(F.silu(block.gate(normalized))*block.up(normalized))
        last=x[torch.arange(batch,device=device),lengths-1];logits=model.lm_head(model.norm(last)).float()
    return KVState(model,keys,values,lengths,logits)


@torch.no_grad()
def generate(model,tokenizer,prompts,device,max_new_tokens=64,sample=True,eos=2,generator=None):
    """温度1、完整softmax分布；共享前缀只prefill一次，各采样保留独立KV。"""
    model.eval()
    if any(len(p)+max_new_tokens>model.config.max_seq_len for p in prompts):raise ValueError('Generation budget exceeds context')
    unique=[];lookup={};inverse=[]
    for prompt in prompts:
        key=tuple(prompt)
        if key not in lookup:lookup[key]=len(unique);unique.append(prompt)
        inverse.append(lookup[key])
    state=prefill(model,unique,device)
    if len(unique)!=len(prompts):state=state.expand(torch.tensor(inverse,device=device))
    answers=[[] for _ in prompts];active=torch.ones(len(prompts),dtype=torch.bool,device=device)
    for step in range(max_new_tokens):
        selected=active.nonzero().flatten()
        scores=state.logits.index_select(0,selected)
        next_ids=torch.multinomial(scores.softmax(-1),1,generator=generator).flatten() if sample else scores.argmax(-1)
        tokens=torch.full((len(prompts),),eos,dtype=torch.long,device=device);tokens[selected]=next_ids
        old_active=active.clone()
        for i,token in zip(selected.tolist(),next_ids.tolist()):
            answers[i].append(token)
            if token==eos:active[i]=False
        if not active.any() or step+1==max_new_tokens:break
        state.advance(tokens,old_active)
    texts=[tokenizer.decode(a[:-1] if a and a[-1]==eos else a,skip_special_tokens=False) for a in answers]
    return {'answers':answers,'texts':texts,'eos':[bool(a and a[-1]==eos) for a in answers],
            'unique_prefills':len(unique),'generated_tokens':sum(map(len,answers))}
