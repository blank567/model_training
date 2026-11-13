"""仅在回答位置计算词表投影；提示依然参与注意力与反向传播。"""
import torch
from torch.nn import functional as F
from .generation import precision


def token_scores(model,prompts,answers,device,value_head=None):
    if not prompts or len(prompts)!=len(answers) or any(not p or not a for p,a in zip(prompts,answers)):
        raise ValueError('Nonempty prompt/response pairs required')
    length=max(len(p)+len(a)-1 for p,a in zip(prompts,answers))
    if length>model.config.max_seq_len:raise ValueError('Scoring exceeds context')
    x=torch.zeros((len(prompts),length),dtype=torch.long,device=device);y=torch.zeros_like(x);mask=torch.zeros_like(x,dtype=torch.bool)
    sizes=[]
    for i,(p,a) in enumerate(zip(prompts,answers)):
        ids=torch.tensor(p+a,device=device);n=len(ids)-1;x[i,:n]=ids[:-1];y[i,:n]=ids[1:]
        mask[i,len(p)-1:n]=True;sizes.append(len(a))
    selected=[]
    def choose(module,args):
        hidden=args[0][mask];selected.append(hidden);return (hidden,)
    handle=model.lm_head.register_forward_pre_hook(choose)
    try:
        with precision(device):logits=model(x).logits
        logps=F.log_softmax(logits.float(),-1).gather(-1,y[mask,None]).flatten()
        values=value_head(selected[0].detach().float()).flatten() if value_head is not None else None
    finally:handle.remove()
    return list(logps.split(sizes)),list(values.split(sizes)) if values is not None else None


def sequence_scores(model,prompts,answers,device):
    scores,_=token_scores(model,prompts,answers,device);return torch.stack([s.sum() for s in scores])
