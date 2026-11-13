"""DPO、GRPO和PPO的纯损失函数；采样与模型、Checkpoint IO在独立模块。"""
import torch
from torch.nn import functional as F


def dpo_loss(chosen,rejected,reference_chosen,reference_rejected,beta=.1):
    margin=(chosen-rejected)-(reference_chosen-reference_rejected)
    return -F.logsigmoid(beta*margin),margin.detach()


def group_advantages(rewards,group_size,epsilon=1e-4):
    if len(rewards)%group_size or group_size<2:raise ValueError('Complete groups with at least two responses required')
    values=torch.as_tensor(rewards,dtype=torch.float32).view(-1,group_size)
    std=values.std(-1,unbiased=False,keepdim=True)
    advantage=(values-values.mean(-1,keepdim=True))/(std+epsilon)
    return advantage.flatten(),(std.flatten()>0)


def grpo_response_loss(current,old,reference,advantage,clip=.2,kl_coefficient=.02):
    ratio=(current-old).exp();clipped=ratio.clamp(1-clip,1+clip)
    surrogate=torch.minimum(ratio*advantage,clipped*advantage)
    delta=(reference-current).clamp(-20,20);kl=delta.exp()-delta-1
    loss=(-surrogate+kl_coefficient*kl).mean()
    return loss,kl.mean().detach(),((ratio-1).abs()>clip).float().mean().detach()


def generalized_advantages(rewards,values,gamma=1.,lam=.95):
    """有界生成任务在EOS或max_new_tokens处终止，终点bootstrap为0。"""
    if len(rewards)!=len(values):raise ValueError('One value per generated token required')
    advantage=torch.zeros_like(values);carry=torch.zeros((),dtype=values.dtype,device=values.device)
    for t in range(len(values)-1,-1,-1):
        next_value=values[t+1] if t+1<len(values) else 0.
        delta=rewards[t]+gamma*next_value-values[t];carry=delta+gamma*lam*carry;advantage[t]=carry
    return advantage,advantage+values


def ppo_token_loss(current,old,advantages,values,old_values,returns,clip=.2,value_clip=.2,value_coefficient=.5):
    ratio=(current-old).exp();policy=-torch.minimum(ratio*advantages,ratio.clamp(1-clip,1+clip)*advantages)
    clipped_value=old_values+(values-old_values).clamp(-value_clip,value_clip)
    value_loss=.5*torch.maximum((values-returns).square(),(clipped_value-returns).square())
    return policy+value_coefficient*value_loss,policy.detach(),value_loss.detach()
