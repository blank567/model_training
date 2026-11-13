"""冻结等量旧/新RL数据和旧5万扩展组；显式记录用户授权的PPO内部优化次数。"""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llm_lab.common.io import canonical_hash,read_json,write_json
from llm_reading_independent.comparison import prepare
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['old-source','new-source','output','checkpoint-receipt']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--ppo-updates',type=int,choices=[1,2],default=1);p.add_argument('--authorized',action='store_true');a=p.parse_args()
    checkpoint=read_json(a.checkpoint_receipt)['initial_checkpoint']
    plan=prepare(a.old_source,a.new_source,a.output,checkpoint,a.authorized)
    plan['shared_settings']['ppo_optimization_passes_per_rollout']=a.ppo_updates
    plan['authorization']={'training_authorized':a.authorized,'data_epochs':1,'ppo_internal_updates':a.ppo_updates,
                           'old_50000_extension_requested':True,'all_weights_to_be_preserved':True}
    plan['plan_identity']=canonical_hash({k:v for k,v in plan.items() if k!='plan_identity'})
    write_json(a.output/'plan.json',plan)
    print(json.dumps({'plan':str(a.output/'plan.json'),'identity':plan['plan_identity'],'arms':plan['arms'],
                     'planned_runs':plan['planned_runs'],'training_started':False},ensure_ascii=False))
