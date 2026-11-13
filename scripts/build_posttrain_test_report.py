"""从真实逐题产物生成完整中文报告、CSV、统计区间和静态图，不运行训练。"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import html
import json
import math
import os
from pathlib import Path
import re
import sys
import base64

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib_report'))
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from evaluate_posttrain_comprehensive import cluster_interval,exact_mcnemar,holm
from llm_lab.common.io import read_json,read_jsonl,write_json,sha256_file
from llm_reading_data.build import question_type,length_bucket


def csv_file(path,rows):
    rows=list(rows);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def table(headers,rows):
    def cell(x):return str(x).replace('|','\\|').replace('\n','<br>')
    return '\n'.join(['| '+' | '.join(map(cell,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(cell,row))+' |' for row in rows])+'\n'


def f(value,digits=4):return '—' if value is None else f'{value:.{digits}f}'
def pct(value):return '—' if value is None else f'{value*100:.2f}%'
def pp(value):return f'{value*100:+.2f}'
def pvalue(value):return '—' if value is None else f'{value:.3g}'


def compact_ci(result):return f"{result['percentile_95'][0]*100:+.2f} ～ {result['percentile_95'][1]*100:+.2f}"


def label(key):
    if key=='reading_sft':return '阅读 SFT 50k'
    if key=='pretrained_base':return '预训练 Base'
    if key=='sft100k_full':return '通用 Full SFT 100k（历史2轮）'
    if key=='lora100k_r16_historical':return 'LoRA r16/a32（历史2轮）'
    if key.startswith('lora100k_'):
        return 'LoRA '+key.removeprefix('lora100k_').replace('_ld',' / dropout=').replace('0p','0.')
    if key.startswith('sft20k_'):
        rest=key.removeprefix('sft20k_');lr=rest.split('_')[0].removeprefix('lr')
        batch=re.search(r'_b(\d+)_',rest).group(1)
        epochs='2（历史共享3轮调度）' if 'stop2' in rest else '3（历史）' if 'stop3' in rest else '1'
        return f'SFT 20k / LR={lr} / B={batch} / 轮数={epochs}'
    for method in ['ppo','grpo','dpo']:
        if key.startswith(method+'_'):
            arm=key[len(method)+1:].rsplit('_',1)[0]
            group={'old_matched':'旧匹配组','new_matched':'新匹配组','old_full_50k':'旧全量组'}[arm]
            return method.upper()+' / '+group+' / '+key.rsplit('_',1)[1]
    return key


def mean_groups(rows,field,metric):
    groups=defaultdict(list)
    for r in rows:groups[r[field]].append(r[metric])
    return {k:{'n':len(v),'mean':float(np.mean(v))} for k,v in groups.items()}


def sign_flip_p(reference,contender,replicates=10000):
    """文章整组交换的探索性检验，补充题级McNemar的独立性局限。"""
    right={r['id']:r for r in contender};counts=Counter(r['language'] for r in reference);groups=defaultdict(float)
    for r in reference:
        key=(r['language'],r.get('group_id',r['id']))
        groups[key]+=(right[r['id']]['exact_match']-r['exact_match'])/(len(counts)*counts[r['language']])
    delta=np.asarray(list(groups.values()));observed=abs(delta.sum())
    rng=np.random.default_rng(20261104);hits=0
    for offset in range(0,replicates,500):
        n=min(500,replicates-offset);signs=rng.integers(0,2,(n,len(delta)))*2-1
        hits+=int(np.sum(np.abs(signs@delta)>=observed-1e-14))
    return {'p':(hits+1)/(replicates+1),'replicates':replicates,'clusters':len(delta),
        'method':'exploratory passage-cluster label sign-flip; conditional exchangeability assumption'}


def figure_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in ['C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simhei.ttf','C:/Windows/Fonts/simsun.ttc']:
        if Path(name).exists():
            font_manager.fontManager.addfont(name)
            plt.rcParams['font.family']=font_manager.FontProperties(fname=name).get_name();break
    plt.rcParams.update({'axes.unicode_minus':False,'figure.dpi':150,'savefig.dpi':180,'axes.spines.top':False,
                         'axes.spines.right':False,'font.size':10,'axes.titlepad':12})
    return plt


def generate_figures(folder,plan,reading,generic,comparisons,hist,train_rows,raw_curves):
    plt=figure_setup();folder.mkdir(exist_ok=True)
    def save(fig,name):fig.tight_layout();fig.savefig(folder/name,bbox_inches='tight');plt.close(fig)
    arms=['old_matched','new_matched','old_full_50k'];methods=['ppo','grpo','dpo']
    matrix=np.array([[reading[f'{m}_{a}_best']['generation']['macro_exact_match']*100 for a in arms] for m in methods])
    fig,ax=plt.subplots(figsize=(8,4));im=ax.imshow(matrix,cmap='YlGnBu',vmin=max(0,matrix.min()-3),vmax=matrix.max()+2)
    ax.set_xticks(range(3),['旧匹配组','新匹配组','旧全量组']);ax.set_yticks(range(3),['PPO','GRPO','DPO'])
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{matrix[i,j]:.2f}%',ha='center',va='center',color='white' if im.norm(matrix[i,j])>.6 else 'black',fontweight='bold')
    ax.set_title('固定 2,000 题：验证集选定 best 的精确匹配率\n阅读 SFT 基线 '+pct(reading['reading_sft']['generation']['macro_exact_match'])+'；DPO匹配组best=step0，等同起点')
    fig.colorbar(im,ax=ax,label='中英文 Macro EM（%）');save(fig,'01_reading_best.png')
    fig,ax=plt.subplots(figsize=(10,6));values=comparisons
    for i,row in enumerate(values):
        ci=row['em_interval']['percentile_95'];delta=row['em_interval']['delta_contender_minus_reference']
        ax.plot(np.array(ci)*100,[i,i],color='#275d95',linewidth=2)
        ax.scatter(delta*100,i,color='#275d95',s=35,zorder=3)
    ax.axvline(0,color='#8f8f8f',linestyle='--');ax.set_yticks(range(len(values)),[label(r['key']).replace(' / best','') for r in values]);ax.invert_yaxis()
    ax.set_xlabel('相对阅读 SFT 的 Macro EM 变化（百分点）；95%文章整群配对区间')
    ax.set_title('提升幅度与测试样本不确定性；区间未作多重比较校正');save(fig,'02_paired_delta.png')
    fig,ax=plt.subplots(figsize=(11,5));x=np.arange(9)
    keys=[f'{m}_{a}' for m in methods for a in arms]
    ax.bar(x-.18,[reading[k+'_best']['generation']['macro_exact_match']*100 for k in keys],.36,label='validation best',color='#275d95')
    ax.bar(x+.18,[reading[k+'_last']['generation']['macro_exact_match']*100 for k in keys],.36,label='epoch-1 last',color='#d38a44')
    ax.axhline(reading['reading_sft']['generation']['macro_exact_match']*100,color='gray',linestyle='--',label='阅读SFT')
    ax.set_xticks(x,[label(k+'_best').replace(' / best','').replace(' / ','\n') for k in keys],fontsize=8)
    ax.set_ylabel('Macro EM（%）');ax.set_title('验证选点与训练终点：last 仅用于诊断，不重新选模');ax.legend(ncols=3);save(fig,'03_best_last.png')
    order=sorted(generic,key=lambda k:generic[k]['test']['macro_loss'])
    fig,ax=plt.subplots(figsize=(12,8));y=np.arange(len(order))
    ax.barh(y,[generic[k]['test']['macro_loss'] for k in order],color=['#b18b44' if k.startswith('lora') else '#275d95' for k in order])
    ax.set_yticks(y,[label(k) for k in order],fontsize=8);ax.invert_yaxis();ax.set_xlabel('中英文 Macro 回答交叉熵（越低越好）')
    ax.set_title('相同 1,000 题的参考答案建模；不代表任务正确率');save(fig,'04_instruction_loss.png')
    fig,axes=plt.subplots(1,2,figsize=(13,7),sharey=True)
    for ax,metric,title in [(axes[0],'mean_reference_overlap_rouge_l_f1','参考 ROUGE-L F1（非语义正确率）'),(axes[1],'eos_stop_fraction','EOS 停止率（%）')]:
        scale=100 if metric=='eos_stop_fraction' else 1
        ax.barh(y,[generic[k]['generation'][metric]*scale for k in order],color='#2e867d');ax.set_title(title);ax.set_yticks(y,[label(k) for k in order],fontsize=8)
    axes[0].invert_yaxis();save(fig,'05_instruction_generation.png')
    history={r['key']:r for r in hist['results']}
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    lrkeys=['lr1e-5_mb8_ga4_b32_ep1','lr2e-5_mb8_ga4_b32_ep1','lr5e-5_mb8_ga4_b32_ep1','lr1e-4_mb8_ga4_b32_ep1']
    axes[0].plot([1e-5,2e-5,5e-5,1e-4],[history[k]['test']['macro_loss'] for k in lrkeys],marker='o',color='#275d95');axes[0].set_xscale('log')
    axes[0].set_title('历史学习率对照：相同19,991条、1轮、B32');axes[0].set_xlabel('学习率');axes[0].set_ylabel('历史1,000题 Macro CE')
    bkeys=['lr5e-5_mb8_ga1_b8_ep1','lr5e-5_mb8_ga2_b16_ep1','lr5e-5_mb8_ga4_b32_ep1']
    axes[1].plot([8,16,32],[history[k]['test']['macro_loss'] for k in bkeys],marker='o',color='#2e867d');axes[1].set_xticks([8,16,32]);axes[1].set_xlabel('有效Batch')
    axes[1].set_title('历史Batch对照：相同数据、1轮、LR5e-5');save(fig,'06_sft_controls.png')
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    colors=['#275d95','#2e867d','#d38a44']
    for method,ax in zip(methods,axes):
        for arm,color in zip(arms,colors):
            key=f'{method}_{arm}_best';data=raw_curves.get(key,[])
            steps=[];em=[]
            for row in data:
                if 'validation' in row:
                    steps.append(row['optimizer_step']);em.append(row['validation']['macro_exact_match']*100)
            ax.plot(steps,em,label={'old_matched':'旧匹配','new_matched':'新匹配','old_full_50k':'旧全量'}[arm],color=color,marker='.',markersize=3)
        ax.set_title(method.upper()+' 开发集256题');ax.set_xlabel('optimizer step');ax.legend(fontsize=8)
    axes[0].set_ylabel('Macro EM（%）；选best使用该集合');save(fig,'07_validation_curves.png')
    rows=[r for r in train_rows if r.get('retention_macro_loss') is not None]
    fig,ax=plt.subplots(figsize=(12,7));rows=sorted(rows,key=lambda r:r['retention_macro_loss'])
    ax.barh(range(len(rows)),[r['retention_macro_loss'] for r in rows],color='#7b709d');ax.set_yticks(range(len(rows)),[r['label'] for r in rows],fontsize=8);ax.invert_yaxis()
    ax.set_title('历史预训练验证小窗口保持 CE；非完整预训练测试集');ax.set_xlabel('固定窗口中英文 Macro CE（越低越好）');save(fig,'08_retention.png')


def training_inventory(plan,sources):
    result=[];curves={};saved=read_json(sources/'inventory.json')
    sizes={r['path']:r.get('weights',{}) for r in saved['experiments']}
    for item in plan['models']:
        if item['stage']=='reading_rl' and not item['key'].endswith('_best'):continue
        folder=sources/'training'/Path(item['checkpoint']).parent.name
        summary=read_json(folder/'summary.json');cfg=read_json(folder/'config.json') if (folder/'config.json').exists() else {}
        t=cfg.get('train',{});p=summary.get('plan',{});budget=summary.get('budget',{})
        adapter=summary.get('adapter',{})
        planned_examples=p.get('train_examples',p.get('examples'))
        seen=summary.get('examples_seen',summary.get('progress',{}).get('examples_seen'))
        retention=summary.get('latest_retention') or summary.get('retention_validation')
        metrics=read_jsonl(folder/'metrics.jsonl') if (folder/'metrics.jsonl').exists() else []
        # 取同一训练身份最后记录的保持窗口；不把终点保持Loss冒充best保持Loss。
        if retention is None:
            for r in reversed(metrics):
                if r.get('retention_validation'):retention=r['retention_validation'];break
        row={'key':item['key'],'label':label(item['key']),'stage':item['stage'],'checkpoint':item['checkpoint'],
             'checkpoint_sha256':item['checkpoint_sha256'],'initial_checkpoint':cfg.get('initial_checkpoint',cfg.get('base_checkpoint')),
             'initial_checkpoint_sha256':cfg.get('initial_checkpoint_sha256',cfg.get('base_checkpoint_sha256')),
             'data_manifest':cfg.get('data_manifest',cfg.get('arm_manifest')),'train_examples':planned_examples,
             'examples_seen':seen,'learning_rate':t.get('learning_rate'),'micro_batch_size':t.get('micro_batch_size',t.get('score_micro_batch_size')),
             'gradient_accumulation':t.get('gradient_accumulation_steps',t.get('gradient_accumulation')),
             'effective_batch':p.get('effective_batch_size',t.get('prompt_batch_size')),
             'schedule_epochs':p.get('epochs',t.get('epochs')),'stop_epochs':p.get('stop_epochs',t.get('stop_epochs')),
             'actual_epochs':seen/planned_examples if seen is not None and planned_examples else None,
             'final_step':summary.get('optimizer_step',summary.get('step')),'best_step':summary.get('best_step'),
             'validation_best_loss':summary.get('best_validation_loss'),'adapter_rank':adapter.get('rank',cfg.get('adapter',{}).get('rank')),
             'adapter_alpha':cfg.get('adapter',{}).get('alpha'),'adapter_dropout':cfg.get('adapter',{}).get('dropout'),
             'trainable_parameters':adapter.get('trainable_parameters',summary.get('parameters')),
             'estimated_compute_cny':budget.get('estimated_compute_cny'),'recorded_compute_seconds_all_sessions':budget.get('elapsed_seconds'),
             'retention_macro_loss':retention.get('macro_loss') if retention else None}
        weight_sizes=sizes.get(item['checkpoint'].rsplit('/',1)[0],{})
        row['best_checkpoint_mib']=weight_sizes.get('best.pt',0)/2**20 if 'best.pt' in weight_sizes else None
        row['last_checkpoint_mib']=weight_sizes.get('last.pt',0)/2**20 if 'last.pt' in weight_sizes else None
        if (folder/'wandb_run.json').exists():
            tracking=read_json(folder/'wandb_run.json');row['wandb_run_id']=tracking.get('id');row['wandb_url']=tracking.get('url')
        result.append(row);curves[item['key']]=metrics
    return result,curves


def markdown_html(markdown,root):
    """离线自包含报告，标准表格/段落/图片；不加载外部脚本或字体。"""
    def inline(text):
        text=html.escape(text)
        text=re.sub(r'`([^`]+)`',r'<code>\1</code>',text)
        text=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',text)
        def link(match):
            target=html.unescape(match.group(2))
            if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:',target) and not target.startswith(('https://','http://')):return match.group(0)
            return '<a href="'+html.escape(target,quote=True)+'">'+match.group(1)+'</a>'
        text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,text)
        return text.replace('&lt;br&gt;','<br>')
    lines=markdown.splitlines();body=[];toc=[];index=0;heading_count=0;code=False;in_table=False
    while index<len(lines):
        line=lines[index]
        if line.startswith('```'):
            if in_table:body.append('</tbody></table>');in_table=False
            code=not code;body.append('<pre><code>' if code else '</code></pre>');index+=1;continue
        if code:body.append(html.escape(line)+'\n');index+=1;continue
        if not line.startswith('|') and in_table:body.append('</tbody></table>');in_table=False
        if re.match(r'^#{1,4} ',line):
            level=len(line)-len(line.lstrip('#'));text=line[level+1:];anchor='s'+str(heading_count);heading_count+=1
            body.append(f'<h{level} id="{anchor}">{inline(text)}</h{level}>')
            if level<=2:toc.append((level,anchor,text))
        elif line.startswith('!['):
            match=re.match(r'!\[([^\]]*)\]\(([^)]+)\)',line)
            if match:
                path=root/match.group(2);encoded=base64.b64encode(path.read_bytes()).decode()
                body.append('<figure><img alt="'+html.escape(match.group(1))+'" src="data:image/png;base64,'+encoded+'"></figure>')
        elif line.startswith('|'):
            if re.match(r'^\|[\s:|\-]+$',line):index+=1;continue
            cells=re.split(r'(?<!\\)\|',line.strip('|'))
            if not in_table:body.append('<table><thead><tr>'+''.join('<th>'+inline(c.strip())+'</th>' for c in cells)+'</tr></thead><tbody>');in_table=True
            else:body.append('<tr>'+''.join('<td>'+inline(c.strip().replace('\\|','|'))+'</td>' for c in cells)+'</tr>')
        elif line.startswith('- '):body.append('<p class="item">• '+inline(line[2:])+'</p>')
        elif line.strip():body.append('<p>'+inline(line)+'</p>')
        index+=1
    if in_table:body.append('</tbody></table>')
    navigation=''.join(f'<a class="l{level}" href="#{anchor}">{html.escape(text)}</a>' for level,anchor,text in toc)
    style='''body{font:15px/1.8 "Microsoft YaHei",system-ui,sans-serif;color:#25313e;background:#f4f6f8;margin:0}nav{position:fixed;top:0;bottom:0;width:250px;overflow:auto;background:#15334a;padding:25px 20px;color:white}nav a{display:block;color:#d7e4ef;text-decoration:none;line-height:1.5;padding:6px 0;font-size:13px}nav .l1{font-size:16px;font-weight:bold;margin-bottom:18px}main{margin:0 0 0 290px;padding:38px 48px;max-width:1200px;background:white}h1{font-size:30px}h2{margin-top:42px;border-bottom:2px solid #d4dde6;padding-bottom:8px}h3{margin-top:30px}p{overflow-wrap:anywhere}table{display:block;overflow:auto;border-collapse:collapse;margin:20px 0;width:100%;font-size:13px}th,td{border:1px solid #d5dee7;padding:8px 10px;vertical-align:top}th{background:#eaf0f6;white-space:nowrap}tr:nth-child(even){background:#f8fafc}img{max-width:100%;height:auto}figure{margin:20px 0}code{background:#edf1f5;padding:1px 4px;font-size:12px}pre{background:#f4f6f8;overflow:auto;padding:16px;line-height:1.6}.item{padding-left:10px}a{color:#22658d}@media(max-width:950px){nav{display:none}main{margin:0;padding:22px}}@media print{nav{display:none}main{margin:0;padding:0;max-width:none}body{background:white}h2{break-before:page}table{display:table;width:100%;font-size:9px}tr,figure{break-inside:avoid}a{color:inherit;text-decoration:none}}'''
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>300M 后训练完整测试报告</title><style>'+style+'</style><nav>'+navigation+'</nav><main>'+''.join(body)+'</main></html>'


def main(work):
    work=Path(work);sources=work/'sources_local';data=work/'results_local'
    plan=read_json(sources/'plan.json');summary=read_json(data/'comparison.json')
    if summary['status']!='completed' or summary['plan_identity']!=plan['plan_identity']:raise ValueError('Do not report incomplete/mismatched evaluation')
    summary['prior_attempt_costs']=read_json(data/'prior_attempt_costs.json') if (data/'prior_attempt_costs.json').exists() else {}
    reading={r['key']:r for r in summary['results']['reading_test_2000']}
    generic={r['key']:r for r in summary['results']['instruction_test_1000']}
    predictions={suite:{key:read_jsonl(data/suite/key/'generations.jsonl') for key in values} for suite,values in [('reading_test_2000',reading),('instruction_test_1000',generic)]}
    for suite,models in predictions.items():
        expected=plan['suites'][suite]['expected_examples'];ids=None
        for key,rows in models.items():
            if len(rows)!=expected or len({r['id'] for r in rows})!=expected:raise ValueError('Incomplete/duplicate test predictions')
            row_ids={r['id'] for r in rows}
            if ids is not None and ids!=row_ids:raise ValueError('Models received different test IDs')
            ids=row_ids
    report=work/'report';report.mkdir(exist_ok=True);tables=report/'tables';tables.mkdir(exist_ok=True)
    train_rows,curves=training_inventory(plan,sources);csv_file(tables/'01_model_inventory.csv',train_rows)
    base=predictions['reading_test_2000']['reading_sft'];comparisons=[]
    for key in plan['primary_statistical_family']:
        values=predictions['reading_test_2000'][key]
        em=cluster_interval(base,values);f1=cluster_interval(base,values,'f1')
        comparisons.append({'key':key,'em_interval':em,'f1_interval':f1,'mcnemar':exact_mcnemar(base,values),'exploratory_cluster_sign_flip':sign_flip_p(base,values)})
    adjusted=holm([r['mcnemar']['two_sided_exact_p'] for r in comparisons])
    clustered=holm([r['exploratory_cluster_sign_flip']['p'] for r in comparisons])
    for row,p,q in zip(comparisons,adjusted,clustered):row['mcnemar_holm_p']=p;row['exploratory_cluster_holm_p']=q
    write_json(report/'statistical_analysis.json',{'primary':comparisons})
    csv_file(tables/'02_reading_primary_comparisons.csv',[{'key':r['key'],'delta_em':r['em_interval']['delta_contender_minus_reference'],
        'em_ci95_low':r['em_interval']['percentile_95'][0],'em_ci95_high':r['em_interval']['percentile_95'][1],
        'delta_f1':r['f1_interval']['delta_contender_minus_reference'],**r['mcnemar'],'mcnemar_holm_p':r['mcnemar_holm_p'],
        'exploratory_cluster_p':r['exploratory_cluster_sign_flip']['p'],'exploratory_cluster_holm_p':r['exploratory_cluster_holm_p']} for r in comparisons])
    # 后续正文和附表由同一批完整逐题结果构建。
    build_content(report,plan,summary,reading,generic,predictions,comparisons,train_rows,curves,sources,data)


def build_content(report,plan,summary,reading,generic,predictions,comparisons,train_rows,curves,sources,data):
    tables=report/'tables';rpred=predictions['reading_test_2000'];gpred=predictions['instruction_test_1000']
    inventory={r['key']:r for r in train_rows};reading_manifest=read_json(sources/'data_manifests/reading_test_2000.json')
    instruction_manifest=read_json(sources/'data_manifests/instruction_test_1000.json')
    audit=read_json(data/'report_data_audit.json')
    hist=read_json(sources/'historical/sft_test_comparison_20261028_171853/comparison.json')
    historical_lora=read_json(sources/'historical/lora_vs_full_final_20261028_191900/comparison.json')
    historical_scaled=read_json(sources/'historical/sft_100k_final_20261028_172855/comparison.json')
    ablation=read_json(sources/'controls/lora_rank_dropout_20261101_134339/comparison.json')
    methods=['ppo','grpo','dpo'];arms=['old_matched','new_matched','old_full_50k']
    main_keys=['reading_sft']+[f'{m}_{a}_best' for m in methods for a in arms]
    baseline=reading['reading_sft']['generation'];test_base=rpred['reading_sft']
    main_rows=[];language_rows=[];source_rows=[];subgroups=[];errors=[];per_item=[]
    for key,value in reading.items():
        g=value['generation'];t=value['test'];row={'key':key,'label':label(key),'checkpoint_step':value['metadata']['checkpoint_step'] if 'metadata' in value else value['checkpoint_step'],
            'examples':g['examples'],'correct':g['correct'],'macro_em':g['macro_exact_match'],'macro_f1':g['macro_f1'],
            'zh_em':g['by_language']['zh']['exact_match'],'en_em':g['by_language']['en']['exact_match'],
            'macro_ce':t['macro_loss'],'zh_ce':t['zh']['loss'],'en_ce':t['en']['loss'],'eos':g['eos_rate'],
            'extractive':g['extractive_rate'],'empty':g['empty_rate'],'special':g['special_token_rate'],
            'mean_tokens':g['mean_generated_tokens'],'repeat_3gram':g['mean_repeated_3gram_fraction'],
            'evaluated_identical_to':value.get('reused_identical_model_results'),'actual_gpu_evaluation':value['actual_gpu_evaluation_repeated']}
        main_rows.append(row)
        for lang,stats in g['by_language'].items():language_rows.append({'key':key,'language':lang,**stats})
        for source,stats in g['by_source'].items():source_rows.append({'key':key,'source':source,**stats})
        for reason,count in g['errors'].items():errors.append({'key':key,'category':reason,'count':count,'fraction':count/2000})
        values=rpred[key]
        for r in values:
            r['question_type']=question_type(r['question'],r['language'])
            r['answer_length_bucket']=length_bucket(max(0,r['gold_answer_tokens']-1))
            r['prompt_length_bucket']='0_255' if r['prompt_tokens']<256 else '256_511' if r['prompt_tokens']<512 else '512_767' if r['prompt_tokens']<768 else '768_plus'
        for field in ['question_type','answer_length_bucket','prompt_length_bucket']:
            for language in ['zh','en']:
                subset=[r for r in values if r['language']==language]
                for name,stats in mean_groups(subset,field,'exact_match').items():
                    group=[r for r in subset if r[field]==name]
                    subgroups.append({'key':key,'language':language,'dimension':field,'group':name,'examples':stats['n'],'exact_match':stats['mean'],
                        'f1':float(np.mean([r['f1'] for r in group])),'eos':float(np.mean([r['stop_reason']=='eos' for r in group]))})
        if key in plan['primary_statistical_family']:
            original={r['id']:r for r in test_base}
            for r in values:
                b=original[r['id']]
                per_item.append({'key':key,'id':r['id'],'source':r['source'],'language':r['language'],'group_id':r['group_id'],
                    'question':r['question'],'references':r['references'],'sft_answer':b['answer'],'model_answer':r['answer'],
                    'sft_em':b['exact_match'],'model_em':r['exact_match'],'delta_em':r['exact_match']-b['exact_match'],
                    'sft_f1':b['f1'],'model_f1':r['f1'],'eos':r['stop_reason']=='eos','error_category':r['error_category']})
    csv_file(tables/'03_reading_all_checkpoints.csv',main_rows)
    csv_file(tables/'04_reading_by_language.csv',language_rows);csv_file(tables/'05_reading_by_source.csv',source_rows)
    csv_file(tables/'06_reading_by_question_and_answer_length.csv',subgroups);csv_file(tables/'07_reading_errors.csv',errors)
    csv_file(tables/'08_reading_18000_paired_answers.csv',per_item)
    rowmap={r['key']:r for r in main_rows}
    regimes=[];best_last=[]
    for method in methods:
        for left,right,meaning in [('old_matched','new_matched','新匹配减旧匹配'),('old_matched','old_full_50k','旧全量减旧匹配')]:
            a=f'{method}_{left}_best';b=f'{method}_{right}_best'
            interval=cluster_interval(rpred[a],rpred[b]);f1=cluster_interval(rpred[a],rpred[b],'f1')
            regimes.append({'method':method,'comparison':meaning,'reference':a,'contender':b,'delta_em':interval['delta_contender_minus_reference'],
                'ci95_low':interval['percentile_95'][0],'ci95_high':interval['percentile_95'][1],'delta_f1':f1['delta_contender_minus_reference'],
                'reference_examples':inventory[a]['examples_seen'],'contender_examples':inventory[b]['examples_seen']})
        for arm in arms:
            a=f'{method}_{arm}_best';b=f'{method}_{arm}_last'
            interval=cluster_interval(rpred[a],rpred[b]);best_last.append({'method':method,'arm':arm,'best':a,'last':b,
                'best_step':rowmap[a]['checkpoint_step'],'last_step':rowmap[b]['checkpoint_step'],
                'best_em':rowmap[a]['macro_em'],'last_em':rowmap[b]['macro_em'],'delta_em_last_minus_best':interval['delta_contender_minus_reference'],
                'ci95_low':interval['percentile_95'][0],'ci95_high':interval['percentile_95'][1],
                'best_ce':rowmap[a]['macro_ce'],'last_ce':rowmap[b]['macro_ce'],
                'same_model_tensors':reading[a]['tensor_sha256']==reading[b]['tensor_sha256']})
    csv_file(tables/'09_data_regime_comparisons.csv',regimes);csv_file(tables/'10_best_vs_last.csv',best_last)
    generic_rows=[];generic_groups=[];loss_intervals=[]
    full_loss_rows=read_jsonl(data/'instruction_test_1000/sft100k_full/test_rows.jsonl')
    import evaluate_sft_models as metrics
    for key,value in generic.items():
        rows=gpred[key];g=value['generation'];t=value['test']
        bylang={lang:[r for r in rows if r['language']==lang] for lang in ['zh','en']}
        macro_rouge=float(np.mean([np.mean([r['reference_overlap_rouge_l_f1'] for r in rs]) for rs in bylang.values()]))
        generic_rows.append({'key':key,'label':label(key),'examples':len(rows),'macro_ce':t['macro_loss'],
            'zh_ce':t['zh']['loss'],'en_ce':t['en']['loss'],'zh_ppl':t['zh']['perplexity'],'en_ppl':t['en']['perplexity'],
            'zh_teacher_token_accuracy':t['zh']['teacher_forced_token_accuracy'],'en_teacher_token_accuracy':t['en']['teacher_forced_token_accuracy'],
            'rouge_overall':g['mean_reference_overlap_rouge_l_f1'],'rouge_macro_language':macro_rouge,
            'eos':g['eos_stop_fraction'],'empty':g['empty_fraction'],'repeat_3gram':g['mean_repeated_3gram_fraction'],
            'exact_reference_match':g['normalized_exact_match_fraction'],
            'constraints_covered':g['supported_constraints']['covered_examples'],
            'supported_constraints_pass':g['supported_constraints']['all_supported_constraints_pass_fraction'],
            'avg_generated_tokens':float(np.mean([r['generated_tokens'] for r in rows]))})
        for field in ['language','task']:
            for name in sorted({r[field] for r in rows}):
                subset=[r for r in rows if r[field]==name]
                generic_groups.append({'key':key,'dimension':field,'group':name,'examples':len(subset),
                    'reference_ce':t[name]['loss'] if field=='language' else t['tasks'][name]['loss'],
                    'rouge':float(np.mean([r['reference_overlap_rouge_l_f1'] for r in subset])),
                    'eos':float(np.mean([r['stop_reason']=='eos' for r in subset])),
                    'repeat_3gram':float(np.mean([r['repeated_3gram_fraction'] for r in subset])),
                    'exact_reference_match':float(np.mean([r['normalized_exact_match'] for r in subset]))})
        if key.startswith('lora'):
            contender=read_jsonl(data/'instruction_test_1000'/key/'test_rows.jsonl')
            interval=metrics.paired_interval(full_loss_rows,contender,seed=20261104,samples=2000)
            loss_intervals.append({'key':key,'delta_ce_lora_minus_full':t['macro_loss']-generic['sft100k_full']['test']['macro_loss'],
                'ci95_low':interval['percentile_95'][0],'ci95_high':interval['percentile_95'][1],
                'fair_epoch_budget':key=='lora100k_r16_historical'})
    csv_file(tables/'11_instruction_1000_all_models.csv',generic_rows);csv_file(tables/'12_instruction_by_language_task.csv',generic_groups)
    csv_file(tables/'13_full_vs_lora_paired_loss.csv',loss_intervals)
    grows={r['key']:r for r in generic_rows}
    history_rows=[{'key':r['key'],'macro_ce':r['test']['macro_loss'],'zh_ce':r['test']['zh']['loss'],'en_ce':r['test']['en']['loss'],
        'retention_ce':r['retention_validation']['macro_loss'],'generated_examples':r['generation']['examples'],
        'eos':r['generation']['eos_stop_fraction'],'checkpoint_step':r['checkpoint_step']} for r in hist['results']]
    csv_file(tables/'14_historical_sft_controls.csv',history_rows)
    probes=[];intrusion=[]
    for suite,values in [('reading_test_2000',reading),('instruction_test_1000',generic)]:
        for key,value in values.items():
            if value['actual_gpu_evaluation_repeated']:
                probe=value['numerical_probe'];probes.append({'suite':suite,'key':key,'examples':probe['examples'],
                    'max_prefill_relative_l2':max(probe['prefill_relative_l2']),
                    'prefill_argmax_matches':sum(probe['prefill_argmax_matches']),
                    'first16_exact_matches':sum(probe['first16_tokens_exactly_match_native'])})
            english=[r for r in predictions[suite][key] if r['language']=='en']
            han=sum(bool(re.search(r'[\u4e00-\u9fff]',r['answer'])) for r in english)
            intrusion.append({'suite':suite,'key':key,'english_examples':len(english),'answers_containing_han':han,
                              'fraction':han/len(english),'not_automatic_semantic_error':True})
    csv_file(tables/'15_cached_native_numerical_probes.csv',probes)
    csv_file(tables/'16_english_answers_containing_han.csv',intrusion)
    cross=read_json(data/'cross_domain_loss/comparison.json')
    if cross['status']!='completed' or len(cross['results'])!=10:raise ValueError('Cross-domain check incomplete')
    cross_results={r['key']:r for r in cross['results']}
    cross_base=read_jsonl(data/'cross_domain_loss/reading_sft/test_rows.jsonl');cross_rows=[]
    for key,value in cross_results.items():
        values=read_jsonl(data/'cross_domain_loss'/key/'test_rows.jsonl')
        ci=metrics.paired_interval(cross_base,values,seed=20261104,samples=2000)
        cross_rows.append({'key':key,'label':label(key),'instruction_macro_ce':value['test']['macro_loss'],
            'instruction_zh_ce':value['test']['zh']['loss'],'instruction_en_ce':value['test']['en']['loss'],
            'reading_macro_em':reading[key]['generation']['macro_exact_match'],
            'delta_instruction_ce_vs_reading_sft':value['test']['macro_loss']-cross_results['reading_sft']['test']['macro_loss'],
            'ce_ci95_low':ci['percentile_95'][0],'ce_ci95_high':ci['percentile_95'][1],
            'examples':len(values),'generation_not_measured_in_this_check':True})
    csv_file(tables/'17_cross_domain_instruction_loss.csv',cross_rows)
    method_pairs=[]
    for arm in arms:
        for a,b in [('ppo','grpo'),('ppo','dpo'),('grpo','dpo')]:
            left=f'{a}_{arm}_best';right=f'{b}_{arm}_best';ci=cluster_interval(rpred[left],rpred[right])
            method_pairs.append({'arm':arm,'reference':left,'contender':right,'delta_em':ci['delta_contender_minus_reference'],
                'ci95_low':ci['percentile_95'][0],'ci95_high':ci['percentile_95'][1],
                'status':'exploratory; no multiple-comparison adjustment; DPO has fewer examples'})
    csv_file(tables/'18_algorithm_pairwise_exploratory.csv',method_pairs)
    lora_controls=[]
    selected_key='lora100k_'+ablation['selected']['key'];selected_rows=read_jsonl(data/'instruction_test_1000'/selected_key/'test_rows.jsonl')
    for key in generic:
        if key.startswith('lora100k_') and key not in [selected_key,'lora100k_r16_historical']:
            ci=metrics.paired_interval(selected_rows,read_jsonl(data/'instruction_test_1000'/key/'test_rows.jsonl'),seed=20261104,samples=2000)
            lora_controls.append({'reference':selected_key,'contender':key,'delta_ce':generic[key]['test']['macro_loss']-generic[selected_key]['test']['macro_loss'],
                'ci95_low':ci['percentile_95'][0],'ci95_high':ci['percentile_95'][1],
                'same_data_epoch_update_budget':True,'test_not_used_to_reselect':True})
    csv_file(tables/'19_lora_rank_dropout_paired_ce.csv',lora_controls)
    sft_controls=[];sft_reference='sft20k_lr5e-5_mb8_ga4_b32_ep1'
    sft_reference_rows=read_jsonl(data/'instruction_test_1000'/sft_reference/'test_rows.jsonl')
    for key in generic:
        if key.startswith('sft20k_') and key!=sft_reference:
            ci=metrics.paired_interval(sft_reference_rows,read_jsonl(data/'instruction_test_1000'/key/'test_rows.jsonl'),seed=20261104,samples=2000)
            sft_controls.append({'reference':sft_reference,'contender':key,'delta_ce':generic[key]['test']['macro_loss']-generic[sft_reference]['test']['macro_loss'],
                'ci95_low':ci['percentile_95'][0],'ci95_high':ci['percentile_95'][1],
                'history_multi_epoch_and_schedule_changed':'stop' in key,'not_used_to_reselect':True})
    csv_file(tables/'20_sft_controls_paired_ce.csv',sft_controls)
    if len(per_item)!=18000:raise ValueError('Expected nine complete 2000-row paired comparisons')
    generate_figures(report/'figures',plan,reading,generic,comparisons,hist,train_rows,curves)
    plt=figure_setup();fig,ax=plt.subplots(figsize=(10,5))
    colors={'ppo':'#275d95','grpo':'#2e867d','dpo':'#c98743'}
    cluster_max=max(r['instruction_macro_ce'] for r in cross_rows if not r['key'].startswith('dpo_'))
    for r in cross_rows:
        key=r['key']
        if key in ['dpo_old_matched_best','dpo_new_matched_best']:continue
        ax.scatter(r['instruction_macro_ce'],r['reading_macro_em']*100,s=55,color=colors.get(key.split('_')[0],'#9a5268'))
        short=label(key).replace(' / best','').replace(' / ','-')
        x,y=r['instruction_macro_ce'],r['reading_macro_em']*100
        if key.startswith('dpo_'):target=(x-.02,y+.22);align='right'
        else:
            shift=.18 if key in ['ppo_old_full_50k_best','ppo_new_matched_best'] else .04
            yshift={'grpo_old_matched_best':.18,'ppo_old_full_50k_best':-.18,'grpo_new_matched_best':.15,
                    'ppo_new_matched_best':.05,'reading_sft':-.18}.get(key,0)
            target=(cluster_max+shift,y+yshift);align='left'
        ax.annotate(short,(x,y),xytext=target,textcoords='data',fontsize=8,ha=align,va='center',
            arrowprops={'arrowstyle':'-','color':'#909090','lw':.6})
    ax.set_xlabel('同一通用指令1,000题 Macro 回答CE（越低越好）');ax.set_ylabel('阅读2,000题 Macro EM（%）')
    ax.set_title('专项阅读表现与通用参考答案建模的取舍；不是综合能力排名')
    fig.tight_layout();fig.savefig(report/'figures/09_cross_domain_tradeoff.png',bbox_inches='tight');plt.close(fig)
    reading_top=max(plan['primary_statistical_family'],key=lambda k:reading[k]['generation']['macro_exact_match'])
    generic_top=min(generic,key=lambda k:generic[k]['test']['macro_loss'])
    selected_lora='lora100k_'+ablation['selected']['key']
    findings={'reading_sft_em':baseline['macro_exact_match'],'highest_observed_reading_best':reading_top,
        'highest_observed_reading_em':reading[reading_top]['generation']['macro_exact_match'],
        'lowest_observed_instruction_ce':generic_top,'lowest_observed_instruction_ce_value':generic[generic_top]['test']['macro_loss'],
        'lora_development_selection_unchanged':selected_lora,'logical_questions':sum(len(v)*plan['suites'][k]['expected_examples'] for k,v in predictions.items()),
        'unique_test_questions':3000,'supplemental_teacher_forced_rows':10000,
        'logical_generated_tokens_with_eos':sum(len(r['generated_ids']) for values in predictions.values() for rows in values.values() for r in rows),
        'unique_actual_model_suite_evaluations':sum(r['actual_gpu_evaluation_repeated'] for rs in summary['results'].values() for r in rs),
        'completed_model_suite_entries':len(reading)+len(generic),'no_training_or_model_selection_performed':True}
    write_json(report/'key_findings.json',findings)
    text=report_text(plan,summary,reading,generic,comparisons,train_rows,rowmap,grows,regimes,best_last,history_rows,
        ablation,historical_lora,historical_scaled,audit,reading_manifest,instruction_manifest,loss_intervals,findings,cross_results)
    text+='\n### 10.1 补充数值与语言诊断\n\n'
    text+=f"实际计算的{len(probes)}个模型×测试集条目中，缓存prefill相对原生前向的最大相对L2为{max(r['max_prefill_relative_l2'] for r in probes):.4%}；{sum(r['first16_exact_matches']<r['examples'] for r in probes)}个条目的三题探针出现至少一次首16 Token序列差异。所有正式比较均使用统一缓存协议，历史原生生成单列；不能据少量探针宣称逐位等价。完整逐模型诊断见附表15。\n\n"
    text+=table(['阅读模型','英文题输出含汉字','英文题数','比例'],[[label(r['key']),r['answers_containing_han'],r['english_examples'],pct(r['fraction'])] for r in intrusion if r['suite']=='reading_test_2000' and r['key'] in main_keys])+'\n'
    text+='含汉字只是可观察的语言混杂标记，英文题出现中文人名等可能合理，不能自动等同错误。结合下方“the足利”等案例，可以定位多语言词表混入和实体错误。\n\n'
    text+='### 10.2 补充跨域保持：同一通用1,000题\n\n'
    text+='此补充计划在执行前另行冻结，固定测试阅读SFT和全部九组验证best，而非只挑阅读测试胜者。只做原生前向参考答案CE，每模型完整1,000题；不生成、不训练、不改变选点。补充共10,000份逐题Loss记录。\n\n'
    text+=table(['模型','通用Macro CE','中文CE','英文CE','相对阅读SFT CE变化','95%配对CE区间'],[
        [r['label'],f(r['instruction_macro_ce']),f(r['instruction_zh_ce']),f(r['instruction_en_ce']),f(r['delta_instruction_ce_vs_reading_sft']),
         f"{r['ce_ci95_low']:+.4f} ～ {r['ce_ci95_high']:+.4f}"] for r in cross_rows])+'\n'
    text+='通用Full100k在相同测试上的CE为'+f(generic['sft100k_full']['test']['macro_loss'])+'；阅读专项SFT为'+f(cross_results['reading_sft']['test']['macro_loss'])+'。参考答案建模在领域迁移后可能退化，但CE并非生成语义准确率，本检查也不等价于完整通用能力评测。\n\n'
    text+='![专项效果与通用回答Loss](figures/09_cross_domain_tradeoff.png)\n\n'
    prior_seconds=sum(r['elapsed_seconds'] for r in summary['prior_attempt_costs'].values());prior_cost=sum(r['estimated_compute_cny'] for r in summary['prior_attempt_costs'].values())
    text+=f"补充保持检查耗时{cross['budget']['elapsed_seconds']/60:.2f}分钟，估算￥{cross['budget']['estimated_compute_cny']:.3f}；先前阅读计算及旧小批/兼容修复尝试已计账{prior_seconds/60:.2f}分钟、￥{prior_cost:.3f}。包含这些已计账尝试，本轮评测合计记录计算时间{(summary['budget']['elapsed_seconds']+cross['budget']['elapsed_seconds']+prior_seconds)/60:.2f}分钟、估算￥{summary['budget']['estimated_compute_cny']+cross['budget']['estimated_compute_cny']+prior_cost:.3f}，均不是平台账单；批量短测与CPU整理开销未逐项计入此合计。\n\n"
    text+='### 10.3 算法间及LoRA更新容量的补充配对对照\n\n'
    text+=table(['同一数据条件','模型变化','阅读EM差/百分点','95%文章群区间/百分点'],[
        [r['arm'],r['reference'].split('_')[0].upper()+'→'+r['contender'].split('_')[0].upper(),pp(r['delta_em']),
         f"{r['ci95_low']*100:+.2f} ～ {r['ci95_high']*100:+.2f}"] for r in method_pairs])+'\n'
    text+='这些算法间比较是探索性补充，没有校正九次相互比较。PPO与GRPO题数相同但每题生成/优化预算不同，DPO题数又较少；因此GRPO观测值最高不等同于在等算力和最优超参数下确定优于PPO。\n\n'
    text+=table(['固定单轮LoRA对照','最终指令CE差（候选−r32/ld0）','95%逐题配对区间'],[
        [label(r['contender']),f(r['delta_ce']),f"{r['ci95_low']:+.4f} ～ {r['ci95_high']:+.4f}"] for r in lora_controls])+'\n'
    text+='LoRA这三项保持数据、1轮、3,125更新和相同学习率/Batch，分别改变rank或dropout。表内差异用于检验已锁定开发选择的外部表现，不回头用最终测试改选超参数。\n\n'
    text+='### 10.4 任务分布中的Full与LoRA\n\n'
    task_keys=['sft100k_full','lora100k_r16_historical','lora100k_r8_ld0',selected_lora]
    text+=table(['既有任务标签','题数']+[label(k)+' CE' for k in task_keys],[
        [task,generic['sft100k_full']['test']['tasks'][task]['examples']]+[f(generic[k]['test']['tasks'][task]['loss']) for k in task_keys]
        for task in sorted(generic['sft100k_full']['test']['tasks'])])+'\n'
    text+='任务CE按该任务真实监督Token汇总，并非中英文等权；任务标签、语言和来源分布相关，例如改写/摘要以中文为主。它能定位模型差异集中在哪里，不能直接比较不同任务的固有难度；生成的逐任务参考重叠、EOS和重复情况见附表12。\n\n'
    text+='### 10.5 SFT参数对照的独立测试配对区间\n\n'
    text+='参考为同19,991条、真正单轮、LR5e-5、B32。以下变化用于核查历史控制变量结果在冻结指令测试上的表现；不再用这次测试改选学习率或Batch。\n\n'
    text+=table(['候选配置','候选−参考CE','95%语言分层配对区间'],[
        [label(r['contender']),f(r['delta_ce']),f"{r['ci95_low']:+.4f} ～ {r['ci95_high']:+.4f}"] for r in sft_controls])+'\n'
    text+='历史多轮行同时改变数据遍历量与调度轨迹，不能作为纯粹单个epoch变量效应；区间仅为当前权重的测试抽样不确定性，未做多重校正。\n\n'
    text+=case_studies(report,rpred,gpred,selected_lora)
    text+=reproduction_appendix(plan,summary,train_rows,audit,data,report)
    (report/'完整测试报告.md').write_text(text,encoding='utf-8')
    (report/'完整测试报告.html').write_text(markdown_html(text,report),encoding='utf-8')
    (report.parent/'README.md').write_text('# 2026-11-04 后训练完整测试\n\n主报告：[完整测试报告](report/完整测试报告.html)。\n\n'
        '已完成：34份Checkpoint、37个模型×测试条目；阅读每份2000题、指令每份1000题，另10个阅读策略各1000题通用CE保持。\n\n'
        'report/ 含完整HTML/Markdown、SFT/LoRA/RL分报告、20张CSV、9张图及案例分析；results_local/为全量逐题原始结果；sources_local/为历史依据及配置。\n\n'
        '阅读Batch16逐文件校验复用v2；全部15个通用模型按Batch128统一从头生成。原尝试路径、参数和摘要见plan及reading_reuse_receipt。\n\n'
        '本地目录不含模型权重或W&B备份。\n',encoding='utf-8')
    def sections(numbers):
        parts=[]
        for number in numbers:
            match=re.search(r'^## '+str(number)+r'\..*?(?=^## |\Z)',text,re.M|re.S)
            if match:parts.append(match.group(0))
        return '\n'.join(parts)
    for name,numbers in [('SFT',[1,2,3,4,7,9,10,11,12]),('LoRA',[1,2,3,4,8,9,10,11,12]),('RL',[1,2,3,4,5,6,9,10,11,12])]:
        standalone='# '+name+' 详细测试报告\n\n日期：2026-11-04。本文由完整报告提取，数据、条件和结论保持一致。代表性案例完整文本见同目录完整报告及case_studies.json。\n\n'+sections(numbers)
        (report/(name+'测试报告.md')).write_text(standalone,encoding='utf-8')
    write_json(report/'report_manifest.json',{'plan_identity':plan['plan_identity'],'evaluator_sha256':summary['source_sha256'],
        'builder_sha256':sha256_file(__file__),'report_date':'2026-11-04','key_findings':findings,
        'files':[{'path':str(p.relative_to(report)),'bytes':p.stat().st_size,'sha256':sha256_file(p)} for p in sorted(report.rglob('*')) if p.is_file() and p.name!='report_manifest.json']})
    print(json.dumps({'report':str(report.resolve()),'findings':findings,'markdown_characters':len(text)},ensure_ascii=False))


def report_text(plan,summary,reading,generic,comparisons,train_rows,rowmap,grows,regimes,best_last,history_rows,
                ablation,historical_lora,historical_scaled,audit,reading_manifest,instruction_manifest,loss_intervals,findings,cross_domain):
    chunks=[]
    def add(text):chunks.append(text.strip()+'\n\n')
    base=reading['reading_sft']['generation'];top=findings['highest_observed_reading_best']
    selected='lora100k_'+ablation['selected']['key'];cost=summary['budget']['estimated_compute_cny']
    add('# 300M 模型后训练完整测试报告')
    add('日期：2026-11-04（北京时间）。范围：同一自研 303,350,784 参数 Decoder 的全参数 SFT、LoRA、阅读专项 SFT、DPO、PPO+RLVR、GRPO+RLVR。报告依据本次真实远程 GPU 测试和已存在的历史逐题记录生成；不以训练 Loss 或开发集指标替代最终测试。')
    add('## 1. 主要结果与适用范围')
    add(f"本次完成 **{len(plan['models'])} 份 Checkpoint、{findings['completed_model_suite_entries']} 个模型×测试集条目**。阅读测试每份 2,000 题，通用指令测试每份 1,000 题，共 {findings['logical_questions']:,} 条完整逐题输出。两套测试共有3,000个不同题ID，59,000是跨模型重复同题的比较次数，不是59,000道新题；补充10,000份通用CE记录同样使用这1,000题。模型张量完全相同的条目复用经摘要验证的结果，最终统计中实际执行 {findings['unique_actual_model_suite_evaluations']} 次不同模型×测试集计算；所有逻辑条目均有完整逐题文件。")
    add(f"阅读 SFT 基线 Macro EM 为 **{pct(base['macro_exact_match'])}**。九组验证集选定的 RL best 中，当前测试观测值最高的是 **{label(top)}**，Macro EM 为 **{pct(reading[top]['generation']['macro_exact_match'])}**，相对 SFT 为 **{pp(reading[top]['generation']['macro_exact_match']-base['macro_exact_match'])} 个百分点**。这是冻结权重在该测试分布上的观测排名，不据此重新挑选 Checkpoint 或声称算法具有普遍优势。")
    add(f"通用指令 1,000 题最低的观测 Macro 回答 CE 来自 **{label(findings['lowest_observed_instruction_ce'])}**，数值 **{f(findings['lowest_observed_instruction_ce_value'])}**。新 LoRA 的事前开发集选型仍是 **{label(selected)}，alpha=64、LR=2e-4、epochs=1**；本次测试不会改写原选型。")
    add(f"同一通用1,000题的保持检查中，Full100k CE为{f(generic['sft100k_full']['test']['macro_loss'])}，阅读专项SFT为{f(cross_domain['reading_sft']['test']['macro_loss'])}，阅读最高观测best的GRPO旧全量为{f(cross_domain[top]['test']['macro_loss'])}。因此专项阅读分数提升伴随通用参考分布建模的取舍，不能总结为全能力同时上升；这项CE检查本身也不等价于生成语义正确率。")
    add('两组匹配 DPO 的 best 均停留在 step 0；本次模型张量摘要核验确认其与阅读 SFT 起点相同。因此其 best 测试分数是起点成绩，不能写成“DPO 训练后获得该提升”。训练终点 last 的结果在第 6 节完整列出。')
    add('结论范围仅覆盖给定材料的抽取式阅读理解，以及指令参考答案建模和有限格式诊断。没有测试通用数学证明、编程、长上下文、多轮对话、实时知识或可靠的综合推理能力；RLVR 在这里指可验证奖励设置，PPO/GRPO 是采用该奖励的优化算法，并非另一个独立模型类别。')
    add('## 2. 模型与训练链路')
    add('共同的预训练 Base → 通用 Full SFT 100k（历史实际两轮）→ 阅读专项 Full SFT 50k（单轮）→ 分别独立初始化九组 RL。LoRA 实验则从同一预训练 Base 注入 Adapter；不把前一算法结果接给后一算法。所有本次新增操作均为只读推理和统计，不增加训练轮数，不修改原冻结配置、best/last、优化器状态或 W&B 训练身份。')
    add(table(['模型组','训练数据与更新量','初始化/比较条件','本次测试'],[
        ['预训练 Base','历史约 3B Token；所测为原 best','303,350,784 参数，共用本项目分词器','阅读2,000＋指令1,000'],
        ['通用 Full SFT 20k ×8','19,991条；历史LR/Batch/轮数对照','同一预训练Base；完整保留历史1/2/3轮身份','新指令1,000＋原历史1,000 CE/64生成'],
        ['通用 Full SFT 100k','100,000条；历史实际2轮，共6,250更新','共享3轮调度的2轮停止点；不是新单轮对照','阅读2,000＋指令1,000'],
        ['历史 LoRA r16','100,000条；实际2轮，共6,250更新','同Base、同100k数据；历史调度与Full匹配','指令1,000；核查既有1,000题结果'],
        ['新 LoRA ×4','r8/a16、r32/a64；r32 dropout0/0.05/0.1；各1轮','LR2e-4、B32、3,125更新；开发集选型已锁定','全部各1,000题；选定r32另测阅读2,000'],
        ['阅读 SFT 50k','50,000条，1轮、1,563更新','从通用Full100k初始化；所有RL的共同起点','完整阅读2,000'],
        ['PPO / GRPO / DPO 各3组','旧匹配、新匹配、旧全量；均数据1轮','分别从同一阅读SFT best独立初始化','每组best与last各2,000题']]))
    add('原本未完成的两组 r16 学习率筛选（状态 paused）、在线初始化失败且没有权重的 r8 尝试，以及模拟中断记录，不列为完成的正式模型。报告将其留在原始 inventory 供核查；本次不会续训这些历史任务。')
    add('### 2.1 RL 参数和数据量')
    add(table(['设置','PPO+RLVR','GRPO+RLVR','DPO'],[
        ['actor学习率','1e-6','1e-6','5e-6'],['每批问题数/反向微批','16/16','16/16','16/16'],
        ['每题策略回答','1','4','初始SFT离线生成并筛选负例'],['每批内部更新','2','1','1'],
        ['数据遍历/stop_epochs','1 / null','1 / null','1 / null'],['回答/上下文','64 / 1024 Token','64 / 1024 Token','64 / 1024 Token'],
        ['特有系数','clip0.2、KL0.02、线性value head','clip0.2、KL0.02、组归一化优势','beta0.1'],
        ['旧/新匹配的实际题数','25,000 / 25,000','25,000 / 25,000','20,462 / 20,462'],
        ['旧全量实际题数','50,000','50,000','40,871合格偏好对'],
        ['旧/新匹配optimizer更新','3,126 / 3,126','1,563 / 1,563','1,279 / 1,279'],
        ['旧全量optimizer更新','6,250','3,125','2,555']]))
    add('因此“都训练一轮”不代表算力、生成次数、Token 数或更新次数相同。GRPO 每题四个回答，PPO 每个 rollout 两次内部优化；DPO 筛选后题数减少。跨算法结果是当前已训练配置的比较，不是固定所有超参数及计算量之后的算法定理。')
    add('## 3. 测试集、来源与隔离证据')
    add('### 3.1 阅读理解最终测试：2,000题')
    stat=reading_manifest['statistics']['test']
    add(table(['项目','实际值'],[
        ['中文/英文','1,000 / 1,000题'],['来源','DRCD 500、CMRC2018 500、SQuAD 1,000'],
        ['文章分组/独立材料数',f"{stat['articles']}个文章分组；本次精确审计840个规范化材料"],
        ['回答监督Token（含EOS）',f"中文{stat['answer_tokens']['zh']:,}、英文{stat['answer_tokens']['en']:,}；中文占{pct(stat['zh_answer_token_fraction'])}"],
        ['最大提示/参考回答',f"{stat['max_prompt_tokens']} / {stat['max_answer_tokens_with_eos']} Token（回答含EOS）"],
        ['中文规范','DRCD原始繁体的材料、问题、答案联合转简体并重校跨度；其余中文按发布规范验收'],
        ['选题规则','事前按来源、文章组抽取；不依赖模型测试得分'],['用途','从未进入这九组训练或best选点；开发集256题与最终测试分开']]))
    add('来源使用已冻结的官方评测文件：DRCD dev/test、CMRC2018 dev、SQuAD validation 中预先隔离的文章分组。源文件 revision、URL、SHA256 和许可声明保存在随包 data_manifests 的 origin 清单；本报告不把自定义指标冒充各官方排行榜成绩。')
    add('旧训练组来自阅读 SFT 的 DRCD/CMRC/SQuAD 训练来源，新组来自 DuReader-robust 与 Quoref。最终测试仍来自前一类来源，所以它能够比较本项目目标分布上的效果，但不是来源中立的“旧/新语料质量”裁判。')
    add('### 3.2 通用指令最终测试：1,000题')
    st=instruction_manifest['statistics']
    add(table(['维度','分布'],[
        ['样本数','中文743，英文257'],['回答监督Token','中文57,107，英文57,118；按Token近似均衡，题数不均衡'],
        ['任务标签','改写235、摘要235、基于材料问答177、格式177、一般问答176'],
        ['主要来源','smoltalk_chinese 699、alpaca_gpt4_zh 44、四个smol_smoltalk来源共257'],
        ['最大输入','512 Token；每题完整提示保留，生成预算256'],
        ['质量边界','多为已筛选指令参考回答，并非所有回答都有人类逐题正确性裁决；任务标签含启发式判断']]))
    add('该测试在 LoRA 调参前冻结。历史 r16 与 Full100k 已在此做过每模型1,000题完整生成，本次先核验原 comparison.json，再以统一缓存协议复测并补齐新r8/r32四组及SFT历史候选。新rank/dropout的配置选择只来自开发集2,000条，未因本次测试重新调整。')
    add('这是复用既有冻结测试，不是新抽取的未曾打开测试集。自动rank/dropout选型代码不读取其内容和得分，但历史r16结果已经对开发者暴露，不能证明所有人工决策都绝对未受历史结果影响。该局限与未进入梯度训练的精确隔离证据分别披露。')
    add('不能把“中文743题、英文257题”的总体生成均值当作语言等权结果。因此表格同时给出总体参考ROUGE-L与中英文等权Macro ROUGE-L；CE一律先按各语言监督Token汇总，再对两种语言取等权均值。')
    add('本次逐文抽查还发现具体质量局限：要求只用miss/misses两词的题，其参考却用中文解释；一条标为grounded_qa的样本只给材料等待后续提问；扬州改写参考混入“easiliy”。这些都说明格式/去重/隔离验收不等于逐题语义无误。本次保留原冻结数据与评分并在第11节完整展示，不做看过模型输出之后的回填或清洗；不能以这些非随机案例估计整个测试集坏样本比例。')
    add('### 3.3 本次精确隔离复核')
    isol=[]
    for arm,value in audit['reading_arms'].items():
        o=value['source_vs_final_test_exact_isolation'];isol.append([arm,o['question']['overlapping_keys'],o['context']['overlapping_keys'],o['group_id']['overlapping_keys']])
    add(table(['阅读训练来源','重叠问题键','重叠材料键','重叠组键'],isol))
    add(table(['通用训练来源','训练题数','与最终1,000题重叠的规范化提示'],[[Path(r['manifest']).parent.name,r['examples'],r['vs_final_test']['prompt']['overlapping_keys']] for r in audit['instruction_training']]))
    add('阅读旧匹配组审计其整个50k父来源，所以也覆盖其25k子集；DPO偏好题来自这些父来源及其批准的筛选子集。审计验证发布split的SHA和行数后复核已知精确键。它不证明没有近义改写、翻译等语义重叠，也不证明3B预训练语料没有包含公开测试材料。分组标识受来源定义影响，跨来源语义相同的文章仍可能未被精确键发现。')
    add('### 3.4 DPO数据的实际边界')
    dpo_stats=[]
    for arm,value in audit['reading_arms'].items():
        pref=value.get('dpo_preferences',{});dpo_stats.append([arm,pref.get('records',{}).get('examples',pref.get('statistics',{}).get('examples','见manifest')),pref.get('statistics',{})])
    add('DPO chosen 是原人工金标准；rejected 是同一初始阅读SFT生成后由保守规则接收的负回答，不是额外人工成对偏好。规则会排除缺失EOS、空答、特殊Token、疑似别名/包含关系及高重叠回答；“不是原文跨度”的格式负例不等同于经人工证实的语义错误。')
    add('采用用户批准的现有候选：旧匹配20,462对，新匹配20,462对，旧全量40,871对；停止补齐和额外采样。旧/新匹配中中文题均8,256、英文均12,206；对应中文答案监督Token占比约44.58%与44.71%，并非严格50%。没有复制题目或引入后来取消的补充候选。')
    add('## 4. 评测实现和指标含义')
    add('### 4.1 可复现的推理条件')
    add('远程单张 vGPU-48GB，CUDA BF16，seed42，TF32关闭，eval模式、无梯度、无采样、无重复惩罚。阅读生成批大小16、通用指令生成批大小128，按完整提示Token长度排序；每题独立EOS停止。阅读最多64个新Token、保留所有Token候选；通用指令最多256个新Token，仅禁止PAD/UNK，允许EOS。原生CE前向微批固定8。')
    add('阅读22条模型记录完整结束后逐文件校验复用；通用部分在实测Batch128对相同128题短测约4.48倍加速、峰值约6.11GiB后，另冻结新协议并对全部15模型从头统一生成。旧小批通用尝试保留且不混入本次比较。短测首32 Token序列119/128完全一致，其余体现BF16批大小/贪心近似差异，因此没有把不同批大小的通用结果混排，也不声称逐位等价。')
    add('各模型使用同一测试ID集合和分词器。Prompt、system、分词边界与既有模板一致，没有把gold作为标注或助手续写额外拼入生成前缀；材料本身含有答案是抽取式任务的正常设置。原始Decoder前向计算teacher-forced损失；只读KV适配器加速逐Token生成。每份真实模型做3题、16 Token数值诊断，记录BF16缓存/原始前向的相对L2与首Token/生成一致性。BF16不同批次或kernel不保证逐位一致；本次正式排名只使用同一新协议，历史生成单列。')
    add('CPU的变长批解码、PAD/UNK禁止、原模型张量不变、样本ID配对、缺题拒绝及Holm校正检查在本地和远程均3项通过；这是评测正确性证据。模型效果的证据则来自完整GPU逐题输出，两者分别报告。')
    add('### 4.2 回答损失与Perplexity')
    add('对于每题，先将完整提示与参考回答+EOS拼接并错位一次：提示与PAD目标为-100，回答和EOS计入交叉熵。语言Loss = 该语言所有回答目标的负对数概率之和 / 真实监督Token数；Macro CE = (CE_zh + CE_en)/2。各语言PPL = exp(CE_language)，不把两种语言的PPL直接当作能力评分。')
    add('teacher-forced Token准确率是在给定真实前文条件下预测目标Token的比例；它与模型自行生成完整正确答案的概率不同。参考答案Loss降低通常表示更贴近该答案分布，不能单独证明事实正确、约束遵循或推理能力提高。')
    add('### 4.3 阅读EM、F1和错误归类')
    add('EM对生成答案与发布的全部人工别名分别做NFKC、大小写折叠、空白与少量外层引号/句尾符号规范化，再判断是否完整相等。保留小数点、负号、C++和单位等有意义标点，不奖励简单复制整段材料。Macro EM对中英文得分等权；本测试题数恰好相等，因此也等于总体EM。')
    add('F1使用项目声明的中文字符/词片段及英文空格分词，与任一金标准取最大重叠F1；不是官方SQuAD或CMRC评分器。EM=0可能是错误，也可能是金标准别名不全、正确但带解释或边界差异。参考重叠与原文跨度判断仅用于诊断，不以程序标签冒充完整人工语义判断。')
    add('错误类别按固定优先级：correct → empty → special_token → length_limit → partial_overlap → wrong_material_span → not_a_material_span。它们互斥，便于计数；例如部分重叠也可能是错误实体或答案过长，不能只凭类别判断病因。')
    add('### 4.4 通用生成与统计区间')
    add('ROUGE-L F1是中文规范化字符/英文单词的最长公共子序列参考重叠；参考完全匹配只是字符串关系，不是通用任务准确率。EOS率、空答率、平均输出Token和重复3gram比例用于解释生成行为；重复3gram也会统计正常短语复现，不能直接当成死循环发生率。格式约束只评分可解析的显式规则，并给出覆盖题数，未覆盖的要求不会自动算通过。')
    add('阅读主要差异区间：同一道题配对、按语言分层、按原文章group_id整群重采样2,000次、固定seed20261104，报告95%百分位区间。这样同文章多道题会一起抽样，避免把2,000题全部当作相互独立。主比较家族事前固定为九组best分别对阅读SFT；给出题级exact McNemar及九比较Holm校正。题级检验的独立性假设对共享文章可能不成立，所以另附文章整组符号交换检验10,000次及其Holm结果，明确标为探索性敏感性分析。')
    add('区间未对九组同时比较作校正，只反映当前模型权重下的测试抽样不确定性，不包含训练seed、预训练数据或超参数选择的方差；不能据单次单seed实验声称普遍显著。通用CE配对区间沿用语言分层的逐题重采样，原辅助集分组/近似重复仍是局限。')
    add('## 5. 阅读SFT与九组RL：主要最终测试')
    keys=['reading_sft']+[f'{m}_{a}_best' for m in ['ppo','grpo','dpo'] for a in ['old_matched','new_matched','old_full_50k']]
    add(table(['验证选定模型','步数','Macro EM','中文EM','英文EM','Macro F1','EOS','回答CE'],[
        [label(k),rowmap[k]['checkpoint_step'],pct(rowmap[k]['macro_em']),pct(rowmap[k]['zh_em']),pct(rowmap[k]['en_em']),f(rowmap[k]['macro_f1']),pct(rowmap[k]['eos']),f(rowmap[k]['macro_ce'])] for k in keys]))
    add('![验证选定模型的阅读测试](figures/01_reading_best.png)')
    add('### 5.1 相对阅读SFT的配对差异')
    add(table(['模型','EM变化/百分点','95%文章群区间/百分点','新增正确/丢失正确','McNemar Holm p','探索性文章群Holm p'],[
        [label(r['key']).replace(' / best',''),pp(r['em_interval']['delta_contender_minus_reference']),compact_ci(r['em_interval']),
         f"{r['mcnemar']['gained_correct']} / {r['mcnemar']['lost_correct']}",pvalue(r['mcnemar_holm_p']),pvalue(r['exploratory_cluster_holm_p'])] for r in comparisons]))
    add('![阅读配对变化及区间](figures/02_paired_delta.png)')
    positive=[r for r in comparisons if r['em_interval']['percentile_95'][0]>0]
    negative=[r for r in comparisons if r['em_interval']['percentile_95'][1]<0]
    crossing=[r for r in comparisons if r not in positive and r not in negative]
    add(f"九组中，{len(positive)}组的描述性95%文章群区间完全在0以上，{len(negative)}组在0以下，其余{len(crossing)}组包含0或为相同模型的零区间。区间与经多重校正的p值应一起阅读；零区间的step0 DPO不是极高确定性的学习成功，而是同一组模型张量的结果复用。")
    add('阅读优化目标会改变整个答案分布，因此EM上升而参考CE略升可以同时出现：前者只看贪心第一答案是否完全匹配，后者度量全部参考目标Token的对数概率。不能用CE单一指标否定EM变化，也不能用EM改善掩盖概率校准或部分匹配质量下降。')
    add('### 5.2 旧/新语料和数据量对照')
    add(table(['算法','对照','实际训练题数','EM变化/百分点','95%文章群区间/百分点','F1变化'],[
        [r['method'].upper(),r['comparison'],f"{r['reference_examples']:,} → {r['contender_examples']:,}",pp(r['delta_em']),f"{r['ci95_low']*100:+.2f} ～ {r['ci95_high']*100:+.2f}",f(r['delta_f1'])] for r in regimes]))
    add('这六项为补充描述性比较，没有另行做多重校正。旧→新同时改变来源、文章长度、答案特征与训练题是否已在领域SFT见过；旧匹配→旧全量同时改变样本量、更新量和训练算力。不能分别简化为“纯粹去重的因果效果”或“只增加数据的唯一影响”。DPO还因保守筛选与其他算法的实际题数不同，必须保留20,462/40,871这一事实。')
    add('### 5.3 分来源结果')
    source_names=['DRCKnowledgeTeam/DRCD','ymcui/cmrc2018','rajpurkar/squad']
    add(table(['模型','DRCD 500题 EM','CMRC 500题 EM','SQuAD 1,000题 EM'],[
        [label(k)]+[pct(reading[k]['generation']['by_source'][s]['exact_match']) for s in source_names] for k in keys]))
    add('按来源表能识别总体提升是否集中于一个数据源；500题子集的不确定性通常比总体更大。中文由两个来源混合，英文只有SQuAD，因此语言差异也包含来源与难度差异，不能解释成纯语言能力差距。完整分语言F1、来源F1、按提示长度、按问题类型和金标准答案长度的结果见CSV附表。')
    add('## 6. best与单轮终点last：稳定性和退化')
    add(table(['算法/数据','best步→last步','best EM','last EM','last−best/百分点','best CE→last CE','张量相同'],[
        [r['method'].upper()+'/'+r['arm'],f"{r['best_step']} → {r['last_step']}",pct(r['best_em']),pct(r['last_em']),pp(r['delta_em_last_minus_best']),f"{f(r['best_ce'])} → {f(r['last_ce'])}",'是' if r['same_model_tensors'] else '否'] for r in best_last]))
    add('![验证最优与单轮终点比较](figures/03_best_last.png)')
    add('这些last结果是预先声明的终点诊断，不拿最终测试在best与last之间重新选择。即便last在最终测试上略高，也保留原验证集best选择；否则就会让最终测试参与选模。对两组step0 DPO，best的张量与SFT起点完全相同，last才体现本次DPO更新的实际终点。')
    add('![256题开发集训练轨迹](figures/07_validation_curves.png)')
    add('DPO可能在偏好Loss下降、偏好margin扩大时，贪心阅读EM或F1下降。这一现象与“提高chosen相对rejected的分数”不保证在所有候选中生成正确跨度相容。当前可疑因素包括规则负例中的格式差异、答案长度/边界、学习率、数据分布和偏好与任务目标的错位；本实验没有独立干预去验证哪一项是主因，报告不作单因果归因。')
    add('旧全量DPO在256题开发集的best EM从初始69/256提高到70/256，仅多对1题；其开发F1却由约0.45485降至0.40644。事前规则先比较EM、再以F1打破同分，因此选择step1300符合原规则，却暴露了小开发集和字典序选点的噪声。本次最终测试上它低于阅读SFT；后续可在未用的开发数据上扩大评测或预设最小改进门槛，而不以这次测试改写历史best。')
    add('单轮也可能出现开发集或测试性能退化；epochs=1不是无过拟合保证。更稳妥的后续实验是仍保持单轮，用独立开发集调整学习率、KL/beta、偏好过滤和数据覆盖，不在最终测试上反复选Checkpoint，也不自动追加第二轮。')
    add('## 7. 全参数SFT详细测试')
    add('### 7.1 原19,991条参数控制实验')
    add('下面保留原历史测试结果：每模型1,000条参考答案CE、只抽64条生成。该历史主测试曾被明确用于选配置，因此属于已经暴露的选型结果；本次统一指令1,000题完整生成另列于7.3/8节，不把这64题写成1,000题生成。')
    add(table(['历史配置','best步','中文CE','英文CE','Macro CE','生成题数','EOS','保持CE'],[
        [r['key'],r['checkpoint_step'],f(r['zh_ce']),f(r['en_ce']),f(r['macro_ce']),r['generated_examples'],pct(r['eos']),f(r['retention_ce'])] for r in history_rows]))
    add('![历史学习率及Batch控制](figures/06_sft_controls.png)')
    add('相同单轮、B32下，学习率1e-5→2e-5→5e-5→1e-4的历史CE逐步下降；这说明在该预算内更大学习率更快适应该参考分布，并不证明继续增大学习率一定更优。相同数据量、LR5e-5下，有效Batch8/16/32对应更新2,499/1,250/625步；比较保持了一轮题量，但更新次数不同。')
    add('历史两轮/三轮实验使用共享三轮调度，best实际为1,250与1,200步，标签里的“3轮”是实验预算而非所测best已经训练三轮。它们的历史Macro CE仅差约0.00318，不能凭这一小差距声称广泛能力有确定差别。当前所有新训练仍遵守真正单轮配置；本报告展示历史事实，不会恢复多轮默认值。')
    add('### 7.2 从约20k扩充到100k的历史独立辅助测试')
    add(table(['历史模型','辅助1,000题 CE','已用于选型的旧测试 CE','辅助生成题数','辅助EOS'],[
        [r['key'],f(r['auxiliary_test']['macro_loss']),f(r['primary_test_already_used_for_selection']['macro_loss']),r['generation']['examples'],pct(r['generation']['eos_stop_fraction'])] for r in historical_scaled['results']]))
    add('数据扩充后辅助分布CE降低，而原主测试略有回升；数据量、任务组成和长度分布共同变化，不能当成纯样本数控制实验。该辅助测试生成也只有64条，本次的统一新1,000题完整生成补充了更大样本的生成行为证据。')
    add('### 7.3 本次统一1,000题：所有已完成Full SFT与Base')
    sftkeys=[k for k in generic if k.startswith('sft') or k=='pretrained_base']
    add(table(['模型','Macro CE','中文CE','英文CE','Macro ROUGE-L','总体EOS','重复3gram','生成题数'],[
        [label(k),f(grows[k]['macro_ce']),f(grows[k]['zh_ce']),f(grows[k]['en_ce']),f(grows[k]['rouge_macro_language']),pct(grows[k]['eos']),pct(grows[k]['repeat_3gram']),1000] for k in sftkeys]))
    add('本表与原历史测试的样本分布不同，绝对Loss不宜跨表直接相减；同一表内所有模型接收同一批题。历史被选配置不会按本次测试重新排名后替换。SFT100k、Base和新选定LoRA也在阅读测试中作迁移基线；其通用训练没有阅读专项50k阶段。')
    add(table(['迁移基线','阅读Macro EM','阅读F1','阅读回答CE'],[
        [label(k),pct(reading[k]['generation']['macro_exact_match']),f(reading[k]['generation']['macro_f1']),f(reading[k]['test']['macro_loss'])] for k in ['pretrained_base','sft100k_full',selected,'reading_sft']]))
    add('## 8. LoRA详细测试与Full方法比较')
    add('### 8.1 是否已经测试、这次补测什么')
    add('已核实历史r16学习率筛选胜者与Full100k的comparison：status=completed、generated_examples_per_model=1000、final_test_was_not_used_for_selection=true，双方均保留1,000条generations与test_rows。因此“LoRA从未跑最终测试”不适用于历史r16；缺失的是后来新增r8/r32 rank/dropout实验的最终生成评测。此次四组均完整补齐1,000题。')
    add('### 8.2 开发集选型与最终测试分开')
    unique={r['key']:r for r in ablation['rank_results']+ablation['dropout_results']}
    add(table(['新LoRA配置','可训练参数','开发Macro CE（选型）','主验证CE','本次最终CE','最终Macro ROUGE-L','最终EOS'],[
        [r['key'],f"{r['adapter']['trainable_parameters']:,}",f(r['development_loss']),f(r['primary_validation_loss']),
         f(grows['lora100k_'+r['key']]['macro_ce']),f(grows['lora100k_'+r['key']]['rouge_macro_language']),pct(grows['lora100k_'+r['key']]['eos'])] for r in unique.values()]))
    add('四组均固定同一Base、100k训练集、LR2e-4、B32、3,125更新、真正1轮；目标为168个attention/MLP Linear层。r8约3,489,792个可训练参数，占Base约1.15%；r32约13,959,168，占约4.60%。rank32相对rank8的开发CE改善约0.08724；dropout0/0.05/0.1之间开发CE差仅约0.00159～0.00361，小差距需要结合最终指标与不确定性解读。')
    add('这些是新增可训练Adapter参数，LoRA总参数为303,350,784底座加Adapter；不能把“只训练约4.60%底座规模的参数”写成“整个模型只有约14M参数”。PPO训练还带独立线性value head，推理比较只使用actor，不把它当另一个完整303M critic。')
    add('### 8.3 同一1,000题上的Full与LoRA')
    lorakeys=[k for k in generic if k.startswith('lora')]
    add(table(['模型','Macro CE','相对Full CE变化','95%逐题配对CE区间','实际数据遍历'],[
        [label(k),f(grows[k]['macro_ce']),f(next(r for r in loss_intervals if r['key']==k)['delta_ce_lora_minus_full']),
         f"{next(r for r in loss_intervals if r['key']==k)['ci95_low']:+.4f} ～ {next(r for r in loss_intervals if r['key']==k)['ci95_high']:+.4f}",
         '历史2轮' if k=='lora100k_r16_historical' else '1轮'] for k in lorakeys]))
    add('正CE差表示该LoRA对参考答案的建模损失高于Full。历史r16与Full都实际遍历两轮、同100k数据，适合作为已有训练方案间的比较；新r8/r32只遍历一轮，Full100k为历史两轮，二者不构成等训练预算的Full-vs-LoRA单变量实验。此处不额外训练一个Full单轮模型补齐条件，也不拿不匹配的条件给方法下普遍优劣结论。')
    show={r['key']:r for r in train_rows}
    add(table(['存储与更新规模','可训练参数','best文件/MiB','可恢复last/MiB'],[
        [label(k),f"{show[k]['trainable_parameters']:,}",f(show[k]['best_checkpoint_mib'],2),f(show[k]['last_checkpoint_mib'],2)]
        for k in ['sft100k_full','lora100k_r16_historical','lora100k_r8_ld0',selected]]))
    add('LoRA的best仅保存Adapter，重建推理仍依赖原Base及其身份；Full的best保存完整模型。少训练参数能显著减少优化器和Checkpoint开销，但Base前向及传播仍有成本，不能仅凭参数比例承诺同倍数训练加速。')
    add('原历史相同测试的Full CE为'+f(historical_lora['results'][0]['test']['macro_loss'])+'，r16 CE为'+f(historical_lora['results'][1]['test']['macro_loss'])+'，差'+f(historical_lora['delta_lora_minus_full'])+'。历史原始生成使用全前缀重算，本次使用已记录数值诊断的统一KV协议；teacher-forced CE仍由原生前向计算。')
    add('![统一指令测试回答Loss](figures/04_instruction_loss.png)')
    add('![统一指令生成行为](figures/05_instruction_generation.png)')
    add('### 8.4 生成行为与格式覆盖')
    add(table(['模型','总体ROUGE-L','语言等权ROUGE-L','EOS','空答','重复3gram','格式覆盖题数','覆盖要求通过率'],[
        [label(k),f(grows[k]['rouge_overall']),f(grows[k]['rouge_macro_language']),pct(grows[k]['eos']),pct(grows[k]['empty']),pct(grows[k]['repeat_3gram']),grows[k]['constraints_covered'],pct(grows[k]['supported_constraints_pass'])] for k in ['sft100k_full']+lorakeys]))
    add('格式通过率的分母只包括能够被当前规则解析的题，不是1,000题综合指令准确率；任务标签为format_following也可能包含规则不支持的约束。ROUGE较高与重复较少都不能证明完整正确，改写/摘要尤其需要人工语义与忠实度审查。逐语言、逐任务分项及所有输出保存在CSV/JSONL，不只展示均值。')
    add('## 9. 能力保持、成本与实验质量')
    add('### 9.1 预训练能力保持的有限证据')
    add('![历史固定小窗口保持Loss](figures/08_retention.png)')
    add('保持Loss来自历史固定预训练validation小窗口：每语言2批、批大小8、每条512 Token，即各8,192个目标Token。它不是完整预训练测试集，也不是通用知识正确率。图中部分为训练终点记录，并非对应best的独立测试；原历史SFT对照表则保存当时每个best的保持结果。不要把两类位置混用来选模。')
    add('历史Full100k保持Macro CE约4.9241，预训练Base原比较约3.2655，提示有分布保持退化；新单轮LoRA的保持Loss更低时，也不能归因为方法本身更优，因为训练轮数与可训练参数不同。本次另外补测阅读SFT与九组RL best在同一通用1,000题上的参考CE，见10.2；没有对全部RL进行通用生成语义评审，不能据阅读EM改进声称全能力提升。')
    add('### 9.2 训练与测试成本')
    rows=[r for r in train_rows if r['stage'] in ['reading_rl','reading_sft','lora','sft_100k']]
    add(table(['训练方案','记录计算时间/分钟','估算GPU费用/元','最终更新','本次所测best'],[
        [r['label'],f(r['recorded_compute_seconds_all_sessions']/60,2) if r['recorded_compute_seconds_all_sessions'] is not None else '—',
         f(r['estimated_compute_cny'],3),r['final_step'],r['best_step']] for r in rows]))
    add(f"主统一生成与CE测试账本记录用时 **{summary['budget']['elapsed_seconds']/60:.2f}分钟**，按￥3.03/小时估算 **￥{cost:.3f}**；补充跨域检查及合计另见10.2。这些是项目进程计算时间估算，不含停机后的计费、不代表平台账单；中断恢复组使用budget累计时间，避免把只记录最后一段的elapsed_seconds当总训练时长。")
    add('### 9.3 监控、权重和可复核性')
    add('所有正式模型按Checkpoint文件SHA固定；评测前后另核对加载后模型张量摘要不变。相同张量的不同文件容器会有不同文件SHA，因此额外记录tensor SHA，并只在同一测试协议下复用结果。训练best、完整可恢复last和冻结配置均保留，没有删权重，没有重新训练。')
    add('此前第一组DPO的W&B显示crashed时，本地已完成1,279步、20,462题；云端曾只接收510步。已在原run补传77条真实指标并核验finished/1,279，未改模型张量。W&B图表是否完整与模型是否完成训练分别验证；报告以Checkpoint、summary和逐题文件为主，不仅凭网页状态。')
    add('## 10. 如何解释这些实验，下一步如何避免错误结论')
    add('1）对“RL有没有提升”，先看同起点、同一最终测试上的配对EM和来源分项，再看F1、生成长度与last退化；不只比较训练reward或开发集峰值。')
    add('2）对“新数据是不是更好”，同时考虑来源迁移、旧组已见题、实际Token和筛选后的题数。新组低分不等于人工标注更差，旧组高分也不证明复用训练题是更好的普遍策略。')
    add('3）对“LoRA能否替代Full”，当前最公平的已有历史对照是r16与Full100k的相同两轮预算；新r8/r32之间的单轮对照能够研究更新容量和dropout。Full两轮与LoRA一轮只能作为已交付模型的实际比较。')
    add('4）对“DPO为什么不理想”，本次足以显示部分配置没有超过起点、终点可能退化，但不足以证明DPO方法对该任务无效。后续若继续，优先在开发集用更保守学习率/beta、人工核查负例和抽取边界，而非自动增加epoch。')
    add('5）本次测试之后，这两个测试集的汇总和案例已经对开发者暴露。后续若据此调数据、奖励、学习率或模型，应该另保留新的未用测试；不能继续把同一测试声称为完全未见的最终检验。本次没有更新任何训练配置或模型选择指针。')
    add('建议后续优先补充：多seed复现实验、来源外留出阅读集、盲法人工语义评分、通用能力保持的统一复测，以及同100k数据/同1轮的Full基线。它们均未在本轮自动启动。')
    return ''.join(chunks)


def case_studies(report,rpred,gpred,selected_lora):
    chunks=['\n## 11. 代表性逐题案例与错误模式\n\n']
    baseline={r['id']:r for r in rpred['reading_sft']};chosen=[];seen=set()
    models=['reading_sft','ppo_old_full_50k_best','grpo_old_full_50k_best','dpo_old_full_50k_best','dpo_new_matched_last']
    lookup={k:{r['id']:r for r in rpred[k]} for k in models}
    for method in ['ppo','grpo','dpo']:
        key=method+'_old_full_50k_best'
        for language in ['zh','en']:
            for direction in ['gained','lost']:
                values=[r for r in rpred[key] if r['language']==language and r['id'] not in seen and
                    (baseline[r['id']]['exact_match']==0 and r['exact_match']==1 if direction=='gained' else baseline[r['id']]['exact_match']==1 and r['exact_match']==0)]
                if not values:continue
                row=sorted(values,key=lambda r:r['id'])[0];seen.add(row['id'])
                chosen.append({'selection':method.upper()+' '+direction+' '+language,'row':row})
    for language in ['zh','en']:
        values=[r for r in rpred['reading_sft'] if r['language']==language and not r['exact_match'] and r['f1']>0 and r['id'] not in seen]
        if values:
            row=sorted(values,key=lambda r:r['id'])[0];seen.add(row['id']);chosen.append({'selection':'partial_overlap '+language,'row':row})
    chunks.append('案例不是随机估计样本，也不代替上面的全量统计。按预先可解释的关系选取：三个算法旧全量best相对SFT的新增正确/丢失正确，各取中英文ID排序第一例，再补中英文部分重叠各一例；同一题不重复。该规则会同时展示获益和受损题，避免只给成功案例。归类是评分关系，不是完整人工语义正确性裁决。\n\n')
    notes={
        'reading_069023edec0a4f69b2c438c4':'SFT已经列出正确地理范围，但多带了“面积达170万平方公里”；PPO/GRPO主要改善答案停止边界。DPO旧全量输出把地理列表末尾变成人物词“努尔哈赤”，这已经是内容错误，不能归入同一种边界现象。',
        'reading_07ebb60dbe3bd2e313b5d482':'材料明确区分英国2004年与波兰/西班牙2006年。PPO/GRPO选择了后者，暴露国家与年份的绑定错误；这一受损例是真正的材料关系误读，而非只多了标点。',
        'reading_03af11c23748fd0f1e183491':'问题问夏季常见现象，材料直接写thunderstorms。PPO/GRPO从“high heat indices”转为该跨度，显示第一答案实体选择改善；不能仅凭这一题推断外部气象知识增加。',
        'reading_0b680ad70d06bc82259811db':'原材料将Cromwell的盟友写为the Scots。PPO输出“the足利”，DPO输出“theating”，均未保留该实体；GRPO在本例仍正确。它提示平均分之外应检查跨语言片段混入和无依据实体。',
        'reading_047f826c493c87f91c68fd72':'材料以“曾任”描述电子音乐社创社社长，以“现任”描述大宇音乐音效职位，但发布答案同时接受这两个职位。这是金标准别名存在时间关系争议的明确例子。本次没有修改答案或评分，GRPO所给现职位恰好是材料支持的回答；不将所有发布别名视为同样无争议。',
        'reading_05cc512154e9db6f08b57d59':'材料给出低地苏格兰语与中世纪佛兰芒语两种词源说法，发布别名只覆盖前者及其简称。GRPO输出“天主教语”未获材料支持；同时，参考也没有穷尽材料里的全部可辩护说法，说明别名完整性并非完美。',
        'reading_0b10b07b41cec6c1002f1b0b':'GRPO恢复了关键实体Turing machine；PPO的Avicenna machine虽然自定义F1达到2/3，核心名称仍然错误。这说明F1较高不能替代对关键实体的核对。',
        'reading_216198fbeaca7b76abe5fb65':'GRPO只在experience前多了一个未配对引号，答案主要内容与SFT相同，却被保守EM判0。这一例不能据此声称实质知识退化。另外发布别名considerable impact与列举经验/理念/武器并非完全相同的关系，金标准语义也有歧义；均保持原评分并披露。',
        'reading_0a0bad2eae30512578352b0f':'“洛杉矶湖人队”与金标准“洛杉矶湖人”在此指向同队，但前者EM为0。该题的RL增益主要是与人工跨度边界对齐，不宜描述成之前完全不懂球队知识。',
        'reading_00714d284115d8890c771309':'DPO只答“体重”，丢掉了决定原因的“过轻”；尽管字符重叠F1不低，它没有完整回答为何无法入伍。PPO/GRPO保留了关键限定。',
        'reading_007ed3122f434df415d0abbd':'材料包含After 1971，DPO选出1971，而其他几种策略给出1948/1949。此例表明DPO整体退化不意味着每道题都退化；仍需用全量净增益衡量。',
        'reading_038814f4cc4b92d5dab146dd':'参考为617 ft；DPO将数值缩到“6”，其他三个best保持617 ft。这里是数值与单位信息丢失，不能因为它输出得更短就认为抽取格式更好。'}
    saved=[]
    for number,item in enumerate(chosen,1):
        row=item['row'];ident=row['id'];context=row['context'];position=-1
        for answer in row['references']:
            position=context.find(answer)
            if position>=0:break
        start=max(0,position-70) if position>=0 else 0
        snippet=context[start:start+260];snippet=('…' if start else '')+snippet+('…' if start+260<len(context) else '')
        entries=[{'model':k,**lookup[k][ident]} for k in models]
        saved.append({'selection':item['selection'],'id':ident,'question':row['question'],'context':context,'references':row['references'],'models':entries,
                      'text_review_note':notes.get(ident),'review_is_independent_human_judgment':False})
        chunks.append(f"### 11.{number} 阅读：{item['selection']}\n\n")
        chunks.append(f"题ID：`{ident}`；来源：{row['source']}。\n\n问题：{row['question']}\n\n材料摘录：{snippet}\n\n金标准别名：{' / '.join(row['references'])}\n\n")
        chunks.append(table(['模型','实际输出','EM','F1','结束方式'],[[label(e['model']),e['answer'],int(e['exact_match']),f(e['f1']),e['stop_reason']] for e in entries])+'\n')
        chunks.append('分析：')
        if ident in notes:
            chunks.append(notes[ident]+'\n\n')
        elif 'gained' in item['selection']:
            chunks.append('该算法的贪心答案从未命中发布别名变为完整命中；看其他模型是否同步纠正，可区分公共任务难点与特定策略变化。材料和参考只用于事后核对，生成输入没有拼入参考答案。\n\n')
        elif 'lost' in item['selection']:
            chunks.append('原SFT已能给出金标准，本轮策略改变了第一答案并失去EM。这是需要计入净收益的受损题；更高平均分不意味着每题都改善。F1非零时还应检查答案边界、额外解释或别名不足，不能自动判定全部语义错误。\n\n')
        else:
            chunks.append('输出与参考部分重叠而非完全相同。可能是遗漏关键限定、输出较长材料或人工别名不充分；当前规则能显示边界问题，但无法单凭F1证明语义正确或错误。\n\n')
    instruction_notes={
        '04b252d9f0b0573d3b8874c6727d380132c9bb59174055ac35cd4df04665af3b':'输入要求两个句子且只用miss/misses，但参考本身用中文解释，未满足词汇限制。Full没有解释出时态区别并重复措辞；LoRA几乎回抄问题。该中文限制未被当前规则解析，supported_check_count=0不能理解成格式通过；同时本题参考与要求不一致，不宜用重叠量当严格指令正确性。',
        '01e1e071e0bf48605d316c0776d6716e4fd22df9472dd02ac0a499ee8fab3ded':'这一条只是提供救援工具材料并准备后续提问，并没有具体阅读问题；参考本来也是请求继续提问。LoRA的简短确认在功能上与参考相容，Full额外重复Jaws of Life。两者ROUGE约0.205并不证明回答语义很差，反而说明grounded_qa启发式标签和参考重叠的边界。',
        '005575d6d494bfb327da3c348b6c4b12cf14fc5023e80f2f7afe9b48e95fcefd':'Full删去“不断”“各种各样”等冗词，同时保留旅游需求上升与形式/目的地多样两个信息；LoRA保留更多原有修饰，仅局部换词。此例Full在简洁改写上更贴合要求，且双方均无重复循环，说明并非所有样本都存在生成退化。',
        '005a3e1c0894b5f8d7666a1372ccaa082e4cfbc3718ead88fff2790dd8dd82af':'任务明确要求第一人称、第三人称各写一遍。Full只给A的第一人称一句，遗漏第二版本及B；LoRA没有两个人称版本，混合归属并重复《三国演义》、遗漏《平凡的世界》。主要问题是多重要求和信息归属，不是仅凭ROUGE高低判胜。',
        '0f90e38c86c5f57f5a8438247b988f5de6255d5b371b57e513ce4640b146347d':'Full多条bullet反复讲同一种服务渠道；LoRA在第一条中反复生成“电子邮件”。双方达到256 Token上限而未EOS，属于可以直接从输出确认的重复与任务覆盖不足；比单看平均Loss更能暴露实际可用性问题。',
        '0008cd33a5915e3ec43b06cfa47c00eb923c97c75b8b3f6e785f73f23d9119fc':'两者都输出至少三个bullet，所以形式检查同时通过；LoRA三个bullet多次重复目标/边界，内容多样性较弱。该检查没有核实提示里的研究依据，也没有评判建议的事实或医学有效性；这里仅比较可观察的格式与重复。',
        '00d5e89051ad713d10759c2719da3611d0cf9deb93846be03662d027be0ea5a4':'提示关注四周计划的第一周、户外活动及成员家属。两个输出均引入未给出的workshop设定并重复，LoRA重复更多，双方达到长度上限。它们有“明确目标”等相关起点，但没有持续围绕具体第一周场景形成清晰、紧凑的优先事项。',
        '2fe38b77b8c5f138f9029651a97f5c07bae3c1453bf0963998525c5dc6db4bde':'Full输出引号后连续[ ]，是明显无内容推进的循环；LoRA保留原文并作轻微编辑，正常EOS，在本题更可用。参考还混入英文拼写easiliy，暴露现有参考质量问题。这个反例说明Full平均CE更低不代表每一题都更好。',
        'f9cd99935cde34d61819515966577ef11c23eeb1427322a421c09516a645a32f':'要求20字以内、最多两词，两个模型却长串重复“努力/坚持”并耗尽Token预算，显然违反长度要求。该中文规则在现有checker中覆盖为0；不应把没有自动检查项解读为遵循指令。'}
    full={r['id']:r for r in gpred['sft100k_full']};lora={r['id']:r for r in gpred[selected_lora]}
    general=[];used=set()
    for language in ['zh','en']:
        for task in ['format_following','grounded_qa','rewriting','summarization','short_qa_or_other']:
            values=[r for r in full.values() if r['language']==language and r['task']==task]
            if not values:continue
            values.sort(key=lambda r:r['id'])
            # 格式题优先展示可执行规则；其余按固定ID次序取例。
            covered=[r for r in values if r['constraint_checks']['supported_check_count']]
            row=(covered or values)[0];general.append(('language/task stratified',row));used.add(row['id'])
    for key,values in [('full',list(full.values())),('lora',list(lora.values()))]:
        row=max((r for r in values if r['id'] not in used),key=lambda r:(r['repeated_3gram_fraction'],r['id']))
        general.append(('highest_repetition_'+key,row));used.add(row['id'])
    chunks.append(f'### 11.{len(chosen)+1} 通用指令：Full与事前选定LoRA的输出\n\n')
    chunks.append('下列按语言×已有任务标签选首个ID；格式题优先有可执行规则者，再补Full和LoRA各自重复3gram最高的未展示题。高重复案例用于定位风险，不代表总体发生比例。Markdown为阅读便利截取较长文本，完整输入、参考、输出和Token在原JSONL及case_studies.json中。\n\n')
    for number,(reason,row) in enumerate(general,1):
        a=full[row['id']];b=lora[row['id']]
        saved.append({'selection':reason,'suite':'instruction_test_1000','id':row['id'],'full':a,'lora':b,
            'text_review_note':instruction_notes.get(row['id']),'review_is_independent_human_judgment':False})
        def shorten(text,limit=650):return text if len(text)<=limit else text[:limit]+'…（正文截取，完整文本见JSONL）'
        chunks.append(f"#### 通用案例{number}：{row['language']} / {row['task']} / {reason}\n\n")
        chunks.append('题ID：`'+row['id']+'`\n\n输入：'+shorten(row['prompt'],900)+'\n\n参考：'+shorten(row['reference'])+'\n\n')
        chunks.append(table(['模型','实际输出','参考ROUGE-L','重复3gram','结束'],[
            ['Full100k（历史2轮）',shorten(a['answer']),f(a['reference_overlap_rouge_l_f1']),pct(a['repeated_3gram_fraction']),a['stop_reason']],
            [label(selected_lora)+'（1轮）',shorten(b['answer']),f(b['reference_overlap_rouge_l_f1']),pct(b['repeated_3gram_fraction']),b['stop_reason']]])+'\n')
        if row['id'] in instruction_notes:chunks.append('逐文分析：'+instruction_notes[row['id']]+'\n\n')
        if a['constraint_checks']['supported_check_count']:
            chunks.append('可执行约束结果：Full '+json.dumps(a['constraint_checks']['checks'],ensure_ascii=False)+'；LoRA '+json.dumps(b['constraint_checks']['checks'],ensure_ascii=False)+'。这些检查只覆盖列出的明确形式要求。\n\n')
        elif row['id'] in instruction_notes:pass
        elif reason.startswith('highest_repetition'):
            chunks.append('分析：本例按重复指标选出，需要查看重复内容是否是有意义的列表复现、材料引用，还是无进展地循环。达到max_new_tokens也仅代表预算用尽，不能自动归因为EOS训练失败。\n\n')
        else:
            chunks.append('分析：比较是否保留输入含义、遗漏限定、添加无依据事实或未遵循任务。表中ROUGE只是参考重叠，本报告没有额外调用裁判模型给语义正确率打分；两模型的训练遍历量也不同。\n\n')
    write_json(report/'case_studies.json',saved)
    write_json(report/'case_review_notes.json',{'scope':'text-grounded case review by the assistant; not independent human adjudication',
        'test_references_or_scores_changed':False,'notes':notes,'instruction_notes':instruction_notes})
    return ''.join(chunks)


def reproduction_appendix(plan,summary,train_rows,audit,data,report):
    text=['\n## 12. 文件索引、复现和审计附录\n\n']
    text.append('### 12.1 交付文件\n\n')
    text.append(table(['位置','内容'],[
        ['完整测试报告.html','自包含离线HTML，含目录、完整表格和内嵌图；建议首先打开'],
        ['完整测试报告.md','可搜索/编辑的中文报告正文'],['SFT测试报告.md / LoRA测试报告.md / RL测试报告.md','从完整报告提取的独立方法报告与共同评测条件'],
        ['tables/01_model_inventory.csv','正式模型、初始化、训练量、学习率/Batch/轮数/步数、参数和费用'],
        ['tables/02_reading_primary_comparisons.csv','九组best相对SFT的配对区间、增益/受损计数、校正p值'],
        ['tables/03～07_reading*.csv','所有best/last及基线的总体、语言、来源、题型/答案长、错误分布'],
        ['tables/08_reading_18000_paired_answers.csv','18,000条SFT与九组best逐题对照、参考和双方回答'],
        ['tables/09_data_regime_comparisons.csv / 10_best_vs_last.csv','旧新/全量对照和终点退化'],
        ['tables/11～13_instruction*.csv / full_vs_lora*.csv','15模型的1,000题完整测试汇总、语言/任务、配对CE区间'],
        ['tables/14_historical_sft_controls.csv','历史8组SFT与Base的真实旧测试结果'],
        ['tables/15_cached_native_numerical_probes.csv / 16_english_answers_containing_han.csv','真实GPU数值探针与英文题输出含汉字的透明诊断'],
        ['tables/17_cross_domain_instruction_loss.csv','阅读SFT及全部九组RL best在相同通用1,000题的完整CE保持检查'],
        ['tables/18_algorithm_pairwise_exploratory.csv / 19_lora_rank_dropout_paired_ce.csv','算法间探索性配对差异；固定单轮LoRA rank/dropout测试CE区间'],
        ['tables/20_sft_controls_paired_ce.csv','独立指令测试上SFT学习率、Batch及历史训练量对照的配对CE区间'],
        ['case_studies.json','报告全部代表案例的完整文本，不只保留正文摘录'],
        ['statistical_analysis.json / key_findings.json','可复算统计结果及摘要，不包含新选模指针'],
        ['../results_local/<suite>/<model>/generations.jsonl','每模型完整1,000或2,000题输入、参考、输出、结束方式及逐题分数'],
        ['../results_local/<suite>/<model>/test_rows.jsonl','每题答案Loss总和、Token数、teacher-forced命中数'],
        ['../sources_local/','历史完整测试、冻结配置、训练指标、数据manifest与逐文件来源校验清单'],
        ['report_manifest.json','报告文件大小、SHA256、计划身份与工具身份']]))
    text.append('\n### 12.2 复现实验条件\n\n')
    text.append('原评测目录：`'+plan['output_dir']+'`。本地仅保存评测结果和报告素材；没有下载模型权重，也没有重新启动此前取消的W&B备份。再次执行评测需要原远程Checkpoint和数据；已经完成的模型条目可按身份校验跳过，不应删除结果强制重跑。\n\n')
    text.append('```bash\ncd /root/autodl-tmp/model_training\nsource scripts/env_remote.sh\n.venv/bin/python -u scripts/evaluate_posttrain_comprehensive.py \\\n  --plan '+plan['output_dir']+'/plan.json\n```\n\n')
    text.append('本地从已同步结果重建报告：\n\n```powershell\n$env:TEMP="E:\\codex\\tmp"\n$env:TMP=$env:TEMP\n$env:PYTHONPATH="E:\\codex\\model_training\\src;E:\\codex\\model_training\\scripts"\n& E:\\conda_envs\\llm\\python.exe scripts/build_posttrain_test_report.py --work runs/assistant/inference/'+report.parent.name+'\n```\n\n')
    environment=read_json(data/'runtime_environment.json')
    text.append(table(['环境项','记录值'],[[k,v] for k,v in environment.items() if k!='source_fingerprints']))
    text.append('\n### 12.3 身份摘要与模型路径\n\n')
    text.append(table(['身份','SHA256 / 值'],[['本次计划',plan['plan_identity']],['评测工具',summary['source_sha256']],
        ['复用阅读评测工具',plan.get('reused_suites',{}).get('reading_test_2000',{}).get('source_evaluator_sha256','同当前工具')],
        ['冻结SFT源码',plan['source_fingerprints']['sft']],['冻结RL源码',plan['source_fingerprints']['rl']],
        ['阅读测试split',summary['results']['reading_test_2000'][0]['test_identity']['split_sha256']],
        ['指令测试split',summary['results']['instruction_test_1000'][0]['test_identity']['split_sha256']],
        ['共同分词器',summary['results']['reading_test_2000'][0]['test_identity']['tokenizer_sha256']]]))
    text.append('\n阅读结果的旧新路径及逐文件SHA见reading_reuse_receipt.json；原目录完整保留。15个通用模型全部按新Batch128协议重跑，没有从被替换的Batch16通用尝试中拼接输出。\n\n')
    text.append('\n完整Checkpoint列表的文件SHA在CSV及plan中，以下路径不作搬动或改写：\n\n')
    text.append(table(['模型键','远程Checkpoint'],[[m['key'],m['checkpoint']] for m in plan['models']]))
    text.append('\n### 12.4 完整性和明确局限\n\n')
    checks=[['每份模型题数','阅读2,000 / 指令1,000，ID唯一、各模型集合一致'],
        ['权重状态','SHA冻结；模型加载后的前后张量摘要相等；无optimizer更新'],
        ['断点与历史身份','best/last、原配置、训练日志保留；终点只作诊断'],
        ['测试数据','split与tokenizer SHA核验；本次精确训练/测试键重叠为0'],
        ['训练次数','本次0次训练；没有修改epochs或stop_epochs'],
        ['报告统计','全量逐题数据可查；CI、p值、分母、语言权重与案例选择规则已披露'],
        ['能力范围','抽取阅读及有限指令参考测试；不是全领域能力基准'],
        ['不确定性','单训练seed；缺少独立复现；文章相关、参考别名和语义重叠边界仍在'],
        ['公平性','Full100k历史2轮 vs 新LoRA1轮不等预算；算法间LR、题数、采样和算力不同'],
        ['测试后开发','结果已暴露；未来依据本报告调参需新的未用测试，不再声称该测试完全未见']]
    text.append(table(['检查项','结论/边界'],checks)+'\n')
    return ''.join(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--work',type=Path,required=True)
    main(parser.parse_args().work)
