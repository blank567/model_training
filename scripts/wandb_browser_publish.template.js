// 仅在用户提供、已登录的同一个 W&B 项目页执行。
// 不读取 Cookie/Token，不删除 run/history，不修改 config/group/displayName。
async function publishWandbLayout(payload) {
 const host = 'https://forge.coreweave.com';
 const prefix = '/wandb/mblankm-peking-university/bilingual-training-lab/';
 if (location.origin !== host || !location.pathname.startsWith(prefix)) throw Error('Unexpected project page');
 if (payload.entity !== 'mblankm-peking-university' || payload.project !== 'bilingual-training-lab') throw Error('Unexpected payload project');
 if (window.__WBO?.publication?.state === 'running') throw Error('Publication is already running');
 if (window.__WBO?.publication?.state === 'done') {
  console.log('WBO_RESULT ' + JSON.stringify({publication: window.__WBO.publication})); return;
 }
 const state = {state:'running', date:payload.date, summaryRuns:0, summaryMetrics:0, workspaces:[], report:null};
 window.__WBO ??= {}; window.__WBO.publication = state;
 const queryRuns = 'query ChartRunSnapshot($e:String!,$p:String!){project(name:$p,entityName:$e){id runs(first:100){edges{node{id name displayName state summaryMetrics}} pageInfo{hasNextPage}}}}';
 const variables = {e:payload.entity, p:payload.project};
 const gql = async(query, variables) => {
  const response = await fetch('/api/wandb/graphql', {method:'POST', credentials:'same-origin', signal:AbortSignal.timeout(30000), headers:{'Content-Type':'application/json'}, body:JSON.stringify({query, variables})});
  if(!response.ok)throw Error('HTTP '+response.status);
  const result=await response.json(); if(result.errors)throw Error(JSON.stringify(result.errors)); return result.data;
 };
 const readRuns = async() => {
  const result=await gql(queryRuns,variables);
  if(!result.project || result.project.runs.pageInfo.hasNextPage)throw Error('Incomplete project inventory');
  return result.project.runs.edges.map(({node})=>({...node, summary:typeof node.summaryMetrics==='string'?JSON.parse(node.summaryMetrics):node.summaryMetrics}));
 };
 const updateSummary='mutation ChartSummary($id:String,$summaryMetrics:JSONString){upsertBucket(input:{id:$id,summaryMetrics:$summaryMetrics}){bucket{id}}}';
 const upsertView='mutation ChartView($id:ID,$entityName:String,$projectName:String,$type:String,$name:String,$displayName:String,$description:String,$spec:String!){upsertView(input:{id:$id,entityName:$entityName,projectName:$projectName,type:$type,name:$name,displayName:$displayName,description:$description,spec:$spec,createdUsing:WANDB_SDK}){view{id name displayName type spec} inserted}}';
 const makeView=async(item,type,name)=>{
  const result=await gql(upsertView,{entityName:payload.entity,projectName:payload.project,type,name,displayName:item.displayName,description:item.description||'',spec:JSON.stringify(item.spec)});
  const view=result.upsertView.view;
  if(JSON.stringify(JSON.parse(view.spec))!==JSON.stringify(item.spec))throw Error('View readback differs: '+name);
  return {id:view.id,name:view.name,title:view.displayName};
 };
 try {
  const before=await readRuns(), byName=new Map(before.map(run=>[run.name,run]));
  window.__WBO.summaryBackup=before.filter(run=>Object.hasOwn(payload.summary_updates,run.name));
  console.log('WBO_RESULT '+JSON.stringify({backup:window.__WBO.summaryBackup,namespace:payload.namespace}));
  for(const [name, additions] of Object.entries(payload.summary_updates)){
   const run=byName.get(name);if(!run||run.state!=='finished')throw Error('Missing or unfinished target run: '+name);
   for(const [key,value] of Object.entries(additions)){
    if(!key.startsWith(payload.namespace+'/'))throw Error('Metric outside allowed namespace');
    if(Object.hasOwn(run.summary,key)&&JSON.stringify(run.summary[key])!==JSON.stringify(value))throw Error('Conflicting existing metric: '+name+'/'+key);
   }
   const merged={...run.summary,...additions};
   if(Object.keys(additions).some(key=>!Object.hasOwn(run.summary,key))){await gql(updateSummary,{id:run.id,summaryMetrics:JSON.stringify(merged)});}
   state.summaryRuns++;state.summaryMetrics+=Object.keys(additions).length;
  }
  const after=await readRuns(), afterMap=new Map(after.map(run=>[run.name,run]));
  if(before.length!==after.length)throw Error('Run count changed during publication');
  for(const original of before){
   const current=afterMap.get(original.name);
   if(!current||current.id!==original.id||current.state!==original.state||current.displayName!==original.displayName)throw Error('Run identity changed');
   for(const [key,value] of Object.entries(original.summary)){
    if(JSON.stringify(current.summary[key])!==JSON.stringify(value))throw Error('Original summary changed: '+original.name+'/'+key);
   }
   for(const [key,value] of Object.entries(payload.summary_updates[original.name]||{})){
    if(JSON.stringify(current.summary[key])!==JSON.stringify(value))throw Error('Final test metric verification failed');
   }
  }
  state.originalSummaryPreserved=true;state.runsBefore=before.length;state.runsAfter=after.length;
  state.report=await makeView(payload.report,'runs','labcharts'+payload.date);
  state.report.url=host+prefix+'reports/'+encodeURIComponent(state.report.title.replaceAll(' ','-'))+'--'+state.report.id;
  for(let i=0;i<payload.workspaces.length;i++){
   const item=payload.workspaces[i], name='nw-labcharts'+payload.date+String.fromCharCode(97+i)+'-v';
   const view=await makeView(item,'project-view',name);
   view.url=host+prefix+'workspace?nw='+name.replaceAll('nw-','').replaceAll('-v','');
   state.workspaces.push(view);
  }
  state.state='done';state.finishedAt=new Date().toISOString();
  console.log('WBO_RESULT '+JSON.stringify({publication:state}));
 }catch(error){state.state='error';state.error=error.message;console.log('WBO_ERROR '+JSON.stringify(state));}
}
