// 只新建引用已有云端指标的报告与工作区。
// 禁止向 run、history、summary、config 或 group 发起写操作；不上传本地最终测试结果。
async function createWandbChartViews(payload) {
 const host='https://forge.coreweave.com', prefix='/wandb/mblankm-peking-university/bilingual-training-lab/';
 if(location.origin!==host||!location.pathname.startsWith(prefix))throw Error('Unexpected project page');
 if(payload.entity!=='mblankm-peking-university'||payload.project!=='bilingual-training-lab'||payload.summary_updates)throw Error('Payload outside views-only scope');
 if(window.__WBO?.viewsPublication?.state==='running')throw Error('Already running');
 if(window.__WBO?.viewsPublication?.state==='done'){console.log('WBO_RESULT '+JSON.stringify({views_publication:__WBO.viewsPublication}));return;}
 window.__WBO??={};const state={state:'running',mode:'views-only',workspaces:[],report:null};__WBO.viewsPublication=state;
 const gql=async(query,variables)=>{
  const response=await fetch('/api/wandb/graphql',{method:'POST',credentials:'same-origin',signal:AbortSignal.timeout(30000),headers:{'Content-Type':'application/json'},body:JSON.stringify({query,variables})});
  if(!response.ok)throw Error('HTTP '+response.status);
  const result=await response.json();if(result.errors)throw Error(JSON.stringify(result.errors));return result.data;
 };
 const inventoryQuery='query ChartViewsInventory($e:String!,$p:String!){project(name:$p,entityName:$e){runs(first:100){edges{node{id name displayName state}} pageInfo{hasNextPage}} allViews(viewType:"project-view"){edges{node{id name displayName spec}}} reports:allViews(viewType:"runs"){edges{node{id name displayName}}}}}';
 const vars={e:payload.entity,p:payload.project};
 const upsert='mutation CreateChartView($entityName:String,$projectName:String,$type:String,$name:String,$displayName:String,$description:String,$spec:String!){upsertView(input:{entityName:$entityName,projectName:$projectName,type:$type,name:$name,displayName:$displayName,description:$description,spec:$spec,createdUsing:WANDB_SDK}){view{id name displayName type spec} inserted}}';
 const canonical=value=>JSON.stringify(value,(_,v)=>v&&typeof v==='object'&&!Array.isArray(v)?Object.fromEntries(Object.keys(v).sort().map(k=>[k,v[k]])):v);
 try{
  const before=(await gql(inventoryQuery,vars)).project;
  if(before.runs.pageInfo.hasNextPage)throw Error('Incomplete run inventory');
  state.beforeRunCount=before.runs.edges.length;
  const originalViews=before.allViews.edges.map(x=>x.node),knownNames=new Map([...originalViews,...before.reports.edges.map(x=>x.node)].map(x=>[x.name,x]));
  __WBO.viewLayoutBackup=originalViews;
  console.log('WBO_RESULT '+JSON.stringify({view_backup:originalViews}));
  const make=async(item,type,name)=>{
   if(knownNames.has(name))throw Error('View name already exists; refusing overwrite: '+name);
   const result=await gql(upsert,{entityName:payload.entity,projectName:payload.project,type,name,displayName:item.displayName,description:item.description||'',spec:JSON.stringify(item.spec)});
   const view=result.upsertView.view;
   if(canonical(JSON.parse(view.spec))!==canonical(item.spec))throw Error('View spec differs: '+name);
   return{id:view.id,name:view.name,title:view.displayName};
  };
  state.report=await make(payload.report,'runs','labcharts'+payload.date);
  state.report.url=host+prefix+'reports/'+encodeURIComponent(state.report.title.replaceAll(' ','-'))+'--'+state.report.id;
  for(let i=0;i<payload.workspaces.length;i++){
   const item=payload.workspaces[i],name='nw-labcharts'+payload.date+String.fromCharCode(97+i)+'-v';
   const view=await make(item,'project-view',name);
   view.url=host+prefix+'workspace?nw='+name.replaceAll('nw-','').replaceAll('-v','');state.workspaces.push(view);
  }
  const after=(await gql(inventoryQuery,vars)).project;
  if(canonical(before.runs)!==canonical(after.runs))throw Error('Run inventory changed');
  const afterViews=new Map(after.allViews.edges.map(x=>[x.node.id,x.node]));
  for(const original of originalViews){if(canonical(afterViews.get(original.id))!==canonical(original))throw Error('Original workspace changed');}
  state.originalViewsPreserved=true;state.runsUnchanged=true;state.afterRunCount=after.runs.edges.length;
  state.state='done';state.finishedAt=new Date().toISOString();console.log('WBO_RESULT '+JSON.stringify({views_publication:state}));
 }catch(error){state.state='error';state.error=error.message;console.log('WBO_ERROR '+JSON.stringify(state));}
}
