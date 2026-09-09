async page => {
  await page.emulateMedia({reducedMotion:'no-preference'});
  const graph = {nodes:[{id:'alpha',label:'Alpha',type:'entity',description:'Test entity',retrieved:false,position:{x:0,y:0}}],
    edges:[],total_nodes:1,total_edges:0,truncated:false,warnings:[],layout_key:'browser-fixture'};
  const makeRun = (id, selected='lightrag') => ({
    id, created_at:'2026-09-08T12:00:00Z', query:'Browser-only routing fixture',
    method_id:'adaptive',options:{},top_k:5,status:'success',answer:'UI fixture answer.',
    latency_ms:1200,retrieval_ms:1100,generation_ms:100,error:null,
    retrieval:{context_text:'Fixture',entities:[],relationships:[],chunks:[{content:'Early evidence fixture'}],references:[],graph:{...graph,hit_node_ids:['alpha']},warnings:[],
      metadata:{selected_method:selected,routing_probabilities:{vector:.18,lightrag:.67,pathrag:.15}}}
  });
  const graphRequests=[];
  const historical=makeRun('history-fixture','pathrag');
  await page.addInitScript(({run}) => {
    const originalFetch=window.fetch.bind(window);
    window.__routingRequests=0;
    window.fetch=(input,options)=>{
      if(input!=='/api/query/stream') return originalFetch(input,options);
      const id='live-'+(++window.__routingRequests);
      const current={...run,id};
      return Promise.resolve(new Response(new ReadableStream({
        start(controller){
          const encoder=new TextEncoder();
          const emit=event=>controller.enqueue(encoder.encode(JSON.stringify(event)+'\n'));
          window.__emitStage=stage=>{
            if(stage==='routing') emit({type:'routing',id,metadata:current.retrieval.metadata});
            if(stage==='retrieval') emit({type:'retrieval',id,retrieval:current.retrieval});
            if(stage==='result'){emit({type:'result',run:current});controller.close();}
            if(stage==='error'){emit({type:'error',error:{message:'Simulated request failure'}});controller.close();}
          };
        }
      }),{headers:{'Content-Type':'application/x-ndjson'}}));
    };
  },{run:makeRun('live')});
  await page.unroute('**/api/**');
  await page.route('**/api/**', async route => {
    const path=route.request().url().replace(/^https?:\/\/[^/]+\/api/, '').split('?')[0];
    let data;
    if(path==='/health') data={api_key_configured:true,imported_profile:true,retrieval_ready:true,runtime_issues:[],llm_model:'fixture',embedding_model:'fixture'};
    else if(path==='/dataset/status') data={dataset:{corpus_name:'Medical',subset:'medical',character_count:1052158,word_count:174610}};
    else if(path==='/index/status') data={status:'ready',reusable:true,error_type:null};
    else if(path==='/retrieval/methods') data={methods:[{id:'adaptive',name:'Adaptive',description:'Browser fixture',options:[]},{id:'vector',name:'Vector',description:'Browser fixture',options:[]}]};
    else if(path==='/runs') data={runs:[historical]};
    else if(path==='/runs/history-fixture') data=historical;
    else if(path==='/graph'){graphRequests.push(route.request().url()); data=graph;}
    else { await route.fulfill({status:404,json:{error:{message:'Unmocked API request'}}}); return; }
    await route.fulfill({json:data});
  });
  await page.setViewportSize({width:1560,height:1000});
  await page.goto('http://127.0.0.1:5180/');
  await page.waitForFunction(()=>document.querySelector('.adaptive-routing')?.dataset.state==='idle');
  await page.getByRole('textbox',{name:'提出你的问题'}).fill('Browser-only routing fixture');
  await page.evaluate(()=>window.__routingTest={release:false,failure:false});
  await page.getByRole('button',{name:'开始查询',exact:false}).click();
  await page.waitForFunction(()=>document.querySelector('.adaptive-routing')?.dataset.state==='waiting');
  if(await page.locator('.routing-candidate.is-selected').count()) throw Error('Stale winner during waiting');
  if(await page.locator('.routing-score-track.is-loading').count()!==3) throw Error('All three candidates must load');
  if((await page.locator('.adaptive-routing').innerText()).includes('%')) throw Error('Invented scores before response');
  await page.locator('.adaptive-routing').screenshot({path:'output/playwright/adaptive-loading.png'});
  await page.evaluate(()=>{
    const panel=document.querySelector('.adaptive-routing');
    window.__routingFrames=[];
    const observe=()=>{
      const frame={time:performance.now(),state:panel.dataset.state,
        scores:[...panel.querySelectorAll('[data-score-visible=true]')].map(n=>n.dataset.method),
        selected:panel.querySelector('.is-selected')?.dataset.method??null};
      const previous=window.__routingFrames.at(-1);
      if(!previous || JSON.stringify({...frame,time:0})!==JSON.stringify({...previous,time:0})) window.__routingFrames.push(frame);
    };
    window.__routingObserver=new MutationObserver(observe);
    window.__routingObserver.observe(panel,{subtree:true,attributes:true,childList:true});
    observe();window.__emitStage('routing');
  });
  await page.waitForFunction(()=>document.querySelector('.adaptive-routing')?.dataset.state==='complete');
  const frames=await page.evaluate(()=>{window.__routingObserver.disconnect();return window.__routingFrames;});
  for(let count=1;count<=3;count++) if(!frames.some(f=>f.scores.length===count && f.state==='revealing' && f.selected===null)) throw Error('Scores did not reveal sequentially');
  const start=frames.find(f=>f.state==='revealing');
  const end=frames.find(f=>f.state==='complete');
  const duration=end.time-start.time;
  if(duration<800 || duration>1250) throw Error('Reveal duration outside target: '+duration);
  if(end.selected!=='lightrag') throw Error('Wrong final route');
  if(!(await page.locator('.routing-candidate[data-method=lightrag]').innerText()).includes('67.0%')) throw Error('Wrong probability');
  await page.locator('.adaptive-routing').screenshot({path:'output/playwright/adaptive-complete.png'});
  await page.waitForFunction(()=>document.querySelector('.graph-count')?.textContent.includes('1 节点') &&
    document.querySelector('.graph-panel')?.textContent.includes('Medical / lightrag'));
  if(!graphRequests.some(url=>url.includes('method_id=lightrag'))) throw Error('Selected graph was not fetched before answer');
  if(await page.locator('.answer-content').count()) throw Error('Answer arrived before release');
  if(!(await page.getByRole('button',{name:'开始查询',exact:false}).isDisabled())) throw Error('Query stopped while answer pending');
  await page.evaluate(()=>window.__emitStage('retrieval'));
  await page.getByRole('heading',{name:'知识图谱 · 检索命中'}).waitFor();
  await page.getByText('Early evidence fixture',{exact:true}).waitFor();
  if(await page.locator('.answer-content').count()) throw Error('Evidence waited for final answer');
  await page.screenshot({path:'output/playwright/adaptive-before-answer.png',fullPage:true});
  await page.evaluate(()=>window.__emitStage('result'));
  await page.locator('.answer-content').waitFor();
  if(await page.locator('.adaptive-routing').getAttribute('data-state')!=='complete') throw Error('Final result restarted routing animation');
  await page.getByRole('button',{name:'开始查询',exact:false}).waitFor({state:'visible'});
  await page.evaluate(()=>window.__routingTest={release:false,failure:true});
  await page.getByRole('button',{name:'开始查询',exact:false}).click();
  await page.waitForFunction(()=>document.querySelector('.adaptive-routing')?.dataset.state==='waiting');
  if(await page.locator('.routing-candidate.is-selected').count()) throw Error('Winner not cleared for repeat request');
  await page.evaluate(()=>window.__emitStage('error'));
  await page.waitForFunction(()=>document.querySelector('.adaptive-routing')?.dataset.state==='failed');
  if(await page.locator('.routing-score-track.is-loading').count()) throw Error('Failure left loading active');
  await page.getByRole('button',{name:'查询记录',exact:false}).click();
  await page.getByRole('button',{name:'Browser-only routing fixture',exact:false}).click();
  await page.waitForFunction(()=>document.querySelector('.adaptive-routing')?.dataset.state==='complete');
  if(await page.locator('.routing-candidate[data-method=pathrag].is-selected').count()!==1) throw Error('Historical selection lost');
  await page.getByRole('button',{name:'检索工作台',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('.adaptive-routing')?.dataset.state==='idle');
  await page.emulateMedia({reducedMotion:'reduce'});
  await page.getByRole('textbox',{name:'提出你的问题'}).fill('Reduced-motion fixture');
  await page.evaluate(()=>window.__routingTest={release:true,failure:false});
  await page.getByRole('button',{name:'开始查询',exact:false}).click();
  await page.waitForFunction(()=>document.querySelector('.adaptive-routing')?.dataset.state==='waiting');
  await page.evaluate(()=>window.__emitStage('routing'));
  await page.waitForFunction(()=>document.querySelector('.adaptive-routing')?.dataset.state==='complete');
  await page.evaluate(()=>{window.__emitStage('retrieval');window.__emitStage('result');});
  await page.locator('.answer-content').waitFor();
  await page.setViewportSize({width:390,height:844});
  await page.waitForFunction(()=>document.documentElement.scrollWidth<=window.innerWidth);
  if(await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth)) throw Error('Mobile horizontal overflow');
  await page.locator('.adaptive-routing').screenshot({path:'output/playwright/adaptive-mobile.png'});
  return {mockQueries:await page.evaluate(()=>window.__routingRequests),realApiCalls:0,revealMilliseconds:Math.round(duration),
    scoresBeforeAnswer:true,graphBeforeAnswer:true,evidenceBeforeAnswer:true,noAnimationReplay:true,
    sequentialScores:true,repeatRequest:true,failure:true,history:true,reducedMotion:true,mobile:true};
}
