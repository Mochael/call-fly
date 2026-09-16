import {test,expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';

test('two independent browser callers hear audio; stopping and restarting one preserves the other',async({browser})=>{
  test.skip(!process.env.EXPECT_MODAL,'Requires the deployed Modal worker pool');
  test.setTimeout(300_000);
  const contexts=await Promise.all([browser.newContext({permissions:['microphone']}),browser.newContext({permissions:['microphone']})]);
  const pages=await Promise.all(contexts.map(context=>context.newPage()));
  const errors=[];
  try{
    await Promise.all(pages.map(async page=>{
      page.on('pageerror',error=>errors.push(error.message));
      await page.addInitScript(()=>{
        window.callRuntime=null;window.callFrames=[];window.playedSamples=0;window.duplexCaptures=0;
        window.addEventListener('eric-ready',e=>window.callRuntime=e.detail);
        window.addEventListener('eric-frame',e=>window.callFrames.push(e.detail));
        window.addEventListener('eric-playback',e=>window.playedSamples+=e.detail.samples);
        window.addEventListener('eric-capture',e=>{if(e.detail.duringPlayback)window.duplexCaptures++;});
      });
      await page.goto(process.env.TEST_BASE_URL||'http://localhost:8765');
      await page.getByRole('button',{name:/Call fly/}).click();
    }));
    await Promise.all(pages.map(page=>expect.poll(()=>page.evaluate(()=>window.callFrames.length),{timeout:210_000}).toBeGreaterThan(200)));
    const reports=await Promise.all(pages.map(page=>page.evaluate(()=>({
      runtime:window.callRuntime,frames:window.callFrames.length,audio_s:window.playedSamples/24000,
      corrections:window.callFrames.filter(f=>f.reservoir_logit_rms>0).length,duplexCaptures:window.duplexCaptures,
    }))));
    const shared=reports.every(r=>r.runtime.serving==='upstream-masked-streaming');
    const identity=r=>`${r.worker_id}:${r.session_slot}`;
    if(shared){
      expect(new Set(reports.map(r=>identity(r.runtime))).size).toBe(2);
    }else expect(new Set(reports.map(r=>r.runtime.worker_id)).size).toBe(2);
    // A one-container preview can require both callers to share the same GPU.
    // Production may distribute them across already-loaded containers.
    if(process.env.EXPECT_SHARED){
      expect(new Set(reports.map(r=>r.runtime.worker_id)).size).toBe(1);
      expect(reports.every(r=>r.runtime.serving==='upstream-masked-streaming')).toBeTruthy();
    }
    expect(new Set(reports.map(r=>r.runtime.call_id)).size).toBe(2);
    for(const report of reports){
      expect(report.runtime.codec).toBe('torch-cuda');expect(report.audio_s).toBeGreaterThan(14);
      expect(report.corrections).toBeGreaterThan(100);expect(report.duplexCaptures).toBeGreaterThan(5);
    }
    await pages[0].getByRole('button',{name:/End call/}).click();
    const before=await pages[1].evaluate(()=>window.callFrames.length);
    await expect.poll(()=>pages[1].evaluate(()=>window.callFrames.length),{timeout:10000}).toBeGreaterThan(before+50);
    await pages[0].evaluate(()=>{window.callFrames=[];});
    await pages[0].getByRole('button',{name:/Call fly/}).click();
    await expect.poll(()=>pages[0].evaluate(()=>window.callFrames.length),{timeout:30000}).toBeGreaterThan(20);
    expect(await pages[0].evaluate(()=>window.callFrames[0].index)).toBe(1);
    const restart=await pages[0].evaluate(()=>window.callRuntime);
    expect(restart.call_id).not.toBe(reports[0].runtime.call_id);
    if(shared){
      expect(restart.serving).toBe('upstream-masked-streaming');
      expect(identity(restart)).not.toBe(identity(reports[1].runtime));
      if(process.env.EXPECT_SHARED)expect(restart.worker_id).toBe(reports[1].runtime.worker_id);
    }else expect(restart.worker_id).not.toBe(reports[1].runtime.worker_id);
    await Promise.all(pages.map(async page=>{
      await expect(page.locator('#error')).toBeHidden();
      await page.getByRole('button',{name:/End call/}).click();
    }));
    expect(errors).toEqual([]);
    await writeFile(process.env.EXPECT_SHARED?'artifacts/moshi/shared-browser.json':shared?'artifacts/moshi/shared-production-browser.json':'artifacts/moshi/concurrent-browser.json',JSON.stringify({passed:true,calls:reports,restart},null,2));
  }finally{await Promise.all(contexts.map(context=>context.close()));}
});
