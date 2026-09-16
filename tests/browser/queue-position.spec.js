import {test,expect} from '@playwright/test';
import path from 'node:path';

test('queue positions update, microphone stays unsent until admission, and cancel works',async({page})=>{
  let socket;
  const sent=[];
  await page.route('**/api/health',r=>r.fulfill({json:{ready:true,backend:'moshi',stage:'Ready'}}));
  await page.route('**/api/call-diagnostics',r=>r.fulfill({json:{ok:true}}));
  await page.route('**/api/connectome/*',r=>r.fulfill({path:path.resolve('dist/vercel/neural/connectome',r.request().url().split('/').pop())}));
  await page.routeWebSocket('**/api/conversation',ws=>{
    socket=ws;ws.onMessage(m=>sent.push(m));
    ws.send(JSON.stringify({type:'warming',phase:'queued',people_ahead:2,message:'Internal capacity message'}));
  });
  await page.goto('/');
  await page.getByRole('button',{name:'Call fly',exact:true}).click();
  await expect(page.locator('#status')).toContainText('2 people ahead of you.');
  await expect(page.locator('#hint')).toContainText(/Waiting [1-9]/);
  for(const [ahead,copy] of [[1,'1 person ahead of you.'],[0,"You're next."]]) {
    socket.send(JSON.stringify({type:'warming',phase:'queued',people_ahead:ahead}));
    await expect(page.locator('#status')).toContainText(copy);
  }
  expect(sent).toEqual([]);
  socket.send(JSON.stringify({type:'ready',max_call_seconds:300}));
  await expect(page.locator('#hint')).toContainText('Speak naturally');
  await expect.poll(()=>sent.filter(m=>typeof m!=='string').length).toBeGreaterThan(0);
  await page.getByRole('button',{name:'End call',exact:true}).click();
  await expect(page.locator('#status')).toHaveText('You ended the conversation.');
  // Rejoin, then cancel while still queued.
  sent.length=0;
  await page.getByRole('button',{name:'Call fly',exact:true}).click();
  await expect(page.locator('#status')).toContainText('2 people ahead of you.');
  await page.getByRole('button',{name:'End call',exact:true}).click();
  expect(sent.some(m=>typeof m==='string'&&JSON.parse(m).type==='stop')).toBeTruthy();
  expect(sent.filter(m=>typeof m!=='string')).toHaveLength(0);
});
