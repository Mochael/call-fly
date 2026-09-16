import {test, expect} from '@playwright/test';

for (const ending of ['limit', 'overload', 'disconnect']) {
  test(`call ending remains visible after health polling: ${ending}`, async ({page}) => {
    let socket;
    const stopped=[];
    await page.routeWebSocket('**/api/conversation', ws => {
      socket=ws;
      ws.onMessage(message => {
        if (typeof message === 'string') stopped.push(JSON.parse(message));
      });
      ws.send(JSON.stringify({type:'ready', max_call_seconds:300}));
    });
    await page.goto('/');
    await page.getByRole('button', {name:/Call fly/}).click();
    await expect(page.locator('#hint')).toContainText('Speak naturally');
    let expected;
    if (ending==='limit') {
      expected='The five-minute call limit was reached. Start a new call to keep chatting.';
      socket.send(JSON.stringify({type:'session_end', code:'session_limit', message:expected}));
    } else if (ending==='overload') {
      expected='Moshi is falling behind live audio. Close other heavy apps and restart the call.';
      socket.send(JSON.stringify({type:'error', code:'input_backlog', message:expected}));
    } else {
      expected='The speech connection closed unexpectedly (code 1011). Start a new call.';
      socket.close({code:1011});
    }
    await expect(page.locator('#status')).toHaveText(expected);
    await expect(page.getByRole('button',{name:/Call fly/})).toBeEnabled();
    // At least two health polls used to overwrite the reason with "Ready".
    await page.waitForTimeout(5500);
    await expect(page.locator('#status')).toHaveText(expected);
    if (ending==='limit') expect(stopped.some(s=>s.reason==='session_limit')).toBeTruthy();
    if (ending==='overload') await expect(page.locator('#error')).toHaveText(expected);
  });
}

test('warns before the configured call limit while keeping the call active', async ({page}) => {
  await page.routeWebSocket('**/api/conversation', ws => {
    ws.onMessage(() => {});
    ws.send(JSON.stringify({type:'ready', max_call_seconds:31}));
  });
  await page.goto('/');
  await page.getByRole('button', {name:/Call fly/}).click();
  await expect(page.locator('#hint')).toContainText('Call limit in', {timeout:10000});
  await expect(page.getByRole('button',{name:/End call/})).toBeVisible();
  await page.getByRole('button',{name:/End call/}).click();
  await expect(page.locator('#status')).toHaveText('You ended the conversation.');
});

test('disconnect diagnostics survive a failed upload and are retried', async ({page}) => {
  let socket, available=false;
  const saved=[];
  await page.route('**/api/call-diagnostics', async route=>{
    if(!available) {await route.fulfill({status:503,body:'offline'});return;}
    const record=route.request().postDataJSON();
    const response=await route.fetch();
    if(response.ok())saved.push(record);
    await route.fulfill({response});
  });
  await page.routeWebSocket('**/api/conversation', ws=>{
    socket=ws;ws.onMessage(()=>{});
    ws.send(JSON.stringify({type:'ready',call_id:'abcdef12',max_call_seconds:300}));
  });
  await page.goto('/');
  await page.getByRole('button',{name:/Call fly/}).click();
  await expect(page.locator('#hint')).toContainText('Speak naturally');
  socket.close({code:1011});
  await expect(page.locator('#status')).toContainText('closed unexpectedly');
  await expect.poll(()=>page.evaluate(()=>JSON.parse(localStorage.getItem('eric-pending-call-diagnostics')||'[]').length)).toBeGreaterThan(0);
  available=true;
  await expect.poll(()=>saved.some(r=>r.event==='socket_close'&&r.call_id==='abcdef12'&&r.close_code===1011),{timeout:10000}).toBeTruthy();
  await expect.poll(()=>saved.some(r=>r.event==='client_end'&&r.reason==='connection_error'),{timeout:10000}).toBeTruthy();
  await expect.poll(()=>page.evaluate(()=>JSON.parse(localStorage.getItem('eric-pending-call-diagnostics')||'[]').length)).toBe(0);
});

test('waiting shows elapsed time, sends no microphone audio and can be cancelled', async ({page}) => {
  const received=[];
  await page.routeWebSocket('**/api/conversation', ws=>{
    ws.onMessage(message=>received.push(message));
    ws.send(JSON.stringify({type:'warming',phase:'allocating',message:'Waiting for an available call worker…'}));
  });
  await page.goto('/');
  await page.getByRole('button',{name:/Call fly/}).click();
  await expect(page.locator('#status')).toContainText('Waiting for an available call worker');
  await expect(page.locator('#hint')).toContainText(/Waiting [1-9]\d*s/,{timeout:5000});
  expect(received).toEqual([]);
  await page.getByRole('button',{name:/End call/}).click();
  await expect(page.getByRole('button',{name:/Call fly/})).toBeEnabled();
  await expect(page.locator('#status')).toHaveText('You ended the conversation.');
  expect(received.some(message=>typeof message==='string'&&JSON.parse(message).type==='stop')).toBeTruthy();
});
