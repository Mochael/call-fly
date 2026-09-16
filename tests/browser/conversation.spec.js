import {test, expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';

test('real Moshi duplex audio, live model features, simultaneous capture, stop and restart', async ({page}) => {
  test.setTimeout(240_000);
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.addInitScript(() => {
    window.runtimeReady=null;window.addEventListener('eric-ready',e=>window.runtimeReady=e.detail);
    window.testMetrics = []; window.graphFrames = []; window.frames = []; window.playedSamples = 0; window.duplexCaptures = 0;
    window.addEventListener('eric-connectome', e => window.graphFrames.push(e.detail));
    window.addEventListener('eric-metrics', e => window.testMetrics.push(e.detail));
    window.addEventListener('eric-frame', e => window.frames.push(e.detail));
    window.addEventListener('eric-playback', e => { window.playedSamples += e.detail.samples; });
    window.addEventListener('eric-capture', e => { if (e.detail.duringPlayback) window.duplexCaptures++; });
  });
  await page.setViewportSize({width: 1440, height: 1000});
  await page.goto('/');
  const start = page.getByRole('button', {name: /Call fly/});
  await expect(start).toBeEnabled({timeout: 90_000});
  await start.click();
  await expect(page.locator('[data-neural="input"]')).toHaveText('Live model reservoir', {timeout: process.env.EXPECT_MODAL ? 150_000 : 20_000});
  await expect.poll(() => page.evaluate(() => window.duplexCaptures), {timeout: 30_000}).toBeGreaterThan(10);
  await expect.poll(() => page.evaluate(() => window.frames.length), {timeout: 45_000}).toBeGreaterThan(375);
  await expect(page.locator('#error')).toBeHidden();
  expect(await page.evaluate(() => window.playedSamples)).toBeGreaterThan(24000*25);
  const samples = await page.evaluate(() => ({
    runtime:window.runtimeReady,
    logitRms:window.frames.map(f=>f.reservoir_logit_rms),
    features: window.frames.slice(-30).map(f => f.features),
    text: window.frames.map(f => f.text).join(''),
    last: window.testMetrics.at(-1), captures: window.duplexCaptures, graph: window.graphFrames,
  }));
  if(process.env.EXPECT_MODAL){
    expect(samples.runtime.execution).toBe('torch');expect(samples.runtime.gpu).toMatch(/NVIDIA|Tesla/);
    expect(samples.runtime.codec).toBe('torch-cuda');
    expect(samples.runtime.region).toMatch(/^us-/);
  }
  expect(samples.logitRms.some(value=>value>0)).toBeTruthy();
  expect(samples.features.every(f => f.length === 64 && f.every(Number.isFinite))).toBeTruthy();
  expect(new Set(samples.features.map(f => JSON.stringify(f))).size).toBeGreaterThan(10);
  expect(samples.text.trim().length).toBeGreaterThan(10);
  expect(samples.last.queued_frames).toBeLessThan(5);
  expect(samples.graph.length).toBeGreaterThan(80);
  expect(samples.graph.every(f => f.neurons === 166700)).toBeTruthy();
  expect(new Set(samples.graph.map(f => f.rms)).size).toBeGreaterThan(20);
  expect(samples.graph.some(f=>f.bursts>0)).toBeTruthy();
  expect(new Set(samples.graph.map(f=>f.bursts)).size).toBeGreaterThan(10);
  await writeFile('artifacts/moshi/full-connectome-browser.json', JSON.stringify({
    runtime:samples.runtime, meanLogitRms:samples.logitRms.reduce((a,b)=>a+b,0)/samples.logitRms.length,
    moshi: samples.last, duplexCaptures: samples.captures,
    graphUpdates: samples.graph.length,
    meanGraphMs: samples.graph.reduce((sum,f)=>sum+f.compute_ms,0)/samples.graph.length,
    peakGraphMs: Math.max(...samples.graph.map(f=>f.compute_ms)),
    burstCountRange:[Math.min(...samples.graph.map(f=>f.bursts)),Math.max(...samples.graph.map(f=>f.bursts))],
    meanBursts:samples.graph.reduce((sum,f)=>sum+f.bursts,0)/samples.graph.length,
  }, null, 2));
  expect(samples.graph.some(f=>f.bursts>0)).toBeTruthy();
  await expect(page.locator('.transcript-panel')).toBeHidden();
  await page.screenshot({path: 'artifacts/moshi/browser-duplex.png', fullPage: true});
  console.log('MOSHI BROWSER', JSON.stringify({text: samples.text, metrics: samples.last, duplexCaptures: samples.captures}));
  await page.getByRole('button', {name: /End call/}).click();
  await expect(page.locator('[data-neural="input"]')).toHaveText('No conversation input');
  await expect(page.locator('[data-neural="input-meter"]')).toHaveJSProperty('value', 0);
  await expect(page.locator('[data-neural="active"]')).toHaveText('0', {timeout: 5000});
  await expect(page.locator('[data-neural="state"]')).toHaveText('IDLE');
  await expect(page.locator('[data-neural="updates"]')).toHaveText('0');
  await expect(page.locator('[data-neural="compute"]')).toHaveText('—');
  await expect.poll(async () => (await (await page.request.get('/api/health')).json()).busy).toBeFalsy();
  await page.evaluate(() => { window.frames = []; });
  await start.click();
  await expect.poll(() => page.evaluate(() => window.frames.length), {timeout: 15_000}).toBeGreaterThan(10);
  expect(await page.evaluate(() => window.frames[0].index)).toBe(1);
  await page.getByRole('button', {name: /End call/}).click();
  await expect(page.locator('#error')).toBeHidden();
  expect(errors).toEqual([]);
});

test('mobile layout and keyboard-accessible call control', async ({page}) => {
  await page.setViewportSize({width: 390, height: 844});
  await page.goto('/');
  await expect(page.getByRole('button', {name: /Call fly/})).toBeEnabled();
  await page.keyboard.press('Tab');
  await expect(page.getByRole('button', {name: /Call fly/})).toBeFocused();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({path: 'artifacts/moshi/browser-mobile.png', fullPage: true});
});
