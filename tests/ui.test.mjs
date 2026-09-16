import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';

const source = await readFile(new URL('../dist/legacy-app.js', import.meta.url), 'utf8');
const flush = () => new Promise(resolve => setImmediate(resolve));

function fixture({denied = false} = {}) {
  class Element {
    constructor() { this.dataset = {}; this.children = []; this.hidden = false; this.listeners = {}; this.textContent = ''; }
    addEventListener(name, fn) { this.listeners[name] = fn; }
    append(...items) { for (const item of items) { item.parent = this; this.children.push(item); } }
    replaceChildren() { this.children = []; }
    get firstElementChild() { return this.children[0]; }
    remove() { this.parent.children.splice(this.parent.children.indexOf(this), 1); }
  }
  const elements = new Map();
  const get = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  get('#scene-fallback').hidden = true;
  get('#error').hidden = true;
  const timers = new Map(); let timerId = 0, trackStops = 0;
  const sockets = [], contexts = [], startedSources = [];
  class AudioContext {
    constructor() {
      this.state = 'running'; this.currentTime = 0; this.sampleRate = 48000;
      this.destination = {}; this.audioWorklet = {addModule: async () => {}}; contexts.push(this);
    }
    async resume() {}
    async close() { this.state = 'closed'; }
    createAnalyser() { return {connect() {}, disconnect() {}, getFloatTimeDomainData(a) { a.fill(0); }}; }
    createMediaStreamSource() { return {connect() {}, disconnect() {}}; }
    createBuffer(channels, length, rate) { return {duration: length / rate, copyToChannel() {}}; }
    createBufferSource() {
      return {connect(to) { this.connectedTo = to; }, disconnect() {}, start() { startedSources.push(this); }, stop() { this.stopped = true; }};
    }
  }
  class Socket {
    static OPEN = 1;
    constructor() { this.readyState = 1; this.sent = []; sockets.push(this); }
    send(data) { this.sent.push(data); }
    close() { this.readyState = 3; this.onclose?.(); }
    emit(data) { this.onmessage({data: typeof data === 'object' && !(data instanceof ArrayBuffer) ? JSON.stringify(data) : data}); }
  }
  const sandbox = {
    document: {querySelector: get, createElement: () => new Element()},
    window: {isSecureContext: true, addEventListener() {}, dispatchEvent() {}},
    navigator: {mediaDevices: {getUserMedia: async () => {
      if (denied) throw Object.assign(new Error('denied'), {name: 'NotAllowedError'});
      return {getTracks: () => [{stop: () => trackStops++}]};
    }}},
    fetch: async () => ({ok: true, json: async () => ({ready: true, stage: 'Ready when you are.'})}),
    AudioContext, AudioWorkletNode: class { constructor() { this.port = {}; } connect() {} disconnect() {} },
    WebSocket: Socket, location: {protocol: 'http:', host: 'localhost:8765'},
    setTimeout: fn => { timers.set(++timerId, fn); return timerId; }, clearTimeout: id => timers.delete(id), setInterval() {},
    Float32Array, ArrayBuffer, DataView, console: {warn() {}},
  };
  const context = vm.createContext(sandbox);
  // Deliberately unavailable 3D module exercises the graceful fallback as well.
  new vm.Script(source, {filename: 'app.js', importModuleDynamically: async () => { throw new Error('No WebGL in this test'); }}).runInContext(context);
  return {get, sockets, contexts, startedSources, timers, stops: () => trackStops};
}

test('voice, streamed transcript, audio routing, cleanup and restart survive missing 3D', async () => {
  const f = fixture(); await flush();
  assert.equal(f.get('#scene-fallback').hidden, false);
  assert.equal(f.get('#call').disabled, false);
  await f.get('#call').listeners.click();
  const ws = f.sockets[0];
  assert.equal(f.get('#connection').textContent, 'CALLING');
  ws.emit({type: 'transcript', role: 'assistant', text: 'Hello!'});
  ws.emit({type: 'audio_start', sample_rate: 24000});
  ws.emit(new Float32Array(2400).buffer);
  assert.equal(f.get('#status').textContent, 'Eric is speaking…');
  assert.equal(f.get('#connection').textContent, 'LIVE');
  assert.equal(typeof f.startedSources[0].connectedTo.getFloatTimeDomainData, 'function');
  ws.emit({type: 'turn_end'});
  f.startedSources[0].onended();
  for (const fn of f.timers.values()) fn(); f.timers.clear();
  assert.equal(f.get('#status').textContent, "I'm listening…");
  ws.emit({type: 'transcript', role: 'user', text: '<script>example</script>'});
  ws.emit({type: 'transcript', role: 'assistant', text: 'Nice'});
  ws.emit({type: 'transcript', role: 'assistant', text: 'Nice to meet you.'});
  const rows = f.get('#transcript').children;
  assert.equal(rows.length, 3);
  assert.equal(rows[1].children[1].textContent, '<script>example</script>');
  assert.equal(rows[2].children[1].textContent, 'Nice to meet you.');
  assert.equal(f.get('#transcript-empty').hidden, true);
  await f.get('#call').listeners.click();
  assert.equal(f.stops(), 1);
  assert.equal(f.contexts[0].state, 'closed');
  assert.equal(ws.sent.at(-1), 'stop');
  assert.equal(f.get('#connection').textContent, 'STANDBY');
  assert.equal(f.get('#transcript').children.length, 3);
  await f.get('#call').listeners.click();
  assert.equal(f.get('#transcript').children.length, 0);
  assert.equal(f.get('#transcript-empty').hidden, false);
  assert.equal(f.sockets.length, 2);
  await f.get('#call').listeners.click();
});

test('microphone denial returns to a usable idle screen', async () => {
  const f = fixture({denied: true}); await flush();
  await f.get('#call').listeners.click();
  assert.equal(f.get('#error').hidden, false);
  assert.match(f.get('#error').textContent, /permission was denied/);
  assert.equal(f.get('#call').dataset.active, 'false');
  assert.equal(f.get('#call').disabled, false);
  assert.equal(f.get('#connection').textContent, 'STANDBY');
  assert.equal(f.contexts[0].state, 'closed');
  assert.equal(f.sockets.length, 0);
});
