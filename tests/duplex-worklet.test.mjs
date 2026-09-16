import {test} from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
const source = await readFile(new URL('../dist/duplex-mic-worklet.js', import.meta.url), 'utf8');
for (const sampleRate of [24000, 44100, 48000]) {
  test(`continuous microphone resampling at ${sampleRate} Hz preserves timing and level`, () => {
    const frames = []; let Processor;
    const context = vm.createContext({sampleRate, Float32Array,
      AudioWorkletProcessor: class {constructor() {this.port={postMessage: f => frames.push(f)};}},
      registerProcessor: (_, cls) => { Processor = cls; },
    });
    new vm.Script(source).runInContext(context);
    const processor = new Processor();
    const input = new Float32Array(sampleRate*.8).fill(.2);
    for(let i=0; i<input.length; i+=128) processor.process([[input.slice(i,i+128)]]);
    assert.equal(frames.length, 10);
    assert.ok(frames.every(f => f.length === 1920 && f.every(v => Math.abs(v-.2) < 1e-6)));
  });
}
