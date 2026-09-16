import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {conversationDrive, conversationGroups, featureCurrents, CONVERSATION_PARAMETERS} from '../dist/conversation-drive.js';
import {LocomotorSim} from '../dist/neural/locomotor.js';

const circuit = JSON.parse(await readFile(new URL('../dist/neural/locomotor_circuit.json', import.meta.url)));
const groups = conversationGroups(circuit);

test('silence and disconnected calls supply no drive; louder audio increases bounded input', () => {
  for (const phase of ['idle', 'connecting', 'transcribing', 'listening', 'speaking']) {
    assert.equal(conversationDrive({phase, level: 0}).current, 0);
  }
  for (const phase of ['listening', 'speaking']) {
    assert.equal(conversationDrive({phase, level: NaN}).current, 0);
    assert.ok(conversationDrive({phase, level: .03}).current > conversationDrive({phase, level: .012}).current);
    assert.equal(conversationDrive({phase, level: 100}).current, .24);
  }
});

test('duplex input combines live model features, microphone and playback with deterministic currents', () => {
  const features = Array.from({length: 64}, (_,i) => Math.sin(i*1.3));
  const input = conversationDrive({phase: 'duplex', features, micLevel: .04, outputLevel: .03});
  assert.ok(input.drives.every(d => d.current > 0));
  const currents = featureCurrents(circuit.neurons.length, input.features);
  assert.deepEqual(currents, featureCurrents(circuit.neurons.length, input.features));
  assert.notDeepEqual(currents, featureCurrents(circuit.neurons.length, features.map(v => -v)));
  assert.ok(currents.every(v => Number.isFinite(v) && v >= 0 && v <= .121));
  const sim = new LocomotorSim(circuit, CONVERSATION_PARAMETERS);
  sim.feedbackEnabled = false; sim.drive.set(currents); sim.step(3000);
  assert.ok(sim.totalSpikes > 0);
  sim.drive.fill(0); sim.step(1000);
  assert.ok(Math.max(...sim.rates) < 1);
});

test('real circuit responds to distinct conversation inputs and settles after input stops', () => {
  assert.ok(groups.microphone.length && groups.processing.length && groups.voice.length);
  assert.equal(new Set(Object.values(groups).flat()).size, Object.values(groups).flat().length);
  for (const phase of ['listening', 'thinking', 'synthesizing', 'speaking']) {
    const sim = new LocomotorSim(circuit, CONVERSATION_PARAMETERS);
    sim.feedbackEnabled = false;
    sim.step(400);
    assert.equal(sim.totalSpikes, 0);
    const {group, current} = conversationDrive({phase, level: .04});
    for (const i of groups[group]) sim.drive[i] = current;
    sim.step(5000);
    assert.ok(sim.totalSpikes > 0, phase);
    assert.ok(groups[group].some(i => sim.rates[i] > 1), phase);
    sim.drive.fill(0);
    sim.step(1000);
    assert.ok(Math.max(...sim.rates) < 1, `${phase} should settle`);
  }
});
