import {LocomotorSim} from './neural/locomotor.js';
import {conversationGroups, conversationDrive, featureCurrents, CONVERSATION_PARAMETERS} from './conversation-drive.js';

let circuit, sim, running = false, pulseUntil = 0, pulseSide = 0;
let groups, inputs, lastWall = 0, signal = {}, signalAt = -Infinity;

function reset() {
  sim = new LocomotorSim(circuit, CONVERSATION_PARAMETERS);
  sim.feedbackEnabled = false;
  groups = [0, 1].map(leg => circuit.neurons.flatMap((n, i) =>
    n.role === 'motor' && n.leg === leg && n.motorChannel === 'tibia_flexor' ? [i] : []));
  inputs = conversationGroups(circuit);
  pulseUntil = 0;
  lastWall = performance.now();
}

function snapshot(peaks = sim.rates, stimulation = sim.drive) {
  const rates = Float32Array.from(peaks), drive = Float32Array.from(stimulation);
  postMessage({type: 'frame', rates, drive, simMs: sim.simMs, totalSpikes: sim.totalSpikes},
    [rates.buffer, drive.buffer]);
}

onmessage = ({data}) => {
  try {
    if (data.type === 'init') {
      circuit = data.circuit;
      reset(); running = data.running;
      snapshot();
    } else if (data.type === 'running') {
      running = data.value; lastWall = performance.now();
    } else if (data.type === 'conversation') {
      signal = data.signal; signalAt = performance.now();
    } else if (data.type === 'reset' && sim) {
      reset(); snapshot();
    } else if (data.type === 'pulse' && sim && running) {
      pulseSide = data.side === 1 ? 1 : 0;
      pulseUntil = sim.simMs + 220;
    }
  } catch (error) {
    running = false;
    postMessage({type: 'error', message: error.message});
  }
};

setInterval(() => {
  if (!sim || !running) return;
  const now = performance.now();
  const steps = Math.min(50, Math.max(1, Math.round(now-lastWall)));
  lastWall = now;
  const peaks = new Float32Array(sim.n), stimulation = new Float32Array(sim.n);
  // Expire input if the main thread stops reporting audio or call state.
  const input = conversationDrive(now-signalAt < 250 ? signal : {});
  const projected = featureCurrents(sim.n, input.features);
  for (let step = 0; step < steps; step++) {
    sim.drive.set(projected);
    if (input.group) for (const i of inputs[input.group]) sim.drive[i] = input.current;
    for (const extra of input.drives || []) {
      for (const i of inputs[extra.group]) sim.drive[i] = Math.min(.25, sim.drive[i]+extra.current);
    }
    if (sim.simMs < pulseUntil) {
      for (const i of groups[pulseSide]) sim.drive[i] = .25;
    }
    sim.step(1);
    for (let i = 0; i < sim.n; i++) {
      peaks[i] = Math.max(peaks[i], sim.rates[i]);
      stimulation[i] = Math.max(stimulation[i], sim.drive[i]);
    }
  }
  snapshot(peaks, stimulation);
}, 40);
