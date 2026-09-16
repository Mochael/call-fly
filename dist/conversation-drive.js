// Engineered input mapping, not a claim about fly speech or language circuits.
// Lower recurrent gain keeps the display responsive after sustained audio input.
export const CONVERSATION_PARAMETERS = {synapticGain: 1.2};

export function conversationGroups(circuit) {
  const indices = predicate => circuit.neurons.flatMap((n, i) => predicate(n) ? [i] : []);
  return {
    microphone: indices(n => n.role === 'sensory' && [0, 1].includes(n.leg)),
    processing: indices(n => n.role === 'descending'),
    voice: indices(n => n.role === 'motor' && [0, 1].includes(n.leg) && n.motorChannel === 'tibia_flexor'),
  };
}

function envelope(level, floor, ceiling) {
  return Math.sqrt(Math.max(0, Math.min(1, ((Number.isFinite(level) ? level : 0)-floor)/(ceiling-floor))));
}

export function conversationDrive(signal = {}) {
  switch (signal.phase) {
    case 'duplex': {
      const microphone = .16*envelope(signal.micLevel, .008, .08);
      const voice = .16*envelope(signal.outputLevel, .0015, .045);
      const features = Array.isArray(signal.features) && signal.features.length === 64
        ? signal.features.map(v => Number.isFinite(v) ? Math.max(-1, Math.min(1,v)) : 0) : [];
      return {group: null, current: Math.max(microphone, voice, ...features.map(v => Math.abs(v)*.12)),
        label: features.length ? 'Live Moshi features + audio' : 'Waiting for live Moshi features',
        features, drives: [{group: 'microphone', current: microphone}, {group: 'voice', current: voice}]};
    }
    case 'listening': return {group: 'microphone', current: .24*envelope(signal.level, .008, .08), label: 'Your microphone'};
    case 'thinking': return {group: 'processing', current: .09, label: 'Generating reply'};
    case 'synthesizing': return {group: 'processing', current: .055, label: 'Preparing voice'};
    case 'speaking': return {group: 'voice', current: .24*envelope(signal.level, .0015, .045), label: 'Eric’s audio output'};
    case 'transcribing': return {group: null, current: 0, label: 'Recognizing your speech'};
    case 'connecting': return {group: null, current: 0, label: 'Connecting'};
    default: return {group: null, current: 0, label: 'No conversation input'};
  }
}

// An untrained, fixed sparse projection. No anatomical meaning is assigned to
// a model feature: every neuron receives a distinct combination of channels.
export function featureCurrents(count, features = []) {
  const result = new Float32Array(count);
  if (features.length !== 64) return result;
  for (let i = 0; i < count; i++) {
    let seed = Math.imul(i+1, 2654435761) >>> 0, sum = 0;
    for (let j = 0; j < 8; j++) {
      seed = (Math.imul(seed, 1664525)+1013904223) >>> 0;
      sum += features[(seed >>> 16) % 64] * (seed & 256 ? 1 : -1);
    }
    result[i] = Math.max(0, Math.min(.12, .065*sum/Math.sqrt(8)));
  }
  return result;
}
