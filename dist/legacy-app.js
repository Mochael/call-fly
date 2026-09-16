const callButton = document.querySelector('#call');
const label = document.querySelector('#button-label');
const status = document.querySelector('#status');
const hint = document.querySelector('#hint');
const error = document.querySelector('#error');
const stateLight = document.querySelector('#state-light');
const connection = document.querySelector('#connection');
const emptyTranscript = document.querySelector('#transcript-empty');
const transcript = document.querySelector('#transcript');
const stateLabels = {listening: "I'm listening…", transcribing: 'One moment…', thinking: 'Eric is thinking…', speaking: 'Eric is speaking…'};

let ready = false;
let session = null;
let nextId = 0;
let flyScene = null;
let sceneState = 'idle';
let neuralPanel = null;

import('./neural-panel.js').then(({createNeuralPanel}) => createNeuralPanel(document.querySelector('#neural-panel'), conversationSignal))
  .then(panel => { neuralPanel = panel; })
  .catch(() => {
    const message = document.querySelector('#neural-error');
    message.hidden = false;
    message.textContent = 'The neural display is unavailable. You can still talk to Eric.';
  });
window.addEventListener('pagehide', () => neuralPanel?.dispose());

function audioLevel() {
  if (!session) return 0;
  if (session.listening) return session.micLevel || 0;
  if (!session.outputAnalyser) return 0;
  session.outputAnalyser.getFloatTimeDomainData(session.outputSamples);
  let sum = 0;
  for (const sample of session.outputSamples) sum += sample * sample;
  return Math.sqrt(sum / session.outputSamples.length);
}

function conversationSignal() {
  if (!session) return {phase: 'idle', level: 0};
  if (session.listening) return {phase: 'listening', level: session.micLevel || 0};
  // Read the output analyser at playback time, not when PCM/text arrives.
  if (session.sources.size) return {phase: 'speaking', level: audioLevel()};
  return {phase: session.processingPhase || 'connecting', level: 0};
}

// Rendering is optional: a missing WebGL context must never break a voice call.
import('./fly-scene.js').then(({createFlyScene}) => {
  flyScene = createFlyScene(document.querySelector('#fly-scene'), audioLevel);
  flyScene.setState(sceneState, Boolean(session));
}).catch(err => {
  document.querySelector('#scene-fallback').hidden = false;
  console.warn('3D view unavailable:', err);
});

function setState(value) {
  if (session && ['thinking', 'transcribing'].includes(value)) session.processingPhase = value;
  sceneState = stateLabels[value] ? value : session ? 'connecting' : 'idle';
  stateLight.dataset.state = sceneState;
  flyScene?.setState(sceneState, Boolean(session));
  connection.textContent = session ? (sceneState === 'connecting' ? 'CALLING' : 'LIVE') : 'STANDBY';
  connection.dataset.active = String(Boolean(session));
  status.textContent = stateLabels[value] || value;
}

function showError(message) { error.textContent = message; error.hidden = false; }

async function checkHealth() {
  if (session) return;
  try {
    const response = await fetch('/api/health');
    if (!response.ok) throw new Error('Server unavailable');
    const health = await response.json();
    if (session) return;
    ready = health.ready;
    callButton.disabled = !ready;
    label.textContent = ready ? 'Start conversation' : 'Getting ready';
    status.textContent = health.stage;
    if (health.error) showError('Eric could not start. Check the local server and restart it.');
  } catch {
    if (session) return;
    ready = false;
    callButton.disabled = true;
    label.textContent = 'Connecting';
    status.textContent = 'Waiting for the local server…';
  }
}

function addText(s, role, text) {
  if (!text || session !== s) return;
  emptyTranscript.hidden = true;
  if (role === 'user' || !s.assistantElement) {
    const row = document.createElement('p');
    row.className = `turn ${role}`;
    const name = document.createElement('strong');
    name.textContent = role === 'user' ? 'YOU' : 'ERIC';
    const content = document.createElement('span');
    content.textContent = text;
    row.append(name, content);
    transcript.append(row);
    if (role === 'user') s.assistantElement = null;
    else s.assistantElement = content;
    while (transcript.children.length > 24) transcript.firstElementChild.remove();
  } else s.assistantElement.textContent = text;
  transcript.scrollTop = transcript.scrollHeight;
}

function resetCapture(s) {
  s.micLevel = 0;
  s.frames = []; s.preRoll = []; s.speech = false; s.voiced = 0; s.silence = 0; s.frameCount = 0;
}

function maybeListen(s) {
  if (session !== s || !s.turnComplete || s.sources.size) return;
  clearTimeout(s.listenTimer);
  s.listenTimer = setTimeout(() => {
    if (session !== s || s.sources.size || !s.turnComplete) return;
    resetCapture(s);
    s.listening = true;
    s.assistantElement = null;
    setState('listening');
    hint.textContent = 'Speak naturally. Eric replies when you pause.';
  }, 300); // Let the loudspeaker and room echo settle before listening.
}

function playChunk(s, buffer) {
  if (session !== s) return;
  const floats = new Float32Array(buffer);
  if (!floats.length) return;
  error.hidden = true; // A successful reply clears any earlier recoverable voice glitch.
  s.listening = false;
  s.processingPhase = 'waiting';
  setState('speaking');
  const audio = s.context.createBuffer(1, floats.length, s.sampleRate);
  audio.copyToChannel(floats, 0);
  const source = s.context.createBufferSource();
  source.buffer = audio;
  source.connect(s.outputAnalyser);
  s.sources.add(source);
  const when = Math.max(s.context.currentTime + 0.08, s.playAt);
  s.playAt = when + audio.duration;
  source.onended = () => { source.disconnect(); s.sources.delete(source); maybeListen(s); };
  source.start(when);
}

function encodePCM(frames, sourceRate) {
  const total = frames.reduce((count, frame) => count + frame.length, 0);
  const input = new Float32Array(total);
  let offset = 0;
  for (const frame of frames) { input.set(frame, offset); offset += frame.length; }
  // Area-average resampling is adequate for microphone speech; wire format is mono 16 kHz PCM16 LE.
  const ratio = sourceRate / 16000;
  const length = Math.floor(input.length / ratio);
  const output = new ArrayBuffer(length * 2);
  const view = new DataView(output);
  for (let i = 0; i < length; i++) {
    const start = Math.floor(i * ratio), end = Math.max(start + 1, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j++) sum += input[j] || 0;
    const value = Math.max(-1, Math.min(1, sum / (end - start)));
    view.setInt16(i * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
  }
  return output;
}

function capture(s, frame) {
  if (session !== s || !s.listening || s.socket?.readyState !== WebSocket.OPEN) return;
  const duration = frame.length / s.context.sampleRate;
  let energy = 0;
  for (const v of frame) energy += v * v;
  const rms = Math.sqrt(energy / frame.length);
  s.micLevel = rms;
  s.preRoll.push(frame);
  while (s.preRoll.length > Math.ceil(0.25 / duration)) s.preRoll.shift();
  if (!s.speech) {
    s.voiced = rms > 0.012 ? s.voiced + duration : 0;
    if (s.voiced < 0.08) return;
    s.speech = true;
    s.frames = [...s.preRoll];
    s.frameCount = s.frames.length;
  } else { s.frames.push(frame); s.frameCount++; }
  s.silence = rms < 0.008 ? s.silence + duration : 0;
  if (s.silence >= 0.65 || s.frameCount * duration >= 25) {
    s.listening = false;
    s.turnComplete = false;
    const payload = encodePCM(s.frames, s.context.sampleRate);
    resetCapture(s);
    s.socket.send(payload);
    setState('transcribing');
    hint.textContent = 'End the conversation any time.';
  }
}

async function stopConversation(message = 'Ready when you are.') {
  const s = session;
  session = null;
  if (s) {
    clearTimeout(s.listenTimer);
    s.listening = false;
    s.stream?.getTracks().forEach(track => track.stop());
    for (const source of s.sources) { try { source.stop(); } catch {} }
    s.captureNode?.disconnect();
    s.inputNode?.disconnect();
    s.outputAnalyser?.disconnect();
    if (s.socket?.readyState === WebSocket.OPEN) s.socket.send('stop');
    s.socket?.close();
    if (s.context.state !== 'closed') await s.context.close();
  }
  callButton.dataset.active = 'false';
  callButton.disabled = !ready;
  label.textContent = 'Start conversation';
  hint.textContent = 'Your microphone stays off until you start.';
  setState(message);
}

async function startConversation() {
  error.hidden = true;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    showError('Microphone access needs localhost or HTTPS. Open this app at http://localhost:8765.');
    return;
  }
  transcript.replaceChildren();
  emptyTranscript.hidden = false;
  const context = new AudioContext();
  const s = {id: ++nextId, context, sources: new Set(), playAt: 0, sampleRate: 24000,
    listening: false, turnComplete: false, assistantElement: null};
  session = s;
  s.outputAnalyser = context.createAnalyser();
  s.outputAnalyser.fftSize = 256;
  s.outputSamples = new Float32Array(s.outputAnalyser.fftSize);
  s.outputAnalyser.connect(context.destination);
  resetCapture(s);
  label.textContent = 'End conversation';
  callButton.dataset.active = 'true';
  setState('Allow your microphone to begin.');
  try {
    await context.resume();
    const stream = await navigator.mediaDevices.getUserMedia({audio: {
      echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1,
    }});
    if (session !== s) { stream.getTracks().forEach(track => track.stop()); return; }
    s.stream = stream;
    await context.audioWorklet.addModule('/mic-worklet.js');
    if (session !== s) return;
    s.inputNode = context.createMediaStreamSource(stream);
    s.captureNode = new AudioWorkletNode(context, 'microphone-capture');
    s.captureNode.port.onmessage = event => capture(s, event.data);
    s.inputNode.connect(s.captureNode);
    s.captureNode.connect(context.destination); // Processor emits silence, never microphone monitoring.
    setState('Connecting to Eric…');
    const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/conversation`);
    s.socket = socket;
    socket.binaryType = 'arraybuffer';
    socket.onmessage = event => {
      if (session !== s) return;
      if (typeof event.data !== 'string') { playChunk(s, event.data); return; }
      const data = JSON.parse(event.data);
      switch (data.type) {
        case 'state': setState(data.state); break;
        case 'transcript':
          if (data.role === 'assistant') s.processingPhase = 'synthesizing';
          addText(s, data.role, data.text); break;
        case 'audio_start': s.sampleRate = data.sample_rate; s.turnComplete = false; break;
        case 'turn_end': s.turnComplete = true; maybeListen(s); break;
        case 'no_speech': hint.textContent = 'I didn’t catch that. Try speaking a little louder.'; break;
        case 'error': showError(data.message); break;
        case 'metrics': window.dispatchEvent(new CustomEvent('eric-metrics', {detail: data})); break;
      }
    };
    socket.onerror = () => { if (session === s) showError('Connection interrupted. Please try again.'); };
    socket.onclose = () => { if (session === s) stopConversation('Conversation ended.'); };
  } catch (err) {
    if (session !== s) return;
    const denied = err.name === 'NotAllowedError';
    showError(denied ? 'Microphone permission was denied. Allow microphone access in your browser, then try again.' : `Could not start the microphone: ${err.message}`);
    await stopConversation();
  }
}

callButton.addEventListener('click', () => session ? stopConversation() : startConversation());
window.addEventListener('pagehide', () => { if (session) stopConversation(); });
checkHealth();
setInterval(checkHealth, 2500);
