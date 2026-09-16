import {recordDiagnostic, flushDiagnostics} from './call-diagnostics.js';
const callButton = document.querySelector('#call');
const label = document.querySelector('#button-label');
const status = document.querySelector('#status');
const hint = document.querySelector('#hint');
const error = document.querySelector('#error');
const stateLight = document.querySelector('#state-light');
let ready = false, session = null, flyScene = null, neuralPanel = null;
let lastEndMessage = '';

function outputLevel(s) {
  if (!s?.outputAnalyser) return 0;
  s.outputAnalyser.getFloatTimeDomainData(s.outputSamples);
  return Math.sqrt(s.outputSamples.reduce((sum, v) => sum+v*v, 0)/s.outputSamples.length);
}
function conversationSignal() {
  const s = session;
  if (!s?.connected) return {phase: s ? 'connecting' : 'idle'};
  while (s.featureQueue.length && s.featureQueue[0].when <= s.context.currentTime) {
    const item = s.featureQueue.shift();
    s.features = item.features; s.featuresAt = performance.now();
    if(item.brain)window.dispatchEvent(new CustomEvent('eric-brain',{detail:item.brain}));
  }
  return {phase: 'duplex', micLevel: s.micLevel || 0, outputLevel: outputLevel(s),
    features: performance.now()-s.featuresAt < 500 ? s.features : []};
}

import('./full-connectome-panel.js').then(({createNeuralPanel}) =>
  createNeuralPanel(document.querySelector('#neural-panel'), conversationSignal, {duplex: true}))
  .then(panel => { neuralPanel = panel; })
  .catch(() => {
    const message = document.querySelector('#neural-error');
    message.hidden = false; message.textContent = 'The neural display is unavailable. Voice conversation still works.';
  });
import('./fly-scene.js').then(({createFlyScene}) => {
  flyScene = createFlyScene(document.querySelector('#fly-scene'), () => Math.max(session?.micLevel || 0, outputLevel(session)));
  flyScene.setState('idle', false);
}).catch(() => { document.querySelector('#scene-fallback').hidden = false; });

function showError(message) { error.textContent = message; error.hidden = false; }
async function checkHealth() {
  if (session) return;
  try {
    const response = await fetch('/api/health');
    if (!response.ok) throw new Error('Server unavailable');
    const health = await response.json();
    if (session) return;
    void flushDiagnostics();
    ready = health.ready && health.backend === 'moshi';
    callButton.disabled = !ready;
    label.textContent = ready ? 'Call fly' : 'Getting ready';
    status.textContent = health.backend === 'moshi' ? (lastEndMessage || health.stage) : 'Start the Moshi server to use this page.';
    if (health.error) showError('Moshi could not start. Check the local server log.');
  } catch {
    if (session) return;
    ready = false; callButton.disabled = true;
    status.textContent = 'Waiting for the local Moshi server…';
  }
}
function updateActivity() {
  const s = session;
  if (!s) return;
  if (!s.connected) {
    if(s.waitingAt) {
      status.textContent = s.waitingMessage || 'Connecting your call…';
      hint.textContent = `Waiting ${Math.floor((performance.now()-s.waitingAt)/1000)}s. You can cancel with End call.`;
    }
    return;
  }
  const output = outputLevel(s), now = performance.now();
  if (output > .004) s.spokeAt = now;
  const speaking = now-s.spokeAt < 180;
  const listening = s.micLevel > .012;
  if (s.maxSeconds && s.inputSamples) {
    const remaining = Math.max(0, Math.ceil(s.maxSeconds-s.inputSamples/24000));
    if (remaining <= 30) hint.textContent = `Call limit in ${remaining} seconds. You can start a new call afterward.`;
  }
  status.textContent = speaking ? (listening ? 'Listening and speaking…' : 'Eric is speaking—and listening.') : 'Eric is listening…';
  const state = speaking ? 'speaking' : 'listening';
  stateLight.dataset.state = state;
  flyScene?.setState(state, true);
}

function playChunk(s, buffer) {
  if (session !== s || !s.connected) return;
  const samples = new Float32Array(buffer);
  s.outputFrames++;
  if (!samples.length) return;
  // Keep a small fixed cushion; never let playback silently drift seconds behind.
  if (s.playAt-s.context.currentTime > .8) {
    showError('Playback fell behind live speech. Please restart the call.');
    stopConversation('Call stopped because audio playback fell behind.', 'playback_backlog'); return;
  }
  const audio = s.context.createBuffer(1, samples.length, 24000);
  audio.copyToChannel(samples, 0);
  const source = s.context.createBufferSource(); source.buffer = audio;
  source.connect(s.outputAnalyser); s.sources.add(source);
  const when = Math.max(s.context.currentTime+.06, s.playAt);
  s.playAt = when+audio.duration;
  if (s.pendingFrame) s.featureQueue.push({when, features: s.pendingFrame.features, brain:s.pendingFrame.brain});
  s.pendingFrame = null;
  source.onended = () => { source.disconnect(); s.sources.delete(source); };
  source.start(when);
  window.dispatchEvent(new CustomEvent('eric-playback', {detail: {samples: samples.length,
    buffered_s: s.playAt-s.context.currentTime, context_s: s.context.currentTime, wall_ms: performance.now()}}));
}
function capture(s, samples) {
  if (session !== s || !s.connected || s.socket?.readyState !== WebSocket.OPEN) return;
  s.lastCaptureAt=performance.now();
  s.micLevel = Math.sqrt(samples.reduce((sum,v) => sum+v*v,0)/samples.length);
  if (s.socket.bufferedAmount > 1920*4*8) {
    showError('The audio connection is falling behind. Please restart the call.');
    stopConversation('Call stopped because the audio connection fell behind.', 'transport_backlog'); return;
  }
  // Microphone frames continue while output audio is playing (true duplex).
  s.socket.send(samples.buffer);
  s.inputSamples += samples.length;
  window.dispatchEvent(new CustomEvent('eric-capture', {detail: {duringPlayback: outputLevel(s) > .004,
    context_s: s.context.currentTime, wall_ms: performance.now()}}));
}

async function stopConversation(message = 'You ended the conversation.', reason = 'user_stop') {
  const s = session; session = null;
  if (!s) return;
  lastEndMessage = message;
  recordDiagnostic(s, 'client_end', reason);
  clearInterval(s.diagnosticTimer);
  if (s) {
    s.connected = false;
    s.stream?.getTracks().forEach(track => track.stop());
    s.captureNode?.disconnect(); s.inputNode?.disconnect();
    for (const source of s.sources) { try { source.stop(); } catch {} }
    s.outputAnalyser?.disconnect();
    s.featureQueue.length = 0; s.features = [];
    if (s.socket?.readyState === WebSocket.OPEN) s.socket.send(JSON.stringify({type:'stop', reason}));
    s.socket?.close();
    if (s.context.state !== 'closed') await s.context.close();
  }
  if (session) return;
  callButton.dataset.active = 'false'; callButton.disabled = !ready;
  label.textContent = 'Call fly'; status.textContent = message;
  hint.textContent = s.callId ? `Call ${s.callId}` : '';
  stateLight.dataset.state = 'idle'; flyScene?.setState('idle', false);
}
async function startConversation() {
  lastEndMessage = '';
  error.hidden = true;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    showError('Microphone access requires localhost or HTTPS.'); return;
  }
  const context = new AudioContext({latencyHint: 'interactive'});
  const s = {context, sources: new Set(), playAt: 0, connected: false, micLevel: 0,
    inputSamples: 0, maxSeconds: 0, clientId:crypto.randomUUID(), startedAt:performance.now(),
    outputFrames:0, lastServerFrame:0, lastFrameAt:null, lastCaptureAt:null,
    featureQueue: [], features: [], featuresAt: -Infinity, spokeAt: -Infinity};
  session = s;
  s.diagnosticTimer=setInterval(()=>recordDiagnostic(s,'client_sample'),5000);
  context.addEventListener('statechange',()=>{if(session===s)recordDiagnostic(s,'audio_state');});
  s.outputAnalyser = context.createAnalyser(); s.outputAnalyser.fftSize = 512;
  s.outputSamples = new Float32Array(512); s.outputAnalyser.connect(context.destination);
  label.textContent = 'End call'; callButton.dataset.active = 'true';
  status.textContent = 'Allow your microphone to begin.';
  flyScene?.setState('connecting', true);
  try {
    await context.resume();
    const stream = await navigator.mediaDevices.getUserMedia({audio: {
      echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1,
    }});
    if (session !== s) { stream.getTracks().forEach(track => track.stop()); return; }
    s.stream = stream;
    stream.getAudioTracks().forEach(track => {
      track.addEventListener('ended', () => {if(session===s)stopConversation('Call ended because microphone access stopped.', 'microphone_ended');});
      track.addEventListener('mute',()=>{if(session===s)recordDiagnostic(s,'microphone_mute');});
      track.addEventListener('unmute',()=>{if(session===s)recordDiagnostic(s,'microphone_unmute');});
    });
    await context.audioWorklet.addModule('/duplex-mic-worklet.js');
    if (session !== s) return;
    s.inputNode = context.createMediaStreamSource(stream);
    s.captureNode = new AudioWorkletNode(context, 'duplex-microphone');
    s.captureNode.port.onmessage = event => capture(s, event.data);
    s.inputNode.connect(s.captureNode); s.captureNode.connect(context.destination);
    s.waitingAt=performance.now();s.waitingMessage='Connecting your call…';
    status.textContent=s.waitingMessage;
    const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/conversation`);
    s.socket = socket; socket.binaryType = 'arraybuffer';
    socket.onmessage = event => {
      if (session !== s) return;
      if (typeof event.data !== 'string') { playChunk(s, event.data); return; }
      const data = JSON.parse(event.data);
      if(data.type==='warming'){
        s.waitingMessage=data.phase==='queued'
          ? 'Eric is busy right now. Your call will start automatically when he’s available.'
          : data.message;
        status.textContent=s.waitingMessage;
      } else if (data.type === 'call_started') {
        s.callId=data.call_id;
      } else if (data.type === 'ready') {
        s.callId=data.call_id||s.callId;
        window.dispatchEvent(new CustomEvent('eric-ready',{detail:data}));
        s.connected = true;
        s.maxSeconds = data.max_call_seconds;
        hint.textContent = 'Speak naturally. You can talk while Eric speaks. Calls last up to five minutes.';
        updateActivity();
      } else if (data.type === 'frame') {
        s.lastServerFrame=data.index; s.lastFrameAt=performance.now();
        if (data.audio_samples) s.pendingFrame = data;
        else { s.features = data.features; s.featuresAt = performance.now();
          if(data.brain)window.dispatchEvent(new CustomEvent('eric-brain',{detail:data.brain})); }
        window.dispatchEvent(new CustomEvent('eric-frame', {detail: data}));
      } else if (data.type === 'metrics') {
        window.dispatchEvent(new CustomEvent('eric-metrics', {detail: data}));
      } else if (data.type === 'error') {
        showError(data.message);
        stopConversation(data.message, 'server_error');
      } else if (data.type === 'session_end') {
        stopConversation(data.message, 'session_limit');
      }
    };
    socket.onerror = () => {
      if (session === s) {
        showError('Connection interrupted. Please start a new call.');
        stopConversation('The connection was interrupted. Start a new call.', 'connection_error');
      }
    };
    socket.onclose = event => {
      recordDiagnostic(s,'socket_close',null,event);
      if (session === s) stopConversation(`The speech connection closed unexpectedly (code ${event.code}). Start a new call.`, 'connection_error');
    };
  } catch (err) {
    if (session !== s) return;
    showError(err.name === 'NotAllowedError' ? 'Microphone permission was denied. Allow access and try again.' : `Could not start the microphone: ${err.message}`);
    await stopConversation('The call could not start.', 'connection_error');
  }
}
callButton.addEventListener('click', () => session ? stopConversation() : startConversation());
document.addEventListener('visibilitychange',()=>{if(session)recordDiagnostic(session,'page_visibility');});
window.addEventListener('pagehide', () => { if (session) stopConversation('Page closed.', 'page_closed'); neuralPanel?.dispose(); });
checkHealth();
setInterval(checkHealth, 2500);
setInterval(updateActivity, 40);
