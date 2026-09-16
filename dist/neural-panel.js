import {conversationGroups, conversationDrive} from './conversation-drive.js';

const COLORS = {motor: '#edad70', premotor: '#67c0a4', descending: '#b89de4', ascending: '#74abe1', sensory: '#91a6af'};
const LABELS = {motor: 'Motor', premotor: 'Premotor', descending: 'Descending', ascending: 'Ascending', sensory: 'Sensory'};

export async function createNeuralPanel(root, readConversation = () => ({phase: 'idle'}), {duplex = false} = {}) {
  const get = name => root.querySelector(`[data-neural="${name}"]`);
  const canvas = get('canvas'), context = canvas.getContext('2d');
  if (!context) throw new Error('Canvas is unavailable');
  const response = await fetch('/neural/locomotor_circuit.json');
  if (!response.ok) throw new Error('The circuit data could not load');
  const circuit = await response.json();
  const inputs = conversationGroups(circuit);
  const raw = circuit.neurons.map(n => n.annotations?.somaLocation);
  const points = raw.map(p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite)
    ? [p[0]+.35*p[1], p[2]] : null);
  const visible = points.filter(Boolean);
  const xs = visible.map(p => p[0]), ys = visible.map(p => p[1]);
  const bounds = {minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys)};
  const edges = circuit.edges.filter(([a,b]) => points[a] && points[b])
    .sort((a,b) => Math.abs(b[2])-Math.abs(a[2])).slice(0, 350);
  const worker = new Worker('/neural-worker.js', {type: 'module'});
  let rates = new Float32Array(circuit.neurons.length), drive = rates.slice();
  let running = !matchMedia('(prefers-reduced-motion: reduce)').matches;
  let selected = -1, hovered = -1, width = 0, height = 0, mapped = [], queued = false, disposed = false;

  get('summary').textContent = `${visible.length.toLocaleString()} positioned neurons · ${circuit.neurons.length.toLocaleString()} simulated`;
  get('edges').textContent = `${edges.length} of ${circuit.edges.length.toLocaleString()} connections shown`;
  get('mapping').textContent = duplex
    ? `64 channels projected from Moshi’s temporal-transformer state continuously stimulate all ${circuit.neurons.length.toLocaleString()} cells through a fixed, untrained mapping. Microphone volume additionally drives ${inputs.microphone.length} front-leg sensory cells; audible output drives ${inputs.voice.length} flexor motor cells. Listening and speaking inputs can overlap. This is a one-way visualization: no circuit output goes back into Moshi.`
    : `Microphone → ${inputs.microphone.length} front-leg sensory cells. Reply/voice generation → ${inputs.processing.length} descending cells. Audible playback → ${inputs.voice.length} front-leg flexor motor cells. These are engineered assignments, not known speech functions.`;

  function inspect(index) {
    const n = circuit.neurons[index];
    if (!n) {
      get('cell-name').textContent = 'Explore a neuron';
      get('cell-detail').textContent = 'Hover or select a point to see its identity and activity.';
      get('cell-rate').textContent = '—';
      return;
    }
    get('cell-name').textContent = n.type;
    const assignment = Object.entries(inputs).find(([group,ids]) => (!duplex || group !== 'processing') && ids.includes(index))?.[0];
    get('cell-detail').textContent = `ID ${n.id} · ${LABELS[n.role] || n.role} · ${n.side || 'side unspecified'} · transmitter: ${n.neurotransmitter?.consensus_nt || 'unclear'}${duplex ? ' · model features' : ''}${assignment ? ` · input: ${assignment}` : ''}`;
    get('cell-rate').textContent = `${Math.round(rates[index])} Hz`;
  }

  function render() {
    queued = false;
    if (disposed || !width || !height) return;
    const c = context;
    c.clearRect(0, 0, width, height);
    c.fillStyle = '#101e1a'; c.fillRect(0, 0, width, height);
    const scale = Math.min((width-38)/(bounds.maxX-bounds.minX), (height-45)/(bounds.maxY-bounds.minY));
    mapped = points.map(p => p ? {x: width/2+(p[0]-(bounds.minX+bounds.maxX)/2)*scale,
      y: height/2+(p[1]-(bounds.minY+bounds.maxY)/2)*scale} : null);
    for (const [a,b] of edges) {
      const p = mapped[a], q = mapped[b];
      c.strokeStyle = `rgba(98,182,155,${.055+Math.min(1,rates[a]/150)*.24})`;
      c.lineWidth = .6;
      c.beginPath(); c.moveTo(p.x,p.y); c.lineTo(q.x,q.y); c.stroke();
    }
    for (let i=0; i<mapped.length; i++) {
      const p = mapped[i]; if (!p) continue;
      const intensity = Math.min(1,rates[i]/150), color = COLORS[circuit.neurons[i].role] || '#9aabaa';
      c.fillStyle = color;
      if (intensity > .05) {
        c.globalAlpha = intensity*.17;
        c.beginPath(); c.arc(p.x,p.y,3+intensity*6,0,2*Math.PI); c.fill();
      }
      c.globalAlpha = .3+.7*intensity;
      c.beginPath(); c.arc(p.x,p.y,1.2+intensity*1.9,0,2*Math.PI); c.fill();
      c.globalAlpha = 1;
      if (drive[i] > 0 || i === selected || i === hovered) {
        c.strokeStyle = drive[i] > 0 ? '#fff2bb' : '#ffffff'; c.lineWidth = 1;
        c.beginPath(); c.arc(p.x,p.y,5,0,2*Math.PI); c.stroke();
      }
    }
    inspect(hovered >= 0 ? hovered : selected);
  }

  function redraw() {
    if (!queued && !disposed) { queued = true; requestAnimationFrame(render); }
  }

  const resize = new ResizeObserver(() => {
    const rect = canvas.getBoundingClientRect(); width = rect.width; height = rect.height;
    const ratio = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width*ratio); canvas.height = Math.round(height*ratio);
    context.setTransform(ratio,0,0,ratio,0,0); redraw();
  });
  resize.observe(canvas);

  function status() {
    get('toggle').textContent = running ? 'Pause' : 'Resume';
    get('state').textContent = running ? 'RUNNING' : 'PAUSED';
    get('pulse').disabled = !running;
    worker.postMessage({type: 'running', value: running && !document.hidden});
  }
  get('toggle').addEventListener('click', () => { running = !running; status(); });
  get('reset').addEventListener('click', () => worker.postMessage({type: 'reset'}));
  get('pulse').addEventListener('click', () => worker.postMessage({type: 'pulse', side: Number(get('side').value)}));

  function nearest(event) {
    const r = canvas.getBoundingClientRect();
    const x = event.clientX-r.left, y = event.clientY-r.top;
    let index = -1, distance = 15;
    mapped.forEach((p,i) => { if (p) { const d = Math.hypot(x-p.x,y-p.y); if (d<distance) { distance=d; index=i; } } });
    return index;
  }
  canvas.addEventListener('pointermove', event => { hovered = nearest(event); redraw(); });
  canvas.addEventListener('pointerleave', () => { hovered = -1; redraw(); });
  canvas.addEventListener('click', event => { selected = nearest(event); redraw(); });
  canvas.addEventListener('keydown', event => {
    if (!['ArrowRight','ArrowLeft','Escape'].includes(event.key)) return;
    event.preventDefault();
    if (event.key === 'Escape') selected = -1;
    else {
      const step = event.key === 'ArrowRight' ? 1 : -1;
      do { selected = (selected+step+points.length)%points.length; } while (!points[selected]);
    }
    hovered = -1; redraw();
  });

  worker.onmessage = ({data}) => {
    if (disposed) return;
    if (data.type === 'error') { get('error').hidden = false; get('error').textContent = data.message; return; }
    if (data.type !== 'frame') return;
    rates = data.rates; drive = data.drive;
    get('active').textContent = rates.filter(rate => rate > 1).length.toLocaleString();
    get('spikes').textContent = data.totalSpikes.toLocaleString();
    get('time').textContent = `${(data.simMs/1000).toFixed(2)} s`;
    redraw();
  };
  worker.onerror = () => { get('error').hidden = false; get('error').textContent = 'The neural simulation could not start. Voice conversation is still available.'; };
  const visibility = () => worker.postMessage({type: 'running', value: running && !document.hidden});
  document.addEventListener('visibilitychange', visibility);
  worker.postMessage({type: 'init', circuit, running: running && !document.hidden});
  function updateInput() {
    if (disposed) return;
    const signal = running && !document.hidden ? readConversation() : {phase: 'idle'};
    const input = conversationDrive(signal);
    get('input').textContent = running ? input.label : 'Simulation paused';
    get('input-meter').value = input.current / .24;
    worker.postMessage({type: 'conversation', signal});
  }
  updateInput();
  const inputTimer = setInterval(updateInput, 40);
  status();
  return {dispose() { disposed = true; clearInterval(inputTimer); worker.terminate(); resize.disconnect(); document.removeEventListener('visibilitychange', visibility); }};
}
