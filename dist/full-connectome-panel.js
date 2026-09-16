import * as THREE from 'three';
import {ActivityBursts} from './activity-bursts.js';
import {OrbitControls} from './vendor/OrbitControls.js';

// Geometry is measured MaleCNS soma position; state is the full FLM recurrence.
export async function createNeuralPanel(root, readConversation = () => ({})) {
  const get = name => root.querySelector(`[data-neural="${name}"]`);
  async function load(name, binary = false) {
    const response = await fetch(`/api/connectome/${name}`);
    if (!response.ok) throw new Error('Full connectome data is unavailable. Run scripts/setup-connectome.sh.');
    return binary ? response.arrayBuffer() : response.json();
  }
  const [manifest, metadata, raw] = await Promise.all([
    load('manifest.json'), load('metadata.json'), load('positions.bin', true),
  ]);
  const coords = new Float32Array(raw), n = manifest.neurons;
  if (coords.length !== n*3 || metadata.ids.length !== n) throw new Error('Connectome geometry count mismatch.');
  const indices = [], center = [0,0,0], lo = [Infinity,Infinity,Infinity], hi = [-Infinity,-Infinity,-Infinity];
  for (let i=0; i<n; i++) {
    if (![coords[i*3], coords[i*3+1], coords[i*3+2]].every(Number.isFinite)) continue;
    indices.push(i);
    for (let a=0; a<3; a++) { lo[a] = Math.min(lo[a], coords[i*3+a]); hi[a] = Math.max(hi[a], coords[i*3+a]); }
  }
  if (!indices.length) throw new Error('No recorded neuron positions.');
  for (let a=0; a<3; a++) center[a] = (lo[a]+hi[a])/2;
  const scale = 2 / Math.max(hi[0]-lo[0], hi[1]-lo[1]);
  const positions = new Float32Array(indices.length*3);
  indices.forEach((id,i) => {
    positions[i*3] = (coords[id*3]-center[0])*scale;
    positions[i*3+1] = -(coords[id*3+1]-center[1])*scale;
    positions[i*3+2] = (coords[id*3+2]-center[2])*scale;
  });
  const canvas = get('canvas');
  const renderer = new THREE.WebGLRenderer({canvas, antialias: false, alpha: false});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  const scene = new THREE.Scene(); scene.background = new THREE.Color('#07151d');
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, .01, 100);
  camera.position.set(0, 0, 6);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true; controls.enablePan = false;
  controls.minDistance = 1.2; controls.maxDistance = 7;
  controls.enableZoom = true; controls.minZoom=.65; controls.maxZoom=5; controls.zoomSpeed=.7;
  controls.saveState();
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const bursts = new ActivityBursts(n), flashes = new Float32Array(indices.length);
  geometry.setAttribute('burst', new THREE.BufferAttribute(flashes, 1).setUsage(THREE.DynamicDrawUsage));
  const material = new THREE.ShaderMaterial({
    uniforms: {age: {value: 10}, pixelRatio: {value: renderer.getPixelRatio()}},
    vertexShader: `attribute float burst; uniform float age; uniform float pixelRatio;
      varying float intensity; varying float direction;
      void main() {
        intensity = abs(burst) * exp(-age / .14);
        direction = burst;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = (1.15 + 4.0*intensity) * pixelRatio;
      }`,
    fragmentShader: `varying float intensity; varying float direction;
      void main() {
        float d=length(gl_PointCoord-vec2(.5)); if(d>.5) discard;
        vec3 base=vec3(.075,.17,.21);
        vec3 lit=direction>=0.0 ? vec3(1.0,.65,.23) : vec3(.16,.8,1.0);
        float glow=(1.0-smoothstep(.05,.5,d));
        gl_FragColor=vec4(mix(base,lit,sqrt(intensity)*glow), .36+.64*intensity*glow);
      }`,
    transparent:true, depthWrite:false,
  });
  const points = new THREE.Points(geometry, material); scene.add(points);
  const selectionGeo = new THREE.BufferGeometry();
  selectionGeo.setAttribute('position', new THREE.Float32BufferAttribute([0,0,0], 3));
  const selectionMat = new THREE.PointsMaterial({color:'#ffffff', size:7, sizeAttenuation:false, depthTest:false});
  const selection = new THREE.Points(selectionGeo, selectionMat); selection.visible=false; scene.add(selection);
  let state = new Float32Array(n), selected = -1, running = !matchMedia('(prefers-reduced-motion: reduce)').matches;
  let disposed = false, ready = true, hadConversation = false;
  let receivedAt = -Infinity, timer, animation, lastRender = 0;
  let failed = false;
  get('summary').textContent = `${n.toLocaleString()} neurons · ${manifest.directed_edges.toLocaleString()} connections`;
  get('edges').textContent = `${indices.length.toLocaleString()} recorded cell-body positions shown. ${(n-indices.length).toLocaleString()} cells have no recorded soma position and still participate in every update. All connections are computed; edge lines are hidden for clarity.`;
  get('mapping').textContent = 'Moshi’s internal features drive the full connectome inside inference every three audio frames. A fitted readout modifies text predictions before sampling; those text tokens also condition speech. This view receives the same reservoir state, compressed for display. Browser rendering cannot change model computation.';
  function inspect() {
    if (selected < 0) {
      get('inspector').hidden=true;
      selection.visible=false; return;
    }
    const id = indices[selected];
    get('cell-name').textContent = metadata.types[id] || metadata.categories[metadata.classes[id]].replaceAll('_',' ');
    get('cell-detail').textContent = `ID ${metadata.ids[id]} · ${metadata.categories[metadata.classes[id]].replaceAll('_',' ')} · ${metadata.sides[id] || 'side unspecified'}`;
    get('cell-rate').textContent = `Δ ${bursts.delta[id] >= 0 ? '+' : ''}${bursts.delta[id].toFixed(4)}`;
    selection.visible=true;
    selection.position.fromArray(positions, selected*3);
    positionInspector();
  }
  function positionInspector() {
    if(selected<0)return;
    const p=selection.position.clone().project(camera), tip=get('inspector');
    if(Math.abs(p.x)>1 || Math.abs(p.y)>1 || Math.abs(p.z)>1){tip.hidden=true;return;}
    tip.hidden=false;
    const w=canvas.clientWidth,h=canvas.clientHeight;
    tip.style.left=`${Math.max(10,Math.min(w-tip.offsetWidth-10,(p.x+1)*w/2+14))}px`;
    tip.style.top=`${Math.max(10,Math.min(h-tip.offsetHeight-58,(1-p.y)*h/2-16))}px`;
  }
  function zoomBy(factor) {
    camera.zoom=THREE.MathUtils.clamp(camera.zoom*factor,controls.minZoom,controls.maxZoom);
    camera.updateProjectionMatrix(); controls.update(); updateZoom();
  }
  function updateZoom(){
    get('zoom-level').textContent=`${Math.round(camera.zoom*100)}%`;
    get('zoom-in').disabled=camera.zoom>=controls.maxZoom;
    get('zoom-out').disabled=camera.zoom<=controls.minZoom;
    positionInspector();
  }
  controls.addEventListener('change',updateZoom);
  get('zoom-in').addEventListener('click',()=>zoomBy(1.25));
  get('zoom-out').addEventListener('click',()=>zoomBy(1/1.25));
  get('home').addEventListener('click',()=>{controls.reset();updateZoom();});
  function status() {
    get('state').textContent = failed ? 'UNAVAILABLE' : !ready ? 'LOADING' : !running ? 'PAUSED' : readConversation().phase==='duplex' ? 'RUNNING' : 'IDLE';
  }
  function clearActivity() {
    state.fill(0); bursts.update(state,true); flashes.fill(0);
    geometry.attributes.burst.needsUpdate=true; receivedAt=-Infinity;
    get('active').textContent='0'; get('updates').textContent='0'; get('compute').textContent='—'; inspect();
  }
  function request() {
    if(disposed)return;
    const conversation=readConversation(),inCall=conversation.phase==='duplex';
    if(hadConversation && !inCall)clearActivity();
    hadConversation=inCall;status();
    const features=conversation.features||[];
    get('input').textContent=!running?'Display paused for reduced motion':inCall?'Live model reservoir':'No conversation input';
    get('input-meter').value=inCall&&features.length?Math.sqrt(features.reduce((s,v)=>s+v*v,0)/features.length):0;
  }
  function receiveBrain(event) {
    if(disposed || !running || readConversation().phase!=='duplex')return;
    const data=event.detail;
    if(data?.source!=='moshi_inference' || data.state_encoding!=='int8-tanh-127' || data.neurons!==n)return;
    let packed;
    try{packed=atob(data.state);}catch{return;}
    if(packed.length!==n)return;
    for(let i=0;i<n;i++){const value=packed.charCodeAt(i);state[i]=(value>127?value-256:value)/127;}
    const events=bursts.update(state,data.updates===0);
    indices.forEach((id,i)=>{flashes[i]=bursts.strength[id];});
    geometry.attributes.burst.needsUpdate=true; receivedAt=performance.now();
    get('active').textContent=events.count.toLocaleString();
    get('updates').textContent=data.updates.toLocaleString();
    get('compute').textContent=`${data.compute_ms.toFixed(0)} ms`;
    const {state:encodedState,...header}=data;
    window.dispatchEvent(new CustomEvent('eric-connectome',{detail:{...header,bursts:events.count,maxDelta:events.maxDelta}}));
    inspect();status();
  }
  window.addEventListener('eric-brain',receiveBrain);
  const raycaster=new THREE.Raycaster(); raycaster.params.Points.threshold=.009;
  let down=null;
  canvas.addEventListener('pointerdown', e=>{down=[e.clientX,e.clientY];});
  canvas.addEventListener('pointerup', e=>{
    if (!down || Math.hypot(e.clientX-down[0],e.clientY-down[1])>5) return;
    const rect=canvas.getBoundingClientRect();
    raycaster.params.Points.threshold=.009/camera.zoom;
    raycaster.setFromCamera(new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,1-(e.clientY-rect.top)/rect.height*2),camera);
    selected=raycaster.intersectObject(points)[0]?.index ?? -1; inspect();
  });
  canvas.addEventListener('keydown',e=>{
    if(['+','=','-','0'].includes(e.key)){e.preventDefault(); if(e.key==='0'){controls.reset();updateZoom();}else zoomBy(e.key==='-'?1/1.25:1.25);return;}
    if(!['ArrowLeft','ArrowRight','Escape'].includes(e.key)) return;
    e.preventDefault();
    selected=e.key==='Escape' ? -1 : (selected+(e.key==='ArrowRight'?1:-1)+indices.length)%indices.length;
    inspect();
  });
  const resize=new ResizeObserver(()=>{
    const {width,height}=canvas.getBoundingClientRect(); if(!width || !height)return;
    renderer.setSize(width,height,false);
    // Anatomical x/y view matches the broad brain silhouette; z remains real depth.
    const aspect=width/height;
    const halfHeight=Math.max((hi[1]-lo[1])*scale*.55,1.1/aspect);
    camera.left=-halfHeight*aspect; camera.right=halfHeight*aspect;
    camera.top=halfHeight; camera.bottom=-halfHeight;
    camera.updateProjectionMatrix();
  }); resize.observe(canvas);
  function render(now) {
    if(disposed)return;
    animation=requestAnimationFrame(render);
    if(document.hidden || now-lastRender<33)return;
    lastRender=now; controls.update();
    material.uniforms.age.value=Math.max(0,(now-receivedAt)/1000);
    renderer.render(scene,camera);
    positionInspector();
  }
  get('compute').textContent='—';
  animation=requestAnimationFrame(render); timer=setInterval(request,200); status(); inspect();
  return {dispose(){ disposed=true; clearInterval(timer); cancelAnimationFrame(animation);
    resize.disconnect(); controls.dispose(); geometry.dispose(); material.dispose(); selectionGeo.dispose(); selectionMat.dispose(); renderer.dispose();
    window.removeEventListener('eric-brain',receiveBrain);
  }};
}
