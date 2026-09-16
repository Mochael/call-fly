import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';

// Original procedural model. Visual direction: fly-wirehead; no upstream scene code.
export function createFlyScene(canvas, sampleAudio = () => 0) {
  const renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: false});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.7));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.35;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#344c3b');
  scene.fog = new THREE.Fog('#344c3b', 14, 35);
  const camera = new THREE.PerspectiveCamera(35, 1, .1, 80);
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(-.1, 1.05, 0);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.minDistance = 7;
  controls.maxDistance = 17;
  controls.minPolarAngle = .35;
  controls.maxPolarAngle = Math.PI / 2.1;
  controls.enableZoom = false; // Let the page scroll naturally, especially on phones.
  const ambient = new THREE.HemisphereLight(0xf3ffd8, 0x24362e, 2.5); scene.add(ambient);
  const key = new THREE.DirectionalLight(0xfff4cc, 4.2);
  key.position.set(-3, 8, 5); key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  Object.assign(key.shadow.camera, {left:-7, right:7, top:6, bottom:-6, near:.5, far:25});
  key.shadow.normalBias = .03; key.shadow.bias = -.0001; scene.add(key);
  const rim = new THREE.DirectionalLight(0xa9d6cc, 2.5); rim.position.set(1, 4, -5); scene.add(rim);
  const screenLight = new THREE.PointLight(0xd7ffc1, 1.5, 5); screenLight.position.set(1.5, 2.1, 1); scene.add(screenLight);
  const material = (color, extra = {}) => new THREE.MeshStandardMaterial({color, roughness:.63, ...extra});
  const dark = material('#142627'), shell = material('#354a46', {flatShading:true, roughness:.5, metalness:.16});
  const jointMat = material('#1b2928', {roughness:.4, metalness:.2});
  const mesh = (geometry, mat, parent, pos = [0,0,0]) => {
    const object = new THREE.Mesh(geometry, mat); object.position.set(...pos);
    object.castShadow = true; object.receiveShadow = true; parent.add(object); return object;
  };
  const oval = (parent, pos, scale, mat, detail = 2) => {
    const o = mesh(new THREE.IcosahedronGeometry(1, detail), mat, parent, pos); o.scale.set(...scale); return o;
  };
  const up = new THREE.Vector3(0,1,0);
  const bone = (parent, a, b, radius, mat) => {
    const o = mesh(new THREE.CylinderGeometry(radius*.72, radius, 1, 7), mat, parent);
    pose(o,a,b); return o;
  };
  function pose(o,a,b) {
    const start = new THREE.Vector3(...a), end = new THREE.Vector3(...b);
    o.position.copy(start).lerp(end,.5); const direction = end.sub(start);
    o.scale.y = direction.length(); o.quaternion.setFromUnitVectors(up,direction.normalize());
  }
  function lines(parent, pairs, color, opacity = 1) {
    const geo = new THREE.BufferGeometry().setFromPoints(pairs.map(p => new THREE.Vector3(...p)));
    const o = new THREE.LineSegments(geo,new THREE.LineBasicMaterial({color,transparent:opacity<1,opacity}));
    parent.add(o); return o;
  }
  // Matte specimen table and a recessed instrument pad.
  mesh(new THREE.PlaneGeometry(90,90),material('#344b38'),scene).rotation.x = -Math.PI/2;
  mesh(new THREE.BoxGeometry(7.4,.22,4.75),material('#82917a'),scene,[0,.13,0]);
  mesh(new THREE.BoxGeometry(7.48,.045,4.83),material('#b0bb9a'),scene,[0,.26,0]);
  mesh(new THREE.BoxGeometry(4.25,.027,3.5),material('#718978'),scene,[-1.1,.296,.07]);
  const grid = new THREE.GridHelper(4,16,0x9ead94,0x93a28b); grid.position.set(-1.1,.313,.07); grid.scale.z=.84;
  grid.material.transparent=true;grid.material.opacity=.24;scene.add(grid);
  for (const x of [-3.55,3.55]) for (const z of [-2.2,2.2]) {
    mesh(new THREE.CylinderGeometry(.065,.065,.018,16),material('#4d6152',{metalness:.6}),scene,[x,.292,z]);
  }
  // The body is narrow, charcoal and unbanded; a fly has one pair of wings.
  const fly = new THREE.Group(); fly.position.set(-1.1,1.24,.36); fly.rotation.y=.13; scene.add(fly);
  oval(fly,[-.66,-.025,0],[.83,.32,.36],dark,3);
  const segmentLines=[];
  for(let i=0;i<5;i++) {
    const x=-.45-i*.18, r=Math.sqrt(Math.max(.02,1-((x+.66)/.86)**2));
    for(let j=0;j<32;j++) {
      const a=j*Math.PI*2/32,b=(j+1)*Math.PI*2/32;
      segmentLines.push([x,Math.cos(a)*.325*r-.025,Math.sin(a)*.365*r],[x,Math.cos(b)*.325*r-.025,Math.sin(b)*.365*r]);
    }
  }
  lines(fly,segmentLines,'#55665a',.65);
  oval(fly,[-.03,.07,0],[.54,.43,.43],shell,3);
  const head = new THREE.Group(); head.position.set(.55,.13,0); fly.add(head);
  oval(head,[0,0,0],[.31,.31,.36],material('#526158'),2);
  const eyeMat = material('#951e2c',{roughness:.33,metalness:.14});
  const facetGeo = new THREE.IcosahedronGeometry(.025,0);
  for(const side of [-1,1]) {
    oval(head,[.09,.035,side*.253],[.23,.3,.185],eyeMat,3);
    const facets = new THREE.InstancedMesh(facetGeo,material('#ad2935',{roughness:.45}),150);
    const matrix = new THREE.Matrix4();
    for(let i=0;i<150;i++) {
      const y=1-2*(i+.5)/150, angle=i*2.3999632297, r=Math.sqrt(1-y*y);
      matrix.makeTranslation(.09+.233*r*Math.cos(angle),.035+.302*y,side*(.253+.189*Math.abs(r*Math.sin(angle))));
      facets.setMatrixAt(i,matrix);
    }
    head.add(facets);
    // Short antennae and their feather-like arista, rather than insect horns.
    const base=[.265,.06,side*.095],tip=[.405,.02,side*.12];
    bone(head,base,tip,.027,jointMat);
    oval(head,tip,[.048,.047,.027],dark,1);
    const antenna=[];
    antenna.push(tip,[.51,.22,side*.18]);
    for(let j=0;j<6;j++) {const k=j/6;antenna.push([.41+k*.1,.035+k*.17,side*(.12+k*.06)],[.37+k*.1,.055+k*.17,side*(.17+k*.06)]);}
    lines(head,antenna,'#142627');
  }
  const mouth = new THREE.Group(); head.add(mouth);
  bone(mouth,[.22,-.17,0],[.35,-.33,0],.038,jointMat);
  oval(mouth,[.35,-.33,0],[.052,.035,.07],dark,2);
  // Evenly spaced bristles with independent angular/height samples, not a spiral.
  // Best-candidate sampling avoids clumps while keeping the distribution natural.
  const hairs=[],roots=[];
  let hairSeed=173;
  const hairRandom=()=>((hairSeed=Math.imul(hairSeed,1664525)+1013904223>>>0)/4294967296);
  for(let i=0;i<165;i++) {
    let root,bestDistance=-1;
    for(let attempt=0;attempt<20;attempt++) {
      const angle=hairRandom()*Math.PI*2,y=.12+.87*hairRandom(),r=Math.sqrt(1-y*y);
      const candidate=new THREE.Vector3(.54*r*Math.cos(angle),.43*y,.43*r*Math.sin(angle));
      let nearest=Infinity;
      for(const existing of roots) nearest=Math.min(nearest,candidate.distanceToSquared(existing));
      if(nearest>bestDistance){root=candidate;bestDistance=nearest;}
    }
    roots.push(root);
    const normal=new THREE.Vector3(root.x/(.54*.54),root.y/(.43*.43),root.z/(.43*.43)).normalize();
    const base=root.clone().add(new THREE.Vector3(-.03,.07,0));
    const tip=base.clone().addScaledVector(normal,.075+hairRandom()*.035);
    hairs.push(base.toArray(),tip.toArray());
  }
  lines(fly,hairs,'#10201e');
  const wings=[];
  for(const side of [-1,1]) {
    // Seat the wing roots directly on the thorax, without a raised back plate.
    const wing=new THREE.Group();wing.position.set(-.13,.43,side*.21);fly.add(wing);
    const shape=new THREE.Shape();
    shape.moveTo(0,0);shape.bezierCurveTo(-.43,.08,-1.45,.31,-1.98,.55);
    shape.bezierCurveTo(-2.35,.72,-2.17,1.02,-1.86,1.03);
    shape.bezierCurveTo(-1.05,1.02,-.38,.45,0,0);
    const geo=new THREE.ShapeGeometry(shape,18);geo.rotateX(Math.PI/2);
    if(side===-1) geo.scale(1,1,-1);
    const membrane=mesh(geo,material('#b8d5c7',{transparent:true,opacity:.4,side:THREE.DoubleSide,roughness:.27,metalness:.25,depthWrite:false}),wing);
    membrane.castShadow=false;
    const veins=[];
    const v=(x,z)=>[x,.003,z*side];
    [[-2.08,.72],[-1.95,.97],[-1.63,.94],[-1.3,.79]].forEach(([x,z])=>veins.push(v(0,0),v(x,z)));
    veins.push(v(-.6,.27),v(-.71,.46),v(-1.12,.39),v(-1.3,.8),v(-1.6,.49),v(-1.63,.94));
    lines(wing,veins,'#687c70',.78);wings.push(wing);
  }
  const forelegs=[];
  for(const side of [-1,1]) for(let i=0;i<3;i++) {
    const x=.22-i*.32;
    const a=[x,-.12,side*.3],b=[x+.27-i*.26,-.35,side*.65],c=[x+.48-i*.28,-.86,side*(.9+i*.07)],d=[c[0]+.17,-.91,c[2]+side*.13];
    const bones=[bone(fly,a,b,.032,jointMat),bone(fly,b,c,.018,jointMat),bone(fly,c,d,.011,dark)];
    const joint=oval(fly,b,[.04,.04,.04],shell,1);
    if(i===0) forelegs.push({side,a,b,c,d,bones,joint});
  }
  // A freestanding phone: screen texture is drawn from the actual call state.
  const phone=new THREE.Group();phone.position.set(1.58,1.67,-.19);phone.rotation.set(-.12,-.25,0);scene.add(phone);
  function roundedShape(w,h,r) {
    const s=new THREE.Shape(),x=-w/2,y=-h/2;
    s.moveTo(x+r,y);s.lineTo(x+w-r,y);s.quadraticCurveTo(x+w,y,x+w,y+r);s.lineTo(x+w,y+h-r);s.quadraticCurveTo(x+w,y+h,x+w-r,y+h);s.lineTo(x+r,y+h);s.quadraticCurveTo(x,y+h,x,y+h-r);s.lineTo(x,y+r);s.quadraticCurveTo(x,y,x+r,y);return s;
  }
  const casing=new THREE.ExtrudeGeometry(roundedShape(1.37,2.65,.15),{depth:.12,bevelEnabled:true,bevelSegments:3,steps:1,bevelSize:.025,bevelThickness:.025,curveSegments:10});
  mesh(casing,material('#222d29',{metalness:.65,roughness:.3}),phone);
  const screenCanvas=document.createElement('canvas');screenCanvas.width=512;screenCanvas.height=1024;
  const ctx=screenCanvas.getContext('2d');const texture=new THREE.CanvasTexture(screenCanvas);texture.colorSpace=THREE.SRGBColorSpace;
  mesh(new THREE.PlaneGeometry(1.23,2.47),new THREE.MeshBasicMaterial({map:texture,toneMapped:false}),phone,[0,0,.148]);
  mesh(new THREE.BoxGeometry(.33,.052,.012),material('#101916'),phone,[0,1.16,.161]);
  mesh(new THREE.BoxGeometry(.022,.32,.07),material('#66736a',{metalness:.7}),phone,[.705,.5,.04]);
  mesh(new THREE.BoxGeometry(1.35,.08,.8),material('#4b6252',{metalness:.2}),scene,[1.6,.35,-.34]);
  bone(scene,[1.6,.39,-.5],[1.6,1.18,-.47],.085,material('#4b6252'));
  const labelCanvas=document.createElement('canvas');labelCanvas.width=512;labelCanvas.height=80;
  const lc=labelCanvas.getContext('2d');lc.fillStyle='#304b3a';lc.font='24px monospace';lc.fillText('Eric the fruit fly',15,45);
  const labelTexture=new THREE.CanvasTexture(labelCanvas);
  const deskLabel=mesh(new THREE.PlaneGeometry(2.5,.39),new THREE.MeshBasicMaterial({map:labelTexture,transparent:true}),scene,[-1.05,.321,1.66]);deskLabel.rotation.x=-Math.PI/2;
  let state='idle',active=false,started=0,gestureStart=-Infinity,smoothLevel=0,lastScreen=-1,disposed=false;
  const gestureDuration=2200;
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const handset=new Path2D('M7 3H4a1 1 0 0 0-1 1c0 9.4 7.6 17 17 17a1 1 0 0 0 1-1v-3l-5-2-2 2a14 14 0 0 1-7-7l2-2-2-5Z');
  const screenWords={idle:'Ready when you are',connecting:'Calling Eric…',listening:'Listening to you',speaking:'Eric is speaking',thinking:'Thinking…',transcribing:'Listening back…'};
  function drawScreen(now) {
    const accepted=active&&(reduced.matches||now-gestureStart>=gestureDuration/2);
    const connected=accepted&&state!=='connecting';
    ctx.fillStyle='#111e1a';ctx.fillRect(0,0,512,1024);
    ctx.textAlign='center';ctx.fillStyle='#9eaf98';ctx.font='22px monospace';ctx.fillText(connected?'LIVE CONVERSATION':active?'INCOMING CALL':'VOICE CALL',256,112);
    ctx.fillStyle='#e2eacb';ctx.font='58px sans-serif';ctx.fillText('Human',256,225);
    ctx.fillStyle='#a0b39a';ctx.font='24px sans-serif';ctx.fillText(active&&!accepted?'Human is calling…':accepted&&!connected?'Connecting…':screenWords[state]||screenWords.idle,256,275);
    if(connected) {const secs=Math.floor((now-started)/1000);ctx.font='22px monospace';ctx.fillText(`${Math.floor(secs/60).toString().padStart(2,'0')}:${(secs%60).toString().padStart(2,'0')}`,256,326);}
    if(connected) {
      for(let i=0;i<25;i++) {
        const height=8+smoothLevel*155*(.4+.6*Math.abs(Math.sin(i*.8+now*.007)));
        ctx.fillStyle='#d4ed9e';ctx.beginPath();ctx.roundRect(95+i*13,465-height/2,5,height,3);ctx.fill();
      }
      ctx.fillStyle='#b5c2aa';ctx.font='23px sans-serif';ctx.fillText('Call in progress',256,590);
    } else if(active) {
      ctx.font='23px sans-serif';ctx.fillStyle='#b5c2aa';ctx.fillText(accepted?'Connecting…':'Waiting for Eric to answer…',256,590);
    } else {
      ctx.font='32px sans-serif';ctx.fillStyle='#bccbb2';
      ['1','2','3','4','5','6','7','8','9','*','0','#'].forEach((n,i)=>ctx.fillText(n,144+(i%3)*112,413+Math.floor(i/3)*90));
    }
    ctx.fillStyle=accepted?'#b96751':'#c8e89a';ctx.beginPath();ctx.arc(256,839,51,0,Math.PI*2);ctx.fill();
    ctx.save();ctx.translate(256,839);ctx.rotate(accepted?Math.PI*.75:0);ctx.scale(2,2);ctx.translate(-12,-12);
    ctx.strokeStyle=accepted?'#ffe4d3':'#294030';ctx.lineWidth=1.8;ctx.lineCap='round';ctx.lineJoin='round';
    ctx.stroke(handset);ctx.restore();
    ctx.fillStyle='#8f9f87';ctx.font='18px monospace';ctx.fillText(active?(connected?'CONNECTED':accepted?'CONNECTING':'INCOMING CALL'):'VOICE CALL',256,945);
    texture.needsUpdate=true;
  }
  let w=0,h=0;
  const observer=new ResizeObserver(()=>{
    const rect=canvas.getBoundingClientRect(); if(!rect.width||!rect.height)return;
    w=rect.width;h=rect.height;renderer.setSize(w,h,false);camera.aspect=w/h;
    camera.position.set(6.7,5.05,8.8).multiplyScalar(w/h<1?1.2:1);
    camera.fov=w/h<1?47:35;camera.updateProjectionMatrix();controls.update();
  });observer.observe(canvas);
  const onLost=event=>{event.preventDefault();document.querySelector('#scene-fallback').hidden=false;renderer.setAnimationLoop(null);};
  canvas.addEventListener('webglcontextlost',onLost);
  const onRestored=()=>{document.querySelector('#scene-fallback').hidden=true;if(!document.hidden)renderer.setAnimationLoop(frame);};
  canvas.addEventListener('webglcontextrestored',onRestored);
  function frame(now) {
    if(disposed||!w||!h)return;
    const motion=reduced.matches?0:1,t=now/1000;
    smoothLevel+=(Math.min(1,sampleAudio()*7)-smoothLevel)*.2;
    fly.position.y=1.24+motion*Math.sin(t*1.6)*.008;
    head.rotation.z=motion*(Math.sin(t*1.2)*.025+(state==='listening'?-.065:0));
    mouth.rotation.z=motion*smoothLevel*Math.sin(t*20)*.12;
    wings.forEach((wing,i)=>{wing.rotation.x=motion*Math.sin(t*(state==='speaking'?19:2)+i)*(.012+smoothLevel*.035);});
    const elapsed=(now-gestureStart)/gestureDuration;
    const reach=active&&motion&&elapsed>=0&&elapsed<1?Math.sin(Math.PI*elapsed):0;
    scene.updateMatrixWorld(true);
    forelegs.forEach(leg=>{
      if(leg.side!==1)return;
      const target=phone.localToWorld(new THREE.Vector3(0,-.79,.19));fly.worldToLocal(target);
      const c=new THREE.Vector3(...leg.c).lerp(target,reach);
      const b=new THREE.Vector3(...leg.b).lerp(new THREE.Vector3(.99,-.1,1.05),reach);
      const d=new THREE.Vector3(...leg.d).lerp(target.clone().add(new THREE.Vector3(.06,-.025,0)),reach);
      pose(leg.bones[0],leg.a,b.toArray());pose(leg.bones[1],b.toArray(),c.toArray());pose(leg.bones[2],c.toArray(),d.toArray());leg.joint.position.copy(b);
    });
    if(now-lastScreen>100){drawScreen(now);lastScreen=now;}
    controls.update();renderer.render(scene,camera);
  }
  const visibility=()=>renderer.setAnimationLoop(document.hidden?null:frame);
  document.addEventListener('visibilitychange',visibility);visibility();
  return {
    setState(value,isActive=active) {
      if(isActive&&!active){gestureStart=performance.now();started=gestureStart;}
      active=isActive;state=value;lastScreen=-1;
    },
    dispose() {
      disposed=true;renderer.setAnimationLoop(null);observer.disconnect();controls.dispose();
      document.removeEventListener('visibilitychange',visibility);canvas.removeEventListener('webglcontextlost',onLost);canvas.removeEventListener('webglcontextrestored',onRestored);
      const geometries=new Set(),materials=new Set(),textures=new Set();
      scene.traverse(o=>{if(o.geometry)geometries.add(o.geometry);if(o.material)(Array.isArray(o.material)?o.material:[o.material]).forEach(m=>{materials.add(m);if(m.map)textures.add(m.map);});});
      geometries.forEach(g=>g.dispose());textures.forEach(t=>t.dispose());materials.forEach(m=>m.dispose());renderer.dispose();
    },
  };
}
