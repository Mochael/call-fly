// Only public browser assets and checked connectome geometry enter the output.
import {cp,mkdir,readFile,writeFile,rm} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {backendConfiguration} from '../deploy/vercel-gateway.js';

const root=new URL('../',import.meta.url),out=new URL('dist/vercel/',root);
await rm(out,{recursive:true,force:true});await mkdir(out,{recursive:true});
for(const name of ['index.html','style.css','app.js','call-diagnostics.js','activity-bursts.js',
  'full-connectome-panel.js','duplex-mic-worklet.js','fly-scene.js','vendor']) {
  await cp(new URL('dist/'+name,root),new URL(name,out),{recursive:true});
}
await mkdir(new URL('neural/',out),{recursive:true});
for(const name of ['FULL_CONNECTOME.md','FLM_LICENSE.txt']) {
  await cp(new URL('dist/neural/'+name,root),new URL('neural/'+name,out));
}
const geometry=new URL('neural/connectome/',out);await mkdir(geometry,{recursive:true});
let local=true;
try {await readFile(new URL('.runtime/connectome/manifest.json',root));}catch(error) {
  if(error.code!=='ENOENT')throw error;
  local=false;
}
const config=local?null:backendConfiguration();
async function asset(name) {
  if(local)return readFile(new URL('.runtime/connectome/'+name,root));
  const response=await fetch(config.url+'/api/connectome/'+name,{
    headers:{'x-voice-service-token':config.token},redirect:'error',signal:AbortSignal.timeout(30_000),
  });
  if(!response.ok)throw new Error(`Connectome asset ${name} unavailable (HTTP ${response.status}).`);
  return Buffer.from(await response.arrayBuffer());
}
const rawManifest=await asset('manifest.json'),manifest=JSON.parse(rawManifest);
if(manifest.neurons!==166700 || manifest.directed_edges!==25582938)throw new Error('Unexpected connectome graph.');
await writeFile(new URL('manifest.json',geometry),rawManifest);
for(const name of ['metadata.json','positions.bin']) {
  const data=await asset(name);
  if(createHash('sha256').update(data).digest('hex')!==manifest.geometry_files?.[name])throw new Error(`Connectome checksum mismatch: ${name}`);
  if(name==='positions.bin' && data.length!==manifest.neurons*12)throw new Error('Invalid geometry size.');
  if(name==='metadata.json' && JSON.parse(data).ids.length!==manifest.neurons)throw new Error('Invalid neuron metadata.');
  await writeFile(new URL(name,geometry),data);
}
console.log('Built browser assets and verified public connectome geometry for Vercel.');
