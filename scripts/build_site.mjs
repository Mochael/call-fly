// Keep the local static app intact. Produce a separate Worker + assets build.
import {cp,mkdir,readFile,writeFile,rm,readdir} from 'node:fs/promises';
const root=new URL('../',import.meta.url),out=new URL('dist/client/',root);
await rm(out,{recursive:true,force:true});await mkdir(out,{recursive:true});
for(const name of ['index.html','style.css','app.js','call-diagnostics.js','activity-bursts.js','full-connectome-panel.js','duplex-mic-worklet.js','fly-scene.js','vendor']){
  await cp(new URL('dist/'+name,root),new URL(name,out),{recursive:true});
}
await mkdir(new URL('neural/connectome/',out),{recursive:true});
for(const name of ['FULL_CONNECTOME.md','FLM_LICENSE.txt'])await cp(new URL('dist/neural/'+name,root),new URL('neural/'+name,out));
for(const name of ['manifest.json','metadata.json','positions.bin'])await cp(new URL('.runtime/connectome/'+name,root),new URL('neural/connectome/'+name,out));
await mkdir(new URL('dist/server/',root),{recursive:true});
await cp(new URL('deploy/site-worker.js',root),new URL('dist/server/index.js',root));
await mkdir(new URL('dist/.openai/',root),{recursive:true});
await cp(new URL('.openai/hosting.json',root),new URL('dist/.openai/hosting.json',root));
console.log('Built private web gateway and connectome assets.');
