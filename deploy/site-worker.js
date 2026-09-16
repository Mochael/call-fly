// The Sites audience gate protects this same-origin gateway. The backend key
// exists only in server-side environment variables, never browser source.
const allowed=new Set(['/api/conversation','/api/call-diagnostics']);
export default {
  async fetch(request,env){
    const url=new URL(request.url);
    if(url.pathname==='/api/health'){
      const configured=!!(env.VOICE_BACKEND_URL&&env.VOICE_SERVICE_TOKEN);
      return Response.json({ready:configured,backend:'moshi',busy:false,full_duplex:true,
        connectome_in_model:true,sample_rate:24000,frame_samples:1920,max_call_seconds:300,
        stage:configured?'Ready for a live conversation.':'Cloud voice service setup is still in progress.',error:null},
        {headers:{'Cache-Control':'no-store'}});
    }
    if(url.pathname.startsWith('/api/connectome/')){
      const name=url.pathname.slice('/api/connectome/'.length);
      if(!['manifest.json','metadata.json','positions.bin'].includes(name))return new Response('Not found',{status:404});
      url.pathname='/neural/connectome/'+name;
      return env.ASSETS.fetch(new Request(url,request));
    }
    if(allowed.has(url.pathname)){
      if(!env.VOICE_BACKEND_URL||!env.VOICE_SERVICE_TOKEN)return Response.json({message:'Voice service is not configured.'},{status:503});
      const target=new URL(url.pathname,env.VOICE_BACKEND_URL);
      if(target.protocol!=='https:')return new Response('Invalid voice service configuration',{status:503});
      const headers=new Headers(request.headers);
      headers.set('x-voice-service-token',env.VOICE_SERVICE_TOKEN);
      headers.delete('cookie');headers.delete('authorization');headers.delete('host');
      return fetch(new Request(target,{method:request.method,headers,body:request.body,redirect:'manual',duplex:'half'}));
    }
    return env.ASSETS.fetch(request);
  }
};
