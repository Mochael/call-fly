// Only timing/counters/device state. No audio, transcripts, or model features.
const KEY='eric-pending-call-diagnostics';
let flushing=false;
function pending() { try{const rows=JSON.parse(localStorage.getItem(KEY)||'[]');return Array.isArray(rows)?rows:[];}catch{return [];} }
function save(records) { try{localStorage.setItem(KEY,JSON.stringify(records.slice(-64)));}catch{} }
export async function flushDiagnostics() {
  if(flushing)return;
  flushing=true;
  try {
    for(const record of pending()) {
      const controller=new AbortController(), timer=setTimeout(()=>controller.abort(),3000);
      try {
        const response=await fetch('/api/call-diagnostics',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify(record),keepalive:true,signal:controller.signal});
        if(!response.ok && (response.status<400 || response.status>=500))break;
        save(pending().filter(r=>r.event_id!==record.event_id));
      } catch {break;} finally {clearTimeout(timer);}
    }
  } finally {flushing=false;}
}
export function recordDiagnostic(s,event,reason=null,close=null) {
  const now=performance.now(), track=s.stream?.getAudioTracks()[0];
  const known=['running','suspended','interrupted','closed'];
  const record={event,event_id:crypto.randomUUID(),client_id:s.clientId,call_id:s.callId||null,reason,
    elapsed_ms:Math.max(0,now-s.startedAt),input_frames:Math.round(s.inputSamples/1920),
    output_frames:s.outputFrames,last_server_frame:s.lastServerFrame,
    frame_gap_ms:s.lastFrameAt===null?null:Math.max(0,now-s.lastFrameAt),
    capture_gap_ms:s.lastCaptureAt===null?null:Math.max(0,now-s.lastCaptureAt),
    playback_buffer_s:Math.max(0,s.playAt-s.context.currentTime),
    socket_buffer_bytes:s.socket?.bufferedAmount||0,socket_state:s.socket?.readyState??3,
    audio_state:known.includes(s.context.state)?s.context.state:'unknown',
    microphone_state:track?.readyState||'unknown',microphone_muted:track?.muted||false,hidden:document.hidden,
    close_code:close?.code??null,close_clean:close?.wasClean??null};
  save([...pending(),record]);
  void flushDiagnostics();
}
