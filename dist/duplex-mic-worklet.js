// Continuous mono PCM at 24 kHz, in 80 ms frames. No turn detector or output gate.
class DuplexMicrophone extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frame = new Float32Array(1920);
    this.offset = 0;
    this.ratio = sampleRate / 24000;
    this.sum = 0; this.weight = 0;
  }
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;
    for (const sample of channel) {
      let remaining = 1;
      while (remaining > 1e-8) {
        const take = Math.min(remaining, this.ratio-this.weight);
        this.sum += sample*take; this.weight += take; remaining -= take;
        if (this.weight >= this.ratio-1e-8) {
          this.frame[this.offset++] = Math.max(-1, Math.min(1, this.sum/this.ratio));
          this.sum = 0; this.weight = 0;
          if (this.offset === this.frame.length) {
            this.port.postMessage(this.frame, [this.frame.buffer]);
            this.frame = new Float32Array(1920); this.offset = 0;
          }
        }
      }
    }
    return true; // Output remains silent: never monitor the microphone.
  }
}
registerProcessor('duplex-microphone', DuplexMicrophone);
