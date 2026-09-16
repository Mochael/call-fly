class MicrophoneCapture extends AudioWorkletProcessor {
  constructor() { super(); this.buffer = new Float32Array(2048); this.offset = 0; }
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel) {
      for (let i = 0; i < channel.length; i++) {
        this.buffer[this.offset++] = channel[i];
        if (this.offset === this.buffer.length) {
          this.port.postMessage(this.buffer, [this.buffer.buffer]);
          this.buffer = new Float32Array(2048);
          this.offset = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('microphone-capture', MicrophoneCapture);
