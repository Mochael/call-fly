// Display events derived from changes in continuous FLM state, not action potentials.
export class ActivityBursts {
  constructor(count) {
    this.previous = new Float32Array(count);
    this.noise = new Float32Array(count);
    this.cooldown = new Uint8Array(count);
    this.strength = new Float32Array(count);
    this.delta = new Float32Array(count);
  }
  reset(state) {
    this.previous.fill(0); if (state) this.previous.set(state);
    this.noise.fill(0); this.cooldown.fill(0); this.strength.fill(0); this.delta.fill(0);
  }
  update(state, reset = false) {
    if (reset) { this.reset(state); return {count:0, maxDelta:0}; }
    let count=0, maxDelta=0;
    for (let i=0;i<state.length;i++) {
      const delta=state[i]-this.previous[i], change=Math.abs(delta);
      // Absolute floor plus a slowly adapting per-neuron change baseline.
      // No top-k selection or random flashes: steady activity gives no events.
      const threshold=Math.max(.025, 2.5*this.noise[i]);
      this.strength[i]=0; this.delta[i]=delta;
      if (this.cooldown[i]) this.cooldown[i]--;
      else if (change>threshold) {
        this.strength[i]=Math.sign(delta)*Math.min(1,.7+(change-threshold)/.12);
        this.cooldown[i]=2; count++;
      }
      this.noise[i]=.92*this.noise[i]+.08*change;
      this.previous[i]=state[i]; maxDelta=Math.max(maxDelta,change);
    }
    return {count,maxDelta};
  }
}
