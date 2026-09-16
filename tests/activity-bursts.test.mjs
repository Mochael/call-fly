import {test} from 'node:test';
import assert from 'node:assert/strict';
import {ActivityBursts} from '../dist/activity-bursts.js';

test('steady high activation gives one transient and then no flashes',()=>{
  const b=new ActivityBursts(3), state=new Float32Array([.2,-.3,.0001]);
  assert.equal(b.update(state).count,2);
  assert.ok(b.strength[0]>0 && b.strength[1]<0);
  for(let i=0;i<30;i++)assert.equal(b.update(state).count,0);
  assert.equal(b.strength.every(x=>x===0),true);
});
test('noise is suppressed, changes are neuron-specific, and reset does not flash',()=>{
  const b=new ActivityBursts(3);
  for(let i=0;i<10;i++)assert.equal(b.update([i%2*.01,0,0]).count,0);
  assert.equal(b.update([.01,.3,0]).count,1);
  assert.equal(b.strength[0],0); assert.ok(b.strength[1]>0); assert.equal(b.strength[2],0);
  assert.equal(b.update([0,0,0],true).count,0);
  assert.ok(b.strength.every(x=>x===0));
});
test('refractory period prevents continuous strobing and counts are not fixed',()=>{
  const b=new ActivityBursts(2);
  assert.equal(b.update([.4,0]).count,1);
  assert.equal(b.update([-.4,0]).count,0);
  assert.equal(b.update([.4,0]).count,0);
  assert.equal(b.update([-.4,.4]).count,2);
});
