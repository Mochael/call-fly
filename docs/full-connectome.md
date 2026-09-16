# Full MaleCNS conversation visualization

## Data and attribution

This view uses the **same full retained MaleCNS v1.0 graph as [nftechie/flm](https://github.com/nftechie/flm)**, pinned at commit `7251a8921db4f891c39bd75ee5ad827f7031a24b`:

- 166,700 retained neurons; 25,582,938 directed neuron-pair connections.
- 124,177,617 measured synaptic contacts across those connections.
- Retention: nonempty superclass annotation, excluding status `Glia`, with both endpoints retained. This is FLM's retained graph, not every segment in the raw reconstruction.
- Every retained connection participates in every numerical update. No motor-circuit cropping, edge threshold, or randomized replacement graph.
- 139,662 recorded cell-body positions are displayed. The other 27,038 neurons still participate in the simulation; no positions are invented for them.
- Soma coordinates are published 8 nm voxel positions. The view centers/scales them, uses x horizontally and negative y vertically, and retains z as depth. Individual neuron arbors and synapses are not drawn.

Data: [MaleCNS official downloads](https://male-cns.janelia.org/download/), **CC BY 4.0**. Credit: FlyEM at HHMI Janelia, University of Cambridge Department of Zoology, MRC Laboratory of Molecular Biology, and Google Research.

Graph preparation and recurrence are adapted from FLM, copyright 2026 Alex Wormuth, **MIT**; see [the preserved license](../dist/neural/FLM_LICENSE.txt). The earlier Fly Typist locomotor visualization remains available only in the legacy voice app, under its separate attribution.

## What the flashes mean

The default display now shows **change events**, not sustained state magnitude. For neuron i, each received graph update computes `delta = state_i - previous_state_i`. An orange/cyan flash means a sharp rise/fall. An event requires `abs(delta) > max(0.025, 2.5 * recent_average_abs_delta)`. The baseline is updated as `0.92*baseline + 0.08*abs(delta)`. After an event, that neuron waits two updates before it can flash again.

Events flash immediately when the snapshot arrives and fade exponentially with an 140 ms time constant. Timing is limited to graph snapshots (4.17 Hz); no sub-step spike timings are invented. Dim points retain the anatomy. The counter reports events this update across all 166,700 computed cells; the inspector shows the selected cell's latest delta. A reset clears display history without manufacturing a flash.

These are **visual events derived from continuous states, not simulated biological action potentials**. There is no random firing or fixed top-k population. Constant high activation stops flashing, while changing states can produce different numbers and locations of events. All FLM state computation remains unchanged.

## Computation

We preserve FLM's incoming-normalized, unsigned adjacency and recurrence:

```
W[post, pre] = contact_count(pre → post) / total_incoming_contacts(post)
x_next = tanh(W @ (0.6*x + 0.4*input))
```

Moshi's 64 temporal-transformer features (already reduced from its hidden state) replace FLM's language-model token embeddings. A fixed seeded 64→128 projection is RMS-normalized, then mapped to all neurons using seeded feature bins and signs, following FLM's input construction. These projections are engineered, not measured fly language pathways. No learned FLM adapter or language backbone is imported.

Every update computes every edge on the CPU inside the model inference worker, once every three audio frames. A seeded, bias-free pooling maps all neuron states to 128 features; an offline-fitted readout adds a bounded correction to Moshi text scores before sampling. Sampled text also conditions speech. Browser activity cannot influence this computation.

Snapshots of that same state are quantized to signed 8-bit values and streamed with the call. The display follows playback timing. Ending a call clears state. No online learning or additional reservoir-feedback projection is used. See the repository's `docs/moshi-reservoir.md` for calibration and validation.

## Reproduce

From the repo root:

```
./scripts/setup-moshi.sh
./scripts/setup-connectome.sh
# Calibrate the readout: see docs/moshi-reservoir.md.
./scripts/start.sh
```

The preparation script downloads two official Feather tables (~1 GB), verifies FLM's SHA-256 source hashes, retains the same sorted exact int64 IDs, and verifies the expected node, connection and contact counts. The generated `.runtime/connectome/manifest.json` records provenance and hashes for graph arrays and geometry. Runtime verifies these files before loading the graph. Large source and derived files stay in ignored `.runtime/`, not Git.

## Verification

The graph recurrence previously matched eight sequential upstream FLM updates exactly. Current tests additionally check physical edge removal, zero readout on disconnection, bounded changes to model scores before sampling, model-driven cadence and reset. Real-model ablation results and browser metrics are saved under ignored `artifacts/moshi/`; these demonstrate causal participation and runtime behavior, not superior conversational quality.
