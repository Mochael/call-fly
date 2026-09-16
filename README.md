# Call fly

Talk with Eric through a live, full-duplex voice model while watching the activity of a fly-connectome reservoir that participates in generating the conversation.

**Moshi supplies the speech and language capabilities. The fly connectome adds a small adjustment to its text-token predictions.** The visualization displays the same reservoir state used for that adjustment. It is not a separate animation driven by audio volume.

This is an experimental integration inspired by [Alex Wormuth's Fly Language Model (FLM)](https://github.com/nftechie/flm). It demonstrates that an anatomical network can influence a pretrained model's output. It does **not** demonstrate better conversation, biological fly understanding, or self-improvement.

## The complete path

```text
Microphone → Mimi audio encoder → Moshi's streaming temporal model
                                          │
                          current 4,096-number internal state
                                          │
                              fixed 64-feature summary
                                          │
                            fixed input mapping to neurons
                                          │
                         full retained MaleCNS connectome
                           + previous reservoir activity
                                          │
                             fixed 128-feature pooling
                                          │
                              fitted 128 × 4,096 adapter
                                          │
                         small, bounded text-score correction
                                          │
Moshi's original text scores ──────────────┤
                                          ↓
                                  text-token sampling
                                          ↓
                    Moshi audio generator → Mimi decoder → speaker
```

The reservoir state also goes to the browser's 3D viewer. Model computation runs on the server and does not depend on rendering the visualization.

Moshi processes incoming and outgoing audio continuously, in 80 ms frames. The microphone remains active while Eric speaks; there is no separate speech-to-text → text LLM → text-to-speech pipeline in the current app.

## Where the fly connections and weights come from

The source is the public **MaleCNS v1.0** anatomical reconstruction, downloaded from [the official FlyEM dataset](https://male-cns.janelia.org/download/). We use the same retention rule as FLM: neurons with a nonempty superclass annotation, excluding `Glia`, and connections whose endpoints are both retained.

| Quantity | Retained graph |
| --- | ---: |
| Neurons | 166,700 |
| Directed neuron-pair connections | 25,582,938 |
| Synaptic contacts across those connections | 124,177,617 |
| Neurons with recorded cell-body positions | 139,662 |

“Full” refers to this full retained FLM graph, not every segment in the raw reconstruction. Neurons without recorded positions still participate in computation.

The dataset provides a contact count for each connected neuron pair. Following FLM, we divide each count by the total retained incoming contacts of the receiving neuron:

```text
W[receiver, sender] = contacts(sender → receiver) / total_incoming_contacts(receiver)
```

For example, 20 contacts from A and 80 from C become incoming weights of 0.2 and 0.8. These weights preserve relative anatomical connectivity. They are **not measured electrical synaptic strengths**. The implementation uses unsigned weights and does not incorporate neurotransmitter-specific excitation/inhibition, conduction delays, membrane physiology, or biological spikes.

The fixed recurrence is:

```text
next_activity = tanh(W × (0.6 × previous_activity + 0.4 × input_drive))
```

The contact counts come from anatomy; the normalization and update rule are modeling choices adapted from FLM. Every retained edge participates in an update. The graph advances once every three Moshi frames, approximately **4.17 times per second**. This cadence is an engineering choice, not biological time.

Source filenames, checksums, filtering and matrix construction are recorded in [prepare_connectome.py](scripts/prepare_connectome.py) and [the data documentation](docs/full-connectome.md). The FLM reference revision is `7251a8921db4f891c39bd75ee5ad827f7031a24b`.

## How Moshi connects to those neurons

There is no known mapping from Moshi's features to fly speech neurons in this project. The input and output interfaces are engineered:

1. **Summarize Moshi:** reduce its 4,096-number temporal hidden state to 64 bounded features using a fixed signed block projection and scale normalization.
2. **Drive the graph:** a seeded random 64→128 projection creates input channels. Fixed channel assignments and signs distribute them across all neurons. Seed: `7301`.
3. **Read the graph:** fixed signed pooling groups neuron activity into 128 values and normalizes their scale. Seed: `7302`.
4. **Apply the learned adapter:** a fitted 128×4,096 matrix converts those values into a vector compatible with Moshi's text prediction head.

The random mappings are generated reproducibly and remain fixed. They are not labels for concepts such as food, happiness or speech. Negative signs in these mappings do not identify inhibitory biological neurons.

The readout becomes `0.03 × tanh(pooled_activity × adapter)`. Moshi's existing text head projects it to vocabulary scores. We subtract its mean across the vocabulary and cap its root-mean-square magnitude at **0.10 logit units** before adding it to the original scores. The latest correction is reused between graph updates.

Code: [model_reservoir.py](server/model_reservoir.py), [connectome.py](server/connectome.py), [PyTorch hook](server/moshi_torch_engine.py), [shared inference](server/moshi_batch_engine.py).

## What was actually trained

**Only the readout adapter was fitted.** Moshi, the connectome weights, and the input/pooling mappings stayed frozen. We did not import FLM's adapter, fine-tune Moshi, train a new voice, or train the connectome to understand language.

### The audio and training examples

The development calibration used three generated speech clips, approximately 41, 46 and 42 seconds long. The first two were training inputs; the third was held out for validation. These local clips and their AIFF source files are not distributed in this repository. The original generator command, voice and source text were not preserved in a reproducible generation script, so we cannot establish that exact provenance from the retained records. They should not be presented as a public benchmark dataset.

The training script played each clip into frozen Moshi as microphone input while Moshi generated its own stream. The reservoir observed Moshi's internal state but did not change generation during calibration. At each reservoir update, the script saved a pair:

- **Input:** the 128 pooled reservoir values.
- **Target:** the corresponding original 4,096-number Moshi hidden state.

This produced **361 training pairs and 176 validation pairs**. These are correlated observations from roughly 87 seconds of training audio, not hundreds of independent conversations. No transcripts or correct-answer labels were supplied.

### The loss and fitting procedure

Let `R` be the matrix of reservoir observations, `H` the recorded Moshi states, and `A` the adapter. We solve:

```text
minimize over A:  sum((R × A − H)²) + 100 × sum(A²)

A = solve(Rᵀ × R + 100 × I, Rᵀ × H)
```

This is bias-free ridge regression: a linear least-squares fit with a penalty on large weights. There is no gradient-based fine-tuning of the original model. The adapter contains **524,288 fitted values**. Its target is the captured Moshi hidden state; the training script adds no separate target normalization.

The fitted adapter and its metadata were actually saved. Separate fits were performed for the local MLX q4 checkpoint and the Modal PyTorch bf16 checkpoint; the artifacts cannot be interchanged. The loader checks model revision, graph manifest, interface, checksum, shape and finite values.

| Held-out state reconstruction error (MSE; lower is better) | MLX q4 | PyTorch bf16 |
| --- | ---: | ---: |
| Fitted adapter | 0.7002 | 0.8476 |
| Always predict zero | 2.0845 | 2.1798 |
| Always predict the training mean | 1.0485 | 1.2355 |

These are small development-set measurements of **state reconstruction**, not speech quality, reasoning, or conversational improvement. The local saved weights and audio hashes were checked against the training manifest; the retained Modal log records successful completion of the bf16 calibration. Those local artifacts are excluded from Git.

### Why this is a limited objective

The adapter learns to recover information already present in Moshi. It does not learn which response would be better. Adding the reconstructed signal back into token selection is an experimental design choice; a good reconstruction score does not prove that this helps generation.

Training implementation: [train_moshi_reservoir.py](scripts/train_moshi_reservoir.py). Calibration details and causal checks: [moshi-reservoir.md](docs/moshi-reservoir.md).

## Does it change the words or the voice?

**It directly changes the probabilities of text tokens.** Moshi's audio generator conditions on the sampled text, so changes can propagate to the generated audio and delivery.

The reservoir does not directly control pitch, timbre, voice identity, or buzzing. Eric currently uses Moshi's standard Moshiko voice. The earlier Qwen cloned-voice pipeline is separate and requires the operator's own recording; no voice-cloning reference is included here.

Nothing learns during a call. Recurrence retains activity over time, and generated tokens enter Moshi's ongoing context, but there is no online optimization, reward loop, or separately trained reservoir-feedback controller.

## What the visualization means

- Each point is a recorded neuron cell-body position, not its complete branching morphology.
- Orange and cyan flashes show sharp rises and falls in the continuous reservoir state used by inference.
- A change must exceed both `0.025` and `2.5 × recent_average_absolute_change`, followed by a two-update cooldown. Steady activity fades into the dim anatomy.
- These are **display events, not biological action potentials or dopamine measurements**. There are no random flashes or fixed top-k active neurons.
- **New bursts** counts change events in the latest displayed update; **Simulation steps** counts completed graph updates in the call; **Last step time** measures graph computation time, not conversation latency.
- Snapshots are quantized for transmission and aligned with audio playback. The adapter reads full-precision state. The browser does not run a second reservoir.

Drag to rotate, scroll/pinch or use +/− to zoom, and select a neuron to inspect its ID and annotations. The graph is idle outside a call. Ending a call resets its state. Camera controls and reduced-motion rendering do not alter inference.

## What has been verified

Removing the graph's connections zeros its readout contribution. A native PyTorch comparison with the same input audio and random seed produced different sampled text and audio with the intact graph versus removed edges. This establishes **causal influence**, not a quality benefit. A shorter earlier MLX comparison changed scores but happened to sample the same output.

The shared server uses upstream Moshi/Mimi batched streaming, execution masks and per-row resets. Four callers share one loaded model on an L40S, with separate model caches and reservoir states. A deterministic isolation check changed/reset/paused peers without changing the unchanged caller's output. A ten-call scaling test used three GPUs. Detailed conditions, timing limits and results are in [deployment.md](docs/deployment.md); these are bounded development tests, not service guarantees.

## Website hosting

**Website: [call-fly.vercel.app](https://call-fly.vercel.app).** It currently requires Vercel sign-in; the public repository does not grant website access.

The frontend and same-origin voice gateway can run on **Vercel**, while Moshi and the connectome run on **Modal**. The gateway keeps the Modal credential on the server and relays live audio and brain updates. See [Vercel deployment](docs/vercel.md) for setup, access controls and verification, and [Modal deployment](docs/deployment.md) for inference capacity.

## Run it yourself

### Local inference on Apple Silicon

Install [uv](https://docs.astral.sh/uv/), allow space for the model (~5.2 GB), the connectome source data (~1 GB), derived arrays and dependencies, then run:

```sh
./scripts/setup-moshi.sh
./scripts/setup-connectome.sh

# Supply your own authorized/generated audio. Last clip is validation.
.runtime/moshi-venv/bin/python scripts/train_moshi_reservoir.py \
  --audio /path/to/train-1.wav /path/to/train-2.wav /path/to/validation.wav \
  --seconds 55

MOSHI_TRANSPORT=local ./scripts/start.sh
```

Open **http://localhost:8765**, click **Call fly**, and allow microphone access. Headphones help prevent speaker feedback. Calls have a five-minute limit; starting again creates fresh state. A fresh clone does not contain an adapter and will not silently substitute a visualizer if calibration is missing.

Local inference uses pinned `kyutai/moshiko-mlx-q4`; cloud inference uses pinned `kyutai/moshiko-pytorch-bf16`. For model versions, cloud setup, service authentication, concurrent sessions and GPU costs, see [deployment.md](docs/deployment.md). Deployment starts paid GPU resources; a configured warm instance incurs charges while idle.

The public source does not grant access to the operator's hosted GPU or change the website's access policy. Use your own deployment and credentials.

### Checks and development

```sh
npm ci
npm run check
npm run test:ui
node --test tests/site-gateway.test.mjs
```

Model-backed tests require prepared data, an adapter and a running server. See [development notes](docs/development.md) for local/model checks, legacy Qwen setup and call diagnostics, and [serving research](docs/moshi-serving-research.md) for upstream references and implementation choices.

## Privacy and repository contents

The browser asks for the microphone only when starting a call. Audio is transmitted to the configured inference service during the call. Application diagnostics record operational metadata such as call IDs, timing, queue sizes and stop reasons; the application does not intentionally persist live call audio, transcripts or hidden-state histories. Generated speech and reservoir snapshots are returned to the caller. Local verification scripts can explicitly save test audio and results.

Credentials, local service configurations, personal voice recordings/transcripts, calibration audio, fitted adapters, model downloads, diagnostic logs and test artifacts are excluded from Git. The website build copies a specific allowlist of browser assets, public connectome geometry and a server-side gateway. Service tokens are runtime secrets, never browser configuration. See [publication and privacy notes](docs/publication.md).

## Credits and licenses

- **Moshi and Mimi:** [Kyutai](https://github.com/kyutai-labs/moshi). Respect the code and model licenses; the pinned Moshiko model cards specify CC BY 4.0 weights.
- **Reservoir design and retained graph:** [FLM, Alex Wormuth](https://github.com/nftechie/flm), MIT. [Preserved notice](dist/neural/FLM_LICENSE.txt).
- **MaleCNS data:** FlyEM at HHMI Janelia, University of Cambridge Department of Zoology, MRC Laboratory of Molecular Biology, and Google Research; CC BY 4.0. [Dataset provenance](docs/full-connectome.md).
- **Rendering:** Three.js and OrbitControls, with their [MIT notice](dist/vendor/THREE-LICENSE.txt).
- **Legacy motor-circuit assets:** separately attributed in [dist/neural/ATTRIBUTION.md](dist/neural/ATTRIBUTION.md).

Public availability of this repository does not replace the licenses attached to third-party code, data or model weights.
