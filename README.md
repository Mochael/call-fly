# Call fly

Meet **Eric the fruit fly**, the character you can call and chat with in this app. Eric listens and talks in real time through a full-duplex voice model, while an interactive brain visualization shows the activity of the fly-connectome reservoir that helps shape his responses.

**Moshi supplies the speech and language capabilities. The fly connectome adds a small adjustment to its text-token predictions.** The visualization displays the same reservoir state used for that adjustment.

This experimental integration is inspired by [Alex Wormuth's Fly Language Model (FLM)](https://github.com/nftechie/flm) and explores how an anatomical network can influence a pretrained model's output.

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

The server computes the reservoir activity and streams it to the browser's 3D viewer alongside the audio.

Moshi processes incoming and outgoing audio continuously, in 80 ms frames. The microphone remains active while Eric speaks, allowing listening and speaking to overlap.

## Where the fly connections and weights come from

The source is the public **MaleCNS v1.0** anatomical reconstruction, downloaded from [the official FlyEM dataset](https://male-cns.janelia.org/download/). We use the same retention rule as FLM: neurons with a nonempty superclass annotation, excluding `Glia`, and connections whose endpoints are both retained.

| Quantity | Retained graph |
| --- | ---: |
| Neurons | 166,700 |
| Directed neuron-pair connections | 25,582,938 |
| Synaptic contacts across those connections | 124,177,617 |
| Neurons with recorded cell-body positions | 139,662 |

“Full” refers to the retained FLM graph described above. All retained neurons participate in computation, including those without recorded positions.

The dataset provides a contact count for each connected neuron pair. Following FLM, we divide each count by the total retained incoming contacts of the receiving neuron:

```text
W[receiver, sender] = contacts(sender → receiver) / total_incoming_contacts(receiver)
```

For example, 20 contacts from A and 80 from C become incoming weights of 0.2 and 0.8. These unsigned weights represent relative anatomical connectivity in a simplified model of continuous neuron activity.

The fixed recurrence is:

```text
next_activity = tanh(W × (0.6 × previous_activity + 0.4 × input_drive))
```

The contact counts come from anatomy; the normalization and update rule are modeling choices adapted from FLM. Every retained edge participates in an update. The graph advances once every three Moshi frames, approximately **4.17 times per second**.

Source filenames, checksums, filtering and matrix construction are recorded in [prepare_connectome.py](scripts/prepare_connectome.py) and [the data documentation](docs/full-connectome.md). The FLM reference revision is `7251a8921db4f891c39bd75ee5ad827f7031a24b`.

## How Moshi connects to those neurons

Fixed numerical mappings carry Moshi's internal features into the reservoir and pool its activity for the adapter:

1. **Summarize Moshi:** reduce its 4,096-number temporal hidden state to 64 bounded features using a fixed signed block projection and scale normalization.
2. **Drive the graph:** a seeded random 64→128 projection creates input channels. Fixed channel assignments and signs distribute them across all neurons. Seed: `7301`.
3. **Read the graph:** fixed signed pooling groups neuron activity into 128 values and normalizes their scale. Seed: `7302`.
4. **Apply the learned adapter:** a fitted 128×4,096 matrix converts those values into a vector compatible with Moshi's text prediction head.

The seeds make these mappings reproducible. Channel assignments, pooling groups and signs remain fixed across calls.

The readout becomes `0.03 × tanh(pooled_activity × adapter)`. Moshi's existing text head projects it to vocabulary scores. We subtract its mean across the vocabulary and cap its root-mean-square magnitude at **0.10 logit units** before adding it to the original scores. The latest correction is reused between graph updates.

Code: [model_reservoir.py](server/model_reservoir.py), [connectome.py](server/connectome.py), [PyTorch hook](server/moshi_torch_engine.py), [shared inference](server/moshi_batch_engine.py).

## Training the readout

**The readout adapter was fitted specifically for Moshi.** Moshi's weights, the connectome weights, and the input/pooling mappings stayed fixed.

### The audio and training examples

The development calibration used three generated speech clips, approximately 41, 46 and 42 seconds long. The first two were training inputs; the third was held out for validation. The clips are local development fixtures. Their original generation recipe was not preserved; reproducing the fit requires supplying your own audio.

The training script played each clip into frozen Moshi as microphone input while Moshi generated its own stream. During calibration, the reservoir observed the original model's internal state. At each reservoir update, the script saved a pair:

- **Input:** the 128 pooled reservoir values.
- **Target:** the corresponding original 4,096-number Moshi hidden state.

This produced **361 training pairs and 176 validation pairs**. These are correlated observations from roughly 87 seconds of training audio, with Moshi itself providing the targets.

### The loss and fitting procedure

Let `R` be the matrix of reservoir observations, `H` the recorded Moshi states, and `A` the adapter. We solve:

```text
minimize over A:  sum((R × A − H)²) + 100 × sum(A²)

A = solve(Rᵀ × R + 100 × I, Rᵀ × H)
```

This is bias-free ridge regression: a linear least-squares fit with a penalty on large weights. It directly solves for the adapter’s **524,288 values**, using the captured Moshi hidden state as the target.

Each checkpoint has its own fitted adapter: one for local MLX q4 and one for Modal PyTorch bf16. The loader validates the saved adapter against the model revision, graph and interface.

| Held-out state reconstruction error (MSE; lower is better) | MLX q4 | PyTorch bf16 |
| --- | ---: | ---: |
| Fitted adapter | 0.7002 | 0.8476 |
| Always predict zero | 2.0845 | 2.1798 |
| Always predict the training mean | 1.0485 | 1.2355 |

These measurements evaluate **state reconstruction** on the held-out development clip. Saved manifests record the audio and adapter hashes for each fit.

### What the objective measures

The adapter learns to recover Moshi’s internal state from reservoir activity. Feeding that reconstruction back into token selection is experimental; its effect on conversation quality remains to be evaluated.

Training implementation: [train_moshi_reservoir.py](scripts/train_moshi_reservoir.py). Calibration details and causal checks: [moshi-reservoir.md](docs/moshi-reservoir.md).

## How it affects speech

**It directly changes the probabilities of text tokens.** Moshi's audio generator conditions on the sampled text, so changes can propagate to the generated audio and delivery.

Eric uses Moshi's standard Moshiko voice. The earlier Qwen cloned-voice pipeline is available separately with your own reference recording.

During a call, all weights stay fixed while reservoir activity and Moshi’s conversation context evolve.

## What the visualization means

- Each point marks a recorded neuron cell-body position.
- Orange and cyan flashes mark sharp rises and falls in the simulation’s continuous activity values, using the same state as inference.
- A change must exceed both `0.025` and `2.5 × recent_average_absolute_change`, followed by a two-update cooldown. Steady activity fades into the dim anatomy.
- **New bursts** counts change events in the latest displayed update; **Simulation steps** counts completed graph updates in the call; **Last step time** measures graph computation time.
- Snapshots are quantized for transmission and aligned with audio playback. The adapter reads full-precision state.

Drag to rotate, scroll/pinch or use +/− to zoom, and select a neuron to inspect its ID and annotations. The graph is idle outside a call. Ending a call resets its state. Camera controls affect the view; inference continues on the server.

## What has been verified

Removing the graph's connections zeros its readout contribution. A native PyTorch comparison with the same input audio and random seed produced different sampled text and audio with the intact graph versus removed edges. This establishes **causal influence** on generation. A shorter earlier MLX comparison changed scores but happened to sample the same output.

The shared server uses upstream Moshi/Mimi batched streaming, execution masks and per-row resets. Four callers share one loaded model on an L40S, with separate model caches and reservoir states. A deterministic isolation check changed/reset/paused peers without changing the unchanged caller's output. A ten-call scaling test used three GPUs. Test conditions, timing limits and results are in [deployment.md](docs/deployment.md).

## Website hosting

**Website: [call-fly.vercel.app](https://call-fly.vercel.app).** Access currently requires Vercel sign-in.

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

Open **http://localhost:8765**, click **Call fly**, and allow microphone access. Headphones help prevent speaker feedback. Calls have a five-minute limit; starting again creates fresh state. Run the calibration step above to create the adapter required for inference.

Local inference uses pinned `kyutai/moshiko-mlx-q4`; cloud inference uses pinned `kyutai/moshiko-pytorch-bf16`. For model versions, cloud setup, service authentication, concurrent sessions and GPU costs, see [deployment.md](docs/deployment.md). Deployment starts paid GPU resources; a configured warm instance incurs charges while idle.

For your own hosted instance, configure a deployment and credentials using the linked setup guide.

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
