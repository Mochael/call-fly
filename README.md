# Call fly — Moshi with a connectome reservoir

A web app with a fly, a phone, and a live speech-to-speech conversation. **Moshi listens while generating speech**, so microphone capture continues while the agent talks. The conversation-history panel is hidden.

The full connectome is now **inside Moshi's prediction path**. An offline-fitted readout adds a bounded correction to text scores before sampling; those text tokens also condition speech. The visualization shows that same reservoir state. All weights remain fixed during calls. See [the implemented architecture and calibration](docs/moshi-reservoir.md).

## Start

On this Mac:

```sh
./scripts/start.sh
```

Open **http://localhost:8765**, click **Call fly**, and allow the microphone. Headphones are recommended; browser echo cancellation is requested, but speaker/room feedback can still affect duplex conversations. Click **End call** to stop playback and release the microphone.

When `.runtime/modal-service.json` contains the deployed Modal URL and service token, this command starts a lightweight local proxy. Moshi and the full connectome run on Modal's GPU service; no inference model loads on the Mac. The token stays on the server. Each GPU serves four calls using upstream Moshi masked streaming. One GPU stays warm; overflow scales to at most three GPUs, with cold-start waits for new containers. See [deployment and measurements](docs/deployment.md) and [the upstream serving research](docs/moshi-serving-research.md).

Use `MOSHI_TRANSPORT=modal ./scripts/start.sh` to require the cloud service, or `MOSHI_TRANSPORT=local ./scripts/start.sh` to explicitly use the existing Apple Silicon model.

Moshi uses its standard Moshiko voice and conversational behavior. Eric remains the visual character; this milestone does not claim a custom fly persona, Qwen voice cloning, or guaranteed spoken buzzes. The prior cloned voice and recordings are preserved separately.

## Fresh setup for local inference

Requires Apple Silicon macOS, [uv](https://docs.astral.sh/uv/), and enough disk space for approximately 5.2 GB of model files plus dependencies. Python 3.12 is managed by uv.

```sh
./scripts/setup-moshi.sh
./scripts/setup-connectome.sh
# Fit the required readout using the calibration command in docs/moshi-reservoir.md.
./scripts/start.sh
```

The isolated `.runtime/moshi-venv` environment uses `requirements-moshi.lock`. No Ollama, reference recording, API key, or hosted inference is required for the Moshi path. Downloads use the public Hugging Face model repository; inference runs locally after download.

The model is `kyutai/moshiko-mlx-q4` at revision `18e4df760a34d5977a34517d7d1580e07acbb2f1`, with `moshi-mlx==0.3.0` and `rustymimi==0.4.1`. The Moshi model weights are CC BY 4.0; see [the model card](https://huggingface.co/kyutai/moshiko-mlx-q4). Moshi MLX code is MIT-licensed.

## How it works

```text
Microphone → Mimi → Moshi temporal model → text predictions → audio depformer → Mimi → speaker
                         ↓                      ↑
                   full connectome → fitted readout
                         ↓
                   3D visualization
```

- **Audio:** mono Float32 PCM over a local WebSocket, 1,920 samples per frame (80 ms). An AudioWorklet resamples continuously; no user-turn segmentation or microphone gate during playback.
- **Inference:** one thread per GPU serializes batched PyTorch inference, including CUDA Mimi. Four independently masked session rows share loaded weights; each has its own model caches and connectome activity. On Apple Silicon, the single-call MLX backend overlaps a separate CPU codec thread with inference. Transient cloud connection failures retry only before a session connects; generated audio is not transcribed or regenerated.
- **Playback:** short PCM frames with a 60 ms scheduling cushion. Buffer limits stop a call explicitly if the machine falls behind instead of accumulating seconds of stale audio.
- **Session:** one active call per browser, with fresh model/codec/reservoir state and a five-minute limit. The final 30 seconds show a countdown. Ending or restarting one cloud call leaves the other session rows running. Start another call after the limit for a new conversation context. Content-free diagnostics record a random call ID, slot, stop reason, frame count, elapsed time and input queue sizes.
- **Privacy:** live transcripts, audio and feature histories are not saved by the server. Diagnostic scripts explicitly write ignored test artifacts. No microphone access before the call button.
- **Access:** the local server binds to loopback and checks host/origin. The private Sites deployment uses a server-side gateway to the authenticated Modal service; GPU credentials never enter browser assets.

## What the neural panel shows

The default view uses **FLM’s full retained MaleCNS v1.0 graph: 166,700 neurons and 25,582,938 directed connections**. Every edge participates in each update. The 3D point cloud displays all 139,662 recorded soma positions; 27,038 cells lack positions but remain in the computation.

Moshi's normalized temporal state (4,096 numbers) is reduced to 64 features, then drives a fixed seeded projection into the full graph. Inference updates the graph once every three audio frames; a fitted 128→4096 readout changes text logits. Snapshots come directly from inference and are aligned with browser playback. No browser-driven simulation runs.

**This mapping is engineered, not a biological correspondence.** The display highlights sharp per-neuron changes as short orange/cyan flashes, then fades back to dim anatomy. It counts new events per update and shows selected neurons' deltas. The change threshold adapts to each neuron's recent changes, with an absolute floor and a two-update cooldown; steady high activity does not keep flashing. These are visual change events, not biological spikes or dopamine. The connectome influences text scores and therefore speech generation; no parameters learn during a call.

Drag to rotate; select a point or use arrow keys to inspect its exact ID and annotations. Scroll/pinch or use the +/− controls to zoom. Selected-neuron details appear beside the neuron inside the graph. Hidden tabs stop drawing, and reduced-motion preferences pause the display. Model computation continues independently. Outside calls the graph is idle. Ending a call clears its state. These controls do not alter Moshi.

Data and provenance: [full-connectome.md](docs/full-connectome.md). Source tables (~1 GB) and derived arrays are verified and stored in ignored `.runtime/`. The older 1,045-neuron motor subset remains only on the legacy Qwen page.

## Verification

```sh
.runtime/moshi-venv/bin/python -m pytest -q tests/test_moshi_server.py tests/test_connectome.py
npm ci
npm run check
npm run test:ui
npm run test:browser
# With the local proxy configured for Modal:
EXPECT_MODAL=1 npm run test:browser
```

Start the local server before browser tests. Browser testing uses the installed Google Chrome, macOS `say`, and FFmpeg to provide prerecorded speech as a fake microphone. `TEST_CHROME_PATH` overrides the Chrome path. The test exercises real Moshi inference, output audio, continuous capture during audible playback, changing model features, visualized activity, cleanup, restart, and mobile layout.

```sh
.runtime/moshi-venv/bin/python scripts/verify_moshi.py --seconds 40
.runtime/moshi-venv/bin/python scripts/benchmark_moshi.py
```

The live API probe measures paced input against output and writes `artifacts/moshi/live-api.wav` and JSON metrics. The offline benchmark loads its own model and measures the sequential path; run it with the server stopped to avoid competing for memory/GPU resources. Test recordings and artifacts are ignored by Git.

### Historical visualizer-only measurements

With all 25.6 million connections active, the earlier visualizer-only 30-second browser test averaged **28.18 ms per graph update** and **57.96 ms per 80 ms Moshi frame**, with zero input backlog at the final checkpoint. All four browser tests passed. The graph state also matched eight sequential updates of the pinned upstream FLM implementation exactly. See [verification details](docs/full-connectome.md#verification-on-this-mac).

### Initial measurements on this Mac

On the M3 Pro (18 GB), a 40-second paced API run completed in 40.07 seconds with 39.92 seconds of generated audio. Model compute averaged about 61 ms per 80 ms frame with no input backlog at the reported checkpoints. Input-frame-to-corresponding-server-frame latency averaged 138 ms (95th percentile 244 ms). These are frame-processing measurements, **not guarantees of response time after a question**, and exclude microphone framing and browser playback buffering. The first sequential implementation averaged 85 ms per frame and was too slow; the live server overlaps codec and inference work.

Performance varies with GPU load and memory pressure. Duplex transport does not guarantee every interruption will produce the desired conversational behavior; that remains model-dependent.

The real Chrome test streamed over 30 seconds of audio with the fly and neural renderer active, averaged 54.28 ms of model compute per frame, and verified 81 microphone frames sent during audible playback. It also verified changing model features, stopping, restarting with fresh state, and the mobile layout. The browser uses its native audio sample rate; the worklet resamples to 24 kHz, and a silent model frame primes new caches before the microphone stream begins.

## Previous cloned-voice version

The original Qwen pipeline is preserved in `server/app.py`, `server/engine.py`, `dist/legacy-app.js`, `dist/legacy.html`, and the original `.venv` environment. Its local voice recordings remain in `voices/eric/` and are ignored by Git.

To run it instead, stop Moshi first, then use:

```sh
./scripts/start-qwen.sh
```

Open **http://localhost:8766/legacy.html**. It still requires Ollama and the previous voice setup. See [the previous pipeline documentation](docs/qwen-voice-legacy.md). Avoid running both models concurrently on this Mac.

## Interface and source layout

The original fly/phone is built with Three.js 0.180.0, vendored with its MIT notice in `dist/vendor/`. Character motion is illustrative. The neural simulator and data are separately attributed above. If WebGL or the neural panel fails, voice calls remain available.

- `server/moshi_engine.py`: pinned model, codec, cache lifecycle and causal reservoir integration.
- `server/moshi_app.py`: bounded duplex transport and serialized workers.
- `dist/app.js`, `dist/duplex-mic-worklet.js`: continuous capture/playback and model telemetry.
- `server/connectome.py`, `scripts/prepare_connectome.py`: full graph, recurrence, provenance and telemetry.
- `dist/full-connectome-panel.js`: 3D full-connectome viewer.
- `dist/neural-panel.js`, `dist/neural-worker.js`, `dist/conversation-drive.js`: legacy motor-circuit viewer.
- `server/model_reservoir.py`, `docs/moshi-reservoir.md`: fitted readout, model hook, causal validation.
- `docs/connectome-reservoir-plan.md`: archived earlier feedback-learning proposal (not active).

### Long-session codec fix

A sustained test exposed the codec's default positional-buffer limit near the end of a five-minute call. Each call now constructs a fresh Mimi tokenizer with `max_seq_len=16384`, instead of relying on `reset()` alone, and primes it before accepting microphone frames. This also avoids carrying codec timing state between calls. The model and PCM session limit remains five minutes. The fixed codec subsequently encoded and decoded 300.64 seconds of audio without the positional-limit error, and the complete browser conversation/reconnect tests passed again.

## Diagnosing an early call ending

The server writes bounded, rotating JSON logs to `.runtime/call-events.jsonl` (2 MB per file, three backups). Records include call ID, server queue depths and processing times, browser input/output frame counts, frame gaps, playback backlog, audio-context state, microphone mute/end state, page visibility and socket close codes. Audio, transcripts, hidden states and feature vectors are not accepted by the diagnostic endpoint.

The browser sends a state sample every five seconds and event records when a call ends, the socket closes, or a device/visibility state changes. Unsent diagnostic events are kept in a bounded local retry buffer (64 events) and retried when the server is available. Abrupt browser/process termination can still prevent a final report; preceding samples remain useful. Call-end messages persist, and show a call ID for correlation. A cause cannot be reconstructed retroactively for calls made before these diagnostics existed.

```
.runtime/moshi-venv/bin/python scripts/diagnose_call.py
.runtime/moshi-venv/bin/python scripts/diagnose_call.py --call-id 01234567
```
