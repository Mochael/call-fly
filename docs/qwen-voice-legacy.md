# Previous Qwen cloned-voice pipeline

A local voice agent with a 3D fly-and-phone interface, using a reference-cloned Qwen voice. Click **Start conversation**, allow the microphone, and speak when Eric is listening. Eric answers out loud and automatically listens again. The button ends the call and releases the microphone. The neural activity panel sits beside the fly on desktop and below it on mobile. Conversation history is hidden.

## The fly interface

An original Three.js model replaces the orb: a charcoal segmented body, red compound eyes, fine bristles, six articulated legs, and one pair of translucent veined wings. Drag the scene to orbit the camera. Starting a call makes the fly reach for the phone; its screen follows the actual connection, listening, thinking, and speaking states. The screen's audio meter samples the real microphone/playback signal. Character motion is illustrative, not a neural simulation.

Visual references: [Fly / Wirehead](https://github.com/mattyhempstead/fly-wirehead) for the darker fly and observation-table direction, and [Fly Typist](https://github.com/tegnike/fly-typist) for the leg-to-device interaction. The character model is original. Three.js 0.180.0 and OrbitControls are vendored in `dist/vendor/`, with the MIT notice in `THREE-LICENSE.txt`, so the interface works without a CDN or build step. The npm dependency pins the matching upstream version.

Reduced-motion preferences disable character movement. If WebGL or the scene module is unavailable, the call button and transcript remain usable.

## Neural activity panel

The separate browser simulation reuses Fly Typist's MaleCNS motor circuit and leaky integrate-and-fire simulator: **1,045 neurons and 17,224 directed connections**, with 880 neurons having measured soma coordinates. It displays those 880 positions and the 350 strongest connections between them. This is a selected motor circuit including the ventral nerve cord, not a complete fly brain. Wiring, synapse counts, and positions come from the anatomical dataset; electrical parameters and conversion of counts to simulated weights are modeling assumptions.

Brightness shows the peak smoothed firing rate during each display interval. Colors identify functional roles; pale yellow rings mark directly stimulated neurons. Hover/click a neuron, or focus the canvas and use arrow keys, to inspect its ID, role, transmitter annotation, and simulated rate. The transmitter label is a dataset annotation, not a measurement of chemical release. Dopamine is not simulated, and no cells in this subset have a dopamine consensus annotation.

The panel follows real conversation signals at 25 updates per second:

- Microphone RMS volume while listening stimulates 25 front-leg sensory cells.
- The actual reply-generation state stimulates 16 descending neurons at current 0.09; preparing the cloned voice uses a lower 0.055 current. These are state indicators, not measurements of model activations.
- RMS measured at the browser playback analyser stimulates 29 front-leg tibia-flexor motor cells while Eric's voice and recorded buzzes play. It does not follow early-arriving text or downloaded audio.
- Audio below a noise floor supplies no stimulus. Ending the call clears the input, and stale inputs expire after 250 ms. The circuit's modeled dynamics determine subsequent firing and decay.

These input assignments are engineered proxies, not biologically identified hearing, speech, or language functions. The selected sensory cells are leg sensors. Conversation signals influence the simulation, but the simulation does not influence the LLM, voice, or character animation. The underlying dataset and relative anatomical weights are unchanged; the simulation's synaptic gain is reduced from 2.4 to 1.2 to avoid persistent activity after input stops in the tested scenarios.

There are no automatic demo pulses. **Test pulse** independently stimulates the selected front-leg motor group, **Pause** freezes time, and **Reset** clears the simulated state (ongoing conversation input can stimulate it again). Reduced-motion preferences start it paused. Computation runs in a worker and pauses in hidden tabs. **What am I seeing?** explains the exact mapping, and selecting a neuron reveals its assigned input, if any.

The vendored simulator is MIT-licensed; the MaleCNS-derived dataset is CC BY 4.0. See [attribution and pinned upstream revision](../dist/neural/ATTRIBUTION.md), [data provenance](../dist/neural/LOCOMOTOR_PROVENANCE.md), and license notices in `dist/neural/`.

## Stack

- Browser: Three.js scene, conversation-driven neural simulation, AudioWorklet microphone capture, automatic speech endpointing, WebSocket, and streamed PCM playback. Transcript elements remain hidden.
- Recognition: faster-whisper `base.en`, int8 on CPU.
- Conversation: Ollama `qwen2.5:1.5b`, streamed, with a short rolling conversation history.
- Voice: `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit`, pinned model revision, MLX-Audio on Apple Silicon. Each phrase is decoded with reference context and sent directly to the browser. There is no output transcription, content validation, or automatic retry.
- Backend: FastAPI, one serialized inference worker and one active call. The conversation pipeline supplies state/audio signals to the browser neural simulation; the circuit does not control the conversation.

Everything runs locally after downloading model weights. No API keys, remote inference, analytics, or transcript storage. The personal voice sample and test artifacts are excluded from Git. Only the fixed opening greeting is cached; answers are generated live.

## Run on this Mac

```sh
./scripts/start-qwen.sh
```

Open **http://localhost:8766/legacy.html**. The page reports startup progress until the models and voice are warm. Keep the server running while calling. Stop it with Ctrl-C.

## Fresh setup

Requires Apple Silicon, Python 3.12 (managed by uv), [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com/download), and FFmpeg. Open Ollama before setup.

```sh
./scripts/setup.sh
mkdir -p voices/eric
ffmpeg -i /path/to/reference.m4a -ar 24000 -ac 1 voices/eric/reference.wav
```

Save the **exact spoken transcript** to `voices/eric/transcript.txt`. This checkout uses the speech-only 15.2–22.25-second excerpt from the supplied recording (“Flies are very cool…” through “…my friends”). `buzz.wav` holds the audible 1.05–1.45-second buzz from the same recording, with playback volume adjusted relative to the speech. The original full recording, transcript, and previous reference are preserved locally; the file in Downloads is untouched. All personal voice files are excluded from Git. Fresh installations without `buzz.wav` use a short synthesized wing-like buzz.

The first run downloads about 1 GB for the LLM plus Qwen weights, the speech codec, and the recognition model. Use the lockfile to reproduce Python dependencies. `npm ci` installs the pinned Three.js package and test tools; the web app itself serves its vendored assets without a build step.

## Behavior and limitations

- This MVP takes turns automatically. It pauses microphone submission while Eric is generating/playing a response to prevent acoustic feedback. It does **not** support spoken barge-in or native full duplex. End conversation stops playback immediately and cancels queued generation.
- Microphone capture requires localhost or HTTPS. This server binds to loopback and is not a public deployment or phone-accessible LAN endpoint. For a remote phone, deploy a secured HTTPS origin and backend with an explicit host/origin allowlist.
- Model startup is slower than warmed calls. Latency depends on memory pressure, reply length, and the reference. Performance figures in `artifacts/` are measurements, not a guarantee.
- The small model is a conversational character, not a reliable source of factual advice.
- The cloned voice is reproduced through reference conditioning, not fine-tuned weights. It may sound different from the hosted Qwen demo.
- The LLM is prompted to use occasional `bzzz`/`zzzz` markers, at most two per reply. Buzz turns are chosen intermittently, never consecutively and never separated by more than two plain replies. If the model omits a requested buzz, a marker is added. Markers remain in the transcript but are removed from Qwen's input and rendered using the short recorded buzz. Internal buzz positions are estimated from the text and moved to nearby quiet audio; no speech-recognition alignment is run.
- Speech recognition is used only for the user's input. Generated speech is played without re-transcribing it or retrying it. The 8-bit model, clean reference, full decoder, and repetition settings remain in place. A complete short phrase still needs to be synthesized before playback; this is not the earlier unstable incremental decoder. Text handling has no added delay; the conversation-history panel is hidden.

### Voice stability

MLX-Audio is pinned to source commit `aef6ebc20ddc775ee62455f383bba141338b7c1c`, which includes the [bounded repetition-history fix](https://github.com/Blaizzy/mlx-audio/pull/914). The installed PyPI wheel previously lacked that fix. The engine uses the pinned private ICL API to apply Qwen's 1.05 repetition penalty rather than the public route's forced 1.5 minimum. Full decoding supplies reference codec context; short fragments and uncontrolled Z spellings are avoided. The greeting cache includes a voice-profile version to invalidate older generated greetings.

## Verification

```sh
.venv/bin/python -m pytest -q
npm run check
npm run test:ui
npm run test:browser
```

Start the local server first and run `npm ci` once. Browser fixtures are generated automatically using macOS `say` and FFmpeg; no microphone hardware is used by the automated tests.

The browser tests use Chromium's fake microphone with **real prerecorded speech**, then exercise real STT, the LLM, and cloned TTS through the production WebSocket. They do not replace model responses with mocks. They also verify the neural circuit's response to stimulation, pause/reset, neuron inspection, actual microphone/generation/playback stimulation, stop-call cleanup, reduced-motion behavior, and mobile layout. A separate API test suite covers malformed recordings, connection exclusivity, origin checks, and session cleanup with a small fake engine.

```sh
.venv/bin/python scripts/probe_voice.py
.venv/bin/python scripts/verify_conversation.py
.venv/bin/python scripts/stress_voice.py --passes 2
```

The probes write ignored audio/JSON results to `artifacts/`. `TEST_CHROME_PATH` overrides the default installed Google Chrome binary used by Playwright.

### Measured on this Mac

On the M3 Pro with 18 GB RAM, the current no-checks configuration generated eight test replies in **2.11–4.90 seconds** before first audio (voice-only). The previous checked configuration took 2.56–5.36 seconds for shorter test phrases and 5.25–9.75 seconds for complete conversations. These are different stochastic runs, not a controlled benchmark. Longer replies, startup, and GPU contention add delay; removing checks does not remove the full-phrase synthesis time. Add about 0.65 seconds for microphone endpoint detection and 0.08 seconds for playback buffering to server timings.

Regression tests ensure speech is synthesized once without calling recognition, alongside cancellation, audible buzz placement, sparse buzz text, transport, UI, and real Chrome conversation tests. The optional offline stress script transcribes saved test WAVs for comparison after generation; that diagnostic is not part of the live playback path. WAVs and JSON reports are saved under ignored `artifacts/`.

After adding the neural panel, all four browser tests passed. That real conversation produced 10.88 seconds of audio, with first server audio at 11.46 seconds. This confirms that longer replies can still have noticeable turn-taking latency even without output checks.

The fixed opening greeting plays immediately from a local cache. Conversation history lasts for the current call and clears when you start a new one. This is a working local MVP; physical microphone quality, room noise, and other apps using the GPU can affect results.

## Layout

`dist/` contains authored static UI files; `server/` contains inference and transport; `scripts/` contains setup and real model probes. `.openai/hosting.json` describes the static frontend only. Publishing those static files alone would not deploy the local speech backend.
